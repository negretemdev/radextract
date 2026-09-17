"""Run Ollama models over chest CT reports and write one CSV row per (report_id, model, run).

--schema picks the extraction schema: chest_ct (schema.py: 3-state status per finding plus nodule and
follow-up details) or binary (schema_binary.py: true/false per finding). Every raw model response is
saved to raw/{schema}/{model}_{report_id}_{run}.json (model tag sanitized for Windows file names).
Existing raw files are reused unless --force is given, so interrupted runs resume.
"""

import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from ollama import Client, ResponseError
from pydantic import ValidationError

SCHEMAS = {"chest_ct": "schema", "binary": "schema_binary", "ctpa": "schema_ctpa"}
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192}
THINKING_NUM_CTX = 16384  # @think variants only: gemma4:26b reasons for 4-8k tokens per report. 32k put the whole model on the CPU. gpt-oss stays at 8192.
MAX_RETRIES = 3
TRANSIENT_RETRIES = 5        # server-side failures (overloaded, 5xx, connection reset) are retried after a pause
TRANSIENT_PAUSE_S = 20

# Model families run WITHOUT constrained format (the JSON schema goes into the prompt instead).
# gpt-oss is prompt-only because format= was silently not enforced at think="medium" (see README).
# Cloud models are always prompt-only: Ollama cloud ignored format= on both gpt-oss and gemma 4.
PROMPT_ONLY_JSON_MODEL_PREFIXES = ("gpt-oss",)

SECTION_HEADER = re.compile(r"^[ \t]*(FINDINGS|IMPRESSION)[ \t]*(:|$)", re.MULTILINE | re.IGNORECASE)
CODE_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


THINK_SUFFIX = "@think"


def split_model_tag(model: str):
    """'gemma4:26b@think' -> ('gemma4:26b', True). The label with the suffix is kept in results and raw file names."""
    if model.endswith(THINK_SUFFIX):
        return model[: -len(THINK_SUFFIX)], True
    return model, False


def thinking_for(model: str):
    """Thinking is fixed per model family; there is no CLI flag. gpt-oss is always "medium" and cannot be
    overridden (high overflows memory on the 16 GB target machine and never finishes). gemma 4 runs with
    thinking off unless the tag carries the @think suffix, which is an explicit per-model opt-in."""
    tag, think_requested = split_model_tag(model)
    name = tag.lower()
    if name.startswith("gpt-oss"):
        return "medium"
    if re.search(r"gemma[-_]?4|qwen3", name):
        return think_requested  # thinking off unless @think; Ollama would otherwise enable qwen3 thinking by default
    return True if think_requested else None


def options_for(model: str) -> dict:
    """Sampling options per model. Only the context size varies: thinking variants get THINKING_NUM_CTX."""
    options = dict(OPTIONS)
    _, think_requested = split_model_tag(model)
    if think_requested:
        options["num_ctx"] = THINKING_NUM_CTX
    return options


def is_cloud_model(model: str) -> bool:
    tag, _ = split_model_tag(model)
    return tag.endswith("-cloud") or tag.endswith(":cloud")


def uses_constrained_format(model: str) -> bool:
    if is_cloud_model(model):
        return False
    tag, _ = split_model_tag(model)
    return not tag.lower().startswith(PROMPT_ONLY_JSON_MODEL_PREFIXES)


def raw_file(raw_dir: Path, model: str, report_id: str, run: int) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9.-]+", "_", model)
    return raw_dir / f"{safe_model}_{report_id}_{run}.json"


def strip_code_fences(text: str) -> str:
    match = CODE_FENCE.search(text)
    return match.group(1) if match else text.strip()


def compact_validation_error(error) -> str:
    if not isinstance(error, ValidationError):
        return f"- (root): {error}"
    lines = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "(root)"
        lines.append(f"- {location}: {item['msg']}")
    return "\n".join(lines)


def parse_and_validate(schema, content: str):
    """Parse the model output, apply the schema's repairs when it defines normalize(), validate.
    Returns (extraction, repairs); raises ValueError for invalid JSON and ValidationError for schema violations."""
    try:
        data = json.loads(strip_code_fences(content))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error}")
    repairs = {}
    if hasattr(schema, "normalize"):
        data, repairs = schema.normalize(data)
    return schema.ReportExtraction.model_validate(data), repairs


def reparse(raw: dict, schema) -> dict:
    """Re-run parsing, repairs and validation on the stored attempts of a raw file; no model call.
    The first attempt that validates wins. The attempt count stays what it was at run time."""
    raw["valid_json"] = False
    raw["extraction"] = None
    for attempt in raw["attempts"]:
        try:
            extraction, repairs = parse_and_validate(schema, attempt["content"])
        except (ValidationError, ValueError) as error:
            attempt["validation_error"] = compact_validation_error(error)
            continue
        attempt["validation_error"] = None
        attempt["repairs"] = repairs
        raw["valid_json"] = True
        raw["extraction"] = extraction.model_dump()
        break
    raw["reparsed"] = True
    return raw


def chat_with_transient_retries(client: Client, tag: str, messages, response_format, think, options):
    """One chat call, retried after a pause when the server (not the model) fails: overloaded, 5xx, connection errors."""
    for transient_attempt in range(1, TRANSIENT_RETRIES + 1):
        try:
            return client.chat(model=tag, messages=messages, format=response_format, think=think, options=options)
        except ResponseError as error:
            transient = error.status_code is not None and (error.status_code >= 500 or error.status_code == 429)
            if not transient or transient_attempt == TRANSIENT_RETRIES:
                raise
            print(f"   server error {error.status_code} ({str(error.error)[:60]}), retrying in {TRANSIENT_PAUSE_S}s", flush=True)
        except (ConnectionError, TimeoutError) as error:
            if transient_attempt == TRANSIENT_RETRIES:
                raise
            print(f"   connection error ({str(error)[:60]}), retrying in {TRANSIENT_PAUSE_S}s", flush=True)
        time.sleep(TRANSIENT_PAUSE_S)


def call_model(client: Client, model: str, report_text: str, schema) -> dict:
    """Call the model, validate, retry on validation errors, and return everything worth saving."""
    tag, _ = split_model_tag(model)
    think = thinking_for(model)
    options = options_for(model)
    constrained = uses_constrained_format(model)
    response_format = schema.ReportExtraction.model_json_schema() if constrained else None
    messages = schema.build_messages(report_text, include_schema=not constrained)
    attempts = []
    extraction = None
    started = time.perf_counter()
    for attempt in range(1, MAX_RETRIES + 2):
        attempt_started = time.perf_counter()
        response = chat_with_transient_retries(client, tag, messages, response_format, think, options)
        content = response.message.content or ""
        record = {
            "attempt": attempt,
            "content": content,
            "thinking": response.message.thinking,
            "latency_s": time.perf_counter() - attempt_started,
            "prompt_eval_count": response.prompt_eval_count,
            "eval_count": response.eval_count,
            "done_reason": response.done_reason,
            "validation_error": None,
        }
        if response.done_reason == "length":
            record["validation_error"] = (f"response cut off by num_ctx={options['num_ctx']} (done_reason=length) after "
                                          f"{len(response.message.thinking or '')} characters of thinking; JSON incomplete or missing")
            attempts.append(record)
            messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": "Your previous response ran out of space while reasoning and the JSON was cut off. "
                 "Keep the reasoning brief this time and return the complete JSON object only."},
            ]
            continue
        try:
            extraction, record["repairs"] = parse_and_validate(schema, content)
            attempts.append(record)
            break
        except (ValidationError, ValueError) as error:
            record["validation_error"] = compact_validation_error(error)
            attempts.append(record)
            messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": "Your JSON failed validation:\n" + record["validation_error"]
                 + "\nReturn the complete corrected JSON object only."},
            ]
    return {
        "schema": schema.__name__,
        "model": model,
        "think": think,
        "constrained_format": constrained,
        "options": options,
        "valid_json": extraction is not None,
        "attempts": attempts,
        "latency_s": time.perf_counter() - started,
        "extraction": extraction.model_dump() if extraction is not None else None,
    }


def normalize_with_offsets(text: str):
    """Lowercase and collapse whitespace. Also return, per normalized character, its offset in the original text."""
    normalized = []
    offsets = []
    for index, char in enumerate(text):
        if char.isspace():
            if normalized and normalized[-1] != " ":
                normalized.append(" ")
                offsets.append(index)
        else:
            lowered = char.lower()
            normalized.append(lowered)
            offsets.extend([index] * len(lowered))
    if normalized and normalized[-1] == " ":
        normalized.pop()
        offsets.pop()
    return "".join(normalized), offsets


def section_at(offset: int, report_text: str) -> str:
    """Section of the report (by its own headers) that contains the character offset."""
    section = "other"
    for match in SECTION_HEADER.finditer(report_text):
        if offset >= match.start():
            section = match.group(1).lower()
    return section


def check_evidence(evidence, report_text: str, normalized_report: str, offsets: list):
    """Return (evidence_ok, section). Both are empty when there is no evidence to check."""
    if evidence is None:
        return "", ""
    normalized_evidence, _ = normalize_with_offsets(evidence)
    position = normalized_report.find(normalized_evidence) if normalized_evidence else -1
    if position == -1:
        return 0, ""
    return 1, section_at(offsets[position], report_text)


def result_columns(schema) -> list[str]:
    columns = ["report_id", "model", "run", "valid_json", "attempts", "latency_s"]
    for column in schema.FLAT_COLUMNS:
        columns.append(column)
        if column.endswith("_evidence"):
            columns += [column + "_ok", column.removesuffix("_evidence") + "_section"]
    return columns


def build_row(report_id: str, model: str, run: int, report_text: str, raw: dict, schema) -> dict:
    row = {
        "report_id": report_id,
        "model": model,
        "run": run,
        "valid_json": int(raw["valid_json"]),
        "attempts": len(raw["attempts"]),
        "latency_s": round(raw["latency_s"], 2),
    }
    if not raw["valid_json"]:
        return row
    extraction = schema.ReportExtraction.model_validate(raw["extraction"])
    normalized_report, offsets = normalize_with_offsets(report_text)
    for column, value in schema.flatten(extraction).items():
        row[column] = value
        if column.endswith("_evidence"):
            evidence_ok, section = check_evidence(value, report_text, normalized_report, offsets)
            row[column + "_ok"] = evidence_ok
            row[column.removesuffix("_evidence") + "_section"] = section
    return row


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama models on structured extraction from chest CT reports.")
    parser.add_argument("--input", type=Path, default=None, help="CSV with report_id, report_text (default: the schema's reports file)")
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument("--models", nargs="+", required=True, help="Ollama model tags, e.g. gpt-oss:20b gemma4:26b")
    parser.add_argument("--schema", choices=SCHEMAS, default="chest_ct")
    parser.add_argument("--ids", type=Path, default=None, help="CSV with a report_id column: run only those reports")
    parser.add_argument("--runs", type=int, default=1, help="repeat each report N times to measure determinism")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--limit", type=int, default=None, help="only the first N reports")
    parser.add_argument("--force", action="store_true", help="re-run even if the raw file exists")
    parser.add_argument("--reparse", action="store_true", help="re-validate existing raw files from their stored attempts (no model calls)")
    parser.add_argument("--allow-cloud", action="store_true", help="permit *-cloud / *:cloud model tags (synthetic data only)")
    args = parser.parse_args()

    cloud_models = [model for model in args.models if is_cloud_model(model)]
    if cloud_models and not args.allow_cloud:
        sys.exit(
            f"ERROR: {', '.join(cloud_models)} is a cloud model: the report text would be sent to Ollama's servers.\n"
            "Real reports must never reach a cloud model. For synthetic test data only, re-run with --allow-cloud."
        )

    schema = importlib.import_module(SCHEMAS[args.schema])
    input_path = args.input or Path(schema.REPORTS_FILE)
    reports = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    if args.ids is not None:
        wanted = pd.read_csv(args.ids, dtype=str, keep_default_na=False)["report_id"]
        reports = reports[reports["report_id"].isin(wanted)]
        if reports.empty:
            sys.exit(f"ERROR: none of the report_ids in {args.ids} are in {input_path}")
    if args.limit is not None:
        reports = reports.head(args.limit)

    client = Client(host=args.host)
    for model in args.models:
        tag, think_requested = split_model_tag(model)
        if think_requested and tag.lower().startswith("gpt-oss"):
            sys.exit(f"ERROR: {model!r}: gpt-oss thinking is fixed at medium on this machine and cannot be changed.")
        try:
            client.show(tag)
        except ResponseError as error:
            sys.exit(f"ERROR: model {tag!r} is not available at {args.host} ({error.error}). Run: ollama pull {tag}")
        except Exception as error:
            sys.exit(f"ERROR: cannot reach Ollama at {args.host}: {error}")

    raw_dir = Path("raw") / getattr(schema, "RAW_DIR_NAME", args.schema)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in args.models:
        print(f"== {model}: schema={args.schema}, think={thinking_for(model)!r}, num_ctx={options_for(model)['num_ctx']}, "
              f"constrained_format={uses_constrained_format(model)}", flush=True)
        for report_number, report in enumerate(reports.itertuples(index=False), start=1):
            for run in range(1, args.runs + 1):
                path = raw_file(raw_dir, model, report.report_id, run)
                if path.exists() and not args.force:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    note = "resumed from raw file"
                    if args.reparse:
                        raw = reparse(raw, schema)
                        path.write_text(json.dumps(raw, indent=1), encoding="utf-8")
                        note = "reparsed from raw file"
                else:
                    raw = call_model(client, model, report.report_text, schema)
                    path.write_text(json.dumps(raw, indent=1), encoding="utf-8")
                    note = ""
                row = build_row(report.report_id, model, run, report.report_text, raw, schema)
                rows.append(row)
                print(
                    f"[{model}] {report_number}/{len(reports)} {report.report_id} run {run}: "
                    f"valid_json={row['valid_json']} attempts={row['attempts']} latency={row['latency_s']:.1f}s {note}",
                    flush=True,
                )
                pd.DataFrame(rows, columns=result_columns(schema)).to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
