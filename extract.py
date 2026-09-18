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
from tqdm import tqdm

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


def parse_and_validate(schema, content: str, field: str = None, report_text: str = ""):
    """Parse the model output, apply the schema's repairs when it defines normalize(), validate.
    Returns (extraction, repairs); raises ValueError for invalid JSON and ValidationError for schema violations.
    In question mode (field given) the answer is one Finding and only the per-finding repairs apply."""
    try:
        data = json.loads(strip_code_fences(content))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {error}")
    repairs = {}
    if field is not None:
        if hasattr(schema, "normalize_finding"):
            data, repairs = schema.normalize_finding(data)
        return schema.Finding.model_validate(data), repairs
    if hasattr(schema, "normalize"):
        data, repairs = schema.normalize(data, report_text)
    return schema.ReportExtraction.model_validate(data), repairs


def reparse_grouped(raw: dict, schema, report_text: str = "") -> dict:
    """Replay a grouped raw file from its stored attempts: each group's first valid attempt, then the cross-field repairs."""
    data = {}
    valid = True
    for group in schema.CALL_GROUPS:
        group_attempts = [attempt for attempt in raw["attempts"] if attempt.get("group") == group["name"]]
        if not group_attempts:
            for field in group["fields"]:
                data[field] = {"mentioned": False, "present": False, "evidence": None}
            continue
        model_class = schema.group_model(group["name"])
        answer = None
        for attempt in group_attempts:
            try:
                parsed = json.loads(strip_code_fences(attempt["content"]))
                parsed, attempt["repairs"] = schema.normalize_group(parsed)
                answer = model_class.model_validate(parsed).model_dump()
                attempt["validation_error"] = None
                break
            except (json.JSONDecodeError, ValidationError) as error:
                attempt["validation_error"] = compact_validation_error(error) if isinstance(error, ValidationError) else f"- (root): invalid JSON: {error}"
        if answer is None:
            valid = False
            for field in group["fields"]:
                data[field] = {"mentioned": False, "present": False, "evidence": None}
            continue
        data.update(answer)
    raw["extraction"] = None
    raw["repairs"] = {}
    if valid:
        data, raw["repairs"] = schema.normalize(data, report_text)
        try:
            raw["extraction"] = schema.ReportExtraction.model_validate(data).model_dump()
        except ValidationError:
            valid = False
    raw["valid_json"] = valid
    raw["reparsed"] = True
    return raw


def reparse(raw: dict, schema, report_text: str = "") -> dict:
    """Re-run parsing, repairs and validation on the stored attempts of a raw file; no model call.
    The first attempt that validates wins. The attempt count stays what it was at run time."""
    if raw.get("grouped"):
        return reparse_grouped(raw, schema, report_text)
    raw["valid_json"] = False
    raw["extraction"] = None
    for attempt in raw["attempts"]:
        try:
            extraction, repairs = parse_and_validate(schema, attempt["content"], raw.get("field"), report_text)
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


def call_model(client: Client, model: str, report_text: str, schema, field: str = None) -> dict:
    """Call the model, validate, retry on validation errors, and return everything worth saving.
    With a field name only that one finding is asked (question mode) and the answer is a single Finding."""
    tag, _ = split_model_tag(model)
    think = thinking_for(model)
    options = options_for(model)
    constrained = uses_constrained_format(model)
    if field is None:
        response_format = schema.ReportExtraction.model_json_schema() if constrained else None
        messages = schema.build_messages(report_text, include_schema=not constrained)
    else:
        response_format = schema.Finding.model_json_schema() if constrained else None
        messages = schema.question_messages(report_text, field, include_schema=not constrained)
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
            extraction, record["repairs"] = parse_and_validate(schema, content, field, report_text)
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
        "field": field,
        "think": think,
        "constrained_format": constrained,
        "options": options,
        "valid_json": extraction is not None,
        "attempts": attempts,
        "latency_s": time.perf_counter() - started,
        "extraction": extraction.model_dump() if extraction is not None else None,
    }


def call_model_grouped(client: Client, model: str, report_text: str, schema) -> dict:
    """Several short calls per report (schema.CALL_GROUPS): a group whose condition is not met is filled as not mentioned
    without a call. Each group is validated and retried on its own; the merged answer gets the schema's cross-field
    repairs and the full validation."""
    tag, _ = split_model_tag(model)
    think = thinking_for(model)
    options = options_for(model)
    constrained = uses_constrained_format(model)
    data = {}
    attempts = []
    calls = []
    valid = True
    started = time.perf_counter()
    for group in schema.CALL_GROUPS:
        if group["when"] is not None and not group["when"](data):
            for field in group["fields"]:
                data[field] = {"mentioned": False, "present": False, "evidence": None}
            calls.append({"group": group["name"], "skipped": True})
            continue
        model_class = schema.group_model(group["name"])
        response_format = model_class.model_json_schema() if constrained else None
        messages = schema.group_messages(report_text, group["name"], include_schema=not constrained)
        answer = None
        for attempt in range(1, MAX_RETRIES + 2):
            attempt_started = time.perf_counter()
            response = chat_with_transient_retries(client, tag, messages, response_format, think, options)
            content = response.message.content or ""
            record = {
                "group": group["name"],
                "attempt": attempt,
                "content": content,
                "thinking": response.message.thinking,
                "latency_s": time.perf_counter() - attempt_started,
                "prompt_eval_count": response.prompt_eval_count,
                "eval_count": response.eval_count,
                "done_reason": response.done_reason,
                "validation_error": None,
            }
            try:
                parsed = json.loads(strip_code_fences(content))
                parsed, record["repairs"] = schema.normalize_group(parsed)
                answer = model_class.model_validate(parsed).model_dump()
                attempts.append(record)
                break
            except (json.JSONDecodeError, ValidationError) as error:
                record["validation_error"] = compact_validation_error(error) if isinstance(error, ValidationError) else f"- (root): invalid JSON: {error}"
                attempts.append(record)
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Your JSON failed validation:\n" + record["validation_error"]
                     + "\nReturn the complete corrected JSON object only."},
                ]
        calls.append({"group": group["name"], "skipped": False, "valid": answer is not None})
        if answer is None:
            valid = False
            for field in group["fields"]:
                data[field] = {"mentioned": False, "present": False, "evidence": None}
            continue
        data.update(answer)
    repairs = {}
    extraction = None
    if valid:
        data, repairs = schema.normalize(data, report_text)
        try:
            extraction = schema.ReportExtraction.model_validate(data)
        except ValidationError:
            valid = False
    return {
        "schema": schema.__name__,
        "model": model,
        "grouped": True,
        "think": think,
        "constrained_format": constrained,
        "options": options,
        "valid_json": valid,
        "attempts": attempts,
        "calls": calls,
        "repairs": repairs,
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


def describe(raw: dict) -> str:
    """One line per report for the console: calls made, retries, failed and skipped groups."""
    calls = len(raw["attempts"])
    retries = sum(1 for attempt in raw["attempts"] if attempt["validation_error"])
    if raw.get("grouped"):
        failed = [call["group"] for call in raw["calls"] if not call.get("skipped") and not call.get("valid")]
        skipped = [call["group"] for call in raw["calls"] if call.get("skipped")]
        status = "FAILED " + ",".join(failed) if failed else "ok"
        return f"calls={calls} retries={retries} {status} {raw['latency_s']:.0f}s" + (f" | skipped: {', '.join(skipped)}" if skipped else "")
    status = "ok" if raw["valid_json"] else "FAILED"
    return f"calls={calls} retries={retries} {status} {raw['latency_s']:.0f}s"


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


def build_question_row(report_id: str, model: str, run: int, report_text: str, raws: dict, schema) -> dict:
    """One row from several single-field answers (question mode); fields that were not asked stay empty."""
    row = {
        "report_id": report_id,
        "model": model,
        "run": run,
        "valid_json": int(all(raw["valid_json"] for raw in raws.values())),
        "attempts": sum(len(raw["attempts"]) for raw in raws.values()),
        "latency_s": round(sum(raw["latency_s"] for raw in raws.values()), 2),
    }
    normalized_report, offsets = normalize_with_offsets(report_text)
    for field, raw in raws.items():
        if not raw["valid_json"]:
            continue
        finding = schema.Finding.model_validate(raw["extraction"])
        for key, value in finding.model_dump().items():
            row[f"{field}_{key}"] = value
        evidence_ok, section = check_evidence(finding.evidence, report_text, normalized_report, offsets)
        row[f"{field}_evidence_ok"] = evidence_ok
        row[f"{field}_section"] = section
    return row


def run_questions(client: Client, model: str, reports, questions: dict, raw_dir: Path, schema, args) -> list[dict]:
    """Question mode: for each report, ask only the listed fields, one call each, and build one row per report."""
    safe_model = re.sub(r"[^A-Za-z0-9.-]+", "_", model)
    rows = []
    progress = tqdm(total=len(reports) * args.runs, desc=model, unit="report", dynamic_ncols=True)
    for report_number, report in enumerate(reports.itertuples(index=False), start=1):
        for run in range(1, args.runs + 1):
            raws = {}
            fresh = 0
            for field in questions[report.report_id]:
                path = raw_dir / f"{safe_model}_{report.report_id}_{run}_q_{field}.json"
                raw = None
                if path.exists() and not args.force:
                    stored = json.loads(path.read_text(encoding="utf-8"))
                    if args.reparse:
                        stored = reparse(stored, schema, report.report_text)
                        path.write_text(json.dumps(stored, indent=1), encoding="utf-8")
                    if stored.get("field") == field:
                        raw = stored
                if raw is None:
                    raw = call_model(client, model, report.report_text, schema, field)
                    path.write_text(json.dumps(raw, indent=1), encoding="utf-8")
                    fresh += 1
                raws[field] = raw
            row = build_question_row(report.report_id, model, run, report.report_text, raws, schema)
            rows.append(row)
            retries = sum(1 for raw in raws.values() for attempt in raw["attempts"] if attempt["validation_error"])
            failed = [field for field, raw in raws.items() if not raw["valid_json"]]
            tqdm.write(f"[{model}] {report.report_id} run {run}: {len(raws)} questions ({fresh} asked now) retries={retries} "
                       + ("FAILED " + ",".join(failed) if failed else "ok") + f" {row['latency_s']:.0f}s")
            progress.update(1)
    progress.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama models on structured extraction from chest CT reports.")
    parser.add_argument("--input", type=Path, default=None, help="CSV with report_id, report_text (default: the schema's reports file)")
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument("--models", nargs="+", required=True, help="Ollama model tags, e.g. gpt-oss:20b gemma4:26b")
    parser.add_argument("--schema", choices=SCHEMAS, default="chest_ct")
    parser.add_argument("--ids", type=Path, default=None, help="CSV with a report_id column: run only those reports")
    parser.add_argument("--questions", type=Path, default=None,
                        help="CSV with report_id and disputed_fields (from compare.py): ask only those fields, one call per field")
    parser.add_argument("--runs", type=int, default=1, help="repeat each report N times to measure determinism")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--limit", type=int, default=None, help="only the first N reports")
    parser.add_argument("--force", action="store_true", help="re-run even if the raw file exists")
    parser.add_argument("--reparse", action="store_true", help="re-validate existing raw files from their stored attempts (no model calls)")
    parser.add_argument("--grouped", action="store_true", help="several short calls per report (schema.CALL_GROUPS) instead of one call")
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
    questions = None
    if args.questions is not None:
        asked = pd.read_csv(args.questions, dtype=str, keep_default_na=False)
        asked = asked[asked["disputed_fields"] != ""]
        questions = {report_id: [field.removesuffix("_present") for field in fields.split(";")]
                     for report_id, fields in zip(asked["report_id"], asked["disputed_fields"])}
        reports = reports[reports["report_id"].isin(questions)]
        if reports.empty:
            sys.exit(f"ERROR: no report in {args.questions} has disputed fields")
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
    if args.grouped and not hasattr(schema, "CALL_GROUPS"):
        sys.exit(f"ERROR: schema {args.schema} does not define CALL_GROUPS; --grouped is not available for it")
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in args.models:
        print(f"== {model}: schema={args.schema}, think={thinking_for(model)!r}, num_ctx={options_for(model)['num_ctx']}, "
              f"constrained_format={uses_constrained_format(model)}" + (", question mode" if questions else "") + (", grouped calls" if args.grouped else ""), flush=True)
        if questions is not None:
            rows += run_questions(client, model, reports, questions, raw_dir, schema, args)
            pd.DataFrame(rows, columns=result_columns(schema)).to_csv(args.output, index=False, encoding="utf-8")
            continue
        progress = tqdm(total=len(reports) * args.runs, desc=model, unit="report", dynamic_ncols=True)
        for report_number, report in enumerate(reports.itertuples(index=False), start=1):
            for run in range(1, args.runs + 1):
                path = raw_file(raw_dir, model, report.report_id, run)
                if args.grouped:
                    path = path.with_name(path.stem + "_grouped.json")
                raw = None
                note = ""
                if path.exists() and not args.force:
                    stored = json.loads(path.read_text(encoding="utf-8"))
                    if args.reparse:
                        stored = reparse(stored, schema, report.report_text)
                        path.write_text(json.dumps(stored, indent=1), encoding="utf-8")
                    try:
                        row = build_row(report.report_id, model, run, report.report_text, stored, schema)
                        raw = stored
                        note = "reparsed from raw file" if args.reparse else "resumed from raw file"
                    except ValidationError:
                        note = "(raw file was from an older schema, re-run)"
                if raw is None:
                    if args.grouped:
                        raw = call_model_grouped(client, model, report.report_text, schema)
                    else:
                        raw = call_model(client, model, report.report_text, schema)
                    path.write_text(json.dumps(raw, indent=1), encoding="utf-8")
                    row = build_row(report.report_id, model, run, report.report_text, raw, schema)
                rows.append(row)
                tqdm.write(f"[{model}] {report.report_id} run {run}: {describe(raw)} {note}")
                progress.update(1)
                pd.DataFrame(rows, columns=result_columns(schema)).to_csv(args.output, index=False, encoding="utf-8")
        progress.close()
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
