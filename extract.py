"""Run Ollama models over chest CT reports and write one CSV row per (report_id, model, run).

Every raw model response is saved to raw/{model}_{report_id}_{run}.json (model tag sanitized for
Windows file names). Existing raw files are reused unless --force is given, so interrupted runs resume.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from ollama import Client, ResponseError
from pydantic import ValidationError

from schema import FINDING_NAMES, FollowUp, LargestNodule, ReportExtraction, build_messages

OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192}
MAX_RETRIES = 3
RAW_DIR = Path("raw")

# Model families run WITHOUT constrained format (the JSON schema goes into the prompt instead).
# On gpt-oss the schema passed as format= was silently not enforced (see README), so it is prompt-only.
PROMPT_ONLY_JSON_MODEL_PREFIXES = ("gpt-oss",)

SECTION_HEADER = re.compile(r"^[ \t]*(FINDINGS|IMPRESSION)[ \t]*(:|$)", re.MULTILINE | re.IGNORECASE)
CODE_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def thinking_for(model: str):
    """Thinking is fixed per model family and deliberately not configurable from the CLI."""
    name = model.lower()
    if name.startswith("gpt-oss"):
        return "medium"  # never "high": it overflows memory on the 16 GB target machine and never finishes
    if re.search(r"gemma[-_]?4", name):
        return False
    return None


def uses_constrained_format(model: str) -> bool:
    return not model.lower().startswith(PROMPT_ONLY_JSON_MODEL_PREFIXES)


def is_cloud_model(model: str) -> bool:
    return model.endswith("-cloud") or model.endswith(":cloud")


def raw_file(model: str, report_id: str, run: int) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9.-]+", "_", model)
    return RAW_DIR / f"{safe_model}_{report_id}_{run}.json"


def strip_code_fences(text: str) -> str:
    match = CODE_FENCE.search(text)
    return match.group(1) if match else text.strip()


def compact_validation_error(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "(root)"
        lines.append(f"- {location}: {item['msg']}")
    return "\n".join(lines)


def call_model(client: Client, model: str, report_text: str) -> dict:
    """Call the model, validate, retry on validation errors, and return everything worth saving."""
    think = thinking_for(model)
    constrained = uses_constrained_format(model)
    response_format = ReportExtraction.model_json_schema() if constrained else None
    messages = build_messages(report_text, include_schema=not constrained)
    attempts = []
    extraction = None
    started = time.perf_counter()
    for attempt in range(1, MAX_RETRIES + 2):
        attempt_started = time.perf_counter()
        response = client.chat(model=model, messages=messages, format=response_format, think=think, options=OPTIONS)
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
        try:
            extraction = ReportExtraction.model_validate_json(strip_code_fences(content))
            attempts.append(record)
            break
        except ValidationError as error:
            record["validation_error"] = compact_validation_error(error)
            attempts.append(record)
            messages = messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": "Your JSON failed validation:\n" + record["validation_error"]
                 + "\nReturn the complete corrected JSON object only."},
            ]
    return {
        "model": model,
        "think": think,
        "constrained_format": constrained,
        "options": OPTIONS,
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


def result_columns() -> list[str]:
    columns = ["report_id", "model", "run", "valid_json", "attempts", "latency_s"]
    for name in FINDING_NAMES:
        columns += [f"{name}_status", f"{name}_evidence", f"{name}_evidence_ok", f"{name}_section"]
    columns.append("nodule_count")
    columns += [f"largest_nodule_{field}" for field in LargestNodule.model_fields]
    columns += ["largest_nodule_evidence_ok", "largest_nodule_section"]
    columns += [f"follow_up_{field}" for field in FollowUp.model_fields]
    columns += ["follow_up_evidence_ok", "follow_up_section"]
    return columns


def build_row(report_id: str, model: str, run: int, report_text: str, raw: dict) -> dict:
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
    extraction = ReportExtraction.model_validate(raw["extraction"])
    normalized_report, offsets = normalize_with_offsets(report_text)
    for name in FINDING_NAMES:
        observation = getattr(extraction, name)
        evidence_ok, section = check_evidence(observation.evidence, report_text, normalized_report, offsets)
        row[f"{name}_status"] = observation.status
        row[f"{name}_evidence"] = observation.evidence
        row[f"{name}_evidence_ok"] = evidence_ok
        row[f"{name}_section"] = section
    row["nodule_count"] = extraction.nodule_count
    for prefix, part in (("largest_nodule", extraction.largest_nodule), ("follow_up", extraction.follow_up)):
        for field, value in part.model_dump().items():
            row[f"{prefix}_{field}"] = value
        evidence_ok, section = check_evidence(part.evidence, report_text, normalized_report, offsets)
        row[f"{prefix}_evidence_ok"] = evidence_ok
        row[f"{prefix}_section"] = section
    return row


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama models on structured extraction from chest CT reports.")
    parser.add_argument("--input", type=Path, default=Path("reports.csv"), help="CSV with report_id, report_text")
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument("--models", nargs="+", required=True, help="Ollama model tags, e.g. gpt-oss:20b gemma4:12b")
    parser.add_argument("--runs", type=int, default=1, help="repeat each report N times to measure determinism")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--limit", type=int, default=None, help="only the first N reports")
    parser.add_argument("--force", action="store_true", help="re-run even if the raw file exists")
    parser.add_argument("--allow-cloud", action="store_true", help="permit *-cloud / *:cloud model tags (synthetic data only)")
    args = parser.parse_args()

    cloud_models = [model for model in args.models if is_cloud_model(model)]
    if cloud_models and not args.allow_cloud:
        sys.exit(
            f"ERROR: {', '.join(cloud_models)} is a cloud model: the report text would be sent to Ollama's servers.\n"
            "Real reports must never reach a cloud model. For synthetic test data only, re-run with --allow-cloud."
        )

    reports = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    if args.limit is not None:
        reports = reports.head(args.limit)

    client = Client(host=args.host)
    for model in args.models:
        try:
            client.show(model)
        except ResponseError as error:
            sys.exit(f"ERROR: model {model!r} is not available at {args.host} ({error.error}). Run: ollama pull {model}")
        except Exception as error:
            sys.exit(f"ERROR: cannot reach Ollama at {args.host}: {error}")

    RAW_DIR.mkdir(exist_ok=True)
    rows = []
    for model in args.models:
        print(f"== {model}: think={thinking_for(model)!r}, constrained_format={uses_constrained_format(model)}")
        for report_number, report in enumerate(reports.itertuples(index=False), start=1):
            for run in range(1, args.runs + 1):
                path = raw_file(model, report.report_id, run)
                if path.exists() and not args.force:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    note = "resumed from raw file"
                else:
                    raw = call_model(client, model, report.report_text)
                    path.write_text(json.dumps(raw, indent=1), encoding="utf-8")
                    note = ""
                row = build_row(report.report_id, model, run, report.report_text, raw)
                rows.append(row)
                print(
                    f"[{model}] {report_number}/{len(reports)} {report.report_id} run {run}: "
                    f"valid_json={row['valid_json']} attempts={row['attempts']} latency={row['latency_s']:.1f}s {note}"
                )
                pd.DataFrame(rows, columns=result_columns()).to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
