"""Run the CTPA extraction over a real export, one row at a time, appending each finished row to the output file.

    uv run run_reports.py --input "C:\\data\\ctpa_export.csv" --limit 20     (pilot: the first 20 rows)
    uv run run_reports.py --input "C:\\data\\ctpa_export.csv"                (every row)

For every row, in the order of the input:
  1. gemma4:26b reads the report (grouped calls).
  2. gpt-oss:20b reads the same report when gemma found an embolism, when the report never states whether there is one,
     or when gemma failed (--second-opinion embolism, the default). "all" sends every report to both models (about four
     times longer), "none" uses gemma alone. Where both read a report and disagree, gpt-oss's answer is kept and the row
     is flagged.
  3. The finished row is appended to the output at once and written to disk: every original column, then row_number,
     extraction_id, needs_review (1 = a person should read it: the two models disagree on a finding, a model failed, or
     the row has no report text), resolved_by, models, disputed_fields, then mentioned, present and the quote for each of
     the 64 findings, then seconds.
Stopping is safe at any time (Ctrl+C or closing the window): every finished row is already in the file, and running the
same command again continues with the next row. Nothing else is stored, no JSON files.

The output is <input name>_extracted.csv next to the input, UTF-8 with a byte-order mark so Excel opens it directly.
The input may be comma, tab, semicolon or pipe separated (guessed from the header line), UTF-8 or Windows-1252, with the
report text in --text-column (default ReportBody). Identical report texts are read once and the answer reused.
Only the Ollama on this machine is used; cloud model tags and other hosts are refused.
"""

import argparse
import csv
import hashlib
import math
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from ollama import Client, ResponseError
from tqdm import tqdm

import schema_ctpa as schema
from evaluate import is_present
from extract import build_row, call_model_grouped, code_version, is_cloud_model, split_model_tag
from resolve import resolve_reports

DELIMITERS = [",", "\t", ";", "|"]
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
FIELD_COLUMNS = [column for column in schema.FLAT_COLUMNS]          # <finding>_mentioned, _present, _evidence
REVIEW_COLUMNS = ["row_number", "extraction_id", "needs_review", "resolved_by", "models", "disputed_fields", "code_version"]


def read_export(path: Path, text_column: str) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        sys.exit("ERROR: this is an Excel file. In Excel use File > Save As > CSV UTF-8, then pass the .csv file.")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            with path.open(encoding=encoding) as handle:
                header = handle.readline()
            delimiter = max(DELIMITERS, key=header.count)
            table = pd.read_csv(path, sep=delimiter, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        sys.exit(f"ERROR: {path} is neither UTF-8 nor Windows-1252 text.")
    if text_column not in table.columns:
        sys.exit(f"ERROR: no column named {text_column!r} in {path}. Columns found: {', '.join(table.columns)}")
    print(f"Read {len(table)} rows from {path.name} ({encoding}, separator {delimiter!r}, {len(table.columns)} columns)")
    return table


def extraction_id(text: str) -> str:
    """A hash of the report text: identifies the report without putting an accession number or MRN anywhere."""
    return "R" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12] if text else ""


def cell(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def finished_rows(output: Path, columns: list) -> dict:
    """row_number -> output record for every complete row already in the output. A last row cut off by a hard stop is
    removed so that it is done again."""
    if not output.exists():
        return {}
    with output.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.reader(handle))
    if not records:
        return {}
    if records[0] != columns:
        sys.exit(f"ERROR: {output} was made from another input or by an older version of this script, so its rows cannot be "
                 "continued. Rename or delete it to start a fresh output, or pass --output with a new file name.")
    complete = [record for record in records[1:] if len(record) == len(columns)]
    if len(complete) != len(records) - 1:
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(complete)
        print(f"Removed {len(records) - 1 - len(complete)} incomplete row(s) left by a hard stop; they will be done again.")
    position = {name: index for index, name in enumerate(columns)}
    return {int(record[position["row_number"]]): dict(zip(columns, record)) for record in complete}


def hold_lock(output: Path):
    """Refuse to start when another run is writing the same output (two windows would interleave rows). The lock belongs
    to this process and ends with it, so a crash or a closed window never leaves a stale lock behind."""
    handle = output.with_name(output.name + ".lock").open("a+")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"ERROR: another run is already writing {output.name}. Let it finish, or stop it (Ctrl+C in its window) first.")
    return handle


def release_lock(handle):
    path = Path(handle.name)
    handle.close()
    try:
        path.unlink()
    except OSError:
        pass   # another run has just taken it over


def append_row(output: Path, record: list):
    """Append one row and force it to disk. Excel on Windows locks a CSV it has open: wait for it to be closed."""
    warned = False
    while True:
        try:
            with output.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(record)
                handle.flush()
                os.fsync(handle.fileno())
            return
        except PermissionError:
            if not warned:
                tqdm.write(f"{output.name} is open in another program (Excel?). Close it; the run waits and then continues.")
                warned = True
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="CTPA extraction over a real export, one finished row appended at a time.")
    parser.add_argument("--input", type=Path, required=True, help="the export: one report per row, all columns are kept")
    parser.add_argument("--text-column", default="ReportBody", help="column with the full report text (default ReportBody)")
    parser.add_argument("--output", type=Path, default=None, help="default: <input name>_extracted.csv next to the input")
    parser.add_argument("--limit", type=int, default=None, help="stop after the first N rows of the input (a pilot)")
    parser.add_argument("--model", default="gemma4:26b", help="model that reads every report (default gemma4:26b)")
    parser.add_argument("--second-model", default="gpt-oss:20b", help="model for the second opinion (default gpt-oss:20b)")
    parser.add_argument("--second-opinion", choices=["embolism", "all", "none"], default="embolism",
                        help="embolism (default): second model on reports where the first found an embolism, the report never "
                             "states whether there is one, or the first failed; all: every report; none: never")
    parser.add_argument("--host", default="http://localhost:11434")
    args = parser.parse_args()

    models = [args.model] + ([] if args.second_opinion == "none" else [args.second_model])
    for model in models:
        if is_cloud_model(model):
            sys.exit(f"ERROR: {model} is a cloud model. Real reports never leave this machine; use a local model tag.")
    if urlparse(args.host).hostname not in LOCAL_HOSTS:
        sys.exit(f"ERROR: {args.host} is not this machine. Real reports only go to the Ollama running locally.")
    client = Client(host=args.host)
    for model in models:
        try:
            client.show(split_model_tag(model)[0])
        except ResponseError as error:
            sys.exit(f"ERROR: model {model} is not in this Ollama ({error.error}). Run: ollama pull {model}")
        except Exception as error:
            sys.exit(f"ERROR: cannot reach Ollama at {args.host}: {error}")

    table = read_export(args.input, args.text_column)
    if args.limit is not None:
        table = table.head(args.limit)
    output = args.output or args.input.with_name(args.input.stem + "_extracted.csv")
    lock = hold_lock(output)
    columns = list(table.columns) + REVIEW_COLUMNS + FIELD_COLUMNS + ["seconds"]
    done = finished_rows(output, columns)
    answers = {}   # extraction_id -> the appended columns, reused for identical report texts
    for row_number, record in done.items():
        if row_number > len(table):
            continue   # done in an earlier run without --limit
        original = table.iloc[row_number - 1]
        if extraction_id(original[args.text_column].strip()) != record["extraction_id"]:
            sys.exit(f"ERROR: row {row_number} of {output.name} does not match row {row_number} of {args.input.name}: the input "
                     "changed since this output was started. Pass --output with a new file name to start a fresh output.")
        if record["extraction_id"] and record["resolved_by"] not in ("no_valid_output", "empty_report"):
            answers[record["extraction_id"]] = {name: record[name] for name in REVIEW_COLUMNS[2:] + FIELD_COLUMNS}
    if not output.exists():
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            csv.writer(handle).writerow(columns)
    version = code_version()
    todo = [number for number in range(1, len(table) + 1) if number not in done]
    print(f"{output.name}: {len(table) - len(todo)} rows already done, {len(todo)} to go. Models: {' + '.join(models)} "
          f"(second opinion: {args.second_opinion}). Stop any time with Ctrl+C; the same command continues.", flush=True)

    progress = tqdm(total=len(table), initial=len(table) - len(todo), unit="row", dynamic_ncols=True)
    flagged = 0
    try:
        for row_number in todo:
            original = table.iloc[row_number - 1]
            text = original[args.text_column].strip()
            key = extraction_id(text)
            started = time.perf_counter()
            if not key:
                appended = {"needs_review": 1, "resolved_by": "empty_report", "models": "", "disputed_fields": ""}
                note = "no report text"
            elif key in answers:
                appended = answers[key]
                note = "same text as an earlier row, answer reused"
            else:
                raw = call_model_grouped(client, args.model, text, schema)
                rows = [build_row(key, args.model, 1, text, raw, schema)]
                first = rows[0]
                ask_second = args.second_opinion == "all" or (args.second_opinion == "embolism" and (
                    first["valid_json"] != 1 or is_present(first.get("pulmonary_embolism_present"))
                    or not is_present(first.get("pulmonary_embolism_mentioned"))))
                used = [args.model]
                if ask_second:
                    raw_second = call_model_grouped(client, args.second_model, text, schema)
                    rows.append(build_row(key, args.second_model, 1, text, raw_second, schema))
                    used.append(args.second_model)
                # where both models read the report, the second model's answer is kept when they disagree (it was right on
                # every disagreement in the first 20 real reports, and 48/50 against gemma's 47/50 on the test set)
                priority = used[::-1]
                final = resolve_reports(pd.DataFrame(rows), pd.DataFrame([{"report_id": key}]), priority, schema).iloc[0]
                appended = {"needs_review": final["needs_review"], "resolved_by": final["resolved_by"],
                            "models": " + ".join(used), "disputed_fields": final["disputed_fields"]}
                appended.update({name: final.get(name) for name in FIELD_COLUMNS})
                if final["resolved_by"] not in ("no_valid_output",):
                    answers[key] = appended
                if final["resolved_by"] == "no_valid_output":
                    note = "no valid answer from " + " or ".join(used)
                else:
                    note = "embolism" if is_present(final.get("pulmonary_embolism_present")) else "no embolism"
                    note += (f", {len(used)} models, {final['resolved_by']}" if len(used) > 1 else "")
                note += " -> REVIEW" if int(final["needs_review"]) else ""
            values = {**{name: original[name] for name in table.columns}, "row_number": row_number, "extraction_id": key,
                      **appended, "code_version": version, "seconds": round(time.perf_counter() - started, 1)}
            append_row(output, [cell(values.get(name)) for name in columns])
            flagged += int(values["needs_review"])
            progress.update(1)
            tqdm.write(f"row {row_number}: {note} ({values['seconds']:.0f} s)")
    except KeyboardInterrupt:
        progress.close()
        release_lock(lock)
        print(f"\nStopped. {progress.n} of {len(table)} rows are in {output}. Run the same command to continue.")
        return
    progress.close()
    release_lock(lock)
    print(f"\nDone: {len(table)} rows in {output}; {flagged} of the rows done in this session need review.")


if __name__ == "__main__":
    main()
