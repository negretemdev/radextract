"""Run the CTPA extraction over a real export, one report per row, and write it back with the answers appended.

    uv run run_reports.py --input "C:\\data\\ctpa_export.csv" --limit 20     (pilot: the first 20 rows)
    uv run run_reports.py --input "C:\\data\\ctpa_export.csv"                (every row; re-running resumes)

Step 1  gemma4:26b, grouped calls, on every report.
Step 2  gpt-oss:20b, grouped calls, only on the reports where gemma found an embolism, did not plainly rule one out, or
        failed. A report where the two disagree is flagged for review; on the 50 test reports this pair flagged 5 and
        every wrong report was among them. --second none skips this step.
Output  <input name>_extracted.csv next to the input: every original column in the original order, then extraction_id,
        <field>_mentioned, <field>_present and <field>_evidence for the 64 fields, resolved_by, needs_review and
        disputed_fields. Written after step 1 and again after step 2, so stopping during step 2 still leaves gemma's
        answers. UTF-8 with a byte-order mark, so Excel opens it directly.
Reading the input: comma, tab, semicolon or pipe separated (guessed from the header line), UTF-8 or Windows-1252, the
        report text in --text-column (default ReportBody). Identical report texts are extracted once.
Privacy everything goes through the Ollama on this machine: cloud model tags and non-local hosts are refused. The working
        files in local_data/ and the raw answers in raw/ contain report text; they stay on this machine (git ignores both).
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

import schema_ctpa as schema
from evaluate import is_present
from extract import is_cloud_model
from resolve import resolve_reports

DELIMITERS = [",", "\t", ";", "|"]
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


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
    """The key of a report in the working files and raw answers: a hash of its text, so re-sorted or re-exported
    files resume correctly and no accession number or MRN appears in a file name."""
    return "R" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12] if text else ""


def main():
    parser = argparse.ArgumentParser(description="CTPA extraction over a real export, answers appended to every row.")
    parser.add_argument("--input", type=Path, required=True, help="the export: one report per row, all columns are kept")
    parser.add_argument("--text-column", default="ReportBody", help="column holding the full report text (default ReportBody)")
    parser.add_argument("--output", type=Path, default=None, help="default: <input name>_extracted.csv next to the input")
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows (a pilot)")
    parser.add_argument("--primary", default="gemma4:26b", help="model for every report (default gemma4:26b)")
    parser.add_argument("--second", default="gpt-oss:20b", help="second opinion on the uncertain reports, or none (default gpt-oss:20b)")
    parser.add_argument("--host", default="http://localhost:11434")
    args = parser.parse_args()

    models = [args.primary] + ([] if args.second == "none" else [args.second])
    for model in models:
        if is_cloud_model(model):
            sys.exit(f"ERROR: {model} is a cloud model. Real reports never leave this machine; use a local model tag.")
    if urlparse(args.host).hostname not in LOCAL_HOSTS:
        sys.exit(f"ERROR: {args.host} is not this machine. Real reports only go to the Ollama running locally.")

    table = read_export(args.input, args.text_column)
    if args.limit is not None:
        table = table.head(args.limit)
    table["report_id"] = table[args.text_column].map(lambda text: extraction_id(text.strip()))
    reports = (table.loc[table["report_id"] != "", ["report_id", args.text_column]]
               .drop_duplicates("report_id").rename(columns={args.text_column: "report_text"}))
    reports["report_text"] = reports["report_text"].str.strip()
    empty = int((table["report_id"] == "").sum())
    print(f"{len(reports)} distinct reports to extract; {len(table) - len(reports) - empty} duplicate texts reuse an answer; "
          f"{empty} rows have no report text")

    work = Path("local_data") / args.input.stem
    work.mkdir(parents=True, exist_ok=True)
    prepared = work / "reports.csv"
    reports.to_csv(prepared, index=False, encoding="utf-8")
    output = args.output or args.input.with_name(args.input.stem + "_extracted.csv")

    def extract(model: str, results_path: Path, ids_path: Path = None):
        command = [sys.executable, "extract.py", "--schema", "ctpa", "--grouped", "--models", model,
                   "--input", str(prepared), "--output", str(results_path), "--host", args.host]
        if ids_path is not None:
            command += ["--ids", str(ids_path)]
        if subprocess.run(command).returncode != 0:
            sys.exit(f"ERROR: {model} stopped. Re-run the same command; finished reports are resumed.")

    def write_output(result_files: list, used_models: list):
        results = pd.concat([pd.read_csv(path, dtype={"report_id": str}) for path in result_files], ignore_index=True)
        final = resolve_reports(results, table, used_models, schema)
        final.loc[final["report_id"] == "", "resolved_by"] = "empty_report"
        final = final.rename(columns={"report_id": "extraction_id"})
        final.to_csv(output, index=False, encoding="utf-8-sig")
        print(f"\nWrote {output}: {len(final)} rows | needs_review {int(final['needs_review'].sum())} | "
              f"{final['resolved_by'].value_counts().to_dict()}", flush=True)

    print(f"\nStep 1: {args.primary} on all {len(reports)} reports", flush=True)
    primary_results = work / "results_primary.csv"
    extract(args.primary, primary_results)
    write_output([primary_results], [args.primary])
    if args.second == "none":
        return

    first = pd.read_csv(primary_results, dtype={"report_id": str})
    uncertain = ((first["valid_json"] != 1) | first["pulmonary_embolism_present"].map(is_present)
                 | ~first["pulmonary_embolism_mentioned"].map(is_present))
    selected = first.loc[uncertain, ["report_id"]]
    print(f"\nStep 2: {args.second} on {len(selected)} of {len(first)} reports (embolism found, not plainly ruled out, "
          "or step 1 failed)", flush=True)
    if selected.empty:
        return
    ids = work / "second_opinion_ids.csv"
    selected.to_csv(ids, index=False)
    second_results = work / "results_second.csv"
    extract(args.second, second_results, ids)
    write_output([primary_results, second_results], models)


if __name__ == "__main__":
    main()
