"""Summarize the review flags of a run_reports.py output without showing any report text or quote.

    uv run flag_summary.py --input "C:\\data\\combined_extracted.csv"

Prints counts only: rows, rows with an embolism, rows flagged for review and why, and for every field on which the two
models disagreed, how many reports, which model said present, and the row numbers (positions in your own file, so you
can look them up there). Safe to paste: it contains field names, numbers and model names, nothing from the reports.
Works on the file as written by run_reports.py and after Excel has re-saved it (TRUE/FALSE).
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

import schema_ctpa as schema

EMBOLISM_FIELDS = [field.removesuffix("_present") for field in schema.KEY_FIELDS]
VOTE = re.compile(r"(?:: |; )([A-Za-z0-9._-]+:[A-Za-z0-9._-]+)=(True|False) \(")   # "gemma4:26b=True (" after ": " or "; "


def parse_disputes(text: str) -> dict:
    """field -> {model: said present} from a disputed_fields cell; field names anchor the parsing, quotes are ignored."""
    starts = sorted((match.start(), name) for name in schema.FINDING_NAMES
                    for match in re.finditer(r"(?:^| \| )" + re.escape(name) + r": (?=[A-Za-z0-9._-]+:[A-Za-z0-9._-]+=(?:True|False) \()", text))
    disputes = {}
    for index, (position, name) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(text)
        segment = text[position:end]
        disputes[name] = {model: value == "True" for model, value in VOTE.findall(segment)}
    return disputes


def main():
    parser = argparse.ArgumentParser(description="Counts of review flags and model disagreements, no report text.")
    parser.add_argument("--input", type=Path, required=True, help="the _extracted.csv written by run_reports.py")
    args = parser.parse_args()

    table = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for column in ["row_number", "needs_review", "resolved_by", "disputed_fields", "pulmonary_embolism_present"]:
        if column not in table.columns:
            raise SystemExit(f"ERROR: {args.input.name} has no column {column!r}; pass the _extracted.csv written by run_reports.py")
    flagged = table[table["needs_review"].str.strip() == "1"]
    embolism = table["pulmonary_embolism_present"].str.strip().str.lower() == "true"
    print(f"Rows: {len(table)} | with an embolism: {int(embolism.sum())} | flagged for review: {len(flagged)} "
          f"(with an embolism: {int((embolism & (table['needs_review'].str.strip() == '1')).sum())})")
    print("resolved_by: " + ", ".join(f"{name} {count}" for name, count in table["resolved_by"].value_counts().items()))
    if "models" in table.columns:
        print("models: " + ", ".join(f"{name or 'none'} {count}" for name, count in table["models"].value_counts().items()))

    by_field = defaultdict(list)                          # field -> [(row_number, {model: present})]
    only_other = []
    for row in flagged.itertuples(index=False):
        disputes = parse_disputes(row.disputed_fields)
        for field, votes in disputes.items():
            by_field[field].append((row.row_number, votes))
        if disputes and not any(field in EMBOLISM_FIELDS for field in disputes):
            only_other.append(row.row_number)
    no_dispute = [row.row_number for row in flagged.itertuples(index=False) if not row.disputed_fields.strip()]

    for title, fields in [("Embolism fields", EMBOLISM_FIELDS), ("Other findings", [f for f in schema.FINDING_NAMES if f not in EMBOLISM_FIELDS])]:
        rows = sorted((field for field in by_field if field in fields), key=lambda field: -len(by_field[field]))
        print(f"\n{title}: disagreements in {sum(len(by_field[field]) for field in rows)} field(s) across "
              f"{len({number for field in rows for number, _ in by_field[field]})} report(s)")
        for field in rows:
            patterns = defaultdict(int)
            for _, votes in by_field[field]:
                patterns[", ".join(f"{model} {'yes' if present else 'no'}" for model, present in votes.items())] += 1
            numbers = ", ".join(number for number, _ in by_field[field])
            print(f"  {field:34s} {len(by_field[field]):3d}  " + " | ".join(f"{pattern}: {count}" for pattern, count in patterns.items())
                  + f"  | rows {numbers}")
    print(f"\nFlagged reports whose disagreements are all outside the embolism fields: {len(only_other)}"
          + (f" (rows {', '.join(only_other)})" if only_other else ""))
    print(f"Flagged for another reason (no valid answer, or no report text): {len(no_dispute)}"
          + (f" (rows {', '.join(no_dispute)})" if no_dispute else ""))


if __name__ == "__main__":
    main()
