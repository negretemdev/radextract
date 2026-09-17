"""Merge two or three models into one final CSV: one row per report with every column of the input CSV
(accession numbers, MRNs, anything else you had) followed by the resolved extraction.

Per scored field the value is the majority across the models that produced valid output for that report
(presence only: asserted or not). With two models that disagree and no third vote the primary model's
value is kept and the report is flagged. Columns added at the end:
  resolved_by    agreement | tiebreaker | unresolved (the worst case over the report's fields)
  needs_review   1 when any field is unresolved, or a deciding third-model value has a quote that is not verbatim
  disputed_fields  the fields that were disputed, with each model's value and quote
"""

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

from evaluate import SCHEMAS, is_present


def main():
    parser = argparse.ArgumentParser(description="Resolve two or three models into one final CSV per report.")
    parser.add_argument("--results", type=Path, default=Path("results.csv"))
    parser.add_argument("--schema", choices=SCHEMAS, default="ctpa")
    parser.add_argument("--input", type=Path, default=None, help="reports CSV whose columns are all kept (default: the schema's reports file)")
    parser.add_argument("--models", nargs="+", default=None, help="models in priority order, first is primary; third and later are tiebreakers (default: order in the file)")
    parser.add_argument("--output", type=Path, default=None, help="default: final_<schema>.csv")
    args = parser.parse_args()

    schema = importlib.import_module(SCHEMAS[args.schema])
    fields = schema.SCORED_FIELDS
    results = pd.read_csv(args.results, dtype={"report_id": str})
    results = results[results["run"] == 1]
    models = args.models or list(dict.fromkeys(results["model"]))
    reports = pd.read_csv(args.input or Path(schema.REPORTS_FILE), dtype=str, keep_default_na=False)
    rows_by_model = {model: results[results["model"] == model].set_index("report_id") for model in models}
    extra_columns = [column for column in schema.FLAT_COLUMNS if column not in fields]

    final_rows = []
    for report in reports.itertuples(index=False):
        report_id = report.report_id
        row = report._asdict()
        voters = [model for model in models if report_id in rows_by_model[model].index and rows_by_model[model].at[report_id, "valid_json"] == 1]
        resolved_by = "agreement"
        needs_review = 0
        disputes = []
        if not voters:
            row["resolved_by"] = "no_valid_output"
            row["needs_review"] = 1
            row["disputed_fields"] = ""
            final_rows.append(row)
            continue
        source_for_field = {}
        for field in fields:
            votes = {model: is_present(rows_by_model[model].at[report_id, field]) for model in voters}
            yes = [model for model, vote in votes.items() if vote]
            no = [model for model, vote in votes.items() if not vote]
            if not yes or not no:
                winner = voters[0]
            elif len(voters) >= 3:
                winners = yes if len(yes) > len(no) else no
                winner = winners[0]
                deciders = [model for model in winners if model not in voters[:2]]
                resolved_by = "tiebreaker" if resolved_by == "agreement" else resolved_by
                for decider in deciders:
                    evidence_ok = rows_by_model[decider].at[report_id, field.replace("_present", "_evidence_ok")]
                    if str(evidence_ok) == "0":
                        needs_review = 1
            else:
                winner = voters[0]
                resolved_by = "unresolved"
                needs_review = 1
            if yes and no:
                name = field.removesuffix("_present")
                described = "; ".join(f"{model}={rows_by_model[model].at[report_id, field]} ({rows_by_model[model].at[report_id, name + '_evidence']})" for model in voters)
                disputes.append(f"{name}: {described}")
            source_for_field[field] = winner
            row[field] = rows_by_model[winner].at[report_id, field]
        for column in extra_columns:
            owner = source_for_field.get(column.rsplit("_", 1)[0] + "_present")
            if owner is None:  # nodule_count, largest_nodule_*, follow_up_* in the chest_ct schema
                owner = voters[0]
            if column in rows_by_model[owner].columns:
                row[column] = rows_by_model[owner].at[report_id, column]
        row["resolved_by"] = resolved_by
        row["needs_review"] = needs_review
        row["disputed_fields"] = " | ".join(disputes)
        final_rows.append(row)

    columns = list(reports.columns) + [column for column in schema.FLAT_COLUMNS if not column.endswith("_evidence_ok")] + ["resolved_by", "needs_review", "disputed_fields"]
    final = pd.DataFrame(final_rows)
    final = final[[column for column in columns if column in final.columns]]
    output = args.output or Path(f"final_{args.schema}.csv")
    final.to_csv(output, index=False, encoding="utf-8")
    print(f"{len(final)} reports | resolved_by: {final['resolved_by'].value_counts().to_dict()} | needs_review: {int(final['needs_review'].sum())}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
