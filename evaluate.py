"""Score results.csv against ground_truth.csv and write summary.csv.

summary.csv has one row per metric and one column per model. Rows named accuracy_<field> also carry
agreement_between_models (fraction of reports where every model gave the identical value on run 1).
With --runs > 1, identical_runs_rate compares the scored fields across runs and
identical_runs_rate_with_evidence additionally requires identical evidence quotes.
Evidence text is not compared with the ground truth: it is scored by the evidence_ok check in extract.py.
"""

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

SCHEMAS = {"chest_ct": "schema", "binary": "schema_binary", "ctpa": "schema_ctpa"}
NUMERIC_FIELDS = {"nodule_count", "largest_nodule_size_mm", "follow_up_interval_months"}


def is_present(value) -> bool:
    """True only for an asserted finding: True/"True" (binary and ctpa schemas) or "present" (chest_ct)."""
    return value is True or str(value).strip() in ("True", "present")


def values_match(field: str, predicted, truth) -> bool:
    if pd.isna(predicted) and pd.isna(truth):
        return True
    if pd.isna(predicted) or pd.isna(truth):
        return False
    if field in NUMERIC_FIELDS:
        return abs(float(predicted) - float(truth)) < 1e-6
    return str(predicted).strip() == str(truth).strip()


def main():
    parser = argparse.ArgumentParser(description="Score extraction results against the ground truth.")
    parser.add_argument("--results", type=Path, default=Path("results.csv"))
    parser.add_argument("--schema", choices=SCHEMAS, default="chest_ct")
    parser.add_argument("--ground-truth", type=Path, default=None, help="defaults to the schema's ground truth file")
    parser.add_argument("--output", type=Path, default=Path("summary.csv"))
    args = parser.parse_args()
    # accuracy_<field> is an exact match (true/false/null or the status string); presence_<field> only asks
    # whether the finding was asserted, so false versus null (negated versus not mentioned) does not count.

    schema = importlib.import_module(SCHEMAS[args.schema])
    ground_truth_path = args.ground_truth or Path(schema.GROUND_TRUTH_FILE)
    results = pd.read_csv(args.results, dtype={"report_id": str})
    truth = pd.read_csv(ground_truth_path, dtype={"report_id": str}).set_index("report_id")
    if results.empty:
        sys.exit(f"ERROR: {args.results} has no rows")
    missing = sorted(set(results["report_id"]) - set(truth.index))
    if missing:
        sys.exit(f"ERROR: no ground truth for report_id(s): {', '.join(missing)}")

    fields = schema.SCORED_FIELDS
    models = list(dict.fromkeys(results["model"]))
    number_of_runs = int(results["run"].max())

    correct = pd.DataFrame(index=results.index)
    for field in fields:
        correct[field] = [
            values_match(field, results.at[index, field], truth.at[results.at[index, "report_id"], field])
            for index in results.index
        ]
    correct.loc[results["valid_json"] == 0, :] = False
    presence_correct = pd.DataFrame(index=results.index)
    for field in fields:
        presence_correct[field] = [
            is_present(results.at[index, field]) == is_present(truth.at[results.at[index, "report_id"], field])
            for index in results.index
        ]
    presence_correct.loc[results["valid_json"] == 0, :] = False

    evidence_ok_columns = [column for column in results.columns if column.endswith("_evidence_ok")]
    evidence_columns = [column for column in results.columns if column.endswith("_evidence")]
    summary = {}
    for model in models:
        rows = results[results["model"] == model]
        model_correct = correct.loc[rows.index]
        summary.setdefault("reports", {})[model] = rows["report_id"].nunique()
        summary.setdefault("rows", {})[model] = len(rows)
        summary.setdefault("valid_json_rate", {})[model] = rows["valid_json"].mean()
        summary.setdefault("mean_attempts", {})[model] = rows["attempts"].mean()
        summary.setdefault("mean_latency_s", {})[model] = rows["latency_s"].mean()
        evidence_checks = pd.concat([rows[column] for column in evidence_ok_columns]).dropna()
        summary.setdefault("evidence_ok_rate", {})[model] = evidence_checks.mean() if len(evidence_checks) else float("nan")
        if number_of_runs > 1:
            identical_values = []
            identical_with_evidence = []
            for _, report_rows in rows.groupby("report_id"):
                identical_values.append((report_rows[fields].astype(str).nunique(dropna=False) == 1).all())
                identical_with_evidence.append((report_rows[fields + evidence_columns].astype(str).nunique(dropna=False) == 1).all())
            summary.setdefault("identical_runs_rate", {})[model] = pd.Series(identical_values).mean()
            summary.setdefault("identical_runs_rate_with_evidence", {})[model] = pd.Series(identical_with_evidence).mean()
        summary.setdefault("accuracy_overall", {})[model] = model_correct[fields].to_numpy().mean()
        summary.setdefault("presence_accuracy_overall", {})[model] = presence_correct.loc[rows.index, fields].to_numpy().mean()
        key_fields = getattr(schema, "KEY_FIELDS", None)
        if key_fields:
            key_correct = presence_correct.loc[rows.index, key_fields]
            summary.setdefault("key_fields_presence_accuracy", {})[model] = key_correct.to_numpy().mean()
            summary.setdefault("reports_all_key_fields_correct", {})[model] = key_correct.all(axis=1).mean()
            key_finding = getattr(schema, "KEY_FINDING", None)
            if key_finding:
                positive = [index for index in rows.index if is_present(truth.at[results.at[index, "report_id"], key_finding])]
                if positive:
                    summary.setdefault("positive_reports_all_key_fields_correct", {})[model] = key_correct.loc[positive].all(axis=1).mean()
        for field in fields:
            summary.setdefault(f"accuracy_{field}", {})[model] = model_correct[field].mean()
        for field in fields:
            summary.setdefault(f"presence_{field}", {})[model] = presence_correct.loc[rows.index, field].mean()
        for field in getattr(schema, "SECONDARY_FIELDS", []):
            matches = [values_match(field, results.at[index, field], truth.at[results.at[index, "report_id"], field]) and results.at[index, "valid_json"] == 1 for index in rows.index]
            summary.setdefault(f"secondary_{field}", {})[model] = pd.Series(matches).mean()

    agreement = {}
    if len(models) > 1:
        first_run = results[results["run"] == 1]
        models_per_report = first_run.groupby("report_id")["model"].nunique()
        complete_ids = models_per_report[models_per_report == len(models)].index
        first_run = first_run[first_run["report_id"].isin(complete_ids)]
        for field in fields:
            pivot = first_run.pivot(index="report_id", columns="model", values=field)
            agreement[field] = (pivot.astype(str).nunique(axis=1, dropna=False) == 1).mean()

    secondary = [key for key in summary if key.startswith("secondary_")]
    if secondary:
        summary["secondary_accuracy_overall"] = {model: pd.Series([summary[key][model] for key in secondary]).mean() for model in models}
    summary_rows = []
    for metric, per_model in summary.items():
        row = {"metric": metric, **per_model}
        field = metric.removeprefix("accuracy_")
        if field in agreement:
            row["agreement_between_models"] = agreement[field]
        summary_rows.append(row)
    columns = ["metric"] + models + (["agreement_between_models"] if agreement else [])
    summary_table = pd.DataFrame(summary_rows, columns=columns)
    summary_table.to_csv(args.output, index=False, encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(summary_table.round(3).to_string(index=False))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
