"""Compare two models field by field and write disputed.csv.

Agreement is judged on presence: a finding counts as asserted when its value is true (or "present" in
the chest_ct schema), so false versus null (negated versus not mentioned) is not a dispute. Those
false-versus-null differences are counted separately. Evidence quotes are ignored on purpose: they vary
between calls even at temperature 0. With --ground-truth the accuracy is also split by agreement.
"""

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

from evaluate import SCHEMAS, is_present, values_match


def main():
    parser = argparse.ArgumentParser(description="Field-by-field disagreement between two models.")
    parser.add_argument("--results", type=Path, default=Path("results.csv"))
    parser.add_argument("--models", nargs=2, default=None, help="the two models to compare (default: first two in the file)")
    parser.add_argument("--schema", choices=SCHEMAS, default="chest_ct")
    parser.add_argument("--ground-truth", type=Path, default=None, help="optional: split accuracy by agreement")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("disputed.csv"))
    args = parser.parse_args()

    schema = importlib.import_module(SCHEMAS[args.schema])
    fields = schema.SCORED_FIELDS
    results = pd.read_csv(args.results, dtype={"report_id": str})
    results = results[results["run"] == args.run]
    models = args.models or list(dict.fromkeys(results["model"]))[:2]
    if len(models) != 2:
        sys.exit("ERROR: need two models to compare")
    first = results[results["model"] == models[0]].set_index("report_id")
    second = results[results["model"] == models[1]].set_index("report_id")
    report_ids = [report_id for report_id in first.index if report_id in second.index]
    if not report_ids:
        sys.exit(f"ERROR: no report_id has rows for both {models[0]} and {models[1]}")
    truth = None
    if args.ground_truth is not None:
        truth = pd.read_csv(args.ground_truth, dtype={"report_id": str}).set_index("report_id")

    secondary_fields = getattr(schema, "SECONDARY_FIELDS", [])
    secondary_total = 0
    disputed_rows = []
    disputes_per_field = {field: 0 for field in fields}
    false_vs_null_total = 0
    agreed_total = agreed_correct = 0
    first_right = second_right = 0
    errors_total = {models[0]: 0, models[1]: 0}
    errors_caught = {models[0]: 0, models[1]: 0}
    for report_id in report_ids:
        disputed_fields = []
        false_vs_null_fields = []
        details = []
        for field in fields:
            value_first = first.at[report_id, field]
            value_second = second.at[report_id, field]
            presence_same = is_present(value_first) == is_present(value_second)
            if presence_same and not values_match(field, value_first, value_second):
                false_vs_null_fields.append(field)
                false_vs_null_total += 1
            if truth is not None:
                truth_present = is_present(truth.at[report_id, field])
                for model, value in ((models[0], value_first), (models[1], value_second)):
                    if is_present(value) != truth_present:
                        errors_total[model] += 1
                        errors_caught[model] += not presence_same
            if presence_same:
                if truth is not None:
                    agreed_total += 1
                    agreed_correct += is_present(value_first) == is_present(truth.at[report_id, field])
                continue
            disputed_fields.append(field)
            disputes_per_field[field] += 1
            details.append(f"{field}: {models[0]}={value_first} | {models[1]}={value_second}")
            if truth is not None:
                if is_present(value_first) == is_present(truth.at[report_id, field]):
                    first_right += 1
                else:
                    second_right += 1
        for field in secondary_fields:
            if not values_match(field, first.at[report_id, field], second.at[report_id, field]):
                secondary_total += 1
                false_vs_null_fields.append(field)
        if disputed_fields or false_vs_null_fields:
            disputed_rows.append({
                "report_id": report_id,
                "n_disputed": len(disputed_fields),
                "disputed_fields": ";".join(disputed_fields),
                "false_vs_null_fields": ";".join(false_vs_null_fields),
                "details": "; ".join(details),
            })

    columns = ["report_id", "n_disputed", "disputed_fields", "false_vs_null_fields", "details"]
    disputed = pd.DataFrame(disputed_rows, columns=columns)
    disputed.to_csv(args.output, index=False, encoding="utf-8")
    total_fields = len(report_ids) * len(fields)
    total_disputed = int(disputed["n_disputed"].sum()) if len(disputed) else 0
    reports_to_review = int((disputed["n_disputed"] > 0).sum()) if len(disputed) else 0
    print(f"{models[0]} vs {models[1]} on {len(report_ids)} reports, {len(fields)} scored fields each")
    print(f"reports with a presence disagreement (to review): {reports_to_review}/{len(report_ids)}")
    print(f"disputed fields (one model asserts the finding, the other does not): {total_disputed}/{total_fields} ({total_disputed / total_fields:.1%})")
    print(f"false-versus-null differences (negated versus not mentioned, not disputes): {false_vs_null_total}")
    if secondary_fields:
        print(f"'mentioned' disagreements (not disputes): {secondary_total}")
    if reports_to_review:
        print("disputed fields per report:", disputed.loc[disputed["n_disputed"] > 0, "n_disputed"].value_counts().sort_index().to_dict())
    agreement = pd.Series({field: 1 - count / len(report_ids) for field, count in disputes_per_field.items()}, name="agreement")
    print("\nper-field presence agreement (lowest first):")
    print(agreement.sort_values().round(3).to_string())
    if truth is not None:
        print(f"\npresence accuracy where the models agree: {agreed_correct}/{agreed_total} = {agreed_correct / max(agreed_total, 1):.1%}")
        print(f"on disputed fields: {models[0]} right {first_right}, {models[1]} right {second_right}")
        for model in models:
            print(f"{model} presence errors caught by disagreement: {errors_caught[model]}/{errors_total[model]}"
                  f" (the rest are errors both models share)")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
