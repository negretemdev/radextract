"""Compare two models field by field and write disputed.csv.

Only the schema's scored fields count. Evidence quotes are ignored: they vary between calls even at
temperature 0, so they cannot take part in agreement. With --ground-truth the accuracy is also split
by whether the two models agreed.
"""

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

from evaluate import SCHEMAS, values_match


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

    disputed_rows = []
    disputes_per_field = {field: 0 for field in fields}
    agreed_total = agreed_correct = 0
    first_right = second_right = both_wrong = 0
    errors_total = {models[0]: 0, models[1]: 0}
    errors_caught = {models[0]: 0, models[1]: 0}
    for report_id in report_ids:
        disputed_fields = []
        details = []
        for field in fields:
            value_first = first.at[report_id, field]
            value_second = second.at[report_id, field]
            if truth is not None:
                for model, value in ((models[0], value_first), (models[1], value_second)):
                    if not values_match(field, value, truth.at[report_id, field]):
                        errors_total[model] += 1
                        errors_caught[model] += not values_match(field, value_first, value_second)
            if values_match(field, value_first, value_second):
                if truth is not None:
                    agreed_total += 1
                    agreed_correct += values_match(field, value_first, truth.at[report_id, field])
                continue
            disputed_fields.append(field)
            disputes_per_field[field] += 1
            details.append(f"{field}: {models[0]}={value_first} | {models[1]}={value_second}")
            if truth is not None:
                if values_match(field, value_first, truth.at[report_id, field]):
                    first_right += 1
                elif values_match(field, value_second, truth.at[report_id, field]):
                    second_right += 1
                else:
                    both_wrong += 1
        if disputed_fields:
            disputed_rows.append({
                "report_id": report_id,
                "n_disputed": len(disputed_fields),
                "disputed_fields": ";".join(disputed_fields),
                "details": "; ".join(details),
            })

    disputed = pd.DataFrame(disputed_rows, columns=["report_id", "n_disputed", "disputed_fields", "details"])
    disputed.to_csv(args.output, index=False, encoding="utf-8")
    total_fields = len(report_ids) * len(fields)
    total_disputed = int(disputed["n_disputed"].sum()) if len(disputed) else 0
    print(f"{models[0]} vs {models[1]} on {len(report_ids)} reports, {len(fields)} scored fields each")
    print(f"reports with any disagreement: {len(disputed)}/{len(report_ids)}")
    print(f"disputed fields: {total_disputed}/{total_fields} ({total_disputed / total_fields:.1%})")
    if len(disputed):
        print("disputed fields per report:", disputed["n_disputed"].value_counts().sort_index().to_dict())
    agreement = pd.Series({field: 1 - count / len(report_ids) for field, count in disputes_per_field.items()}, name="agreement")
    print("\nper-field agreement (lowest first):")
    print(agreement.sort_values().round(3).to_string())
    if truth is not None:
        print(f"\naccuracy where the models agree: {agreed_correct}/{agreed_total} = {agreed_correct / max(agreed_total, 1):.1%}")
        print(f"on disputed fields: {models[0]} right {first_right}, {models[1]} right {second_right}, both wrong {both_wrong}")
        for model in models:
            print(f"{model} errors caught by disagreement: {errors_caught[model]}/{errors_total[model]}"
                  f" (the rest are errors both models share)")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
