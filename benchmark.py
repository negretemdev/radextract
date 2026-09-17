"""One command for the whole benchmark: extract, then evaluate, then compare (when two or more models).

    uv run benchmark.py --schema binary --models gemma4:26b gpt-oss:20b

Outputs: results_<schema>.csv, summary_<schema>.csv and, with two or more models, disputed_<schema>.csv
(the first two models are compared). All other options are passed through to extract.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCHEMAS = {"chest_ct": "ground_truth.csv", "binary": "ground_truth_binary.csv", "ctpa": "ground_truth_ctpa.csv"}


def run(step: list[str]):
    print("\n$ " + " ".join(step), flush=True)
    completed = subprocess.run([sys.executable] + step)
    if completed.returncode != 0:
        sys.exit(f"{step[0]} failed with exit code {completed.returncode}; fix the problem and re-run (finished calls are resumed)")


def main():
    parser = argparse.ArgumentParser(description="Run extract.py, evaluate.py and compare.py in one go.")
    parser.add_argument("--models", nargs="+", required=True, help="Ollama model tags, e.g. gemma4:26b gpt-oss:20b")
    parser.add_argument("--schema", choices=SCHEMAS, default="binary")
    parser.add_argument("--input", type=Path, default=None, help="default: the schema's reports file")
    parser.add_argument("--ids", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reparse", action="store_true", help="re-validate existing raw files, no model calls")
    parser.add_argument("--allow-cloud", action="store_true")
    args = parser.parse_args()

    results = Path(f"results_{args.schema}.csv")
    summary = Path(f"summary_{args.schema}.csv")
    disputed = Path(f"disputed_{args.schema}.csv")
    ground_truth = SCHEMAS[args.schema]

    extract = ["extract.py", "--schema", args.schema, "--output", str(results),
               "--host", args.host, "--runs", str(args.runs), "--models", *args.models]
    if args.input is not None:
        extract += ["--input", str(args.input)]
    if args.ids is not None:
        extract += ["--ids", str(args.ids)]
    if args.limit is not None:
        extract += ["--limit", str(args.limit)]
    if args.force:
        extract.append("--force")
    if args.reparse:
        extract.append("--reparse")
    if args.allow_cloud:
        extract.append("--allow-cloud")
    run(extract)
    run(["evaluate.py", "--schema", args.schema, "--results", str(results), "--ground-truth", ground_truth, "--output", str(summary)])
    if len(args.models) >= 2:
        run(["compare.py", "--schema", args.schema, "--results", str(results), "--models", args.models[0], args.models[1],
             "--ground-truth", ground_truth, "--output", str(disputed)])
    print(f"\nDone. Files: {results}, {summary}" + (f", {disputed}" if len(args.models) >= 2 else "") + f". Send {results} for review.")


if __name__ == "__main__":
    main()
