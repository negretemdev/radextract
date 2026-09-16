"""Explain retries and failures from the raw files: per call, each attempt's done_reason, thinking length,
content length and validation error. By default only calls that retried or failed are listed.

    uv run inspect_raw.py --model gemma4:26b@think
    uv run inspect_raw.py --model gemma4:26b@think --delete-failed   # remove failed raw files so the next run redoes them
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Show why model calls retried or failed.")
    parser.add_argument("--schema", default="binary", help="raw/<schema> directory")
    parser.add_argument("--model", default=None, help="model label as passed to --models; default: every model")
    parser.add_argument("--all", action="store_true", help="list every call, not only retries and failures")
    parser.add_argument("--delete-failed", action="store_true", help="delete raw files with valid_json false")
    args = parser.parse_args()

    files = sorted((Path("raw") / args.schema).glob("*.json"))
    listed = failed = retried = cut_off = 0
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if args.model is not None and raw["model"] != args.model:
            continue
        attempts = raw["attempts"]
        failed += not raw["valid_json"]
        retried += len(attempts) > 1
        cut_off += any(attempt["done_reason"] == "length" for attempt in attempts)
        if not args.all and raw["valid_json"] and len(attempts) == 1:
            continue
        listed += 1
        print(f"{path.name}: valid_json={raw['valid_json']} attempts={len(attempts)} latency={raw['latency_s']:.0f}s think={raw['think']!r} constrained_format={raw['constrained_format']}")
        for attempt in attempts:
            print(f"   attempt {attempt['attempt']}: done_reason={attempt['done_reason']} eval_count={attempt['eval_count']} "
                  f"thinking_chars={len(attempt['thinking'] or '')} content_chars={len(attempt['content'])} latency={attempt['latency_s']:.0f}s")
            if attempt["validation_error"]:
                print("      error: " + attempt["validation_error"][:500].replace("\n", "\n             "))
                print("      content starts: " + attempt["content"][:160].replace("\n", " ") + ("..." if len(attempt["content"]) > 160 else ""))
        if args.delete_failed and not raw["valid_json"]:
            path.unlink()
            print("   deleted")
    total = sum(1 for path in files if args.model is None or json.loads(path.read_text(encoding="utf-8"))["model"] == args.model)
    print(f"\n{total} calls, {retried} retried, {failed} failed, {cut_off} cut off by num_ctx")


if __name__ == "__main__":
    main()
