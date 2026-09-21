# radextract: working notes for Claude

Benchmark and pipeline for structured extraction from radiology reports with local Ollama models.
Development happens on this Mac (M3 Pro, 18 GB, no local models: only Ollama cloud tags, never pull local models here).
Real runs happen on a Windows laptop: RTX 4090 Laptop GPU, 16 GB VRAM, 96 GB RAM, `gemma4:26b` and `gpt-oss:20b` pulled.
The user's time is the scarce resource. Never hand over a laptop command that has not been validated on the cloud models here first.
Real reports must never reach a cloud model; `--allow-cloud` exists only for the synthetic data in this repo.

## How to work with the user
- Short messages, one paragraph or a small table; lead with the number that matters.
- Report accuracy at the report level (reports with every embolism field right, and among PE-positive reports), never only per-field averages: most cells are correct negatives and per-field accuracy hides everything.
- Validate on the cloud first (`gemma4:31b-cloud`, `gpt-oss:20b-cloud`), only on the reports that had errors when possible (`--ids`), then hand over one command.
- Commit and push after every meaningful change; tag states the user may want to return to (`grouped-baseline` = grouped extraction before fine mode).
- The user wants everything binary (true/false), one boolean per finding and per artery, with a verbatim quote per value.

## Pipeline (all pandas, uv project, Python 3.12)
- `schema.py` (chest CT, 3-state, first version), `schema_binary.py` (binary chest CT), `schema_ctpa.py` (the one that matters: CT pulmonary angiogram, 64 fields, each `mentioned` and `present` booleans plus quote).
- `extract.py` runs models over a reports CSV; raw JSON per call under `raw/<schema>/`; resume from raw files; `--reparse` re-scores stored attempts without model calls; `--grouped` = several short gated calls per report; `--fine` = one question per artery field (tested, rejected on the 26B, see below); `--questions` = ask only disputed fields (tiebreaker mode).
- `evaluate.py` scores against the ground truth (`presence_*`, `reports_all_key_fields_correct`, `positive_reports_all_key_fields_correct` are the rows to read). `compare.py` = disagreement between two models on presence. `resolve.py` = one final CSV with all input columns, majority per field, `needs_review` flag. `benchmark.py` = extract, evaluate, compare, resolve in one command. `inspect_raw.py` explains retries.
- Thinking fixed in code: gpt-oss always medium (high overflows the 16 GB machine), gemma and qwen off unless the tag carries `@think` (thinking on the 26B was tested and is worse: 93.7% vs 99% and 25x slower).
- Ollama cloud ignores `format=`; gpt-oss ignores it too; both use prompt-only JSON. Local gemma uses constrained format.
- Model tags contain `:`; raw file names sanitize them. Transient 5xx/429 server errors are retried with a pause.

## CTPA schema conventions (validated by an independent review, 2026-09-18)
- `present` true only when described as present; `mentioned` true when the report addresses the finding (describes or negates). Filtering uses `present`.
- Study quality may be quoted from TECHNIQUE; nothing else. INDICATION/HISTORY/COMPARISON never count.
- `pe_*` fields are false unless `pulmonary_embolism` is present (repaired in code). "No acute PE" + chronic thrombus = PE present, `pe_acute` negated, `pe_chronic` present. `pe_acute` needs the word acute.
- `pe_multiple` counts anatomically distinct arteries (emboli are not countable): a thrombus extending into branches is multiple; "a single filling defect" is negated; unmentioned only when nothing indicates the count.
- Side words never name lobes ("bilateral segmental branches" sets `pe_segmental`, `pe_right`, `pe_left` and no per-lobe field); collective phrases never name segments ("branches of all lobes" sets the six per-lobe fields, no `pe_segment_*`); levels never roll up or down (a subsegmental embolus of a named lobe sets `pe_subsegmental` only); never infer a vessel from the path a clot took ("right middle lobe" in "right middle lobe and right lower lobe segmental arteries" is segmental); a saddle sets `pe_saddle`, the main arteries only when the clot is said to extend into them; the lingula is part of the left upper lobe.
- Code repairs replace retries for what the grammar cannot enforce: present implies mentioned; an unquoted negation becomes unmentioned; `pe_*` cleared without an embolism; a present artery implies its lobe, level and side; an artery field counts only if its quote or the report sentence it comes from names that lobe or segment (word-boundary match; "lateral" must not match "bilateral").
- Only a present finding without a quote triggers a retry.

## Results that decided things (50 synthetic CTPA reports, ground truth hand-filled and validated)
- Single 64-field call: laptop gemma 26B 37/50 reports fully right on PE fields, gpt-oss 20B 35/50 (39 and 38 after the guard and conventions).
- Grouped calls (core, levels, lobar, segmental, named segments right/left, other; artery groups only when an embolism was found): laptop gemma 47/50 in 11 min (14 s/report), gpt-oss 48/50 in 50 min (55 s/report); cloud 31B 49/50 projected. Pair disagrees on 5 reports; disagreement exposes every error of both; no shared errors. Resolved pair (gpt-oss primary): 48/50 with 5 flagged, no wrong report unflagged.
- Fine mode (one question per artery field): neutral on the cloud 31B, worse on the laptop 26B (47 -> 44): isolated questions lose the lobar/segmental allocation context and drop named segments. Giving each fine question the context (levels found, lobar arteries found, sibling definitions) did not help either: neutral on the 31B, 25/26 -> 24/26 on the gpt-oss proxy. Fine mode is closed; grouped mode is the answer.
- Tiebreakers: mistral-small3.2:24b spills (52 s/report) and was wrong on PE location (10% of positives fully right on the full form); qwen3.5:9b and mistral-nemo failed validation constantly under the old validators; a third model re-extracting whole reports never beat gemma alone. Question mode (only the disputed fields) works: 93% right on the hardest 56 fields with the cloud gemma.
- Candidate models that fit 16 GB at q4: dense <= ~13 GB (qwen3.5:9b 6.6 GB, mistral-nemo 7 GB, gemma4:12b 7.6 GB); mistral-small 15 GB borderline; 27B dense models spill and crawl; MoE (gemma4:26b 19 GB) spills gracefully.

- Apple on-device Foundation Model (apple/apple_fm.swift, tag `apple:on-device`, guided generation, runs on the Mac): refuses 27/50 CTPA reports as sensitive content even with the permissive guardrail and minimal instructions. Unusable; helper kept as a record.
- Fine mode with error-driven instructions ("mistakes to avoid" per lobar / per-lobe segmental / named-segment question) is pushed; the user tests it locally on gemma4:26b (`--grouped --fine --force`). Do not run more cloud validations unless asked: they take an hour and the user prefers local runs.
- 2026-09-21 evening: the file sent as the result of that test was byte-identical to the earlier plain fine-mode file (same latencies), so the model was never called; the error-driven fine prompt is still unmeasured. Results rows now carry `source` (called / resumed / reparsed) and `code_version` (git commit): check them first when a file arrives, and a run that resumes everything prints a WARNING.

## Laptop commands
- Pair, grouped (recommended when an hour per 50 reports is fine): `uv run benchmark.py --schema ctpa --models gemma4:26b gpt-oss:20b --grouped`
- gemma alone, grouped (11 min per 50, no flags): `uv run benchmark.py --schema ctpa --models gemma4:26b --grouped`
- Fine-mode retry, gemma alone (14 min per 50): `git pull` then `uv run benchmark.py --schema ctpa --models gemma4:26b --grouped --fine --force`; without `--force` the old `_fine.json` raw files are resumed and nothing runs.
- The file to read is `final_ctpa.csv`; send `results_ctpa.csv` here for scoring.
