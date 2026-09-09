# radextract

Benchmark of Ollama models on structured extraction from chest CT radiology reports.
A Pydantic schema with `Literal` enums fixes the output space, every asserted value must
carry a verbatim quote from the report, and the scripts check those quotes against the
report text rather than trusting the model.

## Files

| File | Purpose |
|---|---|
| `schema.py` | Schema `chest_ct`: Pydantic models (`Observation` with a 3-state status, `LargestNodule`, `FollowUp`, `ReportExtraction`) and the extraction prompt. Adding a finding is one line in `ReportExtraction`; the scripts discover the fields at runtime. |
| `schema_binary.py` | Schema `binary`: the same 20 findings plus `follow_up_recommended`, each a `BinaryObservation` (`present: bool` + quote). false covers both negated and not mentioned. |
| `extract.py` | Runs models over `reports.csv`, writes `results.csv` (one row per report, model and run) and one raw JSON per call in `raw/{schema}/`. `--schema chest_ct` (default) or `--schema binary`. |
| `evaluate.py` | Scores `results.csv` against the schema's ground truth, writes `summary.csv`. |
| `compare.py` | Field-by-field disagreement between two models, writes `disputed.csv`; quotes are ignored on purpose. |
| `reports.csv` | 30 synthetic chest CT reports (`report_id`, `report_text`). Fictional, no patient identifiers. |
| `ground_truth.csv` | Hand-filled reference value for every `chest_ct` field of every report (53 columns). |
| `ground_truth_binary.csv` | The same reference collapsed to true/false per finding (derived from `ground_truth.csv`). |

`.venv/`, `raw/`, `results*.csv`, `summary*.csv` and `disputed*.csv` are gitignored.

## Windows setup (RTX 4090 Laptop GPU, 16 GB VRAM, 96 GB RAM)

1. Install uv in PowerShell, then open a new terminal:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   (`winget install --id=astral-sh.uv -e` also works.)
2. Install Ollama for Windows from https://ollama.com/download and pull the two tags you want to benchmark, for example:
   ```
   ollama pull gpt-oss:20b
   ollama pull gemma4:26b
   ```
   `gemma4:26b` is a mixture-of-experts model (about 4B active parameters per token), so it runs on the 16 GB card even though its weights are 19 GB. `gemma4:12b` (7.6 GB, dense) is the fallback if `ollama ps` shows it spilling.
3. Clone and run. The first `uv run` downloads Python 3.12 (from `.python-version`) and builds `.venv` from `uv.lock`; there is no separate install step:
   ```
   git clone https://github.com/negretemdev/radextract.git
   cd radextract
   uv run extract.py --input reports.csv --output results.csv --models gpt-oss:20b gemma4:26b
   uv run evaluate.py --results results.csv --ground-truth ground_truth.csv --output summary.csv
   ```
   Add `--runs 3` to measure determinism. Model tags are always runtime arguments; nothing is hardcoded.
   For the binary schema add `--schema binary` to both commands (evaluate then scores against `ground_truth_binary.csv`), and to see where two models disagree:
   ```
   uv run compare.py --results results.csv --schema binary --models gemma4:26b gpt-oss:20b --ground-truth ground_truth_binary.csv
   ```

### extract.py options

| Option | Default | Meaning |
|---|---|---|
| `--input` | `reports.csv` | CSV with `report_id`, `report_text` |
| `--output` | `results.csv` | rewritten after every completed call |
| `--models` | required | one or more Ollama tags |
| `--schema` | `chest_ct` | `chest_ct` or `binary` |
| `--ids` | all | CSV with a `report_id` column: run only those reports |
| `--runs` | `1` | repeat each report N times (determinism) |
| `--host` | `http://localhost:11434` | Ollama server |
| `--limit` | all | only the first N reports |
| `--force` | off | re-run even if the raw file exists |
| `--allow-cloud` | off | permit `*-cloud` / `*:cloud` tags (synthetic data only) |

Resume: a call is skipped when `raw/{schema}/{model}_{report_id}_{run}.json` already exists, and its row is rebuilt from that file (the evidence check runs again at rebuild time). Delete a raw file or pass `--force` to redo a call.

### 16 GB VRAM notes

- `num_ctx` is fixed at 8192 in `OPTIONS` in `extract.py`. Do not raise it: the KV cache is what pushes a model off the GPU. Reports are about 400 tokens, the output about 1,500 tokens, and gpt-oss thinking adds 1,000 to 3,000, so 8192 is enough.
- After the first call, run `ollama ps` in another terminal. The model must show `100% GPU`. Any CPU share means it spilled and will run many times slower.
- `gemma4:26b` once showed a CPU share in `ollama ps` on this machine. It is a MoE with only a few billion active parameters, so a partial CPU share can still be fast; if a report takes minutes, switch to `gemma4:12b`.
- gpt-oss is hardcoded to `think="medium"`. Never run it at high on this machine: it overflows memory and never finishes. There is deliberately no CLI flag for thinking.
- Models run one after the other (all reports for model 1, then model 2), so Ollama loads each model once. An idle model unloads after 5 minutes (Ollama's default keep_alive).

## Privacy guard

`extract.py` refuses any model tag ending in `-cloud` or `:cloud` unless `--allow-cloud` is passed, and exits with an error before any call is made. Ollama routes those tags through `localhost` to Ollama's servers, so the report text would leave the machine. **Real reports must never reach a cloud model.** `--allow-cloud` exists only for the synthetic plumbing test on a machine that cannot hold the local models. When in doubt, `ollama ps` shows where a model runs.

## Thinking is fixed in code

`thinking_for()` in `extract.py`:

- tags starting with `gpt-oss` → `think="medium"`
- tags matching `gemma4` / `gemma-4` → `think=False`
- anything else → `think=None` (Ollama's default for that model)

Verified here with ollama 0.6.2 (Python package) against Ollama 0.31.1: `Client.chat()` takes `think: bool | Literal["low", "medium", "high"] | None` and passes it straight through the API. A probe call to `gpt-oss:120b-cloud` with `think="medium"` returned `message.thinking` with 3,000 to 5,000 characters of reasoning and `message.content` with the JSON, so the setting is honoured and the thinking never contaminates the content. `gemma4:31b-cloud` accepted `think=False` and returned no thinking text, so the bool path works too.

## Output columns

`results.csv`: `report_id, model, run, valid_json, attempts, latency_s`, then for each finding `{name}_status, {name}_evidence, {name}_evidence_ok, {name}_section`, then `nodule_count`, `largest_nodule_{size_mm, laterality, lobe, attenuation, calcified, margins, evidence, evidence_ok, section}` and `follow_up_{recommended, modality, interval_months, evidence, evidence_ok, section}`. 102 columns.

- `evidence_ok` (0/1) is computed by the script: whitespace collapsed and case folded, the quote must be a substring of the report. It is blank when there is no quote (`not_mentioned`) or the row is invalid.
- `section` is `findings`, `impression` or `other`, by the report's own `FINDINGS:` / `IMPRESSION:` headers and the character offset of the quote. Narrative reports without headers and anything before `FINDINGS:` (EXAM, INDICATION, TECHNIQUE, COMPARISON) are `other`, so a quote lifted from the indication shows up as `other`.
- `attempts` is 1 plus the number of retries; a call gives up after 3 retries (`valid_json=0`, all other fields blank).

`summary.csv`: one row per metric, one column per model, plus `agreement_between_models` on the `accuracy_<field>` rows when more than one model was run (fraction of reports where every model gave the identical value on run 1). Metrics: `reports, rows, valid_json_rate, mean_attempts, mean_latency_s, evidence_ok_rate, accuracy_overall, accuracy_<field>` for the 30 scored fields (20 statuses, `nodule_count`, 6 `largest_nodule` values, 3 `follow_up` values), and with `--runs > 1` also `identical_runs_rate` (scored fields identical across runs) and `identical_runs_rate_with_evidence` (quotes identical too). Invalid-JSON rows count as wrong for every field. Evidence text is never compared with the ground truth, because several different quotes can be correct; it is scored by `evidence_ok`.

## Schema and ground-truth conventions

The prompt in `schema.py` states these rules; the ground truth was filled by hand with the same rules and every quote in it was checked to be a verbatim substring of its report.

- `present`: the abnormality is explicitly described in FINDINGS, IMPRESSION or the narrative. `absent`: explicitly negated ("no", "without evidence of", "is not identified", "unremarkable", "within normal limits", "no significant", "has resolved"). A whole structure described as normal, clear or unremarkable negates its findings ("the lungs are clear" negates all six lung findings). `not_mentioned`: the report is silent.
- Text in EXAM, INDICATION, TECHNIQUE and COMPARISON lines never counts (three reports mention a finding only in the indication).
- Hedged interpretations of a described abnormality ("likely", "may represent", "suspicious for") are `present`; "cannot be excluded" alone never makes a finding present. "Stable" and "unchanged" abnormalities are present.
- A calcified granuloma (`calcified` yes) or a pulmonary mass counts as a pulmonary nodule. `ground_glass_opacity` is for non-nodular ground-glass; a ground-glass nodule is a nodule with `attenuation` ground_glass. `hepatic_lesion`, `adrenal_nodule`, `thyroid_nodule` include cysts. `osseous_lesion` is a focal bone lesion; degenerative change is not one.
- Sizes in mm; "a x b" uses the largest dimension. `largest_nodule` details are `not_stated` unless the report says them (a "solid nodule" without a calcification statement has `calcified` not_stated).
- `follow_up.recommended` is `no` when the report says no follow-up is needed, `not_stated` when it says nothing.
- Validators (`schema.py`): a quote is required whenever a status is asserted and must be null for `not_mentioned`; `nodule_count` must agree with `pulmonary_nodule`; `largest_nodule` must be empty when there are no nodules; `follow_up` must be empty when `not_stated`. A violation is a validation error, which triggers a retry with the error text appended.

## Things I had to work around

1. **Ollama cloud ignores `format=` schemas.** On `gpt-oss:120b-cloud` with `think="medium"`, passing `ReportExtraction.model_json_schema()` as `format` was silently not enforced: the model invented its own keys and 17 required fields were missing. `gemma4:31b-cloud` did the same (12 required fields missing, fenced JSON), so this is a cloud limitation, not a gpt-oss one. The same calls with no `format` and the JSON schema pasted into the system prompt returned valid extractions. `uses_constrained_format()` in `extract.py` therefore sends every cloud tag through prompt-only JSON (code fences are stripped), and `PROMPT_ONLY_JSON_MODEL_PREFIXES = ("gpt-oss",)` keeps local gpt-oss prompt-only as well, since that path is proven at `think="medium"`. Local gemma gets constrained `format` with the short prompt. To try constrained mode on the local gpt-oss:20b, set that tuple to `()` and compare `valid_json_rate`.
2. **Windows file names.** Model tags contain `:` (illegal in Windows file names), so the raw file name sanitizes the tag: `gpt-oss:20b` → `raw/chest_ct/gpt-oss_20b_R001_1.json`.
3. **Quotes are the weak spot, not statuses.** Even with correct statuses the model rewrites shared negations ("No emphysema or bronchiectasis." quoted as "No bronchiectasis") or splices with "..." ("The visualized liver ... are unremarkable."). The prompt now demands one contiguous span; the `evidence_ok` check catches the rest.
4. **The cloud model is not deterministic even at temperature 0, seed 42.** Three calls on the same report gave identical statuses and values but different quotes each time (0 to 3 `evidence_ok` failures). That is why `summary.csv` reports `identical_runs_rate` and `identical_runs_rate_with_evidence` separately. Local llama.cpp inference should be more deterministic; `--runs 3` will show.
5. **pandas 3.** Strings are a real `str` dtype, empty CSV cells read back as NaN, and `astype(str)` keeps NaN as missing, so comparisons across runs and models use `nunique(dropna=False)`. Reports are read with `dtype=str, keep_default_na=False` so a report is never NaN.
6. **Ollama cloud plumbing test.** With no room for the local models on the Mac, the end-to-end test used the one cloud model already pulled (`gpt-oss:120b-cloud`) with `--allow-cloud` on the synthetic reports. Nothing local was pulled.

## Plumbing test on the Mac (synthetic reports, cloud model)

Mac M3 Pro, Ollama 0.31.1, ollama 0.6.2, pandas 3.0.5, pydantic 2.13.5, Python 3.12. The 3 reports were R001 (two nodules plus follow-up), R002 (normal) and R003 (CT pulmonary angiogram with embolism).

```
$ uv run extract.py --input reports.csv --output results.csv --models gpt-oss:120b-cloud --limit 3 --allow-cloud --force
== gpt-oss:120b-cloud: think='medium', constrained_format=False
[gpt-oss:120b-cloud] 1/3 R001 run 1: valid_json=1 attempts=1 latency=5.4s 
[gpt-oss:120b-cloud] 2/3 R002 run 1: valid_json=1 attempts=1 latency=3.6s 
[gpt-oss:120b-cloud] 3/3 R003 run 1: valid_json=1 attempts=1 latency=3.6s 
Wrote 3 rows to results.csv

$ uv run evaluate.py --results results.csv --ground-truth ground_truth.csv --output summary.csv
                                     metric  gpt-oss:120b-cloud
                                    reports               3.000
                                       rows               3.000
                            valid_json_rate               1.000
                              mean_attempts               1.000
                             mean_latency_s               4.183
                           evidence_ok_rate               0.965
                           accuracy_overall               1.000
           accuracy_pulmonary_nodule_status               1.000
       accuracy_ground_glass_opacity_status               1.000
              accuracy_consolidation_status               1.000
                accuracy_atelectasis_status               1.000
                  accuracy_emphysema_status               1.000
             accuracy_bronchiectasis_status               1.000
           accuracy_pleural_effusion_status               1.000
               accuracy_pneumothorax_status               1.000
       accuracy_pericardial_effusion_status               1.000
               accuracy_cardiomegaly_status               1.000
     accuracy_coronary_calcification_status               1.000
accuracy_mediastinal_lymphadenopathy_status               1.000
      accuracy_hilar_lymphadenopathy_status               1.000
accuracy_aortic_dilation_or_aneurysm_status               1.000
         accuracy_pulmonary_embolism_status               1.000
             accuracy_hepatic_lesion_status               1.000
             accuracy_adrenal_nodule_status               1.000
             accuracy_thyroid_nodule_status               1.000
             accuracy_osseous_lesion_status               1.000
               accuracy_rib_fracture_status               1.000
                      accuracy_nodule_count               1.000
            accuracy_largest_nodule_size_mm               1.000
         accuracy_largest_nodule_laterality               1.000
               accuracy_largest_nodule_lobe               1.000
        accuracy_largest_nodule_attenuation               1.000
          accuracy_largest_nodule_calcified               1.000
            accuracy_largest_nodule_margins               1.000
             accuracy_follow_up_recommended               1.000
                accuracy_follow_up_modality               1.000
         accuracy_follow_up_interval_months               1.000

Wrote summary.csv
```

The two `evidence_ok` failures were both shared statements split by the model ("No pleural effusion or pneumothorax." quoted as "No pneumothorax", "Unremarkable visualized liver and adrenal glands." quoted as "Unremarkable visualized liver."); every status was still correct.

## Full 30-report run on Ollama cloud (reference baseline)

Run on the Mac with `--allow-cloud` on the synthetic reports, one run per model. Ollama cloud ignores `format=`, so all three used prompt-only JSON. `gemma4:26b` has no cloud tag; `gemma4:31b-cloud` stands in for it here.

| Metric | gpt-oss:120b-cloud | gpt-oss:20b-cloud | gemma4:31b-cloud |
|---|---|---|---|
| Valid JSON | 30/30 | 30/30 (one report needed a retry) | 30/30 |
| Accuracy vs ground truth (900 values) | 98.9% | 96.7% | 99.6% |
| Quotes verbatim (`evidence_ok`) | 96.3% | 96.8% | 99.6% |
| Mean latency (cloud, throttled) | 4 s | 53 s | 33 s |

What the errors were:

- All three models write `attenuation: solid` for a nodule the report calls only "calcified" or a "mass" (ground truth `not_stated`, never infer). Clinically defensible; it is the strictest rule in the schema and worth reconsidering.
- Both gpt-oss models take "The lungs are clear." as `not_mentioned` for the individual lung findings instead of `absent` (whole-structure rule). Gemma applied the rule.
- gpt-oss:20b alone made real reading errors: it called a **resolved** consolidation `present` (R024, the comparison-language trap), missed explicit negations of cardiomegaly, aortic dilation and pulmonary embolism, over-negated findings the report never mentioned (R003 osseous lesion, R025 embolism), inferred `calcified: no` five times, and answered `follow_up: no` for normal reports that say nothing about follow-up.
- Gemma's one status error is arguable: "The pulmonary arteries are normal in caliber." read as `absent` for pulmonary embolism (ground truth `not_mentioned`).

Per-field agreement between the three models (run 1) is in the `agreement_between_models` column of `summary.csv`; the least agreed fields were `largest_nodule_calcified` (83%), `adrenal_nodule_status` (87%) and `pulmonary_embolism_status`, `hepatic_lesion_status`, `follow_up_recommended` (90%).

## Binary schema: same 30 reports, same three cloud models

`--schema binary` collapses every finding to `present: true/false` (false = negated or not mentioned) and adds `follow_up_recommended` as a 21st boolean. Quotes are still recorded and checked, but `compare.py` ignores them by design.

| Metric | gemma4:31b-cloud | gpt-oss:20b-cloud | gpt-oss:120b-cloud |
|---|---|---|---|
| Valid JSON | 30/30 | 30/30 | 30/30 (two reports needed a retry) |
| Accuracy vs ground truth (630 values) | 100% | 100% | 100% |
| Quotes verbatim (`evidence_ok`) | 98.7% | 94.3% | 94.0% |
| Mean latency (cloud, throttled) | 22 s | 23 s | 23 s |

gemma4:31b vs gpt-oss:20b: **0 disputed fields in 0 of 30 reports** (3-state schema: 28 disputed fields in 19 of 30 reports). The 3-state disputes were almost entirely `absent` vs `not_mentioned`, `no` vs `not_stated` and inferred nodule attributes; the binary schema has none of those distinctions, so they disappear, and the one genuine 3-state misread (a resolved consolidation called present) did not recur. On this synthetic set a third-model tiebreaker has nothing to do; on real reports, disagreement rows are the ones to read.
