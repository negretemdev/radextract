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
| `benchmark.py` | One command: `extract.py`, then `evaluate.py`, then `compare.py`; writes `results_<schema>.csv`, `summary_<schema>.csv`, `disputed_<schema>.csv`. |
| `schema_ctpa.py` | Schema `ctpa`: 31 CT pulmonary angiogram fields, each `present: true / false / null` (described / negated / not mentioned) plus a quote. |
| `reports_ctpa.csv`, `ground_truth_ctpa.csv` | 50 synthetic CTPA reports (P001–P050) and their hand-filled reference (63 columns). |
| `extract.py` | Runs models over `reports.csv`, writes `results.csv` (one row per report, model and run) and one raw JSON per call in `raw/{schema}/`. `--schema chest_ct` (default), `--schema binary` or `--schema ctpa`; the input CSV defaults to the schema's reports file. |
| `inspect_raw.py` | Explains retries and failures from `raw/`: done_reason, thinking length, validation error per attempt. |
| `evaluate.py` | Scores `results.csv` against the schema's ground truth, writes `summary.csv`. |
| `compare.py` | Field-by-field disagreement between two models on presence (asserted or not), writes `disputed.csv`; false-versus-null differences are counted separately and quotes are ignored on purpose. |
| `reports.csv` | 30 synthetic chest CT reports (`report_id`, `report_text`). Fictional, no patient identifiers. |
| `ground_truth.csv` | Hand-filled reference value for every `chest_ct` field of every report (53 columns). |
| `ground_truth_binary.csv` | The same reference collapsed to true/false per finding (derived from `ground_truth.csv`). |

`.venv/`, `raw/`, `results*.csv`, `summary*.csv` and `disputed*.csv` are gitignored.

## Laptop run checklist (binary schema, two local models)

```powershell
git clone https://github.com/negretemdev/radextract.git   # or: git pull, if already cloned
cd radextract
ollama pull gemma4:26b
ollama pull gpt-oss:20b
uv run benchmark.py --schema binary --models gemma4:26b gpt-oss:20b
```

To benchmark gemma with thinking on as well (gpt-oss and the earlier gemma run are reused from `raw/`, only the new variant is called):

```powershell
uv run benchmark.py --schema binary --models gemma4:26b@think --force
```

`--force` redoes any `@think` raw files produced by an earlier version. The thinking context costs KV-cache memory, and the 26B's 19 GB of weights already exceed the card, so before the first thinking run set these once (PowerShell), then quit Ollama from the tray icon and start it again:

```powershell
[Environment]::SetEnvironmentVariable('OLLAMA_NUM_PARALLEL', '1', 'User')
[Environment]::SetEnvironmentVariable('OLLAMA_FLASH_ATTENTION', '1', 'User')
[Environment]::SetEnvironmentVariable('OLLAMA_KV_CACHE_TYPE', 'q8_0', 'User')
```

One slot instead of several (the cache is allocated per slot) and an 8-bit cache instead of 16-bit. During the first report run `ollama ps` in a second window: the PROCESSOR column should be mostly GPU. If it is still largely CPU, the fallback experiment is `gemma4:12b@think` (7.6 GB dense, fits entirely). If reports take minutes or show `attempts` above 1, ask the raw files why:

```powershell
uv run inspect_raw.py --model gemma4:26b@think
```

It prints, per retried or failed call, each attempt's `done_reason`, thinking length and validation error. `done_reason=length` means thinking plus JSON did not fit in the 8k context; the call is retried with a request to reason briefly, which re-rolls the reasoning and usually fits (on the laptop, gemma4:26b thought about 27,000 characters per report, about 7,500 tokens, versus 3,000 for the cloud 31B, and 5 of the first 8 reports needed this retry). A validation error with `done_reason=stop` means the model broke a schema rule and the retry is doing its job. `--delete-failed` removes failed raw files so a later run redoes only those.

`benchmark.py` runs `extract.py`, then `evaluate.py`, then `compare.py` (first two models) and writes `results_binary.csv`, `summary_binary.csv` and `disputed_binary.csv`. It takes the same options as `extract.py` (`--runs`, `--limit`, `--ids`, `--force`, `--allow-cloud`, `--host`). The three scripts can still be run separately.

- After the first call, run `ollama ps` in a second terminal: each model must show `100% GPU`. If it shows a CPU share and reports take minutes, stop and switch to `gemma4:12b`.
- The run resumes if interrupted: re-run the same `extract.py` command and finished reports are skipped.
- Local gemma is the first model to use constrained `format=` (the cloud ignored it). If gemma calls fail with a grammar or format error, add `"gemma"` to `PROMPT_ONLY_JSON_MODEL_PREFIXES` in `extract.py` and re-run; that sends the schema in the prompt instead.
- The file to look at afterwards is `results_binary.csv` (100 rows: 50 reports x 2 models, every value and quote). `summary_binary.csv` and `disputed_binary.csv` are derived from it and the ground truth in the repo.

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

- `num_ctx` is 8192 in `OPTIONS` in `extract.py` for every model except `@think` variants, which get `THINKING_NUM_CTX` = 16384 (gemma4:26b reasons for 4,000 to 8,000 tokens per report, which fills 8k before any JSON). 32k loaded but `ollama ps` showed `100% CPU`: the cache alone exceeded the card. Ollama-side settings that halve the cache again are in the checklist. gpt-oss stays at 8192 and cannot get the thinking suffix. Reports are about 400 tokens, the output about 1,500 tokens, and gpt-oss thinking adds 1,000 to 3,000, so 8192 is enough for everything else.
- After the first call, run `ollama ps` in another terminal. The model must show `100% GPU`. Any CPU share means it spilled and will run many times slower.
- `gemma4:26b` once showed a CPU share in `ollama ps` on this machine. It is a MoE with only a few billion active parameters, so a partial CPU share can still be fast; if a report takes minutes, switch to `gemma4:12b`.
- gpt-oss is hardcoded to `think="medium"`. Never run it at high on this machine: it overflows memory and never finishes. There is deliberately no CLI flag for thinking.
- Models run one after the other (all reports for model 1, then model 2), so Ollama loads each model once. An idle model unloads after 5 minutes (Ollama's default keep_alive).

## Privacy guard

`extract.py` refuses any model tag ending in `-cloud` or `:cloud` unless `--allow-cloud` is passed, and exits with an error before any call is made. Ollama routes those tags through `localhost` to Ollama's servers, so the report text would leave the machine. **Real reports must never reach a cloud model.** `--allow-cloud` exists only for the synthetic plumbing test on a machine that cannot hold the local models. When in doubt, `ollama ps` shows where a model runs.

## Thinking is fixed in code

`thinking_for()` in `extract.py`:

- tags starting with `gpt-oss` → `think="medium"`
- tags matching `gemma4` / `gemma-4` or `qwen3` (qwen3.5, qwen3.8) → `think=False`; with the `@think` suffix (`gemma4:26b@think`) → `think=True`
- anything else → `think=None` (Ollama's default for that model); `@think` → `think=True`

The `@think` suffix is the only way to change thinking and it is part of the model label: `gemma4:26b` and `gemma4:26b@think` are treated as two different models in `results.csv`, `raw/` and `compare.py`, so both can be benchmarked in one run. The suffix is stripped before the tag is sent to Ollama. `gpt-oss:20b@think` is refused: gpt-oss stays at medium, full stop.

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

## Ambiguous set: R031–R050

Twenty more synthetic reports written to be hard: dictation-style run-on text and abbreviations (RUL, PTX, GGO, LAD, nl), lowercase headers and typos, impression-before-findings, a CONCLUSION instead of IMPRESSION, an ADDENDUM that adds a nodule, findings that contradict the impression (effusion in FINDINGS, "no pleural effusion" in IMPRESSION), and hedges everywhere: "questionable", "equivocal", "possible ... favored to be artifact", "not entirely excluded", "atelectasis versus early consolidation", "bulla versus pneumothorax", "borderline enlarged", "prominent but not pathologically enlarged", "near-complete resolution ... residual", "trace ... physiologic", "no significant effusion; trace fluid", conditional follow-up ("could be considered", "could be obtained"), and findings mentioned only in the indication or comparison line.

Ground-truth conventions for the hard set, applied on top of the prompt rules (the `chest_ct` ground truth covers R001–R030 only; the binary ground truth covers all 50):

- "X versus Y", "questionable X", "equivocal X", "possible X" at a described location: `true` (an abnormality is described; the hedge is interpretation).
- "no definite X ... favored to be artifact", "not entirely excluded", "cannot be excluded": `false`.
- "borderline enlarged" node: `true`; "prominent but not pathologically enlarged", "subcentimeter", "small nonspecific nodes": `false`.
- "near-complete resolution" with residual tissue, "decreased and now small", "trace, physiologic": `true`; "interval resolution", "normalized", "resolved": `false`.
- Findings and impression contradict each other: the FINDINGS description wins.
- Conditional follow-up wording ("could be considered", "could be obtained", "could help differentiate", "continue annual screening"): `true`; "attention on follow-up imaging", "correlate clinically", "discussed with the team", "per Fleischner no follow-up": `false`.
- Not the finding: hyperinflation is not emphysema, bronchial wall thickening is not bronchiectasis, steatosis and hepatic congestion are not hepatic lesions, adrenal thickening without a nodule is not an adrenal nodule, a heterogeneous thyroid is not a thyroid nodule, a compression deformity is neither an osseous lesion nor a rib fracture. A liver laceration is a focal hepatic lesion.

### Splitting false into two booleans

`false` currently merges "explicitly negated" and "not mentioned". For filtering on `true` that never matters, and the quote column already separates the two cases (a `false` with a quote was negated, without one it was silent). Measured on the 30 clean reports: the `present` field had 0 disputes between gemma4:31b and gpt-oss:20b, but a would-be `negated` boolean disagreed on 43 of the 529 agreed-false fields (8%), with gemma matching the ground truth 99.8% of the time and gpt-oss:20b 92%. So if that distinction is needed, add it as a second boolean (`negated: true/false`) and score it separately; it must not take part in the agreement check on `present`.

### Results on the ambiguous set (binary schema, cloud models)

| Metric | gemma4:31b-cloud | gpt-oss:20b-cloud | gpt-oss:120b-cloud |
|---|---|---|---|
| Valid JSON | 20/20 | 20/20 | 20/20 |
| Accuracy vs ground truth (420 values) | 99.3% | 98.3% | 98.8% |
| Quotes verbatim | 98.9% | 99.7% | 99.0% |

gemma4:31b vs gpt-oss:20b on the hard 20: **6 disputed fields in 5 of 20 reports** (1.4% of fields; 4 reports with one field, 1 with two). Where the two agreed, 412 of 414 values were correct; the two agreed-wrong values are both "attention on follow-up imaging" read as a recommendation, which all three models did, so that ground-truth convention is the outlier rather than the models. On the disputes gemma was right 5 times and gpt-oss:20b once. gpt-oss:20b's misses were the classic traps: it let the impression's "No acute pulmonary embolism" override the chronic thromboembolic disease described in the findings, let "no pleural effusion" in the impression override the effusion in the findings, counted "prominent but not pathologically enlarged" nodes as lymphadenopathy, and counted a healing rib fracture as an osseous lesion. Majority vote with the 120B as tiebreaker scores 99.3%, the same as gemma alone. Over all 50 reports the two primaries disagree on 6 of 1,050 fields (0.6%).

## Laptop run: gemma4:26b vs gpt-oss:20b (the real benchmark)

Run on the RTX 4090 laptop on 2026-09-16 with `uv run benchmark.py --schema binary --models gemma4:26b gpt-oss:20b`, 50 reports, one run each. gemma used constrained `format=`, gpt-oss prompt-only JSON.

| Metric | gemma4:26b | gpt-oss:20b |
|---|---|---|
| Valid JSON, first attempt | 50/50 | 49/50 (one retry) |
| Accuracy, all 50 (1,050 values) | 99.0% (11 wrong) | 99.2% (8 wrong) |
| Accuracy, clean 30 | 99.8% | 100% |
| Accuracy, hard 20 | 97.6% | 98.1% |
| Quotes verbatim | 97.0% | 97.5% |
| Median latency per report | 5.2 s | 14.2 s |
| 1,000 reports | 1.4 h | 4.0 h |

Disagreement between the two: 9 fields in 8 of 50 reports (0.9%; seven reports with one field, one with two). Where they agreed, 1,036 of 1,041 values were correct (99.5%). On the disputes gpt-oss was right 6 times, gemma 3. Disagreement exposed 6 of gemma's 11 errors and 3 of gpt-oss's 8; the other 5 are errors both models share and no ensemble can see.

The shared errors: both called chronic thromboembolic disease `false` for pulmonary embolism because the impression said "No acute pulmonary embolism" (R033); both read "attention on follow-up imaging" as a recommendation (R035, R042, a ground-truth convention every model tested rejects, so it is probably wrong); both counted "small mediastinal lymph nodes, nonspecific" as lymphadenopathy (R044) and a healing rib fracture as an osseous lesion (R050).

gemma's own misses were mostly over-calling: calcified nonenlarged hilar nodes as lymphadenopathy, a hilar node filed under mediastinal, "heart size upper normal" as cardiomegaly, adrenal thickening without a nodule as a nodule, and "no dedicated follow-up recommended" for a thyroid nodule overriding "continue annual screening". gpt-oss's were the impression overriding the findings (no effusion, R034), "prominent but not pathologically enlarged" nodes as lymphadenopathy (R038), and one finding lifted from the INDICATION line ("Known hepatic hemangioma", R035), which the `section` column flagged as `other`.

Unlike the cloud comparison, where the 31B gemma was strong enough that gpt-oss added nothing, on the laptop the 26B gemma and gpt-oss:20b are tied on accuracy and their disagreements are informative. The pair costs about 3.7x the time of gemma alone.

## gemma with thinking on (cloud 31B, same 50 reports)

`gemma4:31b-cloud@think` versus the earlier `gemma4:31b-cloud` run with thinking off:

| Metric | thinking off | thinking on |
|---|---|---|
| Accuracy, all 50 (1,050 values) | 99.7% (3 wrong) | 99.8% (2 wrong) |
| Clean 30 | 100% | 100% |
| Hard 20 | 99.3% | 99.5% |
| Quotes verbatim | 98.8% | 99.9% |
| Reasoning per report (characters) | 0 | median 3,100, max 6,800 |
| Responses cut off by the 8k context | 0 | 0 |

The two runs disagreed on a single field in 50 reports (R042 "possibly a focus of atelectasis vs a true nodule", which thinking got right). The two remaining errors are the "attention on follow-up imaging" convention. So on the 31B thinking neither helps nor hurts the statuses in a measurable way, but it fixes the quotes and it is safe within `num_ctx` 8192. The 26B on the laptop had 11 errors with thinking off and is the model with room to gain; that run decides whether `@think` becomes the default for gemma.

## Laptop run: gemma4:26b with thinking on (16k context, 8-bit cache, 26% CPU / 74% GPU)

| Metric | gemma4:26b | gemma4:26b@think | gpt-oss:20b |
|---|---|---|---|
| Valid JSON | 50/50 | 48/50 | 50/50 |
| Reports needing retries | 0 | 20 (six used all 4 attempts) | 1 |
| Accuracy, all 50 (failed reports count as wrong) | 99.0% | 93.7% | 99.2% |
| Accuracy on its valid reports only | 99.0% | 97.6% | 99.2% |
| Clean 30 / hard 20 | 99.8% / 97.6% | 94.9% / 91.9% | 100% / 98.1% |
| Mean time per report | 6 s | 159 s | 17 s |
| Total for 50 reports | 5 min | 132 min | 14 min |

Thinking made the 26B worse on every axis. It fixed 2 of the thinking-off errors and introduced 16 new ones, almost all of the same kind: findings plainly described in the report returned as `false` with no quote (R015 missed cardiomegaly, coronary calcification and the aortic aneurysm; R016 missed six findings), on first attempts that were not cut off. The two failures (R003, R047) burned four attempts and about nine minutes each. Against gpt-oss:20b it disagreed on 64 fields in 12 reports and was right on 3 of them. The cloud 31B had shown no such regression, so this is a property of the 26B on this hardware, not of thinking in general. Conclusion for the laptop: gemma runs with thinking off, and the two-model pipeline is `gemma4:26b` plus `gpt-oss:20b`.

## CTPA schema (`--schema ctpa`)

Built for cohort filtering of CT pulmonary angiogram reports. Every field is `present: true | false | null` with a verbatim quote: true when the finding is explicitly described, false when explicitly negated, null when the report says nothing. Filtering uses `true`; the false/null split is kept because a negated finding and a silent report are different evidence, and it is scored separately so it never inflates the review list.

| Group | Fields |
|---|---|
| Study quality (may be quoted from TECHNIQUE) | `suboptimal_study`, `poor_contrast_opacification`, `motion_artifact` |
| Embolism | `pulmonary_embolism`, `pe_acute`, `pe_chronic`, `pe_saddle`, `pe_main`, `pe_lobar`, `pe_segmental`, `pe_subsegmental`, `pe_right`, `pe_left`, `pe_multiple`, `pe_occlusive`, `pe_nonocclusive` |
| Consequences | `right_heart_strain`, `pulmonary_artery_enlargement`, `perfusion_defect`, `pulmonary_infarct` |
| Lungs, pleura, heart | `consolidation`, `ground_glass_opacity`, `atelectasis`, `pulmonary_nodule`, `emphysema`, `mosaic_attenuation`, `pleural_effusion`, `pneumothorax`, `pericardial_effusion`, `cardiomegaly`, `lymphadenopathy` |

Conventions, stated in the prompt and enforced by validators where possible:

- The `pe_*` fields must be null unless `pulmonary_embolism` is true (validator). "No acute pulmonary embolism" with chronic thrombus described means `pulmonary_embolism` true, `pe_acute` false, `pe_chronic` true, which is the case both local models missed on the earlier set. "Acute component cannot be excluded" leaves `pe_acute` null.
- Levels: `pe_saddle`, `pe_main`, `pe_lobar` (interlobar counts as lobar), `pe_segmental`, `pe_subsegmental`, true for each level described, false for a level explicitly negated. Sides: "bilateral" sets both. `pe_multiple` is true when more than one embolus, filling defect or vessel is involved. A nonocclusive clot makes `pe_nonocclusive` true and `pe_occlusive` false, and the reverse; both true when both are described.
- `suboptimal_study` is true when the report calls itself limited, suboptimal, degraded or nondiagnostic for embolism, false when opacification is adequate or good without a limiting artifact. A motion artifact that "does not limit evaluation" sets `motion_artifact` true and `suboptimal_study` false. A nondiagnostic study leaves `pulmonary_embolism` null.
- `right_heart_strain` is true for RV dilation or enlargement, RV/LV above 1, septal flattening or bowing, IVC reflux, or the words "right heart strain". `pulmonary_artery_enlargement` needs the words enlarged or dilated; a measurement alone does not count. "The lungs are clear" negates the seven parenchymal findings. Septic emboli are nodules, not pulmonary embolism.
- A quote is required for true and false and must be null for null (validator).

`evaluate.py` reports `accuracy_<field>` (exact: true/false/null) and `presence_<field>` (asserted or not). `compare.py` judges disagreement on presence and lists false-versus-null differences in their own column of `disputed.csv`.

### Laptop run

```powershell
git pull
uv run benchmark.py --schema ctpa --models gemma4:26b gpt-oss:20b
```

`benchmark.py` compares the first two models; with a third model listed, run `compare.py --schema ctpa --results results_ctpa.csv --models <a> <b> --ground-truth ground_truth_ctpa.csv` for the other pairs. Send `results_ctpa.csv` for review.
