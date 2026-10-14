"""Local page to settle the disagreements between the two models, one disputed field at a time, as yes/no questions.

    uv run review_disputes.py --input "C:\\data\\combined_extracted.csv"

Opens http://127.0.0.1:8765 in the browser. The page runs on this machine only: the server listens on 127.0.0.1, the
page loads nothing from the internet, and nothing is sent anywhere. For every report flagged for review it shows the
report text and, for each field the two models disagreed on, a plain question ("Is any clot described as non-occlusive?")
with each model's answer and quote highlighted in the report. Answer Yes, No or Can't tell (with an optional note).
Keys: Y, N, C; left and right arrows take the left or right model's answer; up and down arrows move between disputes.

Every answer is written at once to <input name>_adjudication.csv next to the input, so closing the page or stopping the
script (Ctrl+C) loses nothing and the next start continues where you left off. An answer can be changed; the latest one
counts. Choices saved by the first version of this page (left / right / other) are converted on start, after a backup.

"Write result files" (or --write-reviewed without opening the page) writes, next to the input:
  <input name>_reviewed.csv       the extracted file with your answers applied (Can't tell leaves the cell empty),
                                  plus review_status and review_notes
  <input name>_analysis.csv       the original columns except free text (any column with values over 300 characters, such
                                  as the report, findings or impression), row_number, extraction_id, the 64 findings as
                                  1 / 0 (empty when unknown), review_status
  <input name>_who_was_right.csv  per model and per field: answers you settled, how many the model had right, accuracy
Both models' answers come from the disputed_fields column that run_reports.py writes for every disagreement.
"""

import argparse
import csv
import json
import os
import re
import shutil
import threading
import webbrowser
from collections import defaultdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

import schema_ctpa as schema

EMBOLISM_FIELDS = [field.removesuffix("_present") for field in schema.KEY_FIELDS]
ORDER = {name: index for index, name in enumerate(EMBOLISM_FIELDS + [f for f in schema.FINDING_NAMES if f not in EMBOLISM_FIELDS])}
VOTE = re.compile(r"(?:: |; )([A-Za-z0-9._-]+:[A-Za-z0-9._-]+)=(True|False) \(")
ADJUDICATION_COLUMNS = ["decided_at", "row_number", "extraction_id", "field", "question", "answer", "final_present", "note",
                        "left_model", "left_present", "left_quote", "left_correct",
                        "right_model", "right_present", "right_quote", "right_correct"]
# field -> (short name, yes/no question asked on the page)
QUESTIONS = {
    "suboptimal_study": ("Study limited or nondiagnostic", "Does the report call the study limited, suboptimal or nondiagnostic for pulmonary embolism?"),
    "poor_contrast_opacification": ("Poor contrast opacification", "Does the report say the pulmonary arteries are poorly or inadequately opacified?"),
    "motion_artifact": ("Motion artifact", "Does the report describe motion or breathing artifact?"),
    "pulmonary_embolism": ("Pulmonary embolism", "Is there a pulmonary embolism, acute or chronic?"),
    "pe_acute": ("Acute embolism", "Does the report call the embolism acute?"),
    "pe_chronic": ("Chronic embolism", "Does the report call the embolism chronic or chronic-appearing?"),
    "pe_saddle": ("Saddle embolus", "Is there a saddle embolus?"),
    "pe_main": ("Main artery level", "Is there clot in a main pulmonary artery, right or left?"),
    "pe_lobar": ("Lobar level", "Is there clot in a lobar artery (any lobe, or an interlobar artery)?"),
    "pe_segmental": ("Segmental level", "Is there clot in a segmental artery (any lobe)?"),
    "pe_subsegmental": ("Subsegmental level", "Is there clot in a subsegmental artery?"),
    "pe_right": ("Right side", "Is there clot in the arteries of the right lung?"),
    "pe_left": ("Left side", "Is there clot in the arteries of the left lung?"),
    "pe_multiple": ("More than one artery", "Is clot described in more than one artery, or more than one embolus described?"),
    "pe_occlusive": ("Occlusive clot", "Is any clot described as occlusive (completely blocking the artery)?"),
    "pe_nonocclusive": ("Non-occlusive clot", "Is any clot described as non-occlusive (partly blocking the artery)?"),
    "right_heart_strain": ("Right heart strain", "Is there right heart strain (RV dilation, RV/LV above 1, septal flattening or bowing, contrast reflux into the IVC)?"),
    "pulmonary_artery_enlargement": ("Enlarged pulmonary artery", "Is the pulmonary artery described as enlarged or dilated?"),
    "perfusion_defect": ("Perfusion defect", "Is there a perfusion (iodine map) defect?"),
    "pulmonary_infarct": ("Pulmonary infarct", "Is there a pulmonary infarct (including 'infarct versus pneumonia')?"),
    "consolidation": ("Consolidation", "Is there consolidation in the lung?"),
    "ground_glass_opacity": ("Ground-glass opacity", "Is there ground-glass opacity in the lung?"),
    "atelectasis": ("Atelectasis", "Is there atelectasis?"),
    "pulmonary_nodule": ("Pulmonary nodule", "Is there a pulmonary nodule?"),
    "emphysema": ("Emphysema", "Is there emphysema?"),
    "mosaic_attenuation": ("Mosaic attenuation", "Is there mosaic attenuation of the lungs?"),
    "pleural_effusion": ("Pleural effusion", "Is there a pleural effusion?"),
    "pneumothorax": ("Pneumothorax", "Is there a pneumothorax?"),
    "pericardial_effusion": ("Pericardial effusion", "Is there a pericardial effusion?"),
    "cardiomegaly": ("Enlarged heart", "Is the heart, or one of its chambers, enlarged?"),
    "lymphadenopathy": ("Enlarged lymph nodes", "Are there enlarged mediastinal, hilar or axillary lymph nodes?"),
    "pe_main_right": ("Right main pulmonary artery", "Is there clot in the right main pulmonary artery?"),
    "pe_main_left": ("Left main pulmonary artery", "Is there clot in the left main pulmonary artery?"),
    "pe_lobar_interlobar_right": ("Right interlobar artery", "Is there clot in the right interlobar artery?"),
    "pe_lobar_interlobar_left": ("Left interlobar artery", "Is there clot in the left interlobar artery?"),
}
for _key, _words in schema.LOBE_WORDS.items():
    if _key != "lingula":
        QUESTIONS[f"pe_lobar_{_key}"] = (f"{_words.capitalize()} artery (lobar)",
                                         f"Is there clot in the {_words} pulmonary artery itself (the lobar artery, not its segmental branches)?")
    QUESTIONS[f"pe_segmental_{_key}"] = (f"{_words.capitalize()}, segmental", f"Is there clot in a segmental artery of the {_words}?"
                                         + (" (the lingula counts as part of the left upper lobe)" if _key == "left_upper" else ""))
for _name in schema.FINDING_NAMES:
    if _name.startswith("pe_segment_"):
        _lobe, _segment = _name[len("pe_segment_"):].split("_", 1)
        QUESTIONS[_name] = (f"{_segment.replace('_', ' ').capitalize()} segment, {schema.SEGMENT_WORDS[_lobe]}",
                            f"Is there clot in the {_segment.replace('_', ' ')} segmental artery of the {schema.SEGMENT_WORDS[_lobe]}?")
assert set(QUESTIONS) == set(schema.FINDING_NAMES)


def parse_disputes(text: str) -> dict:
    """field -> {model: (present, quote)} from a disputed_fields cell. A field name only starts a dispute when a model's
    vote follows it, so quote text cannot be mistaken for another field."""
    starts = sorted((match.start(), name) for name in schema.FINDING_NAMES
                    for match in re.finditer(r"(?:^| \| )" + re.escape(name) + r": (?=[A-Za-z0-9._-]+:[A-Za-z0-9._-]+=(?:True|False) \()", text))
    disputes = {}
    for index, (position, name) in enumerate(starts):
        segment = text[position:starts[index + 1][0] if index + 1 < len(starts) else len(text)]
        votes = list(VOTE.finditer(segment))
        answers = {}
        for number, vote in enumerate(votes):
            quote = segment[vote.end():votes[number + 1].start() if number + 1 < len(votes) else len(segment)].rstrip()
            quote = quote[:-1] if quote.endswith(")") else quote
            answers[vote.group(1)] = (vote.group(2) == "True", "" if quote.strip() in ("nan", "None", "") else quote.strip())
        if len(answers) == 2:
            disputes[name] = answers
    return disputes


def load(input_path: Path, text_column: str):
    table = pd.read_csv(input_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for column in ["row_number", "needs_review", "disputed_fields", "models"]:
        if column not in table.columns:
            raise SystemExit(f"ERROR: {input_path.name} has no column {column!r}; pass the _extracted.csv written by run_reports.py")
    if text_column not in table.columns:
        raise SystemExit(f"ERROR: no column {text_column!r} with the report text; pass --text-column with its name")
    items, reports, no_answer = [], {}, []
    for row in table[table["needs_review"].str.strip().isin(["1", "1.0"])].to_dict("records"):
        disputes = parse_disputes(row["disputed_fields"])
        if not disputes:
            no_answer.append(row["row_number"])
            continue
        reports[row["row_number"]] = row[text_column]
        listed = [model.strip() for model in row["models"].split("+") if model.strip()]
        for field in sorted(disputes, key=lambda name: ORDER[name]):
            answers = disputes[field]
            left = listed[0] if listed and listed[0] in answers else sorted(answers)[0]
            right = next(model for model in answers if model != left)
            items.append({"id": f"{row['row_number']}|{field}", "row": row["row_number"], "extraction_id": row.get("extraction_id", ""),
                          "field": field, "name": QUESTIONS[field][0], "question": QUESTIONS[field][1], "embolism": field in EMBOLISM_FIELDS,
                          "left": {"model": left, "present": answers[left][0], "quote": answers[left][1]},
                          "right": {"model": right, "present": answers[right][0], "quote": answers[right][1]}})
    items.sort(key=lambda item: (int(float(item["row"])), ORDER[item["field"]]))
    return table, items, reports, no_answer


def make_record(decided_at: str, row, extraction_id: str, field: str, present, note: str,
                left_model: str, left_present: bool, left_quote: str, right_model: str, right_present: bool, right_quote: str) -> dict:
    """present: True (Yes), False (No) or None (Can't tell). A model is right when its answer equals yours."""
    correct = lambda answer: "" if present is None else ("1" if answer == present else "0")
    return {"decided_at": decided_at, "row_number": str(row), "extraction_id": extraction_id, "field": field,
            "question": QUESTIONS[field][1], "answer": "cant_tell" if present is None else ("yes" if present else "no"),
            "final_present": "" if present is None else str(present), "note": note,
            "left_model": left_model, "left_present": str(left_present), "left_quote": left_quote, "left_correct": correct(left_present),
            "right_model": right_model, "right_present": str(right_present), "right_quote": right_quote, "right_correct": correct(right_present)}


def write_records(path: Path, records: list):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ADJUDICATION_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def read_decisions(path: Path) -> dict:
    """id -> latest record. A file saved by the first version (left / right / other) is converted after a backup copy."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        records = list(reader)
        columns = reader.fieldnames or []
    if "answer" not in columns and "chosen" in columns:
        backup = path.with_name(path.stem + "_backup_first_version.csv")
        shutil.copyfile(path, backup)
        converted = []
        for old in records:
            left_present, right_present = old["left_present"] == "True", old["right_present"] == "True"
            present = left_present if old["chosen"] == old["left_model"] else right_present if old["chosen"] == old["right_model"] else None
            converted.append(make_record(old["decided_at"], old["row_number"], old["extraction_id"], old["field"], present, old["note"],
                                         old["left_model"], left_present, old["left_quote"], old["right_model"], right_present, old["right_quote"]))
        write_records(path, converted)
        records = converted
        print(f"Converted {len(records)} earlier choices to yes / no / can't tell; the original file is kept as {backup.name}")
    return {f"{record['row_number']}|{record['field']}": record for record in records}   # the latest answer counts


def write_outputs(table: pd.DataFrame, items: list, decisions: dict, no_answer: list, text_column: str, input_path: Path) -> list:
    """The reviewed file, the 1/0 analysis file and the who-was-right table, each written to a temporary file first."""
    stem = input_path.with_name(input_path.stem)
    reviewed = table.copy()
    position = {row: index for index, row in enumerate(reviewed["row_number"])}
    by_row = defaultdict(list)
    for item in items:
        by_row[item["row"]].append(item)
    status, notes = {}, defaultdict(list)
    for row, row_items in by_row.items():
        states = []
        for item in row_items:
            decision = decisions.get(item["id"])
            if decision is None:
                states.append("pending")
                continue
            if decision["note"]:
                notes[row].append(f"{item['field']}: {decision['note']}")
            index, field = position[row], item["field"]
            def as_cell(value, column: str) -> str:   # keep Excel's TRUE/FALSE if the file was re-saved by Excel
                return "" if value is None else (str(value).upper() if str(reviewed.at[index, column]).isupper() else str(value))
            if decision["answer"] == "cant_tell":
                present, quote = None, ""
                states.append("cant_tell")
            else:
                present = decision["answer"] == "yes"
                quote = next((side["quote"] for side in (item["left"], item["right"]) if side["present"] == present), "")
                states.append("decided")
            reviewed.at[index, field + "_present"] = as_cell(present, field + "_present")
            reviewed.at[index, field + "_mentioned"] = as_cell(None if present is None else (present or bool(quote)), field + "_mentioned")
            reviewed.at[index, field + "_evidence"] = quote
        status[row] = ("pending" if "pending" in states else "reviewed, with can't tell" if "cant_tell" in states else "reviewed")
    reviewed["review_status"] = [status.get(row, "no model answer" if row in no_answer else "no review needed") for row in reviewed["row_number"]]
    reviewed["review_notes"] = [" | ".join(notes.get(row, [])) for row in reviewed["row_number"]]

    long_text = {column for column in table.columns if table[column].str.len().max() > 300}   # report text, findings, impression...
    keep = [column for column in table.columns
            if column != text_column and column not in long_text
            and column not in ("needs_review", "resolved_by", "models", "disputed_fields", "code_version", "seconds")
            and not any(column == field + suffix for field in schema.FINDING_NAMES for suffix in ("_mentioned", "_present", "_evidence"))]
    analysis = reviewed[keep].copy()
    for field in schema.FINDING_NAMES:
        analysis[field] = reviewed[field + "_present"].map(lambda value: "1" if str(value).strip().lower() == "true"
                                                           else "0" if str(value).strip().lower() == "false" else "")
    analysis["review_status"] = reviewed["review_status"]

    tally = defaultdict(lambda: [0, 0, 0])   # (model, field) -> [answers settled, model right, can't tell]
    for item in items:
        decision = decisions.get(item["id"])
        if decision is None:
            continue
        for side in ("left", "right"):
            model = decision[side + "_model"]
            for key in [(model, item["field"]), (model, "ALL embolism fields" if item["embolism"] else "ALL other findings"), (model, "ALL")]:
                if decision["answer"] == "cant_tell":
                    tally[key][2] += 1
                else:
                    tally[key][0] += 1
                    tally[key][1] += decision[side + "_correct"] == "1"
    who = pd.DataFrame([{"model": model, "field": field, "question": QUESTIONS.get(field, ("", ""))[1], "answers_settled": settled,
                         "model_right": right, "accuracy": round(right / settled, 3) if settled else "", "cant_tell": cant}
                        for (model, field), (settled, right, cant) in tally.items()])
    if not who.empty:
        who["order"] = who["field"].map(lambda field: {"ALL": -3, "ALL embolism fields": -2, "ALL other findings": -1}.get(field, ORDER.get(field, 99)))
        who = who.sort_values(["order", "model"]).drop(columns="order")
    else:
        who = pd.DataFrame(columns=["model", "field", "question", "answers_settled", "model_right", "accuracy", "cant_tell"])

    written = []
    for frame, suffix in [(reviewed, "_reviewed.csv"), (analysis, "_analysis.csv"), (who, "_who_was_right.csv")]:
        path = Path(str(stem) + suffix)
        temporary = path.with_name(path.name + ".tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
        written.append(path)
    return written


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dispute review</title>
<style>
:root { --ground:#f3f5f8; --panel:#fff; --ink:#18212c; --muted:#5a6575; --rule:#d8dee6; --left:#1d5fa8; --left-soft:#e3edf9;
  --right:#a85d00; --right-soft:#fdf0dd; --both:#e4e0f5; --accent:#18212c; --done:#2d6a45; }
@media (prefers-color-scheme: dark) { :root { --ground:#0f141a; --panel:#161d26; --ink:#e3e8ee; --muted:#98a3b1; --rule:#27313d;
  --left:#82b2f2; --left-soft:#18283c; --right:#f0a64a; --right-soft:#33240f; --both:#2a2440; --accent:#e3e8ee; --done:#72c593; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--ground); color:var(--ink); font:14.5px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--rule); padding:10px 18px;
  display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; }
header h1 { font-size:16px; margin:0; }
.muted { color:var(--muted); font-size:13px; }
.bar { flex:1 1 160px; height:8px; background:var(--rule); border-radius:4px; overflow:hidden; min-width:120px; }
.bar > div { height:100%; background:var(--done); width:0; }
button, select, input { font:inherit; color:inherit; }
button { border:1px solid var(--rule); background:var(--panel); border-radius:6px; padding:6px 12px; cursor:pointer; }
button:hover { border-color:var(--muted); }
button:focus-visible, select:focus-visible, input:focus-visible { outline:2px solid var(--left); outline-offset:2px; }
main { display:grid; grid-template-columns:minmax(0,1.1fr) minmax(0,1fr); gap:18px; padding:18px; max-width:1500px; margin:0 auto; }
@media (max-width:900px) { main { grid-template-columns:minmax(0,1fr); } }
.panel { background:var(--panel); border:1px solid var(--rule); border-radius:10px; padding:16px; }
#report { white-space:pre-wrap; word-break:break-word; font:13px/1.55 ui-monospace,Menlo,Consolas,monospace; max-height:calc(100vh - 150px); overflow:auto; }
mark { border-radius:3px; padding:0 1px; color:inherit; }
mark.m1 { background:var(--left-soft); box-shadow:inset 0 -2px 0 var(--left); }
mark.m2 { background:var(--right-soft); box-shadow:inset 0 -2px 0 var(--right); }
mark.m3 { background:var(--both); box-shadow:inset 0 -2px 0 var(--muted); }
.name { font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }
.question { font-size:18px; font-weight:600; line-height:1.35; margin:4px 0 14px; text-wrap:balance; }
.cards { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
@media (max-width:560px) { .cards { grid-template-columns:1fr; } }
.card { border:2px solid var(--rule); border-radius:10px; padding:12px; display:grid; gap:6px; align-content:start; }
.card.left { border-color:var(--left); } .card.right { border-color:var(--right); }
.card.agrees { box-shadow:0 0 0 3px var(--accent); }
.says { font-size:15px; } .says b { font-size:17px; letter-spacing:.03em; }
.quote { font-style:italic; color:var(--muted); }
.actions { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-top:16px; }
.actions button { padding:10px 8px; display:grid; gap:2px; }
.actions button b { font-size:16px; } .actions button span { font-size:12px; color:var(--muted); }
.actions button.picked { box-shadow:0 0 0 3px var(--accent); }
#note { width:100%; margin-top:10px; padding:8px 10px; border:1px solid var(--rule); border-radius:6px; background:var(--panel); }
#status { min-height:1.5em; margin-top:8px; }
.keys { margin-top:12px; }
kbd { border:1px solid var(--rule); border-bottom-width:2px; border-radius:4px; padding:0 5px; font-size:12px; }
</style></head><body>
<header>
  <h1>Dispute review</h1><span class="muted" id="file"></span>
  <div class="bar" title="answered"><div id="bar"></div></div><span id="progress" class="muted"></span><span id="tally" class="muted"></span>
  <select id="filter" aria-label="Which disputes"><option value="all">All disputed fields</option><option value="embolism">Embolism fields only</option><option value="open">Not answered yet</option></select>
  <button id="prev">&uarr; Previous</button><button id="next">Next &darr;</button><button id="nextOpen">Next unanswered</button>
  <button id="export">Write result files</button>
</header>
<main>
  <section class="panel"><div class="muted" id="where"></div><div id="report"></div></section>
  <section class="panel">
    <div class="name" id="name"></div><div class="question" id="question"></div>
    <div class="cards"><div class="card left" id="cardL"></div><div class="card right" id="cardR"></div></div>
    <div class="actions">
      <button id="yes"><b>Yes</b><span id="yesWho"></span></button>
      <button id="no"><b>No</b><span id="noWho"></span></button>
      <button id="cant"><b>Can't tell</b><span>leaves the cell empty</span></button>
    </div>
    <input id="note" placeholder="Note (optional)" aria-label="Note">
    <div id="status" class="muted" role="status"></div>
    <div class="keys muted"><kbd>Y</kbd> yes &nbsp; <kbd>N</kbd> no &nbsp; <kbd>C</kbd> can't tell &nbsp; <kbd>&larr;</kbd><kbd>&rarr;</kbd> the left or right model's answer &nbsp; <kbd>&uarr;</kbd><kbd>&darr;</kbd> previous / next</div>
    <div class="muted" id="extra" style="margin-top:10px"></div>
  </section>
</main>
<script>
let S = null, view = [], at = 0;
const $ = id => document.getElementById(id);
const esc = s => s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const reEsc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
function ranges(text, quote) {
  if (!quote) return [];
  const words = quote.trim().split(/\s+/).filter(Boolean); if (!words.length) return [];
  const re = new RegExp(words.map(reEsc).join("\\s+"), "gi"); const out = []; let m;
  while ((m = re.exec(text)) && out.length < 50) { out.push([m.index, m.index + m[0].length]); if (!m[0].length) re.lastIndex++; }
  return out;
}
function marked(text, leftQuote, rightQuote) {
  const cls = new Uint8Array(text.length);
  for (const [a, b] of ranges(text, leftQuote)) for (let i = a; i < b; i++) cls[i] |= 1;
  for (const [a, b] of ranges(text, rightQuote)) for (let i = a; i < b; i++) cls[i] |= 2;
  let html = "", i = 0;
  while (i < text.length) { let j = i; while (j < text.length && cls[j] === cls[i]) j++;
    const seg = esc(text.slice(i, j)); html += cls[i] ? `<mark class="m${cls[i]}">${seg}</mark>` : seg; i = j; }
  return html;
}
const card = a => `<div class="says">${esc(a.model)} says <b>${a.present ? "YES" : "NO"}</b></div>
  <div class="quote">${a.quote ? "&ldquo;" + esc(a.quote) + "&rdquo;" : "(no quote)"}</div>`;
function rebuild() {
  const f = $("filter").value;
  view = S.items.filter(it => f === "all" || (f === "embolism" && it.embolism) || (f === "open" && !S.decisions[it.id]));
  if (at >= view.length) at = Math.max(0, view.length - 1);
}
function tally() {
  const right = {}; let cant = 0;
  for (const it of S.items) { const d = S.decisions[it.id]; if (!d) continue;
    if (d.answer === "cant_tell") { cant++; continue; }
    for (const side of ["left", "right"]) { const m = d[side + "_model"]; right[m] = (right[m] || 0) + (d[side + "_correct"] === "1" ? 1 : 0); } }
  return Object.entries(right).map(([m, n]) => `${m} right ${n}`).concat(cant ? [`can't tell ${cant}`] : []).join(" · ");
}
function show() {
  const answered = S.items.filter(it => S.decisions[it.id]).length;
  $("progress").textContent = `${answered} of ${S.items.length} answered`;
  $("bar").style.width = (S.items.length ? 100 * answered / S.items.length : 0) + "%";
  $("tally").textContent = tally();
  $("extra").textContent = S.no_answer.length ? `Rows flagged without any model answer (check them in the file): ${S.no_answer.join(", ")}` : "";
  if (!view.length) { $("question").textContent = "Nothing to show with this filter."; $("name").textContent = ""; $("cardL").innerHTML = $("cardR").innerHTML = ""; $("report").textContent = ""; return; }
  const it = view[at], d = S.decisions[it.id];
  const sameRow = S.items.filter(x => x.row === it.row);
  $("where").textContent = `Row ${it.row} · question ${sameRow.indexOf(it) + 1} of ${sameRow.length} for this report · ${at + 1} of ${view.length} in this list`;
  $("name").textContent = `${it.name} · ${it.field}`;
  $("question").textContent = it.question;
  $("cardL").innerHTML = card(it.left); $("cardR").innerHTML = card(it.right);
  const yesBy = [it.left, it.right].filter(a => a.present).map(a => a.model), noBy = [it.left, it.right].filter(a => !a.present).map(a => a.model);
  $("yesWho").textContent = yesBy.length ? `as ${yesBy.join(" and ")} said` : ""; $("noWho").textContent = noBy.length ? `as ${noBy.join(" and ")} said` : "";
  const answer = d ? d.answer : null;
  $("yes").classList.toggle("picked", answer === "yes"); $("no").classList.toggle("picked", answer === "no"); $("cant").classList.toggle("picked", answer === "cant_tell");
  $("cardL").classList.toggle("agrees", !!d && d.left_correct === "1"); $("cardR").classList.toggle("agrees", !!d && d.right_correct === "1");
  $("note").value = d ? (d.note || "") : "";
  $("status").textContent = d ? `Saved: ${answer === "cant_tell" ? "can't tell" : answer}` : "Not answered yet";
  $("report").innerHTML = marked(S.reports[it.row] || "", it.left.quote, it.right.quote);
  const first = $("report").querySelector("mark"); if (first) first.scrollIntoView({block: "center"}); else $("report").scrollTop = 0;
}
async function answer(value) {
  if (!view.length) return;
  const it = view[at];
  const r = await fetch("/api/decide", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id: it.id, answer: value, note: $("note").value})});
  const reply = await r.json();
  if (!reply.ok) { $("status").textContent = "Not saved: " + reply.error; return; }
  S.decisions[it.id] = reply.decision;
  if ($("filter").value === "open") { rebuild(); } else { const next = view.findIndex((x, i) => i > at && !S.decisions[x.id]); if (next >= 0) at = next; }
  show();
}
$("yes").onclick = () => answer("yes"); $("no").onclick = () => answer("no"); $("cant").onclick = () => answer("cant_tell");
$("prev").onclick = () => { if (at > 0) { at--; show(); } };
$("next").onclick = () => { if (at < view.length - 1) { at++; show(); } };
$("nextOpen").onclick = () => { const i = view.findIndex((x, k) => k > at && !S.decisions[x.id]); const j = i >= 0 ? i : view.findIndex(x => !S.decisions[x.id]); if (j >= 0) { at = j; show(); } };
$("filter").onchange = () => { at = 0; rebuild(); show(); };
$("export").onclick = async () => { $("status").textContent = "Writing…"; const r = await fetch("/api/export", {method: "POST"}); const reply = await r.json();
  $("status").textContent = reply.ok ? "Wrote " + reply.paths.join(", ") : "Not written: " + reply.error; };
document.addEventListener("keydown", e => {
  if (e.target === $("note") || e.metaKey || e.ctrlKey || e.altKey) return;
  const it = view[at]; const k = e.key.toLowerCase();
  if (k === "y") { e.preventDefault(); answer("yes"); }
  else if (k === "n") { e.preventDefault(); answer("no"); }
  else if (k === "c") { e.preventDefault(); answer("cant_tell"); }
  else if (e.key === "ArrowLeft" && it) { e.preventDefault(); answer(it.left.present ? "yes" : "no"); }
  else if (e.key === "ArrowRight" && it) { e.preventDefault(); answer(it.right.present ? "yes" : "no"); }
  else if (e.key === "ArrowUp") { e.preventDefault(); $("prev").click(); }
  else if (e.key === "ArrowDown") { e.preventDefault(); $("next").click(); }
});
fetch("/api/state").then(r => r.json()).then(state => {
  S = state; $("file").textContent = S.input;
  rebuild(); const first = view.findIndex(it => !S.decisions[it.id]); at = first >= 0 ? first : 0; show();
});
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Local page to settle the two models' disagreements as yes/no questions.")
    parser.add_argument("--input", type=Path, required=True, help="the _extracted.csv written by run_reports.py")
    parser.add_argument("--text-column", default="ReportBody", help="column with the report text (default ReportBody)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.add_argument("--write-reviewed", action="store_true", help="only write the three result files from the saved answers, then stop")
    args = parser.parse_args()

    table, items, reports, no_answer = load(args.input, args.text_column)
    decisions_path = args.input.with_name(args.input.stem + "_adjudication.csv")
    decisions = read_decisions(decisions_path)
    by_id = {item["id"]: item for item in items}
    print(f"{args.input.name}: {len(items)} disputed fields in {len(reports)} reports; {sum(1 for key in decisions if key in by_id)} already answered"
          + (f"; {len(no_answer)} flagged rows have no model answer" if no_answer else ""))
    if args.write_reviewed:
        for path in write_outputs(table, items, decisions, no_answer, args.text_column, args.input):
            print(f"Wrote {path}")
        return
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *arguments):
            pass

        def reply(self, payload, content_type="application/json", code=200):
            body = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.reply(PAGE, "text/html")
            elif self.path == "/api/state":
                self.reply({"input": args.input.name, "items": items, "reports": reports, "no_answer": no_answer,
                            "decisions": {key: value for key, value in decisions.items() if key in by_id}})
            else:
                self.reply({"error": "not found"}, code=404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            request = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/decide":
                item = by_id.get(request.get("id"))
                answer = request.get("answer")
                if item is None or answer not in ("yes", "no", "cant_tell"):
                    return self.reply({"ok": False, "error": "unknown question or answer"})
                present = None if answer == "cant_tell" else answer == "yes"
                record = make_record(datetime.now().isoformat(timespec="seconds"), item["row"], item["extraction_id"], item["field"],
                                     present, (request.get("note") or "").strip(), item["left"]["model"], item["left"]["present"],
                                     item["left"]["quote"], item["right"]["model"], item["right"]["present"], item["right"]["quote"])
                with lock:
                    try:
                        new_file = not decisions_path.exists()
                        with decisions_path.open("a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as handle:
                            writer = csv.DictWriter(handle, fieldnames=ADJUDICATION_COLUMNS)
                            if new_file:
                                writer.writeheader()
                            writer.writerow(record)
                            handle.flush()
                            os.fsync(handle.fileno())
                    except PermissionError:
                        return self.reply({"ok": False, "error": f"{decisions_path.name} is open in another program (Excel?); close it and answer again"})
                    decisions[item["id"]] = record
                return self.reply({"ok": True, "decision": record})
            if self.path == "/api/export":
                with lock:
                    try:
                        paths = write_outputs(table, items, decisions, no_answer, args.text_column, args.input)
                    except PermissionError as error:
                        return self.reply({"ok": False, "error": f"a result file is open in another program (Excel?); close it and try again ({error.filename})"})
                return self.reply({"ok": True, "paths": [path.name for path in paths]})
            self.reply({"error": "not found"}, code=404)

    for port in range(args.port, args.port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"ERROR: no free port between {args.port} and {args.port + 19}")
    url = f"http://127.0.0.1:{port}/"
    print(f"Open {url} (this machine only). Your answers are saved in {decisions_path.name}. Stop with Ctrl+C.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Every answer is saved; run the same command to continue.")


if __name__ == "__main__":
    main()
