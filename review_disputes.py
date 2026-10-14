"""Local page to settle the disagreements between the two models, one disputed field at a time.

    uv run review_disputes.py --input "C:\\data\\combined_extracted.csv"

Opens http://127.0.0.1:8765 in the browser. The page runs on this machine only: the server listens on 127.0.0.1, the
page loads nothing from the internet, and nothing is sent anywhere. For every report flagged for review it shows the
report text and, for each field the two models disagreed on, both answers with their quotes highlighted in the report.
Choose the left model (gemma4:26b), the right model (gpt-oss:20b) or Other (neither, or not sure; add a note).
Keys: left arrow, right arrow, O for Other, up and down arrows to move between disputes.

Every choice is written at once to <input name>_adjudication.csv next to the input, so closing the page or stopping the
script (Ctrl+C) loses nothing; the next start shows your earlier choices and continues. A choice can be changed; the
latest one counts. "Write reviewed file" (or --write-reviewed without opening the page) writes
<input name>_reviewed.csv: the extracted file with your choices applied to <field>_present, _mentioned and _evidence,
plus review_status (no review needed / reviewed / reviewed, with other / pending / no model answer) and review_notes.

Both answers come from the disputed_fields column that run_reports.py writes: for every field the two models
disagreed on, each model's answer and quote. Fields not listed there were answered the same by both models.
"""

import argparse
import csv
import json
import os
import re
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
ADJUDICATION_COLUMNS = ["decided_at", "row_number", "extraction_id", "field", "chosen", "final_present", "note",
                        "left_model", "left_present", "left_quote", "right_model", "right_present", "right_quote"]


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
                          "field": field, "label": schema.FIELD_DEFINITIONS[field], "embolism": field in EMBOLISM_FIELDS,
                          "left": {"model": left, "present": answers[left][0], "quote": answers[left][1]},
                          "right": {"model": right, "present": answers[right][0], "quote": answers[right][1]}})
    items.sort(key=lambda item: (int(float(item["row"])), ORDER[item["field"]]))
    return table, items, reports, no_answer


def read_decisions(path: Path) -> dict:
    decisions = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for record in csv.DictReader(handle):
                decisions[f"{record['row_number']}|{record['field']}"] = record   # the latest choice counts
    return decisions


def write_reviewed(table: pd.DataFrame, items: list, decisions: dict, no_answer: list, output: Path) -> Path:
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
            if decision["chosen"] == "other":
                states.append("other")
                continue
            side = item["left"] if decision["chosen"] == item["left"]["model"] else item["right"]
            index, field = position[row], item["field"]
            def as_cell(value: bool, column: str) -> str:   # keep Excel's TRUE/FALSE if the file was re-saved by Excel
                return str(value).upper() if str(reviewed.at[index, column]).isupper() else str(value)
            reviewed.at[index, field + "_present"] = as_cell(side["present"], field + "_present")
            reviewed.at[index, field + "_evidence"] = side["quote"]
            reviewed.at[index, field + "_mentioned"] = as_cell(side["present"] or bool(side["quote"]), field + "_mentioned")
            states.append("decided")
        status[row] = "pending" if "pending" in states else ("reviewed, with other" if "other" in states else "reviewed")
    reviewed["review_status"] = [status.get(row, "no model answer" if row in no_answer else "no review needed")
                                 for row in reviewed["row_number"]]
    reviewed["review_notes"] = [" | ".join(notes.get(row, [])) for row in reviewed["row_number"]]
    temporary = output.with_name(output.name + ".tmp")
    reviewed.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, output)
    return output


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dispute review</title>
<style>
:root { --ground:#f3f5f8; --panel:#fff; --ink:#18212c; --muted:#5a6575; --rule:#d8dee6; --left:#1d5fa8; --left-soft:#e3edf9;
  --right:#a85d00; --right-soft:#fdf0dd; --both:#e4e0f5; --present:#b42318; --absent:#2d6a45; --accent:#18212c; }
@media (prefers-color-scheme: dark) { :root { --ground:#0f141a; --panel:#161d26; --ink:#e3e8ee; --muted:#98a3b1; --rule:#27313d;
  --left:#82b2f2; --left-soft:#18283c; --right:#f0a64a; --right-soft:#33240f; --both:#2a2440; --present:#f2786c; --absent:#72c593; --accent:#e3e8ee; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--ground); color:var(--ink); font:14.5px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--panel); border-bottom:1px solid var(--rule); padding:10px 18px;
  display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; }
header h1 { font-size:16px; margin:0; }
.muted { color:var(--muted); font-size:13px; }
.bar { flex:1 1 180px; height:8px; background:var(--rule); border-radius:4px; overflow:hidden; min-width:120px; }
.bar > div { height:100%; background:var(--absent); width:0; }
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
.field { font:600 15px/1.3 ui-monospace,Menlo,Consolas,monospace; }
.question { margin:6px 0 14px; }
.cards { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
@media (max-width:560px) { .cards { grid-template-columns:1fr; } }
.card { border:2px solid var(--rule); border-radius:10px; padding:12px; display:grid; gap:8px; align-content:start; }
.card.left { border-color:var(--left); } .card.right { border-color:var(--right); }
.card.chosen { box-shadow:0 0 0 3px var(--accent); }
.model { font-weight:600; }
.pill { display:inline-block; font:600 12px/1 ui-monospace,Menlo,Consolas,monospace; padding:4px 8px; border-radius:999px; }
.pill.yes { color:var(--present); border:1px solid var(--present); } .pill.no { color:var(--absent); border:1px solid var(--absent); }
.quote { font-style:italic; color:var(--muted); }
.actions { display:grid; grid-template-columns:1fr 0.8fr 1fr; gap:10px; margin-top:16px; }
.actions button { padding:10px 8px; font-weight:600; overflow-wrap:anywhere; }
.actions .l { border-color:var(--left); } .actions .r { border-color:var(--right); }
#note { width:100%; margin-top:10px; padding:8px 10px; border:1px solid var(--rule); border-radius:6px; background:var(--panel); }
#status { min-height:1.5em; margin-top:8px; }
.keys { margin-top:14px; }
kbd { border:1px solid var(--rule); border-bottom-width:2px; border-radius:4px; padding:0 5px; font-size:12px; }
</style></head><body>
<header>
  <h1>Dispute review</h1><span class="muted" id="file"></span>
  <div class="bar" title="decided"><div id="bar"></div></div><span id="progress" class="muted"></span>
  <select id="filter" aria-label="Which disputes"><option value="all">All disputed fields</option><option value="embolism">Embolism fields only</option><option value="undecided">Undecided only</option></select>
  <button id="prev">&uarr; Previous</button><button id="next">Next &darr;</button><button id="nextOpen">Next undecided</button>
  <button id="export">Write reviewed file</button>
</header>
<main>
  <section class="panel"><div class="muted" id="where"></div><div id="report"></div></section>
  <section class="panel">
    <div class="field" id="field"></div><div class="question muted" id="question"></div>
    <div class="cards"><div class="card left" id="cardL"></div><div class="card right" id="cardR"></div></div>
    <div class="actions"><button class="l" id="chooseL"></button><button id="chooseO">Other / not sure</button><button class="r" id="chooseR"></button></div>
    <input id="note" placeholder="Note (optional; useful with Other)" aria-label="Note">
    <div id="status" class="muted" role="status"></div>
    <div class="keys muted"><kbd>&larr;</kbd> left model right &nbsp; <kbd>&rarr;</kbd> right model right &nbsp; <kbd>O</kbd> other &nbsp; <kbd>&uarr;</kbd><kbd>&darr;</kbd> previous / next</div>
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
function card(side, answer, chosen) {
  return `<div class="model">${esc(answer.model)}</div>
    <div><span class="pill ${answer.present ? "yes" : "no"}">${answer.present ? "PRESENT" : "ABSENT"}</span></div>
    <div class="quote">${answer.quote ? "&ldquo;" + esc(answer.quote) + "&rdquo;" : "(no quote)"}</div>`;
}
function rebuild() {
  const f = $("filter").value;
  view = S.items.filter(it => f === "all" || (f === "embolism" && it.embolism) || (f === "undecided" && !S.decisions[it.id]));
  if (at >= view.length) at = Math.max(0, view.length - 1);
}
function show() {
  const decided = Object.keys(S.decisions).filter(id => S.byId[id]).length;
  $("progress").textContent = `${decided} of ${S.items.length} decided`;
  $("bar").style.width = (S.items.length ? 100 * decided / S.items.length : 0) + "%";
  if (!view.length) { $("field").textContent = "Nothing to show with this filter."; $("cardL").innerHTML = $("cardR").innerHTML = ""; $("report").textContent = ""; return; }
  const it = view[at], d = S.decisions[it.id];
  const sameRow = S.items.filter(x => x.row === it.row);
  $("where").textContent = `Row ${it.row} · dispute ${sameRow.indexOf(it) + 1} of ${sameRow.length} in this report · item ${at + 1} of ${view.length}`;
  $("field").textContent = it.field;
  $("question").textContent = "Means: " + it.label;
  $("cardL").innerHTML = card("left", it.left); $("cardR").innerHTML = card("right", it.right);
  $("cardL").classList.toggle("chosen", !!d && d.chosen === it.left.model);
  $("cardR").classList.toggle("chosen", !!d && d.chosen === it.right.model);
  $("chooseL").textContent = `\u2190 ${it.left.model} is right`; $("chooseR").textContent = `${it.right.model} is right \u2192`;
  $("chooseO").style.boxShadow = d && d.chosen === "other" ? "0 0 0 3px var(--accent)" : "";
  $("note").value = d ? (d.note || "") : "";
  $("status").textContent = d ? `Saved: ${d.chosen === "other" ? "Other" : d.chosen + " is right"}` : "Not decided yet";
  $("report").innerHTML = marked(S.reports[it.row] || "", it.left.quote, it.right.quote);
  const first = $("report").querySelector("mark"); if (first) first.scrollIntoView({block: "center"}); else $("report").scrollTop = 0;
  $("extra").textContent = S.no_answer.length ? `Rows flagged without any model answer (check them in the file): ${S.no_answer.join(", ")}` : "";
}
async function decide(choice) {
  if (!view.length) return;
  const it = view[at], chosen = choice === "left" ? it.left.model : choice === "right" ? it.right.model : "other";
  const r = await fetch("/api/decide", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id: it.id, chosen, note: $("note").value})});
  const reply = await r.json();
  if (!reply.ok) { $("status").textContent = "Not saved: " + reply.error; return; }
  S.decisions[it.id] = reply.decision;
  const f = $("filter").value;
  if (f === "undecided") { rebuild(); } else { const next = view.findIndex((x, i) => i > at && !S.decisions[x.id]); if (next >= 0) at = next; }
  show();
}
$("chooseL").onclick = () => decide("left"); $("chooseR").onclick = () => decide("right"); $("chooseO").onclick = () => decide("other");
$("prev").onclick = () => { if (at > 0) { at--; show(); } };
$("next").onclick = () => { if (at < view.length - 1) { at++; show(); } };
$("nextOpen").onclick = () => { const i = view.findIndex((x, k) => k > at && !S.decisions[x.id]); const j = i >= 0 ? i : view.findIndex(x => !S.decisions[x.id]); if (j >= 0) { at = j; show(); } };
$("filter").onchange = () => { at = 0; rebuild(); show(); };
$("export").onclick = async () => { $("status").textContent = "Writing\u2026"; const r = await fetch("/api/export", {method: "POST"}); const reply = await r.json();
  $("status").textContent = reply.ok ? `Wrote ${reply.path}` : "Not written: " + reply.error; };
document.addEventListener("keydown", e => {
  if (e.target === $("note")) { if (e.key === "Enter") { e.preventDefault(); decide("other"); } return; }
  if (e.key === "ArrowLeft") { e.preventDefault(); decide("left"); }
  else if (e.key === "ArrowRight") { e.preventDefault(); decide("right"); }
  else if (e.key === "o" || e.key === "O") { e.preventDefault(); $("note").focus(); }
  else if (e.key === "ArrowUp") { e.preventDefault(); $("prev").click(); }
  else if (e.key === "ArrowDown") { e.preventDefault(); $("next").click(); }
});
fetch("/api/state").then(r => r.json()).then(state => {
  S = state; S.byId = Object.fromEntries(S.items.map(it => [it.id, it])); $("file").textContent = S.input;
  rebuild(); const first = view.findIndex(it => !S.decisions[it.id]); at = first >= 0 ? first : 0; show();
});
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Local page to settle the two models' disagreements, one field at a time.")
    parser.add_argument("--input", type=Path, required=True, help="the _extracted.csv written by run_reports.py")
    parser.add_argument("--text-column", default="ReportBody", help="column with the report text (default ReportBody)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.add_argument("--write-reviewed", action="store_true", help="only write <input name>_reviewed.csv from the saved choices, then stop")
    args = parser.parse_args()

    table, items, reports, no_answer = load(args.input, args.text_column)
    decisions_path = args.input.with_name(args.input.stem + "_adjudication.csv")
    reviewed_path = args.input.with_name(args.input.stem + "_reviewed.csv")
    decisions = read_decisions(decisions_path)
    by_id = {item["id"]: item for item in items}
    print(f"{args.input.name}: {len(items)} disputed fields in {len(reports)} reports; {sum(1 for key in decisions if key in by_id)} already decided"
          + (f"; {len(no_answer)} flagged rows have no model answer" if no_answer else ""))
    if args.write_reviewed:
        print(f"Wrote {write_reviewed(table, items, decisions, no_answer, reviewed_path)}")
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
                chosen = request.get("chosen")
                if item is None or chosen not in (item["left"]["model"], item["right"]["model"], "other"):
                    return self.reply({"ok": False, "error": "unknown dispute or choice"})
                final = "" if chosen == "other" else str((item["left"] if chosen == item["left"]["model"] else item["right"])["present"])
                record = {"decided_at": datetime.now().isoformat(timespec="seconds"), "row_number": item["row"],
                          "extraction_id": item["extraction_id"], "field": item["field"], "chosen": chosen, "final_present": final,
                          "note": (request.get("note") or "").strip(),
                          "left_model": item["left"]["model"], "left_present": str(item["left"]["present"]), "left_quote": item["left"]["quote"],
                          "right_model": item["right"]["model"], "right_present": str(item["right"]["present"]), "right_quote": item["right"]["quote"]}
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
                        return self.reply({"ok": False, "error": f"{decisions_path.name} is open in another program (Excel?); close it and choose again"})
                    decisions[item["id"]] = record
                return self.reply({"ok": True, "decision": record})
            if self.path == "/api/export":
                with lock:
                    try:
                        path = write_reviewed(table, items, decisions, no_answer, reviewed_path)
                    except PermissionError:
                        return self.reply({"ok": False, "error": f"{reviewed_path.name} is open in another program (Excel?); close it and try again"})
                return self.reply({"ok": True, "path": str(path)})
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
    print(f"Open {url} (this machine only). Your choices are saved in {decisions_path.name}. Stop with Ctrl+C.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Every choice is saved; run the same command to continue.")


if __name__ == "__main__":
    main()
