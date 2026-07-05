"""Phase 8 — minimal local web app for the expert decision-quality study.

A professor (or colleague) runs ONE command, opens a browser, enters their name,
picks their participant number, and works through their assigned scenarios. For
each scenario they see the information for their assigned condition
(RAW / XAI / XTRAFFIC), pick one intervention, and rate confidence (1-7) and
usefulness of the shown information (1-7). Everything is appended to a JSONL file.

No accounts, no database — a name field and a JSONL file (CLAUDE.md: runnable by a
professor with one command). Progress is derived from the log, so a participant
can close the tab and resume where they left off.

Run:
  python -m xtraffic.evaluation.human_study.app
  # then open http://localhost:8000

Prereqs: scenarios.json + assignment.json must already exist (build them with
  python -m xtraffic.evaluation.human_study.scenarios).

Python 3.9 compatible. Depends only on fastapi + uvicorn (pin added in Phase 8).
"""
from __future__ import annotations

import html
import json
import os
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from ...utils.io_utils import PKG_ROOT

DATA_DIR = os.path.join(PKG_ROOT, "evaluation", "results", "human_study")
RESPONSES_PATH = os.path.join(DATA_DIR, "responses.jsonl")

app = FastAPI(title="XTraffic human study")


# ---------------------------------------------------------------------------
# Data loading (read fresh each request — the study is tiny and this keeps the
# server stateless, so restarting it never loses progress).
# ---------------------------------------------------------------------------
def _load(name: str) -> Any:
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


def _scenarios_by_id() -> Dict[str, Dict[str, Any]]:
    data = _load("scenarios.json")
    return {s["scenario_id"]: s for s in data["scenarios"]}, data


def _answered(evaluator: str) -> Set[str]:
    """scenario_ids this evaluator has already submitted (from the JSONL log)."""
    done: Set[str] = set()
    if not os.path.exists(RESPONSES_PATH):
        return done
    with open(RESPONSES_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("evaluator") == evaluator:
                done.add(r["scenario_id"])
    return done


# ---------------------------------------------------------------------------
# HTML rendering (one plain page; no templates engine, no external assets).
# ---------------------------------------------------------------------------
PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>XTraffic study</title><style>
 body{{font-family:system-ui,Arial,sans-serif;max-width:820px;margin:2rem auto;
      padding:0 1rem;color:#111;line-height:1.5}}
 .card{{border:1px solid #ccc;border-radius:8px;padding:1rem 1.25rem;margin:1rem 0}}
 .num{{font-size:1.4rem;font-weight:700}}
 .muted{{color:#666}} table{{border-collapse:collapse;width:100%}}
 th,td{{border:1px solid #ddd;padding:4px 8px;text-align:left;font-size:.92rem}}
 img{{max-width:100%;border:1px solid #eee;border-radius:6px}}
 .rec{{background:#f6f8fa;border-radius:6px;padding:.5rem .75rem;margin:.4rem 0}}
 label{{display:block;margin:.5rem 0}} .opt{{display:inline-block;margin-right:1rem}}
 button{{background:#1657c9;color:#fff;border:0;border-radius:6px;padding:.6rem 1.2rem;
        font-size:1rem;cursor:pointer}} .pill{{background:#eef;border-radius:12px;
        padding:.1rem .6rem;font-size:.8rem}}
</style></head><body>{body}</body></html>"""


def _likert(name: str, likert_max: int, prompt: str) -> str:
    opts = "".join(
        '<span class="opt"><input type="radio" name="{n}" value="{v}" required> {v}</span>'
        .format(n=name, v=v) for v in range(1, likert_max + 1))
    return '<label><b>{p}</b><br>{o}</label>'.format(p=html.escape(prompt), o=opts)


def _render_condition(cond: str, bundle: Dict[str, Any]) -> str:
    """The information block shown for the assigned condition."""
    p = bundle["prediction"]
    parts = ['<div class="card"><div class="muted">Prediction (Layer 1)</div>'
             '<div class="num">{loc}</div>'
             '{cur} mph now &rarr; <b>{pred} mph</b> in {h} min</div>'.format(
                 loc=html.escape(p["location"]), cur=p["current_speed_mph"],
                 pred=p["predicted_speed_mph"], h=p["horizon_minutes"])]

    if cond in ("XAI", "XTRAFFIC"):
        rows = "".join(
            "<tr><td>{loc}</td><td>{imp}</td><td>{spd} mph</td></tr>".format(
                loc=html.escape(r["location"]), imp=r["importance"],
                spd=r["current_speed_mph"])
            for r in bundle.get("importance_table", []))
        fig_rel = bundle.get("figure")
        fig_ok = bool(fig_rel) and os.path.exists(
            os.path.join(DATA_DIR, "figures", os.path.basename(fig_rel)))
        img = ('<img src="/figure/{fig}">'.format(fig=os.path.basename(fig_rel))
               if fig_ok else '')
        parts.append(
            '<div class="card"><div class="muted">Explanation (Layer 2)</div>'
            '{img}<p class="muted">propagation lag '
            '{lag} min &middot; confidence {conf}</p>'
            '<table><tr><th>contributing location</th><th>importance</th>'
            '<th>current speed</th></tr>{rows}</table></div>'.format(
                img=img, lag=bundle.get("propagation_lag_minutes", "-"),
                conf=bundle.get("explanation_confidence", "-"), rows=rows))

    if cond == "XTRAFFIC":
        adv = bundle.get("advisory", {})
        recs = "".join(
            '<div class="rec"><b>{a}</b> @ {loc} ({t} min)<br>'
            '<span class="muted">expected: {eff}</span></div>'.format(
                a=html.escape(r.get("action", "")), loc=html.escape(r.get("location", "")),
                t=r.get("time_window_minutes", "-"), eff=html.escape(r.get("expected_effect", "")))
            for r in adv.get("recommendations", []))
        parts.append(
            '<div class="card"><div class="muted">Advisory (Layer 3)</div>'
            '<p>{reason}</p>{recs}</div>'.format(
                reason=html.escape(adv.get("reasoning", "")), recs=recs))
    return "".join(parts)


def _start_page(assignment: Dict[str, Any]) -> str:
    n = len(assignment)
    opts = "".join('<option value="eval{i}">Participant {d}</option>'.format(i=i, d=i + 1)
                   for i in range(n))
    body = ('<h1>XTraffic decision study</h1>'
            '<p>Thank you for participating. Enter your name and select your '
            'assigned participant number, then work through each scenario.</p>'
            '<form method="get" action="/study">'
            '<label>Your name<br><input name="name" required></label>'
            '<label>Participant number<br><select name="evaluator">{opts}</select></label>'
            '<button type="submit">Begin</button></form>'.format(opts=opts))
    return PAGE.format(body=body)


def _scenario_page(evaluator: str, name: str, sid: str, cond: str,
                   scn: Dict[str, Any], likert_max: int,
                   done: int, total: int) -> str:
    candidates = scn["conditions"]["RAW"]["candidates"]
    radios = "".join(
        '<label class="opt"><input type="radio" name="choice" value="{c}" required> {label}</label>'
        .format(c=c, label=html.escape(c.replace("_", " ")))
        for c in candidates)
    body = (
        '<div class="pill">{name} &middot; scenario {d}/{t}</div>'
        '<h2>Scenario {sid} <span class="muted">({band}, {cong} congestion)</span></h2>'
        '<p class="muted">A congestion event is predicted. Review the information '
        'below, then choose the single intervention you would deploy.</p>'
        '{info}'
        '<form method="post" action="/submit">'
        '<input type="hidden" name="evaluator" value="{ev}">'
        '<input type="hidden" name="name" value="{name}">'
        '<input type="hidden" name="scenario_id" value="{sid}">'
        '<input type="hidden" name="condition" value="{cond}">'
        '<div class="card"><label><b>Which intervention do you deploy?</b><br>{radios}</label>'
        '{conf}{use}</div>'
        '<button type="submit">Submit &amp; continue</button></form>'
    ).format(
        name=html.escape(name), d=done + 1, t=total, sid=sid,
        band=scn["tod_band"], cong=scn["congestion"],
        info=_render_condition(cond, scn["conditions"][cond]),
        ev=evaluator, cond=cond, radios=radios,
        conf=_likert("confidence", likert_max,
                     "How confident are you in this choice? (1 = not at all, {m} = extremely)".format(m=likert_max)),
        use=_likert("usefulness", likert_max,
                    "How useful was the information shown? (1 = not at all, {m} = extremely)".format(m=likert_max)),
    )
    return PAGE.format(body=body)


def _done_page(name: str) -> str:
    body = ('<h1>All done — thank you, {name}!</h1>'
            '<p>Your responses have been recorded. You may close this tab.</p>'
            .format(name=html.escape(name)))
    return PAGE.format(body=body)


# ---------------------------------------------------------------------------
# Routes.
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _start_page(_load("assignment.json"))


@app.get("/study", response_class=HTMLResponse)
def study(evaluator: str, name: str) -> HTMLResponse:
    assignment = _load("assignment.json")
    if evaluator not in assignment:
        return HTMLResponse(PAGE.format(body="<p>Unknown participant.</p>"), status_code=400)
    by_id, meta = _scenarios_by_id()
    likert_max = meta["likert_max"]
    my = assignment[evaluator]                       # {scenario_id: condition}
    order = [sid for sid in my if sid in by_id]      # assignment order
    done = _answered(evaluator)
    remaining = [sid for sid in order if sid not in done]
    if not remaining:
        return HTMLResponse(_done_page(name))
    sid = remaining[0]
    return HTMLResponse(_scenario_page(
        evaluator, name, sid, my[sid], by_id[sid], likert_max,
        done=len(done), total=len(order)))


@app.post("/submit")
def submit(evaluator: str = Form(...), name: str = Form(...),
           scenario_id: str = Form(...), condition: str = Form(...),
           choice: str = Form(...), confidence: int = Form(...),
           usefulness: int = Form(...)) -> RedirectResponse:
    by_id, _ = _scenarios_by_id()
    scn = by_id[scenario_id]
    record = {
        "evaluator": evaluator,
        "name": name,
        "scenario_id": scenario_id,
        "condition": condition,
        "choice": choice,
        "ground_truth": scn["ground_truth_intervention"],
        "correct": choice == scn["ground_truth_intervention"],
        "confidence": confidence,
        "usefulness": usefulness,
        "tod_band": scn["tod_band"],
        "congestion": scn["congestion"],
        "ts": _now_iso(),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RESPONSES_PATH, "a") as f:              # append-only JSONL log
        f.write(json.dumps(record) + "\n")
    # 303 -> browser re-GETs /study, which serves the next unanswered scenario.
    return RedirectResponse(
        "/study?evaluator={}&name={}".format(evaluator, name), status_code=303)


@app.get("/figure/{fname}")
def figure(fname: str):
    # Serve only from the figures dir (basename-only guards against path escape).
    path = os.path.join(DATA_DIR, "figures", os.path.basename(fname))
    if not os.path.exists(path):
        return HTMLResponse("figure not found", status_code=404)
    return FileResponse(path)


def _now_iso() -> str:
    # Import inside the handler so module import never touches the clock (keeps
    # the module import-safe for the structure smoke test).
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def main() -> None:
    import uvicorn
    print("XTraffic human study -> http://localhost:8000  (Ctrl-C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
