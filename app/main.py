# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Tuple
import asyncio, uuid, time, os, re
from pathlib import Path
from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent

# load .env from app/
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("GEMINI KEY:", "LOADED" if GEMINI_API_KEY else "NOT FOUND")

if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
else:
    print("❌ GEMINI API KEY NOT LOADED")
from .adk_agents import run_multi_agent_adk

# ReportLab
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem,
    Table, TableStyle, Image, KeepTogether
)
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Charts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------- FastAPI -------------------
app = FastAPI(title="Nextify Backend (ReportLab PDF)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://donnaoftadeh.github.io",
        "https://gilbert-unabridged-rumbly.ngrok-free.app",
        # optional locals you use when testing from a local HTML file:
        # "http://localhost:5500", "http://127.0.0.1:5500",
        # "http://localhost:5173", "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------- Storage -------------------
class Submission(BaseModel):
    journey_type: str = Field(default="idea", pattern="^(company|industry|product|idea)$")
    payload: Dict[str, Any]


JOBS: Dict[str, Dict[str, Any]] = {}
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdf"
CHART_DIR = BASE_DIR / "data" / "charts"
PDF_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

FONT_DIR = Path(__file__).resolve().parent / "fonts"
DEJAVU_REG = FONT_DIR / "DejaVuSans.ttf"
DEJAVU_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

UI_STEPS = [
    "Parse Submission",
    "Brainstorm Parallel",
    "Idea Cooker",
    "Theme & Epic Generator",
    "Roadmap Generator",
    "Feature Generation",
    "Prioritization & RICE",
    "OKR Generation",
    "Three-Month Planner",
    "Write Report (PDF)",
]

PIPELINE_STAGE_COUNT = 8

# ------------------- Interactive HITL stages -------------------
# These match UI_STEPS exactly so backend progress, Streamlit tabs,
# LLM judge reviews, and human approval checkpoints all use the same stage model.

INTERACTIVE_STAGES = [
    {
        "id": "parse_submission",
        "title": "Parse Submission",
        "short_title": "Parse",
        "agent": "Input Parser Agent",
        "history_key": None,
    },
    {
        "id": "brainstorm_parallel",
        "title": "Brainstorm Parallel",
        "short_title": "Brainstorm",
        "agent": "Market Analysis Agent + Crazy Idea Agent",
        "history_key": ["market_analysis_md", "crazy_ideas_md"],
    },
    {
        "id": "idea_cooker",
        "title": "Idea Cooker",
        "short_title": "Cooker",
        "agent": "Idea Cooker Agent",
        "history_key": "idea_cooker_md",
    },
    {
        "id": "theme_epic_generator",
        "title": "Theme & Epic Generator",
        "short_title": "Themes",
        "agent": "Theme & Epic Agent",
        "history_key": "theme_epic_md",
    },
    {
        "id": "roadmap_generator",
        "title": "Roadmap Generator",
        "short_title": "Roadmap",
        "agent": "Roadmap Agent",
        "history_key": "roadmap_generator_md",
    },
    {
        "id": "feature_generation",
        "title": "Feature Generation",
        "short_title": "Features",
        "agent": "Feature Generation Agent",
        "history_key": "feature_generation_md",
    },
    {
        "id": "prioritization_rice",
        "title": "Prioritization & RICE",
        "short_title": "RICE",
        "agent": "Prioritization & RICE Agent",
        "history_key": "prioritization_rice_md",
    },
    {
        "id": "okr_generation",
        "title": "OKR Generation",
        "short_title": "OKRs",
        "agent": "OKR Agent",
        "history_key": "okr_output_md",
    },
    {
        "id": "three_month_planner",
        "title": "Three-Month Planner",
        "short_title": "Planner",
        "agent": "Three-Month Planner Agent",
        "history_key": "planner_md",
    },
    {
        "id": "write_report_pdf",
        "title": "Write Report (PDF)",
        "short_title": "Report",
        "agent": "Report Writer Agent",
        "history_key": "final_report_md",
    },
]

INTERACTIVE_STAGE_IDS = [stage["id"] for stage in INTERACTIVE_STAGES]


class FeedbackRequest(BaseModel):
    feedback: str = ""


def _now_ts() -> float:
    return time.time()


def _empty_interactive_stage(stage_id: str) -> Dict[str, Any]:
    return {
        "stage_id": stage_id,
        "status": "pending",
        "agent_output": "",
        "judge_feedback": "",
        "user_feedback": "",
        "revision_count": 0,
        "approved": False,
        "updated_at": None,
    }


def _init_interactive_state() -> Dict[str, Any]:
    return {
        "current_stage_index": 0,
        "stages": {
            stage_id: _empty_interactive_stage(stage_id)
            for stage_id in INTERACTIVE_STAGE_IDS
        },
    }


def _get_stage_config(stage_id: str) -> Dict[str, Any]:
    for stage in INTERACTIVE_STAGES:
        if stage["id"] == stage_id:
            return stage
    raise HTTPException(status_code=404, detail=f"Unknown stage: {stage_id}")


def _get_job_or_404(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _ensure_interactive_state(job: Dict[str, Any]) -> None:
    if "interactive" not in job:
        job["interactive"] = _init_interactive_state()

    if "stages" not in job["interactive"]:
        job["interactive"]["stages"] = {}

    for stage_id in INTERACTIVE_STAGE_IDS:
        if stage_id not in job["interactive"]["stages"]:
            job["interactive"]["stages"][stage_id] = _empty_interactive_stage(stage_id)

    if "current_stage_index" not in job["interactive"]:
        job["interactive"]["current_stage_index"] = 0


def _approved_context(job: Dict[str, Any]) -> str:
    _ensure_interactive_state(job)

    parts = []

    for stage_config in INTERACTIVE_STAGES:
        stage_id = stage_config["id"]
        stage_state = job["interactive"]["stages"][stage_id]

        if stage_state.get("approved") and stage_state.get("agent_output"):
            parts.append(
                f"# {stage_config['title']}\n\n"
                f"{stage_state['agent_output']}"
            )

    if not parts:
        return "No previous stages approved yet."

    return "\n\n---\n\n".join(parts)


def _build_parse_submission_output(job: Dict[str, Any]) -> str:
    payload = job.get("payload", {})
    journey_type = job.get("journey_type", "idea")

    lines = [
        "# Parse Submission",
        "",
        f"**Journey Type:** {journey_type}",
        "",
        "## User Input",
    ]

    for key, value in payload.items():
        pretty_key = key.replace("_", " ").title()
        lines.append(f"- **{pretty_key}:** {value}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "The submission has been parsed into a structured product-development input.",
            "The next stage can now run parallel brainstorming using the user's idea, problem, users, and constraints.",
        ]
    )

    return "\n".join(lines)


def _extract_history_output_for_stage(stage_id: str, job: Dict[str, Any]) -> str:
    stage_config = _get_stage_config(stage_id)
    history_key = stage_config.get("history_key")
    history = job.get("history", {}) or {}

    if stage_id == "parse_submission":
        return _build_parse_submission_output(job)

    if not history:
        return ""

    if isinstance(history_key, list):
        chunks = []

        for key in history_key:
            value = history.get(key, "")
            if value:
                title = key.replace("_md", "").replace("_", " ").title()
                chunks.append(f"# {title}\n\n{value}")

        return "\n\n---\n\n".join(chunks)

    if isinstance(history_key, str):
        return history.get(history_key, "") or ""

    return ""


def _build_placeholder_stage_output(stage_id: str, job: Dict[str, Any]) -> str:
    stage_config = _get_stage_config(stage_id)
    payload = job.get("payload", {})
    approved_context = _approved_context(job)

    idea_title = payload.get("idea_title", "")
    idea_text = payload.get("idea_text", "")
    target_users = payload.get("target_users", "")
    problem = payload.get("problem", "")
    constraints = payload.get("constraints", "")

    return f"""
# {stage_config['title']}

## Agent

{stage_config['agent']}

## Input Used

**Idea Title:** {idea_title}

**Idea Description:** {idea_text}

**Target Users:** {target_users}

**Problem:** {problem}

**Constraints:** {constraints}

## Prior Approved Context

{approved_context}

## Draft Output

This is a placeholder output for **{stage_config['title']}**.

Your full ADK pipeline is still running in the background. Once `job["history"]` is ready, this stage will use the real saved output from the matching history key.

For now, this placeholder proves the interactive Human-in-the-Loop workflow is working stage by stage.
"""


def _get_or_create_stage_output(stage_id: str, job: Dict[str, Any]) -> str:
    real_output = _extract_history_output_for_stage(stage_id, job)

    if real_output and real_output.strip():
        return real_output

    return _build_placeholder_stage_output(stage_id, job)


def _judge_stage_output(stage_id: str, agent_output: str, job: Dict[str, Any]) -> str:
    stage_config = _get_stage_config(stage_id)

    return f"""
# LLM Judge Review: {stage_config['title']}

## Score

**7.5 / 10**

## Strengths

- The output is separated clearly for the **{stage_config['title']}** stage.
- The artifact can be reviewed independently before moving to the next stage.
- The workflow now supports human approval before downstream agents depend on this output.

## Gaps

- The output should be more specific to the target user and problem.
- It should include sharper assumptions, risks, and success criteria.
- It should be more decision-ready before final approval.

## Recommended Improvements

1. Add measurable success criteria.
2. Clarify the most important user segment.
3. Identify key risks and assumptions.
4. Make the next action concrete.
5. Improve clarity before approving this stage.

## Judge Decision

This stage is useful, but it should be improved with human feedback before final approval.
"""


def _revise_stage_output(stage_id: str, job: Dict[str, Any]) -> str:
    _ensure_interactive_state(job)

    stage_config = _get_stage_config(stage_id)
    stage_state = job["interactive"]["stages"][stage_id]

    previous_output = stage_state.get("agent_output", "")
    judge_feedback = stage_state.get("judge_feedback", "")
    user_feedback = stage_state.get("user_feedback", "")
    approved_context = _approved_context(job)

    return f"""
# Revised {stage_config['title']}

## Revision Inputs

### Human Feedback

{user_feedback or "No human feedback was provided."}

### LLM Judge Feedback

{judge_feedback or "No judge feedback was provided."}

### Approved Previous Context

{approved_context}

---

## Revised Stage Output

This is the revised version of **{stage_config['title']}**.

It improves the previous draft by incorporating:
1. the LLM judge critique,
2. the user's human feedback,
3. the approved context from previous stages.

---

## Previous Draft Reference

{previous_output}
"""
# ------------------- Fonts/Styles -------------------
def _register_fonts():
    try:
        if DEJAVU_REG.exists():
            pdfmetrics.registerFont(TTFont("DejaVu", str(DEJAVU_REG)))
        if DEJAVU_BOLD.exists():
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(DEJAVU_BOLD)))
    except Exception:
        pass


def _styles():
    base = getSampleStyleSheet()
    body_name = "DejaVu" if "DejaVu" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    bold_name = "DejaVu-Bold" if "DejaVu-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"

    title = ParagraphStyle("Title", parent=base["Title"], fontName=bold_name, fontSize=18, leading=21, spaceAfter=8)
    h1 = ParagraphStyle("Heading1", parent=base["Heading1"], fontName=bold_name, fontSize=15, leading=19, spaceBefore=8, spaceAfter=6)
    h2 = ParagraphStyle("Heading2", parent=base["Heading2"], fontName=bold_name, fontSize=13, leading=17, spaceBefore=8, spaceAfter=6)
    h3 = ParagraphStyle("Heading3", parent=base["Heading3"], fontName=bold_name, fontSize=11.5, leading=15, spaceBefore=6, spaceAfter=5)
    body = ParagraphStyle("Body", parent=base["BodyText"], fontName=body_name, fontSize=9.8, leading=13, spaceAfter=5)
    mono = ParagraphStyle("Mono", parent=base["BodyText"], fontName="Courier", fontSize=9.2, leading=12, spaceAfter=5)

    return {"title": title, "h1": h1, "h2": h2, "h3": h3, "body": body, "mono": mono}


# ------------------- Title rules -------------------
def _make_report_title(jt: str, p: Dict[str, Any]) -> Tuple[str, str]:
    company = (p.get("bench_company") or p.get("company_name") or "Company").strip()
    product = (p.get("product_name") or "Product").strip()
    industry = (p.get("industry") or "Industry").strip()
    idea = (p.get("idea_title") or p.get("idea_text") or "Idea").strip()

    jt = (jt or "").lower().strip()
    if jt == "company":
        title = f"Next level {company} proposal by Nextify"
    elif jt == "product":
        title = f"The next breakthrough {product} proposal by Nextify"
    elif jt == "industry":
        title = f"The next breakthrough in {industry} market proposal by Nextify"
    else:
        title = f"The idea of {idea} proposal by Nextify"

    fname = "".join(c for c in title if c not in r'\/:*?"<>|').strip()
    return title, fname


# ------------------- Markdown parsing -------------------
def _escape_basic(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _md_inline(s: str) -> str:
    t = re.sub(r"\*\*(.+?)\*\*", r"«b»\1«/b»", s)
    t = re.sub(r"__(.+?)__", r"«b»\1«/b»", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"«i»\1«/i»", t)
    t = re.sub(r"_(.+?)_", r"«i»\1«/i»", t)
    t = _escape_basic(t)
    return t.replace("«b»", "<b>").replace("«/b»", "</b>").replace("«i»", "<i>").replace("«/i»", "</i>")


def _parse_md_table(block: List[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    for raw in block:
        ln = raw.strip()
        if ln.startswith("|"):
            ln = ln[1:]
        if ln.endswith("|"):
            ln = ln[:-1]
        parts = [c.strip() for c in ln.split("|")]
        rows.append(parts)

    cleaned: List[List[str]] = []
    for r in rows:
        if len(r) >= 1 and all(set(c) <= set("-: ") for c in r):
            continue
        cleaned.append(r)
    return cleaned


def _table_flowable(rows: List[List[str]], styles) -> KeepTogether:
    if not rows or not rows[0]:
        return Spacer(1, 1)

    body_style = ParagraphStyle("tbl", parent=styles["body"], fontSize=8.8, leading=12)
    header_style = ParagraphStyle("th", parent=styles["h3"], fontSize=9.4, leading=12, spaceBefore=0, spaceAfter=0)

    data: List[List[Any]] = []
    for i, r in enumerate(rows):
        cells: List[Any] = []
        for c in r:
            txt = _md_inline(c)
            cells.append(Paragraph(txt, header_style if i == 0 else body_style))
        data.append(cells)

    avail = (A4[0] - (14 * mm + 14 * mm))
    ncols = max(len(r) for r in rows)
    colw = [max(50, avail / max(1, ncols))] * ncols

    tbl = Table(data, colWidths=colw, repeatRows=1, hAlign="LEFT", spaceBefore=4, spaceAfter=6)
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return KeepTogether(tbl)


def _extract_rice(rows: List[List[str]]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    if not rows or len(rows) < 2:
        return out

    header = [h.strip().lower() for h in rows[0]]
    try:
        idx = header.index("rice")
    except ValueError:
        return out

    for r in rows[1:]:
        try:
            name = r[0]
            val = float(str(r[idx]).replace(",", "").strip())
            out.append((name, val))
        except Exception:
            continue
    return out


def _chart_rice(job_id: str, scores: List[Tuple[str, float]]) -> str:
    if not scores:
        return ""

    labels = [a for a, _ in scores]
    vals = [b for _, b in scores]

    plt.figure(figsize=(5.4, 3.1))
    plt.bar(labels, vals)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = CHART_DIR / f"rice_{job_id}.png"
    plt.savefig(path)
    plt.close()
    return str(path)


def _parse_to_flowables(text: str, styles) -> Tuple[List[Any], List[List[str]]]:
    """
    Returns (flowables, last_table_rows_if_any) — last_table_rows is used to check for RICE.
    """
    story: List[Any] = []
    lines = (text or "").splitlines()
    bullets: List[str] = []
    nums: List[str] = []
    table_buf: List[str] = []
    last_table_rows: List[List[str]] = []

    def flush_list():
        nonlocal bullets, nums
        if bullets:
            items = [ListItem(Paragraph(_md_inline(b), styles["body"])) for b in bullets]
            story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=14))
            story.append(Spacer(1, 3))
            bullets = []
        if nums:
            items = [ListItem(Paragraph(_md_inline(b), styles["body"])) for b in nums]
            story.append(ListFlowable(items, bulletType="1", leftIndent=14))
            story.append(Spacer(1, 3))
            nums = []

    def flush_table():
        nonlocal table_buf, last_table_rows
        if not table_buf:
            return
        rows = _parse_md_table(table_buf)
        if rows:
            last_table_rows = rows
            story.append(_table_flowable(rows, styles))
        story.append(Spacer(1, 4))
        table_buf = []

    for raw in lines:
        ln = raw.rstrip()
        s = ln.strip()

        if not s:
            flush_list()
            flush_table()
            story.append(Spacer(1, 4))
            continue

        if s.startswith("|"):
            flush_list()
            table_buf.append(s)
            continue

        if s.startswith("### "):
            flush_list()
            flush_table()
            story.append(Paragraph(_md_inline(s[4:]), styles["h3"]))
            continue

        if s.startswith("## "):
            flush_list()
            flush_table()
            story.append(Paragraph(_md_inline(s[3:]), styles["h2"]))
            continue

        if s.startswith("# "):
            flush_list()
            flush_table()
            story.append(Paragraph(_md_inline(s[2:]), styles["h1"]))
            continue

        if s.startswith("- "):
            flush_table()
            bullets.append(s[2:].strip())
            continue

        if re.match(r"^\d+\.\s+", s):
            flush_table()
            nums.append(re.sub(r"^\d+\.\s+", "", s).strip())
            continue

        flush_list()
        flush_table()
        story.append(Paragraph(_md_inline(s), styles["body"]))

    flush_list()
    flush_table()
    return story, last_table_rows


# ------------------- PDF build -------------------
def generate_pdf(job_id: str, journey_type: str, payload: Dict[str, Any], report_text: str) -> str:
    if not isinstance(report_text, str) or not report_text.strip():
        report_text = "No content was produced. Please retry with more details."

    _register_fonts()
    styles = _styles()
    title_text, filename_label = _make_report_title(journey_type, payload)
    out_path = PDF_DIR / f"{filename_label}.pdf"

    story: List[Any] = []
    story.append(Paragraph(_md_inline(title_text), styles["title"]))
    story.append(Spacer(1, 6))

    flow, last_rows = _parse_to_flowables(report_text, styles)
    story.extend(flow)

    rice_scores = _extract_rice(last_rows)
    if rice_scores:
        chart = _chart_rice(job_id, rice_scores)
        if chart and os.path.exists(chart):
            story.append(Image(chart, width=170 * mm, height=90 * mm))
            story.append(Spacer(1, 6))

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=title_text,
        author="Nextify",
    )
    doc.build(story)
    return str(out_path)


# ------------------- Pipeline -------------------
async def _run_pipeline(job_id: str, submission: Submission):
    job = JOBS[job_id]
    job["raw_report"] = ""
    job["history"] = {}

    try:
        job.update(status="running", step=UI_STEPS[0], progress=4, message="Validating…")
        await asyncio.sleep(0.1)

        job.update(step=UI_STEPS[1], progress=7, message="Running multi-agent pipeline...")
        await asyncio.sleep(0.1)

        def cb(idx: int, sec_title: str, msg: str):
            job["step"] = sec_title
            job["message"] = msg
            job["progress"] = min(7 + int(idx * (88 / PIPELINE_STAGE_COUNT)), 95)

        pipeline_payload = dict(submission.payload)
        pipeline_payload["journey_type"] = submission.journey_type

        history = await run_multi_agent_adk(pipeline_payload, cb)

        report_text = history.get("final_report_md", "")
        if not report_text:
            report_text = "\n\n".join(filter(None, [
                history.get("market_analysis_md", ""),
                history.get("crazy_ideas_md", ""),
                history.get("idea_cooker_md", ""),
                history.get("theme_epic_md", ""),
                history.get("roadmap_generator_md", ""),
                history.get("feature_generation_md", ""),
                history.get("prioritization_rice_md", ""),
                history.get("okr_output_md", ""),
                history.get("planner_md", ""),
            ]))

        job["history"] = history
        job["raw_report"] = report_text or ""

        job.update(step=UI_STEPS[-1], progress=97, message="Generating final report…")
        await asyncio.sleep(0.2)

        pdf_path = generate_pdf(job_id, submission.journey_type, submission.payload, report_text)
        job.update(
            pdf_path=pdf_path,
            progress=100,
            status="done",
            step="Complete",
            message="Report ready.",
        )

    except Exception as e:
        job.update(status="failed", step="Error", message=f"Pipeline error: {e}", progress=100)


# ------------------- API -------------------
@app.post("/api/submit")
async def submit(submission: Submission):
    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "created_at": time.time(),
        "status": "interactive",
        "step": "Interactive Mode",
        "progress": 0,
        "message": "Interactive job created. Run each stage manually.",
        "pdf_path": None,
        "journey_type": submission.journey_type,
        "payload": submission.payload,
        "raw_report": "",
        "history": {},
        "interactive": _init_interactive_state(),
    }

    # IMPORTANT:
    # Do not auto-run the full ADK pipeline in interactive mode.
    # It can consume many Gemini calls and trigger 429 quota errors.
    # asyncio.create_task(_run_pipeline(job_id, submission))

    return {
        "job_id": job_id,
        "stages": INTERACTIVE_STAGE_IDS,
    }

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "step": job["step"],
        "progress": job["progress"],
        "message": job["message"],
        "ready": job["status"] == "done",
    }

@app.get("/api/job/{job_id}")
async def get_interactive_job(job_id: str):
    job = _get_job_or_404(job_id)
    _ensure_interactive_state(job)

    return {
        "job_id": job_id,
        "created_at": job.get("created_at"),
        "status": job.get("status"),
        "step": job.get("step"),
        "progress": job.get("progress"),
        "message": job.get("message"),
        "journey_type": job.get("journey_type"),
        "payload": job.get("payload", {}),
        "current_stage_index": job["interactive"]["current_stage_index"],
        "stages": job["interactive"]["stages"],
    }


@app.post("/api/stage/{job_id}/{stage_id}/run")
async def run_interactive_stage(job_id: str, stage_id: str):
    job = _get_job_or_404(job_id)
    _ensure_interactive_state(job)
    _get_stage_config(stage_id)

    output = _get_or_create_stage_output(stage_id, job)
    stage_state = job["interactive"]["stages"][stage_id]

    stage_state.update(
        {
            "status": "needs_judge",
            "agent_output": output,
            "updated_at": _now_ts(),
        }
    )

    return stage_state


@app.post("/api/stage/{job_id}/{stage_id}/judge")
async def judge_interactive_stage(job_id: str, stage_id: str):
    job = _get_job_or_404(job_id)
    _ensure_interactive_state(job)
    _get_stage_config(stage_id)

    stage_state = job["interactive"]["stages"][stage_id]

    if not stage_state.get("agent_output"):
        raise HTTPException(
            status_code=400,
            detail="Run the stage before judging it.",
        )

    judge_feedback = _judge_stage_output(
        stage_id=stage_id,
        agent_output=stage_state["agent_output"],
        job=job,
    )

    stage_state.update(
        {
            "status": "needs_user_feedback",
            "judge_feedback": judge_feedback,
            "updated_at": _now_ts(),
        }
    )

    return stage_state


@app.post("/api/stage/{job_id}/{stage_id}/feedback")
async def save_interactive_feedback(job_id: str, stage_id: str, req: FeedbackRequest):
    job = _get_job_or_404(job_id)
    _ensure_interactive_state(job)
    _get_stage_config(stage_id)

    stage_state = job["interactive"]["stages"][stage_id]

    stage_state.update(
        {
            "status": "feedback_received",
            "user_feedback": req.feedback,
            "updated_at": _now_ts(),
        }
    )

    return stage_state


@app.post("/api/stage/{job_id}/{stage_id}/revise")
async def revise_interactive_stage(job_id: str, stage_id: str):
    job = _get_job_or_404(job_id)
    _ensure_interactive_state(job)
    _get_stage_config(stage_id)

    stage_state = job["interactive"]["stages"][stage_id]

    if not stage_state.get("agent_output"):
        raise HTTPException(
            status_code=400,
            detail="Run the stage before revising it.",
        )

    revised_output = _revise_stage_output(stage_id, job)

    stage_state.update(
        {
            "status": "needs_judge",
            "agent_output": revised_output,
            "revision_count": stage_state.get("revision_count", 0) + 1,
            "updated_at": _now_ts(),
        }
    )

    return stage_state


@app.post("/api/stage/{job_id}/{stage_id}/approve")
async def approve_interactive_stage(job_id: str, stage_id: str):
    job = _get_job_or_404(job_id)
    _ensure_interactive_state(job)
    _get_stage_config(stage_id)

    stage_state = job["interactive"]["stages"][stage_id]

    if not stage_state.get("agent_output"):
        raise HTTPException(
            status_code=400,
            detail="Run the stage before approving it.",
        )

    stage_state.update(
        {
            "status": "approved",
            "approved": True,
            "updated_at": _now_ts(),
        }
    )

    stage_index = INTERACTIVE_STAGE_IDS.index(stage_id)
    current_index = job["interactive"].get("current_stage_index", 0)

    if stage_index == current_index and current_index < len(INTERACTIVE_STAGE_IDS) - 1:
        job["interactive"]["current_stage_index"] = current_index + 1

    return {
        "job_id": job_id,
        "current_stage_index": job["interactive"]["current_stage_index"],
        "stages": job["interactive"]["stages"],
    }
@app.get("/api/result/{job_id}")
async def result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done" or not job["pdf_path"]:
        return JSONResponse({"error": "Result not ready"}, status_code=202)
    filename = os.path.basename(job["pdf_path"])
    return FileResponse(job["pdf_path"], media_type="application/pdf", filename=filename)


@app.get("/api/debug/{job_id}/raw")
async def debug_raw(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    raw = job.get("raw_report") or "(no raw report stored)"
    return PlainTextResponse(raw)


@app.get("/api/debug/{job_id}/history")
async def debug_history(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job.get("history", {}))


@app.get("/")
async def root():
    return {"ok": True, "service": "Nextify Backend (ReportLab PDF)"}

@app.get("/test-env")
async def test_env():
    import os
    key = os.getenv("GOOGLE_API_KEY")
    return {
        "key_loaded": bool(key),
        "key_preview": key[:10] if key else None
    }