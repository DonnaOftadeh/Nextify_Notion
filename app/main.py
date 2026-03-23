# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Tuple
import asyncio, uuid, time, os, re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv("app/.env")

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
        "status": "queued",
        "step": "Queued",
        "progress": 0,
        "message": "Job queued.",
        "pdf_path": None,
        "journey_type": submission.journey_type,
        "raw_report": "",
        "history": {},
    }
    asyncio.create_task(_run_pipeline(job_id, submission))
    return {"job_id": job_id}


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