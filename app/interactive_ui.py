"""
Nextify Interactive UI

Modern LLM-style workflow UI:
- Main area shows current final output only.
- Human feedback, LLM judge feedback, and combined revision are explicit.
- Final Accept & Continue button is at the bottom.
- Sidebar shows workflow, versions, judge feedback, and feedback history.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests
import streamlit as st

from io import BytesIO
from datetime import datetime
import re
import textwrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    KeepTogether,
)
from xml.sax.saxutils import escape

API_BASE = "http://127.0.0.1:8000"

STAGES = [
    {"id": "parse_submission", "title": "Parse Submission", "short_title": "Parse", "agent": "Input Parser Agent", "desc": "Turn the raw idea form into a clean product brief."},
    {"id": "brainstorm_parallel", "title": "Brainstorm Parallel", "short_title": "Brainstorm", "agent": "MarketAnalysisAgent + CrazyIdeaAgent", "desc": "Generate market-grounded and breakthrough ideas from the accepted parsed brief."},
    {"id": "idea_cooker", "title": "Idea Cooker", "short_title": "Cooker", "agent": "IdeaCookerAgent", "desc": "Score, explain, and recommend the best direction."},
    {"id": "theme_epic_generator", "title": "Theme & Epic Generator", "short_title": "Themes", "agent": "ThemeEpicAgent", "desc": "Create themes and epics from the accepted idea direction."},
    {"id": "roadmap_generator", "title": "Roadmap Generator", "short_title": "Roadmap", "agent": "RoadmapAgent", "desc": "Create a phased roadmap."},
    {"id": "feature_generation", "title": "Feature Generation", "short_title": "Features", "agent": "FeatureGenerationAgent", "desc": "Generate MVP and future features."},
    {"id": "prioritization_rice", "title": "Prioritization & RICE", "short_title": "RICE", "agent": "PrioritizationAgent", "desc": "Prioritize features using RICE."},
    {"id": "okr_generation", "title": "OKR Generation", "short_title": "OKRs", "agent": "OKRAgent", "desc": "Generate measurable OKRs."},
    {"id": "three_month_planner", "title": "Three-Month Planner", "short_title": "Planner", "agent": "PlannerAgent", "desc": "Create the 3-month execution plan."},
    {"id": "write_report_pdf", "title": "Write Report", "short_title": "Report", "agent": "ReportWriterAgent", "desc": "Assemble the final product plan."},
]


st.set_page_config(
    page_title="Nextify",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; max-width: 1180px; }
    .hero {
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 20px 22px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 8px 28px rgba(15,23,42,.06);
        margin-bottom: 18px;
    }
    .stage-chip {
        display: inline-block;
        padding: 7px 11px;
        border-radius: 999px;
        margin: 4px 5px 4px 0;
        font-size: 13px;
        border: 1px solid #ddd;
    }
    .approved { background: #ecfdf5; border-color: #a7f3d0; }
    .current { background: #fff7ed; border-color: #fdba74; font-weight: 700; }
    .ready { background: #eff6ff; border-color: #bfdbfe; }
    .pending { background: #f8fafc; color: #64748b; }

    .output-card {
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 28px;
        background: white;
        box-shadow: 0 8px 28px rgba(15,23,42,.05);
        margin-bottom: 20px;
    }
    .summary-card {
        border-left: 5px solid #2563eb;
        background: #eff6ff;
        padding: 14px 16px;
        border-radius: 14px;
        margin: 14px 0;
    }
    .review-card {
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 20px;
        background: #fbfdff;
        margin-top: 18px;
    }
    .accept-card {
        border: 1px solid #bbf7d0;
        border-radius: 22px;
        padding: 20px;
        background: #f0fdf4;
        margin-top: 18px;
    }
    .muted { color: #64748b; font-size: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.caption("AI PRODUCT INNOVATION WORKSPACE")
    st.markdown("# 🚀 Nextify")
    st.markdown(
        "Turn an idea into a researched, evaluated, "
        "and actionable product strategy."
    )



def api_get(path: str) -> Optional[Dict[str, Any]]:
    try:
        res = requests.get(f"{API_BASE}{path}", timeout=180)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not connect to backend: {exc}")
        return None

    if res.status_code >= 400:
        st.error(f"Backend error {res.status_code}: {res.text}")
        return None

    return res.json()


def api_post(path: str, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        res = requests.post(f"{API_BASE}{path}", json=data or {}, timeout=300)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not connect to backend: {exc}")
        return None

    if res.status_code >= 400:
        st.error(f"Backend error {res.status_code}: {res.text}")
        return None

    return res.json()


def clean_pdf_text(text: str) -> str:
    """Return plain text suitable for filenames and simple PDF labels."""
    if not text:
        return ""

    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 - \2", text)
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text.strip()


def _inline_markdown_to_reportlab(text: str) -> str:
    """Convert a small, safe subset of markdown to ReportLab Paragraph markup."""
    if not text:
        return ""

    # Preserve links before escaping.
    links = []

    def store_link(match):
        label = match.group(1)
        url = match.group(2)
        token = f"__NEXTIFY_LINK_{len(links)}__"
        links.append((token, label, url))
        return token

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", store_link, text)
    text = escape(text)

    # Bold + inline code.
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(
        r"`([^`]+)`",
        r'<font name="Courier">\1</font>',
        text,
    )

    for token, label, url in links:
        token_escaped = escape(token)
        label_escaped = escape(label)
        url_escaped = escape(url, {'"': "&quot;"})
        text = text.replace(
            token_escaped,
            f'<link href="{url_escaped}" color="#2563EB">{label_escaped}</link>',
        )

    return text


def _is_markdown_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if not cells:
        return False

    return all(
        bool(re.fullmatch(r":?-{3,}:?", cell))
        for cell in cells
        if cell
    )


def _markdown_table_rows(lines: list[str]) -> list[list[str]]:
    rows = []

    for line in lines:
        if _is_markdown_table_separator(line):
            continue

        cells = [
            cell.strip()
            for cell in line.strip().strip("|").split("|")
        ]

        if cells:
            rows.append(cells)

    return rows


def _build_pdf_styles():
    styles = getSampleStyleSheet()

    navy = colors.HexColor("#0F172A")
    slate = colors.HexColor("#475569")
    blue = colors.HexColor("#2563EB")
    pale_blue = colors.HexColor("#EFF6FF")
    border = colors.HexColor("#E2E8F0")
    soft = colors.HexColor("#F8FAFC")
    green = colors.HexColor("#166534")

    return {
        "brand": ParagraphStyle(
            "NextifyBrand",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=blue,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "cover_title": ParagraphStyle(
            "NextifyCoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=34,
            textColor=navy,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "cover_subtitle": ParagraphStyle(
            "NextifyCoverSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=14,
            leading=20,
            textColor=slate,
            alignment=TA_CENTER,
            spaceAfter=26,
        ),
        "cover_meta": ParagraphStyle(
            "NextifyCoverMeta",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=15,
            textColor=slate,
            alignment=TA_CENTER,
        ),
        "toc_title": ParagraphStyle(
            "NextifyTOCTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=24,
            textColor=navy,
            spaceAfter=16,
        ),
        "toc_item": ParagraphStyle(
            "NextifyTOCItem",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=16,
            textColor=navy,
            leftIndent=4,
            spaceAfter=6,
        ),
        "section_kicker": ParagraphStyle(
            "NextifySectionKicker",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=blue,
            spaceAfter=6,
        ),
        "h1": ParagraphStyle(
            "NextifyH1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=25,
            textColor=navy,
            spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "NextifyH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=navy,
            spaceBefore=9,
            spaceAfter=7,
        ),
        "h3": ParagraphStyle(
            "NextifyH3",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=blue,
            spaceBefore=7,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "NextifyBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14.5,
            textColor=navy,
            spaceAfter=7,
        ),
        "bullet": ParagraphStyle(
            "NextifyBullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=14,
            leftIndent=13,
            firstLineIndent=-7,
            bulletIndent=3,
            textColor=navy,
            spaceAfter=4,
        ),
        "number": ParagraphStyle(
            "NextifyNumber",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=14,
            leftIndent=14,
            firstLineIndent=-9,
            textColor=navy,
            spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "NextifyCallout",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14.5,
            textColor=navy,
            backColor=pale_blue,
            borderColor=border,
            borderWidth=0.5,
            borderPadding=9,
            spaceBefore=5,
            spaceAfter=10,
        ),
        "small": ParagraphStyle(
            "NextifySmall",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=slate,
        ),
        "green": green,
        "navy": navy,
        "slate": slate,
        "blue": blue,
        "pale_blue": pale_blue,
        "border": border,
        "soft": soft,
    }


def _markdown_to_flowables(markdown_text: str, pdf_styles: dict) -> list:
    """Render useful markdown structures as native ReportLab flowables."""
    flowables = []

    if not markdown_text:
        return flowables

    lines = markdown_text.splitlines()
    i = 0

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        if not line:
            flowables.append(Spacer(1, 3 * mm))
            i += 1
            continue

        # Ignore markdown fences but keep their content readable.
        if line.startswith("```"):
            i += 1
            continue

        # Markdown table.
        if "|" in line and i + 1 < len(lines) and _is_markdown_table_separator(lines[i + 1]):
            table_lines = [line, lines[i + 1]]
            i += 2

            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                table_lines.append(lines[i])
                i += 1

            rows = _markdown_table_rows(table_lines)

            if rows:
                max_cols = max(len(row) for row in rows)
                normalized = [
                    row + [""] * (max_cols - len(row))
                    for row in rows
                ]

                page_width = A4[0] - 34 * mm
                col_width = page_width / max_cols

                table_data = []
                for row_index, row in enumerate(normalized):
                    style = pdf_styles["small"]
                    if row_index == 0:
                        style = ParagraphStyle(
                            f"NextifyTableHeader{len(flowables)}",
                            parent=pdf_styles["small"],
                            fontName="Helvetica-Bold",
                            textColor=colors.white,
                        )

                    table_data.append(
                        [
                            Paragraph(_inline_markdown_to_reportlab(cell), style)
                            for cell in row
                        ]
                    )

                table = Table(
                    table_data,
                    colWidths=[col_width] * max_cols,
                    repeatRows=1,
                    hAlign="LEFT",
                )

                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), pdf_styles["navy"]),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pdf_styles["soft"]]),
                            ("GRID", (0, 0), (-1, -1), 0.4, pdf_styles["border"]),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ]
                    )
                )

                flowables.append(table)
                flowables.append(Spacer(1, 4 * mm))

            continue

        # Headings.
        if line.startswith("### "):
            flowables.append(
                Paragraph(
                    _inline_markdown_to_reportlab(line[4:]),
                    pdf_styles["h3"],
                )
            )
            i += 1
            continue

        if line.startswith("## "):
            flowables.append(
                Paragraph(
                    _inline_markdown_to_reportlab(line[3:]),
                    pdf_styles["h2"],
                )
            )
            i += 1
            continue

        if line.startswith("# "):
            flowables.append(
                Paragraph(
                    _inline_markdown_to_reportlab(line[2:]),
                    pdf_styles["h1"],
                )
            )
            i += 1
            continue

        # Bullet points.
        if re.match(r"^[-*]\s+", line):
            content = re.sub(r"^[-*]\s+", "", line)
            flowables.append(
                Paragraph(
                    f"• {_inline_markdown_to_reportlab(content)}",
                    pdf_styles["bullet"],
                )
            )
            i += 1
            continue

        # Numbered lists.
        number_match = re.match(r"^(\d+)[.)]\s+(.*)", line)
        if number_match:
            number = number_match.group(1)
            content = number_match.group(2)
            flowables.append(
                Paragraph(
                    f"<b>{number}.</b> {_inline_markdown_to_reportlab(content)}",
                    pdf_styles["number"],
                )
            )
            i += 1
            continue

        # Important callouts.
        lower = line.lower()
        if (
            lower.startswith("recommendation:")
            or lower.startswith("critical risk:")
            or lower.startswith("risk:")
            or lower.startswith("decision:")
            or lower.startswith("winning concept:")
        ):
            flowables.append(
                Paragraph(
                    _inline_markdown_to_reportlab(line),
                    pdf_styles["callout"],
                )
            )
            i += 1
            continue

        # Normal paragraph.
        flowables.append(
            Paragraph(
                _inline_markdown_to_reportlab(line),
                pdf_styles["body"],
            )
        )
        i += 1

    return flowables


def build_nextify_pdf(
    title: str,
    sections: list[tuple[str, str]],
) -> bytes:
    """
    Build a polished Nextify PDF with:
    - branded cover page
    - document overview
    - table of contents
    - one section per agent/stage
    - native tables, headings and bullets
    - page headers, footers and page numbers
    """
    buffer = BytesIO()
    pdf_styles = _build_pdf_styles()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title=clean_pdf_text(title),
        author="Nextify",
        subject="Nextify Initial Product Requirements Document",
    )

    story = []

    # --------------------------------------------------------
    # COVER PAGE
    # --------------------------------------------------------
    product_name = clean_pdf_text(title)

    if " - " in product_name:
        document_name, product_name = product_name.split(" - ", 1)
    else:
        document_name = "Nextify Product Document"

    story.append(Spacer(1, 28 * mm))
    story.append(Paragraph("NEXTIFY", pdf_styles["brand"]))
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            _inline_markdown_to_reportlab(document_name),
            pdf_styles["cover_title"],
        )
    )
    story.append(
        Paragraph(
            _inline_markdown_to_reportlab(product_name),
            pdf_styles["cover_subtitle"],
        )
    )

    cover_box = Table(
        [
            [
                Paragraph("<b>Prepared by</b>", pdf_styles["small"]),
                Paragraph("Nextify AI Product Innovation Workspace", pdf_styles["small"]),
            ],
            [
                Paragraph("<b>Generated</b>", pdf_styles["small"]),
                Paragraph(datetime.now().strftime("%d %B %Y, %H:%M"), pdf_styles["small"]),
            ],
            [
                Paragraph("<b>Document</b>", pdf_styles["small"]),
                Paragraph("Initial Product Requirements Document", pdf_styles["small"]),
            ],
            [
                Paragraph("<b>Sections</b>", pdf_styles["small"]),
                Paragraph(str(len(sections)), pdf_styles["small"]),
            ],
        ],
        colWidths=[35 * mm, 95 * mm],
        hAlign="CENTER",
    )

    cover_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), pdf_styles["soft"]),
                ("BOX", (0, 0), (-1, -1), 0.6, pdf_styles["border"]),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, pdf_styles["border"]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(Spacer(1, 12 * mm))
    story.append(cover_box)
    story.append(Spacer(1, 16 * mm))
    story.append(
        Paragraph(
            "AI-generated product strategy with human-in-the-loop review, "
            "critical evaluation, and versioned product decisions.",
            pdf_styles["cover_meta"],
        )
    )
    story.append(PageBreak())

    # --------------------------------------------------------
    # DOCUMENT OVERVIEW + CONTENTS
    # --------------------------------------------------------
    story.append(Paragraph("Document Overview", pdf_styles["toc_title"]))
    story.append(
        Paragraph(
            "This Initial PRD consolidates the latest available outputs "
            "from the Nextify product-development workflow. Each section "
            "preserves the agent or stage responsible for the output.",
            pdf_styles["body"],
        )
    )

    overview = Table(
        [
            [
                Paragraph("<b>Product</b>", pdf_styles["small"]),
                Paragraph(_inline_markdown_to_reportlab(product_name), pdf_styles["small"]),
            ],
            [
                Paragraph("<b>Workflow outputs</b>", pdf_styles["small"]),
                Paragraph(str(len(sections)), pdf_styles["small"]),
            ],
            [
                Paragraph("<b>Status</b>", pdf_styles["small"]),
                Paragraph("Initial PRD / Working Product Plan", pdf_styles["small"]),
            ],
        ],
        colWidths=[42 * mm, 105 * mm],
        hAlign="LEFT",
    )

    overview.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), pdf_styles["pale_blue"]),
                ("BOX", (0, 0), (-1, -1), 0.5, pdf_styles["border"]),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, pdf_styles["border"]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(Spacer(1, 4 * mm))
    story.append(overview)
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("Contents", pdf_styles["toc_title"]))

    for number, (section_title, _) in enumerate(sections, start=1):
        story.append(
            Paragraph(
                f"<b>{number:02d}</b>&nbsp;&nbsp;&nbsp;"
                f"{_inline_markdown_to_reportlab(section_title)}",
                pdf_styles["toc_item"],
            )
        )

    # --------------------------------------------------------
    # AGENT / STAGE SECTIONS
    # --------------------------------------------------------
    for number, (section_title, section_content) in enumerate(sections, start=1):
        story.append(PageBreak())

        story.append(
            Paragraph(
                f"SECTION {number:02d}",
                pdf_styles["section_kicker"],
            )
        )

        story.append(
            Paragraph(
                _inline_markdown_to_reportlab(section_title),
                pdf_styles["h1"],
            )
        )

        story.append(
            Table(
                [["NEXTIFY AGENT OUTPUT"]],
                colWidths=[45 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), pdf_styles["pale_blue"]),
                        ("TEXTCOLOR", (0, 0), (-1, -1), pdf_styles["blue"]),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("BOX", (0, 0), (-1, -1), 0.4, pdf_styles["border"]),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                ),
            )
        )
        story.append(Spacer(1, 5 * mm))

        story.extend(
            _markdown_to_flowables(
                section_content,
                pdf_styles,
            )
        )

    # --------------------------------------------------------
    # LAST PAGE
    # --------------------------------------------------------
    story.append(PageBreak())
    story.append(Spacer(1, 32 * mm))
    story.append(Paragraph("NEXTIFY", pdf_styles["brand"]))
    story.append(
        Paragraph(
            "Initial Product Requirements Document",
            pdf_styles["cover_title"],
        )
    )
    story.append(
        Paragraph(
            "Generated from the complete Nextify agent workflow.",
            pdf_styles["cover_subtitle"],
        )
    )
    story.append(Spacer(1, 10 * mm))
    story.append(
        Paragraph(
            "Use this document as a working product artifact. "
            "Validate assumptions, evidence, prioritization, feasibility, "
            "and success metrics before execution.",
            pdf_styles["callout"],
        )
    )

    # --------------------------------------------------------
    # PAGE DECORATION
    # --------------------------------------------------------
    def draw_first_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(pdf_styles["blue"])
        canvas.rect(0, A4[1] - 7 * mm, A4[0], 7 * mm, fill=1, stroke=0)

        canvas.setFillColor(pdf_styles["slate"])
        canvas.setFont("Helvetica", 7.5)
        canvas.drawCentredString(
            A4[0] / 2,
            10 * mm,
            "Nextify • AI Product Innovation Workspace",
        )
        canvas.restoreState()

    def draw_later_pages(canvas, doc):
        canvas.saveState()

        page_number = canvas.getPageNumber()

        canvas.setStrokeColor(pdf_styles["border"])
        canvas.setLineWidth(0.5)
        canvas.line(
            17 * mm,
            A4[1] - 13 * mm,
            A4[0] - 17 * mm,
            A4[1] - 13 * mm,
        )

        canvas.setFillColor(pdf_styles["blue"])
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(
            17 * mm,
            A4[1] - 10 * mm,
            "NEXTIFY",
        )

        canvas.setFillColor(pdf_styles["slate"])
        canvas.setFont("Helvetica", 7.5)
        canvas.drawRightString(
            A4[0] - 17 * mm,
            A4[1] - 10 * mm,
            "Initial Product Requirements Document",
        )

        canvas.setStrokeColor(pdf_styles["border"])
        canvas.line(
            17 * mm,
            13 * mm,
            A4[0] - 17 * mm,
            13 * mm,
        )

        canvas.setFillColor(pdf_styles["slate"])
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(
            17 * mm,
            9 * mm,
            "Generated by Nextify",
        )
        canvas.drawRightString(
            A4[0] - 17 * mm,
            9 * mm,
            f"Page {page_number}",
        )

        canvas.restoreState()

    document.build(
        story,
        onFirstPage=draw_first_page,
        onLaterPages=draw_later_pages,
    )

    buffer.seek(0)
    return buffer.getvalue()

def collect_report_sections(
    job: Optional[Dict[str, Any]],
) -> list[tuple[str, str]]:
    """Collect accepted outputs, falling back to current outputs."""

    if not job:
        return []

    stages = job.get("stages") or {}
    sections = []

    for stage in STAGES:
        stage_id = stage["id"]
        stage_state = stages.get(stage_id) or {}

        content = (
            stage_state.get("accepted_output")
            or stage_state.get("agent_output")
            or ""
        )

        if content.strip():
            sections.append(
                (
                    f"{stage['agent']} — {stage['title']}",
                    content,
                )
            )

    return sections

def rerun_app() -> None:
    st.rerun()


def get_stage(stage_id: str) -> Dict[str, str]:
    for stage in STAGES:
        if stage["id"] == stage_id:
            return stage
    return STAGES[0]


def get_job() -> Optional[Dict[str, Any]]:
    job_id = st.session_state.get("job_id")
    if not job_id:
        return None
    return api_get(f"/api/job/{job_id}")


def reset() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    rerun_app()


def submit_new_job() -> None:
    idea_title = st.session_state.get("idea_title", "").strip()
    idea_text = st.session_state.get("idea_text", "").strip()

    if not idea_title and not idea_text:
        st.warning("Please add at least an idea title or idea description.")
        return

    result = api_post(
        "/api/submit",
        {
            "journey_type": "idea",
            "payload": {
                "idea_title": st.session_state.get("idea_title", ""),
                "idea_text": st.session_state.get("idea_text", ""),
                "target_users": st.session_state.get("target_users", ""),
                "problem": st.session_state.get("problem", ""),
                "constraints": st.session_state.get("constraints", ""),
            },
        },
    )

    if not result:
        return

    job_id = result["job_id"]
    st.session_state.job_id = job_id
    st.session_state.selected_stage_id = "parse_submission"

    with st.spinner("🤖 Nextify is structuring your product idea..."):
        api_post(f"/api/stage/{job_id}/parse_submission/run")

    rerun_app()


def progress(job: Dict[str, Any]) -> Dict[str, Any]:
    approved = sum(1 for s in job["stages"].values() if s.get("approved"))
    current_index = max(0, min(job.get("current_stage_index", 0), len(STAGES) - 1))
    return {"approved": approved, "current_index": current_index, "value": approved / len(STAGES)}


def render_sidebar(job: Optional[Dict[str, Any]]) -> None:
    with st.sidebar:
        st.header("🧭 Workflow")

        if not job:
            st.info("Submit an idea to start.")
            return

        # -----------------------------
        # Progress
        # -----------------------------
        progress_data = progress(job)

        st.progress(progress_data["value"])

        st.caption(
            f"✅ {progress_data['approved']} of {len(STAGES)} stages approved"
        )

        full_report_sections = collect_report_sections(job)

        if full_report_sections:
            product_title = (
                job.get("payload", {}).get("idea_title")
                or "Nextify Product Plan"
            )

            full_report_pdf = build_nextify_pdf(
                title=f"Nextify Product Plan - {product_title}",
                sections=full_report_sections,
            )

            st.download_button(
                label="📘 Download full report",
                data=full_report_pdf,
                file_name="nextify_complete_product_plan.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_full_report_sidebar",
            )

        st.divider()

        selected_stage_id = st.session_state.get(
            "selected_stage_id",
            STAGES[job["current_stage_index"]]["id"],
        )

        selected = get_stage(selected_stage_id)
        selected_state = job["stages"][selected_stage_id]

        st.markdown(f"**Selected:** {selected['title']}")
        st.caption(selected["agent"])

        st.divider()

        for stage in STAGES:
            sid = stage["id"]
            state = job["stages"][sid]

            if state.get("approved"):
                icon = "✅"
            elif state.get("agent_output"):
                icon = "🔵"
            else:
                icon = "⚪"

            if st.button(
                f"{icon} {stage['short_title']}",
                key=f"side_{sid}",
                use_container_width=True,
            ):
                st.session_state.selected_stage_id = sid
                rerun_app()

        st.divider()

        st.markdown("### Selected stage memory")

        with st.expander("Current output", expanded=False):
            if selected_state.get("agent_output"):
                st.markdown(selected_state["agent_output"])
            else:
                st.info("No output yet.")

        with st.expander("Accepted output", expanded=False):
            if selected_state.get("accepted_output"):
                st.markdown(selected_state["accepted_output"])
            else:
                st.info("Not accepted yet.")

        with st.expander("Previous versions", expanded=True):
            versions = selected_state.get("previous_outputs", [])
            if not versions:
                st.info("No previous versions yet.")
            else:
                for item in reversed(versions):
                    with st.expander(
                        f"Version {item.get('version')} — {item.get('reason')}",
                        expanded=False,
                    ):
                        st.markdown(item.get("output", ""))

        with st.expander("LLM judge feedback", expanded=False):
            if selected_state.get("judge_feedback"):
                st.markdown(selected_state["judge_feedback"])
            else:
                st.info("No judge feedback yet.")

        with st.expander("Human feedback history", expanded=False):
            history = selected_state.get("user_feedback_history", [])
            if not history:
                st.info("No human feedback yet.")
            else:
                for item in reversed(history):
                    st.markdown(f"**Feedback {item.get('version')}**")
                    st.write(item.get("feedback", ""))

        st.divider()

        if st.button("🔄 Start new run", use_container_width=True):
            reset()


def render_header(job: Optional[Dict[str, Any]]) -> None:
    if not job:
        return

    p = progress(job)
    current = STAGES[p["current_index"]]

    st.markdown('<div class="hero">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2.4, 1, 1])
    with c1:
        st.markdown("## 🧠 Nextify Interactive AI")
        st.markdown(
            f"""
            <div class="muted">
            Current workflow stage: <strong>{current['title']}</strong><br>
            Active agent: <strong>{current['agent']}</strong><br>
            Job: <code>{job['job_id']}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("Accepted", f"{p['approved']} / {len(STAGES)}")
    with c3:
        st.metric("Step", f"{p['current_index'] + 1} / {len(STAGES)}")

    st.progress(p["value"])

    chips = ""
    for idx, stage in enumerate(STAGES):
        sid = stage["id"]
        state = job["stages"][sid]
        if state.get("approved"):
            cls, icon = "stage-chip approved", "✅"
        elif idx == p["current_index"]:
            cls, icon = "stage-chip current", "🟡"
        elif state.get("agent_output"):
            cls, icon = "stage-chip ready", "🔵"
        else:
            cls, icon = "stage-chip pending", "⚪"
        chips += f'<span class="{cls}">{icon} {stage["short_title"]}</span>'

    st.markdown(chips, unsafe_allow_html=True)
    if job.get("message"):
        st.caption(job["message"])

    st.markdown("</div>", unsafe_allow_html=True)


def render_form() -> None:
    with st.container(border=True):
        st.markdown("### ✨ Start with your product idea")
        st.text_input("Idea title", key="idea_title")
        st.text_area("Idea description", key="idea_text", height=160)
        st.text_input("Target users", key="target_users")
        st.text_area("Problem", key="problem", height=120)
        st.text_input("Constraints", key="constraints")

        st.button("🚀 Submit idea to Parse Agent", type="primary", on_click=submit_new_job, use_container_width=True)


def render_workspace(job: Dict[str, Any]) -> None:
    if "selected_stage_id" not in st.session_state:
        st.session_state.selected_stage_id = STAGES[job["current_stage_index"]]["id"]

    stage_id = st.session_state.selected_stage_id
    stage = get_stage(stage_id)
    state = job["stages"][stage_id]

    stage_index = [s["id"] for s in STAGES].index(stage_id)
    next_stage = STAGES[stage_index + 1] if stage_index < len(STAGES) - 1 else None

    st.markdown(f"## {stage['title']}")
    st.caption(f"{stage['agent']} · {stage['desc']}")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Status", state.get("status", "pending"))
    with m2:
        st.metric("Revisions", state.get("revision_count", 0))
    with m3:
        st.metric("Accepted", "Yes" if state.get("approved") else "No")

    if state.get("error"):
        st.error(state["error"])

    if state.get("applied_feedback_summary"):
        st.markdown(
            f"""
            <div class="summary-card">
            <strong>Latest change</strong><br>
            {state["applied_feedback_summary"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    current_stage_content = (
        state.get("accepted_output")
        or state.get("agent_output")
        or ""
    )

    tab_output, tab_judge, tab_history = st.tabs(
        [
            "📝 Stage Output",
            "⚖️ AI Review",
            "🕘 Versions",
        ]
    )

    with tab_output:
        with st.container(border=True):
            st.subheader(stage["title"])
            st.caption(stage["agent"])

            if current_stage_content:
                st.markdown(current_stage_content)

                current_stage_pdf = build_nextify_pdf(
                    title=f"Nextify - {stage['title']}",
                    sections=[(stage["title"], current_stage_content)],
                )

                st.download_button(
                    label="📄 Download this stage",
                    data=current_stage_pdf,
                    file_name=f"nextify_{stage_id}.pdf",
                    mime="application/pdf",
                    key=f"download_stage_{stage_id}",
                )
            else:
                st.info("This stage has not generated an output yet.")

    with tab_judge:
        with st.container(border=True):
            st.subheader("AI Quality Review")

            if state.get("judge_feedback"):
                st.markdown(state["judge_feedback"])
            else:
                st.info("Run the AI judge to receive a critical evaluation.")

    with tab_history:
        with st.container(border=True):
            st.subheader("Previous Versions")

            previous_versions = state.get("previous_outputs", [])

            if not previous_versions:
                st.info("No previous versions yet.")
            else:
                for version in reversed(previous_versions):
                    with st.expander(
                        f"Version {version.get('version')} - "
                        f"{version.get('reason', 'Revision')}",
                        expanded=False,
                    ):
                        st.markdown(version.get("output", ""))

    q1, q2 = st.columns(2)

    with q1:
        if st.button("Run / regenerate this agent", use_container_width=True):
            with st.spinner(f"🚀 Running {stage['title']}..."):
                api_post(f"/api/stage/{job['job_id']}/{stage_id}/run")
            rerun_app()

    with q2:
        if st.button(
            "Run LLM judge",
            use_container_width=True,
            disabled=not bool(state.get("agent_output")),
        ):
            with st.spinner("⚖️ Running a strict product-quality review..."):
                api_post(f"/api/stage/{job['job_id']}/{stage_id}/judge")
            rerun_app()

    st.markdown('<div class="review-card">', unsafe_allow_html=True)
    st.markdown("### Review and revise")
    st.caption(
        "Revise with human feedback, LLM judge feedback, or both. "
        "Old versions move to the Versions tab and sidebar."
    )

    with st.form(key=f"review_form_{stage_id}", clear_on_submit=True):
        human_feedback = st.text_area(
            "Human feedback",
            height=120,
            placeholder=(
                "Example: Make this more practical, preserve the product idea, "
                "add risks and assumptions, or choose idea 2."
            ),
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            apply_human = st.form_submit_button(
                "Apply human feedback",
                use_container_width=True,
                disabled=not bool(state.get("agent_output")),
            )
        with c2:
            apply_judge = st.form_submit_button(
                "Apply LLM judge feedback",
                use_container_width=True,
                disabled=not bool(
                    state.get("agent_output") and state.get("judge_feedback")
                ),
            )
        with c3:
            apply_both = st.form_submit_button(
                "Apply both",
                use_container_width=True,
                disabled=not bool(state.get("agent_output")),
            )

    if apply_human:
        with st.spinner("✨ Applying human feedback and producing a stronger version..."):
            api_post(
                f"/api/stage/{job['job_id']}/{stage_id}/revise",
                {"mode": "human_only", "feedback": human_feedback},
            )
        rerun_app()

    if apply_judge:
        with st.spinner("✨ Applying the AI review and producing a stronger version..."):
            api_post(
                f"/api/stage/{job['job_id']}/{stage_id}/revise",
                {"mode": "judge_only", "feedback": ""},
            )
        rerun_app()

    if apply_both:
        with st.spinner("✨ Applying both feedback sources and producing a stronger version..."):
            api_post(
                f"/api/stage/{job['job_id']}/{stage_id}/revise",
                {"mode": "both", "feedback": human_feedback},
            )
        rerun_app()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="accept-card">', unsafe_allow_html=True)
    st.markdown("### Final decision for this stage")
    st.caption("Accepting this output sends it as context to the next agent.")

    if next_stage:
        label = f"Accept this output and send to {next_stage['title']}"
    else:
        label = "Accept final report"

    if st.button(
        label,
        type="primary",
        use_container_width=True,
        disabled=not bool(state.get("agent_output")),
    ):
        with st.spinner("Accepting output and preparing the next agent..."):
            api_post(f"/api/stage/{job['job_id']}/{stage_id}/approve")
        if next_stage:
            st.session_state.selected_stage_id = next_stage["id"]
        rerun_app()

    st.markdown("</div>", unsafe_allow_html=True)


job = get_job()
render_sidebar(job)
render_header(job)

if not job:
    render_form()
else:
    with st.expander("Submitted idea", expanded=False):
        payload = job.get("payload", {})
        st.markdown(f"**Idea title:** {payload.get('idea_title', '')}")
        st.markdown(f"**Idea description:** {payload.get('idea_text', '')}")
        st.markdown(f"**Target users:** {payload.get('target_users', '')}")
        st.markdown(f"**Problem:** {payload.get('problem', '')}")
        st.markdown(f"**Constraints:** {payload.get('constraints', '')}")

    render_workspace(job)


if job:
    st.divider()

    st.markdown("## 📘 Nextify Initial PRD")
    st.caption(
        "All accepted product outputs from the complete Nextify journey, "
        "followed by the final product evaluation."
    )

    all_sections = collect_report_sections(job)

    prd_sections = [
        (title, content)
        for title, content in all_sections
        if "Write Report" not in title
    ]

    final_evaluation_sections = [
        (title, content)
        for title, content in all_sections
        if "Write Report" in title
    ]

    if prd_sections or final_evaluation_sections:
        product_title = (
            job.get("payload", {}).get("idea_title")
            or "Nextify Product"
        )

        payload = job.get("payload", {})

        founder_input_md = f"""
# Founder Input

## Idea Title
{payload.get('idea_title', '')}

## Idea Description
{payload.get('idea_text', '')}

## Target Users
{payload.get('target_users', '')}

## Problem
{payload.get('problem', '')}

## Constraints
{payload.get('constraints', '')}
""".strip()

        complete_prd_sections = [
            ("Founder Input — Original Product Brief", founder_input_md),
            *prd_sections,
            *final_evaluation_sections,
        ]

        initial_prd_pdf = build_nextify_pdf(
            title=f"Nextify Initial PRD - {product_title}",
            sections=complete_prd_sections,
        )

        st.download_button(
            label="📥 Download Complete Nextify Initial PRD",
            data=initial_prd_pdf,
            file_name=f"{product_title.replace(' ', '_')}_Nextify_Initial_PRD.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_initial_prd_main",
        )

        with st.expander(
            f"📘 Full Nextify Initial PRD — {product_title}",
            expanded=False,
        ):
            st.markdown(f"# {product_title}")
            st.caption("Nextify Initial Product Requirements Document")

            st.markdown("## Founder Input")
            st.markdown(
                f"**Idea Description:**  \n{payload.get('idea_text', '')}"
            )
            st.markdown(
                f"**Target Users:**  \n{payload.get('target_users', '')}"
            )
            st.markdown(
                f"**Problem:**  \n{payload.get('problem', '')}"
            )
            st.markdown(
                f"**Constraints:**  \n{payload.get('constraints', '')}"
            )

            st.divider()

            for section_title, section_content in prd_sections:
                st.markdown(f"## {section_title}")
                st.markdown(section_content)
                st.divider()

            if final_evaluation_sections:
                st.markdown("## 🧠 Final Product Evaluation")

                for _, section_content in final_evaluation_sections:
                    st.markdown(section_content)
                    st.divider()

        st.markdown("### 🤖 Individual Agent Outputs")

        stages = job.get("stages") or {}

        for stage in STAGES:
            stage_id = stage["id"]
            stage_state = stages.get(stage_id) or {}

            stage_content = (
                stage_state.get("accepted_output")
                or stage_state.get("agent_output")
                or ""
            )

            if not stage_content.strip():
                continue

            with st.expander(
                f"🤖 {stage['agent']} — {stage['title']}",
                expanded=False,
            ):
                st.markdown(stage_content)

    else:
        st.info(
            "The Initial PRD will grow here as each agent generates an output."
        )