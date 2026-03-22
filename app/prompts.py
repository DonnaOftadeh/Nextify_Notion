# app/prompts.py
# Concise prompts. No provenance lines. $ for TAM/SAM/SOM. RICE explanation under the table.

from textwrap import dedent
from typing import Dict, Any

SYSTEM_BASE = dedent("""
You are a precise product strategy and innovation copilot. Write to-the-point,
scan-friendly content that will be turned into a PDF with tables and charts.

FORMATTING RULES (very important):
- Use markdown headings (#, ##, ###) exactly. Keep paragraphs 2–4 sentences.
- Prefer short bullet lists. Avoid long blocks of text.
- When you introduce a concept that has an abbreviation, write the full term first
  followed by the abbreviation in parentheses (e.g., Jobs to be Done (JTBD)).
- Tables MUST be syntactically correct markdown pipe tables. No extra lines,
  no broken rows, no wrapped header cells across multiple lines.
- Keep tables narrow (≤ 4 columns) unless specified otherwise.
- Numeric columns must contain numbers only unless the dollar sign ($) is explicitly requested.
- Use short labels. Avoid fluff.
- If a value is unknown, write a single dash (-).

CONSTRAINTS:
- All content must stay directly relevant to the user’s idea/company/product/industry.
- If you list examples, keep to 3–6 items.
- Be consistent across sections so each step builds on the previous steps.

DO NOT include any 'Context', 'Previous sections', or quoted summaries of earlier content
in your output. Use prior information only to keep names/numbers consistent.
""").strip()

SYSTEM_GUARDRAILS = SYSTEM_BASE

def _kv(k: str, v: Any) -> str:
    v = "" if v is None else str(v)
    return f"- **{k}**: {v}"

def _grounding(j: str, p: Dict[str, Any]) -> str:
    lines = [
        _kv("Journey type", j),
        _kv("Company", p.get("company_name") or p.get("bench_company")),
        _kv("Product", p.get("product_name")),
        _kv("Industry", p.get("industry")),
        _kv("Region", p.get("region")),
        _kv("Idea title", p.get("idea_title") or p.get("idea_text")),
        _kv("Problem", p.get("problem")),
        _kv("Target users", p.get("target_users")),
        _kv("Current stage", p.get("current_stage")),
        _kv("Constraints", p.get("constraints")),
    ]
    return "\n".join(lines)

# ---------------- IDEA ----------------
def idea_problem_snapshot(jt: str, p: Dict[str, Any]) -> str:
    return f"""{SYSTEM_BASE}

Role: Problem and Jobs to be Done (JTBD) Snapshot
Grounding:
{_grounding(jt, p)}

Task:
Write a crisp snapshot of the problem and the core Jobs to be Done (JTBD) across key user roles.

Output:
## Idea Problem Snapshot
Briefly articulate the problem and why it matters now.

### JTBD by User Role
| User Role | Jobs to be Done (JTBD) | Pain |
| --- | --- | --- |
""".strip()

def idea_brainstorm_uses(jt: str, p: Dict[str, Any]) -> str:
    return f"""{SYSTEM_BASE}

Role: Brainstorm Uses and Present-Day Behaviors
Grounding:
{_grounding(jt, p)}

Task:
Brainstorm concrete use patterns and present-day workarounds specific to the idea.

Output:
## Idea Brainstorm Uses
- 5–8 bullet points describing possible uses and current behaviors.
""".strip()

def idea_audience_and_early(jt: str, p: Dict[str, Any]) -> str:
    return f"""{SYSTEM_BASE}

Role: Audience and Early Adopters with Lightweight Market Research
Grounding:
{_grounding(jt, p)}

Task:
Summarize primary segments, early adopters, and quick market notes that matter for the idea.

Output:
## Audience & Early Adopters
| Segment | Why They Care |
| --- | --- |
""".strip()

def idea_competitors_market_size(jt: str, p: Dict[str, Any]) -> str:
    return f"""{SYSTEM_BASE}

Role: Competitor Landscape + Market Size (TAM/SAM/SOM)
Grounding:
{_grounding(jt, p)}

Task:
List direct and adjacent competitors, then estimate market size using Total Addressable Market (TAM),
Serviceable Available Market (SAM), and Serviceable Obtainable Market (SOM).
State assumptions and show explicit calculations. The Value column MUST show dollar amounts with a
leading $ (e.g., $200000000).

Output:
## Competitors (Direct & Adjacent)
- 3–6 bullets: Competitor — brief positioning.

## Market Size (TAM / SAM / SOM)
Assumptions:
| Metric | Assumption | Value |
| --- | --- | --- |
| Global innovation spending | Total global investment relevant to this idea | $... |
| Serviceable % | Fraction of TAM that is serviceable | ... |
| Obtainable % | Fraction of SAM realistically obtainable | ... |

Calculations (include $):
- TAM = <global innovation spending> × 1 = $<number>
- SAM = TAM × <serviceable %> = $<number>
- SOM = SAM × <obtainable %> = $<number>
""".strip()

def idea_assumptions_tests(jt: str, p: Dict[str, Any]) -> str:
    return f"""{SYSTEM_BASE}

Role: Assumptions, Risks, and How to Test
Grounding:
{_grounding(jt, p)}

Task:
Identify 3–6 critical assumptions and risks. Propose lean tests.

Output:
## Assumptions & Risks
- 3–6 key assumptions (bullets)
- 3–6 key risks (bullets)

## How to Test (Lean)
| Assumption | Test | Success Metric | Timebox (months) |
| --- | --- | --- | --- |
- 'Timebox (months)' must be numeric only.
""".strip()

def idea_product_candidates_and_rice(jt: str, p: Dict[str, Any]) -> str:
    return f"""{SYSTEM_BASE}

Role: Product Candidates and Feature Candidates with RICE
Grounding:
{_grounding(jt, p)}

Task:
Describe 2–3 possible product concepts. Then list features with RICE scoring
(Reach, Impact, Confidence, Effort) and include the RICE value.

Output:
## Product Candidates
- Concept A — one-liner
- Concept B — one-liner
- Concept C (optional) — one-liner

## Feature Candidates & RICE
| Feature | Reach | Impact | Confidence | Effort | RICE |
| --- | --- | --- | --- | --- | --- |
- Reach: people/week (number only)
- Impact: 1–5 (number only)
- Confidence: 0–1 (number only)
- Effort: person-weeks (number only)
- RICE: computed value (number only)

### How RICE Was Computed
- Reach = people per week.
- Impact = 1–5 scale.
- Confidence = 0–1.
- Effort = person-weeks.
- RICE = (Reach × Impact × Confidence) ÷ Effort.
""".strip()

def idea_lean_okrs(jt: str, p: Dict[str, Any]) -> str:
    return f"""{SYSTEM_BASE}

Role: Lean Objectives and Key Results (OKR)
Grounding:
{_grounding(jt, p)}

Task:
Create one objective and 3 measurable key results tied to the features above.

Output:
## Lean Objective & Key Results (Next Quarter)
Objective (one sentence).

| Key Result | Baseline | Target | Data Source |
| --- | --- | --- | --- |
- Baseline and Target must be numeric or % (numbers only).
""".strip()

def idea_customer_storyboard(jt: str, p: Dict[str, Any]) -> str:
    return f"""{SYSTEM_BASE}

Role: Customer Journey Storyboard (Discover → Activate → Outcome)
Grounding:
{_grounding(jt, p)}

Task:
Describe the first cohort’s path across three stages. Keep it crisp.

Output:
## Customer Journey Storyboard (Quarter 1)
### Discover
| Element | Description |
| --- | --- |
| Trigger | ... |
| User Action | ... |
| Success Metric | ... |
| Instrumentation | ... |

### Activate
| Element | Description |
| --- | --- |
| Trigger | ... |
| User Action | ... |
| Success Metric | ... |
| Instrumentation | ... |

### Outcome
| Element | Description |
| --- | --- |
| Trigger | ... |
| User Action | ... |
| Success Metric | ... |
| Instrumentation | ... |
""".strip()

def idea_tools(jt: str, p: Dict[str, Any]) -> str:
    return f"""{SYSTEM_BASE}

Role: Tools to Use
Grounding:
{_grounding(jt, p)}

Task:
List practical tools to support discovery, build, and measurement.

Output:
## Tools to Use
| Tool | Why We Use It |
| --- | --- |
""".strip()

def idea_synthesis(jt: str, p: Dict[str, Any]) -> str:
    return f"""{SYSTEM_BASE}

Role: Synthesis Summary
Grounding:
{_grounding(jt, p)}

Task:
Synthesize into a short, actionable summary.

Output:
## Synthesis Summary
- 3–6 bullets tying problem, audience, differentiators, top feature(s), key risk, and next proof points.
""".strip()

def idea_three_month_plan(jt: str, p: Dict[str, Any]) -> str:
    return f"""{SYSTEM_BASE}

Role: Next Three-Month Plan (Milestones)
Grounding:
{_grounding(jt, p)}

Task:
List a crisp milestone plan with numeric durations only. No dates.

Output:
## Next Three-Month Plan
| Milestone | Owner | Metric | Duration (days) |
| --- | --- | --- | --- |
- 'Duration (days)' must contain numbers only.
""".strip()

# -------- compact bundles for product/company/industry --------
def common_problem_snapshot(jt: str, p: Dict[str, Any]) -> str:
    subject = p.get("company_name") or p.get("product_name") or p.get("industry") or "Subject"
    return f"""{SYSTEM_BASE}

Role: Problem and Jobs to be Done (JTBD) Snapshot
Grounding:
{_grounding(jt, p)}

Task:
Summarize the core problem and Jobs to be Done (JTBD) for **{subject}**.

Output:
## Problem Snapshot
Short paragraph (2–4 sentences).

### JTBD by User Role
| User Role | Jobs to be Done (JTBD) | Pain |
| --- | --- | --- |
""".strip()

def common_competitors_and_market(jt: str, p: Dict[str, Any]) -> str:
    return f"""{SYSTEM_BASE}

Role: Competitors and Market Size (TAM/SAM/SOM)
Grounding:
{_grounding(jt, p)}

Task:
Direct/adjacent competitors and market size with explicit assumptions and calculations.
The Value column MUST show dollar amounts with a leading $.

Output:
## Competitors
- 3–6 bullets: Competitor — positioning

## Market Size (TAM / SAM / SOM)
| Metric | Assumption | Value |
| --- | --- | --- |
| Base market size | Estimate relevant to scope | $... |
| Serviceable % | Fraction of TAM serviceable | ... |
| Obtainable % | Fraction of SAM obtainable | ... |

Calculations (include $):
- TAM = <base market size> × 1 = $<number>
- SAM = TAM × <serviceable %> = $<number>
- SOM = SAM × <obtainable %> = $<number>
""".strip()

def common_okrs(jt: str, p: Dict[str, Any]) -> str:
    return f"""{SYSTEM_BASE}

Role: Lean Objective and Key Results
Grounding:
{_grounding(jt, p)}

Task:
One objective and 3 measurable key results.

Output:
## Lean Objective & Key Results
Objective (one sentence).

| Key Result | Baseline | Target | Data Source |
| --- | --- | --- | --- |
""".strip()

def common_plan(jt: str, p: Dict[str, Any]) -> str:
    return f"""{SYSTEM_BASE}

Role: Plan (Execution)
Grounding:
{_grounding(jt, p)}

Task:
Provide a short plan with numeric duration only.

Output:
## Plan (Execution)
| Milestone | Owner | Metric | Duration (days) |
| --- | --- | --- | --- |
""".strip()

def get_prompt_bundle(journey_type: str, payload: Dict[str, Any]) -> Dict[str, str]:
    jt = (journey_type or "").lower().strip()
    if jt == "idea":
        return {
            "1_problem_snapshot":     idea_problem_snapshot(jt, payload),
            "2_brainstorm_uses":      idea_brainstorm_uses(jt, payload),
            "3_audience_early":       idea_audience_and_early(jt, payload),
            "4_competitors_market":   idea_competitors_market_size(jt, payload),
            "5_assumptions_tests":    idea_assumptions_tests(jt, payload),
            "6_features_rice":        idea_product_candidates_and_rice(jt, payload),
            "7_okrs":                 idea_lean_okrs(jt, payload),
            "8_storyboard":           idea_customer_storyboard(jt, payload),
            "9_tools":                idea_tools(jt, payload),
            "10_synthesis":           idea_synthesis(jt, payload),
            "11_three_month_plan":    idea_three_month_plan(jt, payload),
        }
    if jt in ("company", "product", "industry"):
        return {
            "1_problem_snapshot":   common_problem_snapshot(jt, payload),
            "2_competitors_market": common_competitors_and_market(jt, payload),
            "3_okrs":               common_okrs(jt, payload),
            "4_plan":               common_plan(jt, payload),
        }
    return {"error": f"Unknown journey type: {journey_type}"}

def build_agent_prompts(journey_type: str, payload: Dict[str, Any]) -> Dict[str, str]:
    return get_prompt_bundle(journey_type, payload)
