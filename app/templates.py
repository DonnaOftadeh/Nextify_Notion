# app/templates.py
from typing import Dict, List, Tuple

# ---------- small helpers ----------
def _coalesce(*vals, default=""):
    for v in vals:
        if v:
            return v
    return default

def _hdr(title: str) -> str:
    # duplicate line (H2 + plain) helps LLMs keep headings consistent
    return f"## {title}\n{title}"

# ---------- IDEA ----------
def idea_bundle(p: Dict) -> List[Tuple[str, str]]:
    idea        = _coalesce(p.get("idea_title"), p.get("idea_text"), default="the idea")
    problem     = _coalesce(p.get("problem"), default="the core problem")
    audience    = _coalesce(p.get("target_users"), p.get("audience"), default="target users")
    stage       = _coalesce(p.get("current_stage"), default="concept")
    region      = _coalesce(p.get("region"), default="Global")
    constraints = _coalesce(p.get("constraints"), default="minimal constraints")

    return [
        # 1
        ("1. Problem and Jobs to be done (JTBD) Snapshot",
         f"""{_hdr("1. Problem and Jobs to be done (JTBD) Snapshot")}
In ≤150 words, explain the problem **{idea}** addresses for **{audience}** in {region}.
Use the full phrase “jobs to be done (JTBD)” at first mention; JTBD can be used afterward.
List current workarounds users take today. End with a one-sentence “why now” (market/tech/regulation)."""),

        # 2
        ("2. Brainstorm on Current Behaviors and Possibilities",
         f"""{_hdr("2. Brainstorm on Current Behaviors and Possibilities")}
Brainstorm objectively. For each user type (accelerators, founders, investors, kids, educators),
list: current actions, pain points, and opportunities for **{idea}**.
Return a compact table using pipes:
| User type | Current action | Pain point |"""),

        # 3
        ("3. Audience and Early Adopters (Market Research)",
         f"""{_hdr("3. Audience and Early Adopters (Market Research)")}
Define 2–3 micro-segments to reach in Q1 (e.g., Berlin makerspaces, school clubs).
For each: segment, one-line value proposition, acquisition channel.
Return a table:
| Segment | Value proposition | Acquisition channel |
Mark missing data as MISSING to stay realistic."""),

        # 4
        ("4. Competitor Analysis and Market Size",
         f"""{_hdr("4. Competitor Analysis and Market Size")}
Build a competitor matrix (include “do nothing” if relevant):
| Competitor/Alternative | Strength | Gap we exploit |
Add a brief Porter's Five Forces snapshot (one bullet per force).
Then size markets for {region}:
- Total addressable market (TAM) = average revenue per user × number of potential users (state assumptions)
- Serviceable available market (SAM); Serviceable obtainable market (SOM)
Return a second table:
| Metric | Value | Assumptions |"""),

        # 5
        ("5. Assumptions, Risks and How to Test",
         f"""{_hdr("5. Assumptions, Risks and How to Test")}
List the 3 riskiest assumptions blocking **{idea}** at the **{stage}** stage and a one-week test for each.
Each row includes: metric, pass/fail rule, instrument (survey, prototype, landing page), sample prompts.
| Assumption | Test & metric | Pass/Fail rule | Tool |"""),

        # 6
        ("6. Product Candidate and Feature Candidates (RICE)",
         f"""{_hdr("6. Product Candidate and Feature Candidates (RICE)")}
Propose up to 3 product concepts, and 5–7 candidate features.
Score features with the RICE model: Reach (weekly users), Impact (1–5), Confidence (0–1), Effort (person-weeks).
Compute RICE = (Reach × Impact × Confidence) / Effort and sort descending.
Return:
| Feature | Reach | Impact | Confidence | Effort | RICE |"""),

        # 7
        ("7. Lean Objectives and Key Results (Next Quarter)",
         f"""{_hdr("7. Lean Objectives and Key Results (Next Quarter)")}
Write one objective aligned with the problem.
Add 2–3 quantitative key results with baseline→target, and specify the data source for each."""),

        # 8
        ("8. Customer Journey Storyboard (Q1)",
         f"""{_hdr("8. Customer Journey Storyboard (Q1)")}
Describe three frames: Discover → Activate → Outcome (quarter end).
For each frame specify trigger, user action, success metric, instrumentation.
Return:
| Step | What user sees/does | Success metric |"""),

        # 9
        ("9. Tools to Use",
         f"""{_hdr("9. Tools to Use")}
Recommend 5–7 tools across research, prototyping, community, analytics, funding.
Each row: tool and ≤8-word why.
| Tool | Why we use it |"""),

        # 10
        ("10. Synthesis Summary",
         f"""{_hdr("10. Synthesis Summary")}
One short paragraph that combines: user need, market wedge, top feature (by RICE),
key risk & test, and the primary key result for the quarter."""),

        # 11
        ("11. Next Three-Month Plan (Milestones)",
         f"""{_hdr("11. Next Three-Month Plan (Milestones)")}
Provide 4–6 milestones across product and learning.
| Milestone | Owner | Metric | Target date (YYYY-MM-DD) |"""),
    ]

# ---------- COMPANY ----------
def company_bundle(p: Dict) -> List[Tuple[str, str]]:
    company = _coalesce(p.get("company_name"), p.get("bench_company"), default="the company")
    region  = _coalesce(p.get("region"), default="Global")
    return [
        ("1. Problem and Jobs to be done (JTBD) Snapshot",
         f"""{_hdr("1. Problem and Jobs to be done (JTBD) Snapshot")}
For **{company}** in {region}, summarize the problem, user workarounds and why now (≤150 words)."""),
        ("2. Brainstorm on Current Behaviors and Possibilities",
         f"""{_hdr("2. Brainstorm on Current Behaviors and Possibilities")}
| User type | Current action | Pain point |"""),
        ("3. Audience and Early Adopters (Market Research)",
         f"""{_hdr("3. Audience and Early Adopters (Market Research)")}
| Segment | Value proposition | Acquisition channel |"""),
        ("4. Competitor Analysis and Market Size",
         f"""{_hdr("4. Competitor Analysis and Market Size")}
| Competitor/Alternative | Strength | Gap we exploit |
| Metric | Value | Assumptions |"""),
        ("5. Assumptions, Risks and How to Test",
         f"""{_hdr("5. Assumptions, Risks and How to Test")}
| Assumption | Test & metric | Pass/Fail rule | Tool |"""),
        ("6. Product Candidate and Feature Candidates (RICE)",
         f"""{_hdr("6. Product Candidate and Feature Candidates (RICE)")}
| Feature | Reach | Impact | Confidence | Effort | RICE |"""),
        ("7. Lean Objectives and Key Results (Next Quarter)",
         f"""{_hdr("7. Lean Objectives and Key Results (Next Quarter)")}"""),
        ("8. Customer Journey Storyboard (Q1)",
         f"""{_hdr("8. Customer Journey Storyboard (Q1)")}
| Step | What user sees/does | Success metric |"""),
        ("9. Tools to Use",
         f"""{_hdr("9. Tools to Use")}
| Tool | Why we use it |"""),
        ("10. Synthesis Summary",
         f"""{_hdr("10. Synthesis Summary")}"""),
        ("11. Next Three-Month Plan (Milestones)",
         f"""{_hdr("11. Next Three-Month Plan (Milestones)")}
| Milestone | Owner | Metric | Target date (YYYY-MM-DD) |"""),
    ]

# ---------- PRODUCT ----------
def product_bundle(p: Dict) -> List[Tuple[str, str]]:
    product = _coalesce(p.get("product_name"), default="the product")
    region  = _coalesce(p.get("region"), default="Global")
    return [
        ("1. Problem and Jobs to be done (JTBD) Snapshot",
         f"""{_hdr("1. Problem and Jobs to be done (JTBD) Snapshot")}
For **{product}** in {region}, state the problem, workarounds and why now."""),
        ("2. Brainstorm on Current Behaviors and Possibilities",
         f"""{_hdr("2. Brainstorm on Current Behaviors and Possibilities")}
| User type | Current action | Pain point |"""),
        ("3. Audience and Early Adopters (Market Research)",
         f"""{_hdr("3. Audience and Early Adopters (Market Research)")}
| Segment | Value proposition | Acquisition channel |"""),
        ("4. Competitor Analysis and Market Size",
         f"""{_hdr("4. Competitor Analysis and Market Size")}
| Competitor/Alternative | Strength | Gap we exploit |
| Metric | Value | Assumptions |"""),
        ("5. Assumptions, Risks and How to Test",
         f"""{_hdr("5. Assumptions, Risks and How to Test")}
| Assumption | Test & metric | Pass/Fail rule | Tool |"""),
        ("6. Product Candidate and Feature Candidates (RICE)",
         f"""{_hdr("6. Product Candidate and Feature Candidates (RICE)")}
| Feature | Reach | Impact | Confidence | Effort | RICE |"""),
        ("7. Lean Objectives and Key Results (Next Quarter)",
         f"""{_hdr("7. Lean Objectives and Key Results (Next Quarter)")}"""),
        ("8. Customer Journey Storyboard (Q1)",
         f"""{_hdr("8. Customer Journey Storyboard (Q1)")}
| Step | What user sees/does | Success metric |"""),
        ("9. Tools to Use",
         f"""{_hdr("9. Tools to Use")}
| Tool | Why we use it |"""),
        ("10. Synthesis Summary",
         f"""{_hdr("10. Synthesis Summary")}"""),
        ("11. Next Three-Month Plan (Milestones)",
         f"""{_hdr("11. Next Three-Month Plan (Milestones)")}
| Milestone | Owner | Metric | Target date (YYYY-MM-DD) |"""),
    ]

# ---------- INDUSTRY ----------
def industry_bundle(p: Dict) -> List[Tuple[str, str]]:
    industry = _coalesce(p.get("industry"), default="the industry")
    region   = _coalesce(p.get("region"), default="Global")
    return [
        ("1. Problem and Jobs to be done (JTBD) Snapshot",
         f"""{_hdr("1. Problem and Jobs to be done (JTBD) Snapshot")}
For the **{industry}** in {region}, summarize core jobs to be done, pain points and timing."""),
        ("2. Brainstorm on Current Behaviors and Possibilities",
         f"""{_hdr("2. Brainstorm on Current Behaviors and Possibilities")}
| Actor | Current action | Pain point |"""),
        ("3. Audience and Early Adopters (Market Research)",
         f"""{_hdr("3. Audience and Early Adopters (Market Research)")}
| Segment | Value proposition | Acquisition channel |"""),
        ("4. Competitor Analysis and Market Size",
         f"""{_hdr("4. Competitor Analysis and Market Size")}
| Competitor/Alternative | Strength | Gap we exploit |
| Metric | Value | Assumptions |"""),
        ("5. Assumptions, Risks and How to Test",
         f"""{_hdr("5. Assumptions, Risks and How to Test")}
| Assumption | Test & metric | Pass/Fail rule | Tool |"""),
        ("6. Product Candidate and Feature Candidates (RICE)",
         f"""{_hdr("6. Product Candidate and Feature Candidates (RICE)")}
| Feature | Reach | Impact | Confidence | Effort | RICE |"""),
        ("7. Lean Objectives and Key Results (Next Quarter)",
         f"""{_hdr("7. Lean Objectives and Key Results (Next Quarter)")}"""),
        ("8. Customer Journey Storyboard (Q1)",
         f"""{_hdr("8. Customer Journey Storyboard (Q1)")}
| Step | What user sees/does | Success metric |"""),
        ("9. Tools to Use",
         f"""{_hdr("9. Tools to Use")}
| Tool | Why we use it |"""),
        ("10. Synthesis Summary",
         f"""{_hdr("10. Synthesis Summary")}"""),
        ("11. Next Three-Month Plan (Milestones)",
         f"""{_hdr("11. Next Three-Month Plan (Milestones)")}
| Milestone | Owner | Metric | Target date (YYYY-MM-DD) |"""),
    ]

def get_prompt_bundle(journey_type: str, payload: Dict) -> List[Tuple[str, str]]:
    jt = (journey_type or "").lower().strip()
    if jt == "company":
        return company_bundle(payload)
    if jt == "product":
        return product_bundle(payload)
    if jt == "industry":
        return industry_bundle(payload)
    return idea_bundle(payload)
