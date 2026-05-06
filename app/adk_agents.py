"""
Nextify ADK Interactive Agents

Interactive HITL workflow:
Raw idea form
→ Input Parser Agent
→ LLM Judge
→ Reviewer Agent applies human / judge / both feedback
→ user accepts output
→ accepted output becomes input to next agent

Key fixes:
- Parser always prints exact original form content first.
- Brainstorm/Crazy Ideas include real-world analogies and links.
- Reviewer is stage-aware and applies human feedback visibly.
- Brainstorm revisions cannot accidentally become parser output.
- Gemini 503 / 429 retry is supported.
- Local fallback keeps the UI usable if Gemini is overloaded.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, Dict

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
EVAL_MODEL = os.getenv("GEMINI_EVAL_MODEL", "gemini-2.5-flash-lite")
APP_NAME = "nextify_interactive_adk_app"

INTERACTIVE_STAGE_IDS = [
    "parse_submission",
    "brainstorm_parallel",
    "idea_cooker",
    "theme_epic_generator",
    "roadmap_generator",
    "feature_generation",
    "prioritization_rice",
    "okr_generation",
    "three_month_planner",
    "write_report_pdf",
]


# ============================================================
# PROMPTS
# ============================================================

INPUT_PARSER_PROMPT = """
You are the Input Parser Agent for Nextify.

Your job is to turn the raw founder idea form into a clean, structured product brief.

CRITICAL RULES:
- You MUST include the exact original form content first.
- You MUST include the idea title.
- Do not omit any submitted field.
- Do not rewrite the original form fields in the "Original Submitted Form" section.
- Preserve the user's original idea.
- Do not invent a different product.
- After showing the exact form content, add your structured interpretation.
- Return markdown only.
- Do not include internal reasoning.

You receive these fields:
- idea_title
- idea_text
- target_users
- problem
- constraints

Output exactly this structure:

# Parsed Product Brief

## Original Submitted Form

### Idea Title
Copy the exact idea_title value here.

### Idea Description
Copy the exact idea_text value here.

### Target Users
Copy the exact target_users value here.

### Problem
Copy the exact problem value here.

### Constraints
Copy the exact constraints value here.

---

## Agent-Structured Interpretation

### Clean Product Concept
Explain the product clearly using the submitted idea.

### Product Type
Classify the product.

### MVP Boundary
Define the smallest useful MVP.

### Key Assumptions
List key assumptions.

### Key Risks
List key risks.

### Missing Information / Clarifying Questions
List anything the next agent may need later.

### Next Agent Input
Summarize what the Brainstorming Agent should use as source of truth.
"""


MARKET_ANALYSIS_PROMPT = """
You are the MarketAnalysisAgent for Nextify.

You receive:
- the founder idea form
- the previous accepted stage output
- all accepted context so far
- optional human feedback
- optional LLM judge feedback

Your job:
1. Identify the relevant market.
2. Estimate TAM / SAM / SOM with plausible ranges.
3. Identify direct and indirect competitors.
4. Explain market gaps and positioning opportunities.
5. Stay faithful to the founder idea and prior accepted output.
6. Include real-world competitor and analogy product links where relevant.
7. Apply human feedback when provided.

Return markdown only.

Output:

[MARKET_DATA]
## 🌍 Market Overview

## 📊 TAM / SAM / SOM

## 🏁 Competitor Landscape

| Name | Type | URL | What they offer | Strengths | Weaknesses / Gaps |
|---|---|---|---|---|---|

## 🔗 Real-World Analogy Map

| Product | URL | What Nextify can learn from it |
|---|---|---|

## 🎯 Strategic Insights & Positioning
"""


CRAZY_IDEA_PROMPT = """
You are the CrazyIdeaAgent for Nextify.

Your job is to generate bold, novel, creative, but still MVP-buildable product concepts.

You receive:
- the founder idea form
- the previous accepted stage output
- all accepted context so far
- optional human feedback
- optional LLM judge feedback

CRITICAL RULES:
- You MUST apply human feedback when provided.
- You MUST keep creativity and novelty high.
- You MUST make every idea simple enough to start as an MVP.
- You MUST include real-world analogies, inspiration products, and links.
- Use well-known real products only.
- Do not invent fake company names.
- Do not return parser output.
- Do not include “Parsed Product Brief”.
- Stay faithful to the founder idea.
- Return markdown only.

Output exactly:

[CRAZY_IDEAS]
## 🎨 Concept Space

For each concept:

### Concept <number> — <name>

- **Summary:** <2–3 sentence description>
- **Real-world analogies mixed:**
  - **<Product name>:** <official URL> — <what part of this product is being borrowed>
  - **<Product name>:** <official URL> — <what part of this product is being borrowed>
  - **<Product name>:** <official URL> — <what part of this product is being borrowed>
- **Why this mix is novel:** <explain the new combination>
- **Why it might win:**
  - <bullet>
  - <bullet>
- **Simple MVP version:** <small version that can be built first>
- **Future breakthrough version:** <ambitious version later>
- **Risks / downsides:**
  - <bullet>
  - <bullet>

Generate 3–5 concepts.

For the Cursor for Product Managers idea, useful analogy sources may include:
- Cursor: https://www.cursor.com/
- Notion AI: https://www.notion.com/product/ai
- Linear: https://linear.app/
- Jira Product Discovery: https://www.atlassian.com/software/jira/product-discovery
- Productboard: https://www.productboard.com/
- Dovetail: https://dovetail.com/
- Miro: https://miro.com/
- Perplexity: https://www.perplexity.ai/
- Gamma: https://gamma.app/
- Coda: https://coda.io/
- Airtable: https://www.airtable.com/
"""


IDEA_COOKER_PROMPT = """
You are the IdeaCookerAgent for Nextify.

You receive:
- MarketAnalysisAgent output
- CrazyIdeaAgent output
- founder idea form
- previous accepted context

Your job:
1. Evaluate the strongest concepts.
2. Score them with rationale.
3. Recommend one winning concept.
4. Ask the user to approve or give feedback.

Rules:
- Preserve the user's original product direction.
- Do not invent unrelated concepts.
- If the user has selected or preferred a concept in feedback, respect it.
- Use the real-world analogy map from the Brainstorm stage when useful.
- Return markdown only.

Output:

[TRADEOFF_TABLE]
## ⚖️ Tradeoff Analysis

| Concept Name | Differentiation | Market Demand | Feasibility | Strategic Fit | Monetization | Total Score |
|---|---:|---:|---:|---:|---:|---:|

[TRADEOFF_SUMMARY]
## 🧠 Tradeoff Summary

[PRODUCT_SNAPSHOT_MD]
## 🧾 Product Snapshot
- Problem
- Target users
- Concept summary
- Core value proposition
- MVP scope
- Real-world analogies used
- Risks and mitigations
- Example use cases

## User Decision Needed
Tell the user what to approve or how to give feedback.
"""


THEME_EPIC_PROMPT = """
You are the ThemeEpicAgent for Nextify.

You receive the accepted product snapshot or accepted prior stage output.

Your job:
Create 3–5 strategic themes and epics.

Return markdown only.

Output:

[THEME_EPIC_MD]
## 🎯 Strategic Themes

For each theme:
- Theme name
- Why it matters
- User / business value

## 🧩 Epics

For each epic:
- Epic name
- Mapped theme
- What this epic covers
- Why this belongs in roadmap planning
"""


ROADMAP_PROMPT = """
You are the RoadmapAgent for Nextify.

You receive accepted themes and epics.

Your job:
Create a realistic roadmap for an early-stage MVP.

Return markdown only.

Output:

[ROADMAP_GENERATOR_MD]
## 🗺️ Strategic Roadmap

### Phase 1 – Foundation / MVP

### Phase 2 – Validation / Expansion

### Phase 3 – Optimization / Scale

## 🔗 Roadmap Logic
"""


FEATURE_PROMPT = """
You are the FeatureGenerationAgent for Nextify.

You receive the accepted roadmap and product direction.

Your job:
Generate a clean feature list for MVP and near-term roadmap.

Return markdown only.

Output:

[FEATURE_LIST]
## 🧱 Feature List

| id | feature_name | what_it_does | why_it_matters | impact | effort | tags |
|---|---|---|---|---|---|---|

[FEATURE_DETAILS]
## 🔍 Feature Details
"""


PRIORITIZATION_PROMPT = """
You are the PrioritizationAgent for Nextify.

You receive the accepted feature list.

Your job:
Apply RICE prioritization and create a short feature roadmap.

Return markdown only.

Output:

[RICE_TABLE]
## 📊 RICE Prioritization Table

| id | feature_name | reach | impact | confidence | effort | rice_score | priority_rank |
|---|---|---:|---:|---:|---:|---:|---:|

[RICE_SUMMARY]
## 🧠 RICE Summary

[ROADMAP_MD]
## 🗺️ Feature Roadmap
"""


OKR_PROMPT = """
You are the OKRAgent for Nextify.

You receive the accepted product direction, roadmap, features, and priorities.

Your job:
Generate a small, realistic quarterly OKR set.

Return markdown only.

Output:

[OKR_SUMMARY]

[OBJECTIVES]

[KEY_RESULTS]

[MILESTONES_AND_CHECKPOINTS]

[METRICS_AND_INSTRUMENTATION]
"""


PLANNER_PROMPT = """
You are the Three-Month Planner Agent for Nextify.

You receive the accepted OKRs and upstream context.

Your job:
Create a practical 3-month execution plan.

Return markdown only.

Output:

[THREE_MONTH_OVERVIEW]

[MONTHLY_BREAKDOWN]

[WEEKLY_PLAN]

[EXPERIMENTS_AND_LEARNING]

[RISKS_AND_DEPENDENCIES]
"""


REPORT_PROMPT = """
You are the Report Writer Agent for Nextify.

You receive all accepted stage outputs.

Your job:
Assemble all accepted stage outputs into a clean final product plan.

Return markdown only.

Output:

# Final Product Plan

## Product Concept
## Target Users
## Problem
## Recommended Direction
## MVP Scope
## Roadmap
## Prioritized Features
## OKRs
## Three-Month Execution Plan
## Next Steps
"""


EVALUATOR_PROMPT = """
You are the Evaluation & Quality Agent for Nextify.

You evaluate exactly one selected stage output.

Important:
- Evaluate only the current selected stage.
- Do not rewrite the output as a different stage.
- If STAGE_KEY is brainstorm_parallel, your rewritten version must preserve [MARKET_DATA] and [CRAZY_IDEAS].
- If STAGE_KEY is brainstorm_parallel, check whether crazy ideas include real-world analogies and links.
- If STAGE_KEY is parse_submission, your rewritten version must preserve the original submitted form.

Return markdown only.

Output exactly:

[QUALITY_SCORES]
- Overall: <0-10> — <short justification>
- PromptAdherence: <0-10> — <short justification>
- Clarity: <0-10> — <short justification>
- Feasibility: <0-10> — <short justification>
- AlignmentWithIdea: <0-10> — <short justification>

[COMMENT_SUMMARY]
- 3–5 concise bullets

[ISSUES_AND_FLAGS]
- Concrete issues

[IMPROVEMENT_SUGGESTIONS]
- Actionable suggestions

[REWRITTEN_VERSION]
- A revised version of the same selected stage content, not a different stage.
"""


REVIEWER_PROMPT = """
You are the Nextify Reviewer/Rewriter Agent.

You revise exactly one selected stage output using:
- human feedback
- LLM judge feedback
- or both

CRITICAL RULES:
- You MUST apply human feedback when provided.
- You MUST visibly reflect the human feedback in the revised output.
- You must preserve the current stage type.
- You must not return an output for a previous stage.
- Ignore any [REWRITTEN_VERSION] from the judge if it belongs to a different stage.
- Use the judge comments as critique, not as content to blindly copy.
- Preserve the user's original product idea.
- Do not invent an unrelated product.
- Return only the improved stage artifact.
- Return markdown only.

If STAGE_KEY is parse_submission:
Return:
# Parsed Product Brief — Revised Version

## Original Submitted Form
### Idea Title
### Idea Description
### Target Users
### Problem
### Constraints

---

## Agent-Structured Interpretation
### Clean Product Concept
### Product Type
### MVP Boundary
### Key Assumptions
### Key Risks
### Missing Information / Clarifying Questions
### Next Agent Input

If STAGE_KEY is brainstorm_parallel:
You MUST return Brainstorm Parallel output only.
You MUST include both:
[MARKET_DATA]
and
[CRAZY_IDEAS]

You MUST NOT return:
# Parsed Product Brief
# Parsed Product Brief — Revised Version
Original Submitted Form
Agent-Structured Interpretation

For brainstorm_parallel:
- Keep the market analysis.
- Revise the crazy ideas using human feedback.
- Keep novelty and creativity high.
- Make every idea MVP-buildable.
- Add real-world analogies, inspiration products, and links.
- Explain what is being mixed from each analogy product.
- Use well-known real products only.
- Do not invent fake product names.

Required brainstorm_parallel structure:

[MARKET_DATA]
## 🌍 Market Overview
## 📊 TAM / SAM / SOM
## 🏁 Competitor Landscape
## 🔗 Real-World Analogy Map
## 🎯 Strategic Insights & Positioning

[CRAZY_IDEAS]
## 🎨 Concept Space

For each concept:

### Concept <number> — <name>

- **Summary:**
- **Real-world analogies mixed:**
  - **Product:** URL — borrowed element
  - **Product:** URL — borrowed element
  - **Product:** URL — borrowed element
- **Why this mix is novel:**
- **Why it might win:**
- **Simple MVP version:**
- **Future breakthrough version:**
- **Risks / downsides:**

If STAGE_KEY is idea_cooker:
Preserve scored concept comparison.
If human feedback chooses or combines ideas, reflect that in the recommendation.
"""


STAGE_PROMPTS = {
    "parse_submission": INPUT_PARSER_PROMPT,
    "brainstorm_parallel": MARKET_ANALYSIS_PROMPT + "\n\n---\n\n" + CRAZY_IDEA_PROMPT,
    "market_analysis": MARKET_ANALYSIS_PROMPT,
    "crazy_ideas": CRAZY_IDEA_PROMPT,
    "idea_cooker": IDEA_COOKER_PROMPT,
    "theme_epic_generator": THEME_EPIC_PROMPT,
    "roadmap_generator": ROADMAP_PROMPT,
    "feature_generation": FEATURE_PROMPT,
    "prioritization_rice": PRIORITIZATION_PROMPT,
    "okr_generation": OKR_PROMPT,
    "three_month_planner": PLANNER_PROMPT,
    "write_report_pdf": REPORT_PROMPT,
}


# ============================================================
# AGENTS
# ============================================================

input_parser_agent = LlmAgent(name="InputParserAgent", model=MODEL_NAME, instruction=INPUT_PARSER_PROMPT)
market_agent = LlmAgent(name="MarketAnalysisAgent", model=MODEL_NAME, instruction=MARKET_ANALYSIS_PROMPT)
crazy_agent = LlmAgent(name="CrazyIdeaAgent", model=MODEL_NAME, instruction=CRAZY_IDEA_PROMPT)
idea_cooker_agent = LlmAgent(name="IdeaCookerAgent", model=MODEL_NAME, instruction=IDEA_COOKER_PROMPT)
theme_epic_agent = LlmAgent(name="ThemeEpicAgent", model=MODEL_NAME, instruction=THEME_EPIC_PROMPT)
roadmap_agent = LlmAgent(name="RoadmapAgent", model=MODEL_NAME, instruction=ROADMAP_PROMPT)
feature_agent = LlmAgent(name="FeatureGenerationAgent", model=MODEL_NAME, instruction=FEATURE_PROMPT)
prioritization_agent = LlmAgent(name="PrioritizationAgent", model=MODEL_NAME, instruction=PRIORITIZATION_PROMPT)
okr_agent = LlmAgent(name="OKRAgent", model=MODEL_NAME, instruction=OKR_PROMPT)
planner_agent = LlmAgent(name="PlannerAgent", model=MODEL_NAME, instruction=PLANNER_PROMPT)
report_writer_agent = LlmAgent(name="ReportWriterAgent", model=MODEL_NAME, instruction=REPORT_PROMPT)
evaluation_agent = LlmAgent(name="EvaluatorAgent", model=EVAL_MODEL, instruction=EVALUATOR_PROMPT)
reviewer_agent = LlmAgent(name="ReviewerAgent", model=EVAL_MODEL, instruction=REVIEWER_PROMPT)


# ============================================================
# HELPERS
# ============================================================

def _json_pretty(data: Dict[str, Any]) -> str:
    try:
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        return str(data)


def _render_idea_form_md(idea_form: Dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- **Idea Title:** {idea_form.get('idea_title', '')}",
            f"- **Idea Description:** {idea_form.get('idea_text', '')}",
            f"- **Target Users:** {idea_form.get('target_users', '')}",
            f"- **Problem:** {idea_form.get('problem', '')}",
            f"- **Constraints:** {idea_form.get('constraints', '')}",
        ]
    )


def _previous_stage_id(stage_id: str) -> str | None:
    if stage_id not in INTERACTIVE_STAGE_IDS:
        return None

    idx = INTERACTIVE_STAGE_IDS.index(stage_id)

    if idx == 0:
        return None

    return INTERACTIVE_STAGE_IDS[idx - 1]


def _latest_accepted_output(job: Dict[str, Any], stage_id: str) -> str:
    previous_id = _previous_stage_id(stage_id)

    if not previous_id:
        return ""

    stages = job.get("interactive", {}).get("stages", {})
    previous_state = stages.get(previous_id, {})

    return (
        previous_state.get("accepted_output")
        or previous_state.get("agent_output")
        or ""
    )


def _all_accepted_context(job: Dict[str, Any]) -> str:
    stages = job.get("interactive", {}).get("stages", {})
    parts = []

    for stage_id in INTERACTIVE_STAGE_IDS:
        state = stages.get(stage_id, {})
        accepted = state.get("accepted_output")
        if accepted:
            parts.append(f"## {stage_id}\n\n{accepted}")

    return "\n\n---\n\n".join(parts) if parts else "No accepted outputs yet."


def _build_interactive_stage_input(
    *,
    job: Dict[str, Any],
    stage_id: str,
    stage_title: str,
    current_output: str = "",
    human_feedback: str = "",
    judge_feedback: str = "",
    feedback_mode: str = "",
) -> str:
    founder_json = job.get("payload", {}) or {}
    founder_md = _render_idea_form_md(founder_json)
    previous_accepted = _latest_accepted_output(job, stage_id)
    all_context = _all_accepted_context(job)

    parts = [
        f"# STAGE_NAME\n{stage_title}",
        f"## STAGE_KEY\n{stage_id}",
        "## FOUNDER_IDEA_FORM_MARKDOWN",
        founder_md,
        "## FOUNDER_IDEA_FORM_JSON",
        _json_pretty(founder_json),
        "## PREVIOUS_ACCEPTED_OUTPUT",
        previous_accepted or "No previous accepted output.",
        "## ALL_ACCEPTED_CONTEXT",
        all_context,
    ]

    if current_output:
        parts.extend(["## CURRENT_STAGE_OUTPUT", current_output])

    if human_feedback:
        parts.extend(
            [
                "## HUMAN_FEEDBACK",
                human_feedback,
                "## HUMAN_FEEDBACK_INSTRUCTION",
                "This human feedback is non-negotiable. You must visibly apply it in the revised output.",
            ]
        )

    if judge_feedback:
        parts.extend(["## LLM_JUDGE_FEEDBACK", judge_feedback])

    if feedback_mode:
        parts.extend(["## FEEDBACK_MODE", feedback_mode])

    return "\n\n".join(parts)


def _is_temporary_model_error(error_text: str) -> bool:
    lower = (error_text or "").lower()

    return (
        "503" in error_text
        or "unavailable" in lower
        or "high demand" in lower
        or "429" in error_text
        or "resource_exhausted" in lower
        or "quota" in lower
        or "rate limit" in lower
    )


def _looks_like_wrong_stage(stage_id: str, text: str) -> bool:
    t = (text or "").lower()

    if stage_id == "brainstorm_parallel":
        if "parsed product brief" in t or "original submitted form" in t:
            return True
        if "[market_data]" not in t or "[crazy_ideas]" not in t:
            return True

    if stage_id == "parse_submission":
        if "[market_data]" in t or "[crazy_ideas]" in t:
            return True

    return False


def _fallback_parse(job: Dict[str, Any], error_text: str = "") -> str:
    payload = job.get("payload", {}) or {}
    idea_title = payload.get("idea_title", "")
    idea_text = payload.get("idea_text", "")
    target_users = payload.get("target_users", "")
    problem = payload.get("problem", "")
    constraints = payload.get("constraints", "")

    return f"""
# Parsed Product Brief — Local Fallback Version

## Original Submitted Form

### Idea Title
{idea_title or "Not provided."}

### Idea Description
{idea_text or "Not provided."}

### Target Users
{target_users or "Not provided."}

### Problem
{problem or "Not provided."}

### Constraints
{constraints or "Not provided."}

---

## Agent-Structured Interpretation

### Clean Product Concept
{idea_title or "This product"} is an AI-powered product management assistant that helps product managers, founders, and product teams transform messy product inputs into structured product artifacts, decisions, and execution plans.

### Product Type
AI-powered product management workspace / product strategy assistant.

### MVP Boundary
Focus the MVP on one strong workflow:

**idea or messy product input → parsed brief → brainstorm → prioritization → roadmap → feedback loop**

### Key Assumptions
- Product managers and founders have scattered product information across notes, documents, customer feedback, and meetings.
- Users will value an AI assistant that creates structured outputs while preserving human control.
- The first version should focus on one workflow instead of trying to replace every PM tool.
- Human feedback and LLM judge review can increase trust in generated outputs.

### Key Risks
- The product may become too broad if it tries to support too many workflows at once.
- Users may not trust AI-generated product decisions without transparency and editability.
- Integrations with Notion, Slack, Jira, or docs can slow the MVP if added too early.
- The product needs a clear wedge to avoid becoming a generic chatbot.

### Missing Information / Clarifying Questions
- Which first artifact matters most: PRD, roadmap, feature list, OKRs, or prioritization table?
- Should the MVP start with Notion integration or manual copy-paste input?
- Is the first audience solo PMs, startup founders, or product teams inside companies?

### Next Agent Input
Use the exact submitted form and this structured interpretation as source of truth. The Brainstorming Agent should generate market-grounded and creative product directions without changing the original idea.

## System Note
Local fallback was used. Last model error: {error_text}
""".strip()


def _fallback_brainstorm(job: Dict[str, Any], error_text: str = "", human_feedback: str = "") -> str:
    return f"""
[MARKET_DATA]
## 🌍 Market Overview
This product sits in the market for AI-powered product management, product operations, product discovery, and product strategy tools.

The founder idea is best positioned as **Cursor for Product Managers**: an AI-native workspace that helps PMs turn messy product inputs into structured decisions and execution artifacts.

## 📊 TAM / SAM / SOM
- **TAM:** The broader product management, collaboration, and AI productivity software market.
- **SAM:** Product teams, startup founders, innovation teams, product owners, and UX/product discovery teams who already work across Notion, Jira, Slack, docs, spreadsheets, and feedback tools.
- **SOM:** A realistic first wedge is AI-friendly PMs, founders, and small product teams who need faster PRDs, prioritization, roadmap options, and decision documentation.

## 🏁 Competitor Landscape

| Name | Type | URL | What they offer | Strengths | Weaknesses / Gaps |
|---|---|---|---|---|---|
| Notion AI | Indirect | https://www.notion.com/product/ai | AI inside docs and workspace | Strong workspace adoption | Generic, not PM-specific |
| ChatGPT / Claude | Indirect | https://chatgpt.com/ | General AI reasoning and drafting | Flexible and powerful | No persistent PM workflow by default |
| Jira Product Discovery | Adjacent | https://www.atlassian.com/software/jira/product-discovery | Product discovery and prioritization | Strong Atlassian ecosystem | Less generative and less like a PM copilot |
| Productboard | Adjacent | https://www.productboard.com/ | Feedback, insights, roadmap management | Strong PM workflows | Heavy and not AI-native enough for fast drafting |
| Dovetail | Adjacent | https://dovetail.com/ | Research repository and insight synthesis | Strong research synthesis | Not a full PM strategy workspace |

## 🔗 Real-World Analogy Map

| Product | URL | What Nextify can learn from it |
|---|---|---|
| Cursor | https://www.cursor.com/ | Deep AI-native professional workspace for a specific role |
| Notion AI | https://www.notion.com/product/ai | AI embedded inside flexible documents and knowledge bases |
| Linear | https://linear.app/ | Fast, opinionated execution workflow and clean UX |
| Jira Product Discovery | https://www.atlassian.com/software/jira/product-discovery | Product discovery and prioritization structures |
| Productboard | https://www.productboard.com/ | Feedback-to-roadmap workflow |
| Dovetail | https://dovetail.com/ | Research synthesis and insight repository |
| Miro | https://miro.com/ | Visual collaboration and ideation canvas |
| Perplexity | https://www.perplexity.ai/ | Answer engine with sourced exploration |
| Gamma | https://gamma.app/ | Structured AI-generated presentation artifacts |

## 🎯 Strategic Insights & Positioning
- Position as **Cursor for Product Managers**, not another generic AI writer.
- Start with one clear workflow: messy input → product brief → idea options → prioritization → roadmap → feedback loop.
- Keep Notion/Jira/Slack integrations as future advantages, not MVP blockers.
- Use human feedback and LLM judge review as a trust layer.
- Differentiate through product-context memory, decision traceability, and section-level review.

[CRAZY_IDEAS]
## 🎨 Concept Space

### Concept 1 — PRD Autopilot

- **Summary:** PMs paste messy notes, customer feedback, and goals. The system turns them into a structured PRD with assumptions, risks, acceptance criteria, and open questions.
- **Real-world analogies mixed:**
  - **Cursor:** https://www.cursor.com/ — role-specific AI workspace that understands context
  - **Notion AI:** https://www.notion.com/product/ai — AI inside editable docs
  - **Coda:** https://coda.io/ — structured docs and product templates
- **Why this mix is novel:** It combines AI-native role-specific assistance with editable PM documentation and structured product templates.
- **Why it might win:**
  - PRDs are painful, frequent, and easy to demonstrate.
  - It gives a clear wedge before becoming a full PM operating system.
- **Simple MVP version:** Text input box + PRD template generator + editable sections + feedback loop.
- **Future breakthrough version:** Live PRD that updates from Slack, Notion, Jira, user interviews, and analytics.
- **Risks / downsides:**
  - Could be perceived as a document generator unless decision intelligence is emphasized.
  - Needs strong output quality to earn trust.

### Concept 2 — Decision Memory Copilot

- **Summary:** A product decision log that remembers why roadmap decisions were made and links them to evidence, feedback, tradeoffs, and business goals.
- **Real-world analogies mixed:**
  - **Linear:** https://linear.app/ — clean execution workflow and issue history
  - **Productboard:** https://www.productboard.com/ — feedback-to-roadmap evidence trail
  - **Perplexity:** https://www.perplexity.ai/ — sourced answers and visible reasoning references
- **Why this mix is novel:** It turns product decisions into traceable, evidence-backed artifacts rather than scattered meeting notes.
- **Why it might win:**
  - PMs often struggle to justify decisions later.
  - It creates trust through traceability.
- **Simple MVP version:** Manual decision entry + AI-generated rationale + evidence links + risk summary.
- **Future breakthrough version:** Auto-captures decisions from meetings, docs, roadmap changes, and stakeholder comments.
- **Risks / downsides:**
  - Requires habit formation.
  - Needs careful UX so it does not feel like extra admin work.

### Concept 3 — Opportunity Scout

- **Summary:** The AI scans messy inputs and proposes product opportunities, user problems, feature bets, and experiments.
- **Real-world analogies mixed:**
  - **Dovetail:** https://dovetail.com/ — research synthesis and insight clustering
  - **Miro:** https://miro.com/ — opportunity mapping and visual ideation
  - **Jira Product Discovery:** https://www.atlassian.com/software/jira/product-discovery — opportunity prioritization workflow
- **Why this mix is novel:** It combines research synthesis, visual ideation, and product prioritization into a guided opportunity engine.
- **Why it might win:**
  - Helps PMs move from chaos to opportunity spaces.
  - Fits early discovery before PRDs and roadmap decisions.
- **Simple MVP version:** Paste customer feedback and goals → generate 5 opportunities with evidence, priority, and validation experiments.
- **Future breakthrough version:** Continuous opportunity radar connected to support, sales, analytics, and competitor changes.
- **Risks / downsides:**
  - Needs quality inputs to avoid generic opportunities.
  - Must explain why each opportunity matters.

### Concept 4 — Roadmap Debate Agent

- **Summary:** The AI creates multiple roadmap options and argues the tradeoffs like a product leadership review.
- **Real-world analogies mixed:**
  - **ChatGPT:** https://chatgpt.com/ — reasoning and argument generation
  - **Jira Product Discovery:** https://www.atlassian.com/software/jira/product-discovery — prioritization and roadmap framing
  - **Linear:** https://linear.app/ — execution sequencing and delivery clarity
- **Why this mix is novel:** It brings strategic debate into roadmap planning instead of producing one flat recommendation.
- **Why it might win:**
  - Roadmap decisions are political, strategic, and hard to explain.
  - It gives PMs options and tradeoffs, not just answers.
- **Simple MVP version:** Generate three roadmap options: conservative, growth-focused, and innovation-led, each with tradeoffs.
- **Future breakthrough version:** Multi-agent roadmap council with engineering, customer, market, and business perspectives.
- **Risks / downsides:**
  - Needs clear scoring criteria.
  - Could become verbose if not tightly designed.

### Concept 5 — Product Strategy Canvas Builder

- **Summary:** A guided workspace that turns an early idea into positioning, users, problems, MVP scope, features, risks, and OKRs.
- **Real-world analogies mixed:**
  - **Miro:** https://miro.com/ — visual canvas and collaborative strategy mapping
  - **Notion AI:** https://www.notion.com/product/ai — AI-generated structured workspaces
  - **Gamma:** https://gamma.app/ — polished structured output generation
- **Why this mix is novel:** It converts messy strategy thinking into a usable product canvas and shareable artifact.
- **Why it might win:**
  - Great for founders and innovation teams who need structure quickly.
  - Easy to demo and export.
- **Simple MVP version:** Step-by-step form + AI-generated canvas + export to markdown, Notion, or PDF.
- **Future breakthrough version:** Full strategy operating system with memory, benchmarks, and investor-ready outputs.
- **Risks / downsides:**
  - May feel less novel unless the interaction design is excellent.
  - Needs strong templates to be genuinely useful.

## Feedback Applied
{human_feedback or "No human feedback was provided, but the fallback preserves creativity while making each idea MVP-buildable and analogy-rich."}

## Recommendation
Keep the creative range, but make every idea practical by separating:
1. the **simple MVP version**
2. the **future breakthrough version**
3. the **real-world analogy products being mixed**

## System Note
Local fallback was used. Last model error: {error_text}
""".strip()


def _local_fallback_output(
    *,
    stage_id: str,
    stage_title: str,
    job: Dict[str, Any],
    current_output: str = "",
    human_feedback: str = "",
    judge_feedback: str = "",
    feedback_mode: str = "",
    error_text: str = "",
) -> str:
    if stage_id == "parse_submission":
        return _fallback_parse(job, error_text)

    if stage_id == "brainstorm_parallel":
        return _fallback_brainstorm(job, error_text, human_feedback)

    if feedback_mode:
        return f"""
# Revised {stage_title} — Local Fallback Version

## Change Summary
The system attempted to revise this stage using **{feedback_mode}**, but the model was temporarily unavailable.

## Applied Human Feedback
{human_feedback or "No human feedback was provided for this revision."}

## Applied LLM Judge Feedback
{judge_feedback or "No LLM judge feedback was available or selected for this revision."}

## Revised Output
{current_output or _latest_accepted_output(job, stage_id) or _all_accepted_context(job)}

## Practical Improvement Added
- Keep the product direction focused.
- Make the output clearer and easier to approve.
- Preserve the accepted context from previous stages.
- Prepare the content for the next agent.

## System Note
Local fallback was used. Last model error: {error_text}
""".strip()

    return f"""
# {stage_title} — Local Fallback Version

## Source Context
{_latest_accepted_output(job, stage_id) or _all_accepted_context(job)}

## Draft Output
This stage should build on the accepted previous output and create the next product planning artifact.

## Recommended Direction
- Keep the product focused on the approved concept.
- Preserve the target users, problem, constraints, and MVP boundary.
- Make the next output concrete enough for approval and downstream planning.

## System Note
Local fallback was used. Last model error: {error_text}
""".strip()


async def _run_agent_once(
    *,
    agent: LlmAgent,
    input_text: str,
    user_id: str,
    session_id: str,
    session_service: InMemorySessionService,
) -> str:
    max_attempts = 3
    last_error = None

    for attempt in range(1, max_attempts + 1):
        attempt_session_id = f"{session_id}_{attempt}"

        try:
            await session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=attempt_session_id,
            )

            runner = Runner(
                agent=agent,
                app_name=APP_NAME,
                session_service=session_service,
            )

            user_message = types.Content(
                role="user",
                parts=[types.Part(text=input_text)],
            )

            final_text = ""

            async for event in runner.run_async(
                user_id=user_id,
                session_id=attempt_session_id,
                new_message=user_message,
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        part_text = getattr(part, "text", None)
                        if part_text:
                            final_text += part_text

            if final_text.strip():
                return final_text.strip()

            last_error = "Agent returned empty output."

        except Exception as exc:
            last_error = str(exc)

            if not _is_temporary_model_error(last_error):
                raise

            if attempt < max_attempts:
                await asyncio.sleep(3 * attempt)
                continue

    raise RuntimeError(
        f"Gemini/ADK call failed after {max_attempts} attempts. Last error: {last_error}"
    )


async def run_interactive_stage_adk(
    *,
    job: Dict[str, Any],
    stage_id: str,
    stage_title: str,
) -> str:
    session_service = InMemorySessionService()
    user_id = "nextify_interactive_user"

    try:
        if stage_id == "parse_submission":
            input_text = _build_interactive_stage_input(
                job=job,
                stage_id=stage_id,
                stage_title=stage_title,
            )

            output = await _run_agent_once(
                agent=input_parser_agent,
                input_text=input_text,
                user_id=user_id,
                session_id=f"interactive_{stage_id}_{uuid.uuid4().hex}",
                session_service=session_service,
            )

        elif stage_id == "brainstorm_parallel":
            market_input = _build_interactive_stage_input(
                job=job,
                stage_id=stage_id,
                stage_title="Market Analysis",
            )

            crazy_input = _build_interactive_stage_input(
                job=job,
                stage_id=stage_id,
                stage_title="Crazy Ideas",
            )

            market_md = await _run_agent_once(
                agent=market_agent,
                input_text=market_input,
                user_id=user_id,
                session_id=f"market_{uuid.uuid4().hex}",
                session_service=session_service,
            )

            await asyncio.sleep(2)

            crazy_md = await _run_agent_once(
                agent=crazy_agent,
                input_text=crazy_input,
                user_id=user_id,
                session_id=f"crazy_{uuid.uuid4().hex}",
                session_service=session_service,
            )

            output = "\n\n---\n\n".join(
                block for block in [market_md, crazy_md] if block
            ).strip()

        else:
            input_text = _build_interactive_stage_input(
                job=job,
                stage_id=stage_id,
                stage_title=stage_title,
            )

            if stage_id == "idea_cooker":
                agent = idea_cooker_agent
            elif stage_id == "theme_epic_generator":
                agent = theme_epic_agent
            elif stage_id == "roadmap_generator":
                agent = roadmap_agent
            elif stage_id == "feature_generation":
                agent = feature_agent
            elif stage_id == "prioritization_rice":
                agent = prioritization_agent
            elif stage_id == "okr_generation":
                agent = okr_agent
            elif stage_id == "three_month_planner":
                agent = planner_agent
            elif stage_id == "write_report_pdf":
                agent = report_writer_agent
            else:
                raise ValueError(f"Unknown stage_id: {stage_id}")

            output = await _run_agent_once(
                agent=agent,
                input_text=input_text,
                user_id=user_id,
                session_id=f"interactive_{stage_id}_{uuid.uuid4().hex}",
                session_service=session_service,
            )

        if _looks_like_wrong_stage(stage_id, output):
            return _local_fallback_output(
                stage_id=stage_id,
                stage_title=stage_title,
                job=job,
                error_text="Model returned an output that looked like the wrong stage.",
            )

        return output

    except Exception as exc:
        return _local_fallback_output(
            stage_id=stage_id,
            stage_title=stage_title,
            job=job,
            error_text=str(exc),
        )


async def run_interactive_judge_adk(
    *,
    job: Dict[str, Any],
    stage_id: str,
    stage_title: str,
    stage_content: str,
) -> str:
    session_service = InMemorySessionService()
    user_id = "nextify_judge_user"

    founder_json = job.get("payload", {}) or {}
    founder_md = _render_idea_form_md(founder_json)
    previous_accepted = _latest_accepted_output(job, stage_id)
    all_context = _all_accepted_context(job)
    original_prompt = STAGE_PROMPTS.get(stage_id, "")

    input_text = "\n\n".join(
        [
            f"# STAGE_NAME\n{stage_title}",
            f"## STAGE_KEY\n{stage_id}",
            "## ORIGINAL_PROMPT",
            original_prompt,
            "## FOUNDER_IDEA_FORM_MARKDOWN",
            founder_md,
            "## FOUNDER_IDEA_FORM_JSON",
            _json_pretty(founder_json),
            "## PREVIOUS_ACCEPTED_OUTPUT",
            previous_accepted or "No previous accepted output.",
            "## ALL_ACCEPTED_CONTEXT",
            all_context,
            "## STAGE_CONTENT",
            stage_content,
        ]
    )

    try:
        return await _run_agent_once(
            agent=evaluation_agent,
            input_text=input_text,
            user_id=user_id,
            session_id=f"judge_{stage_id}_{uuid.uuid4().hex}",
            session_service=session_service,
        )

    except Exception as exc:
        return f"""
[QUALITY_SCORES]
- Overall: 7 — Fallback judge review generated because Gemini judge was unavailable.
- PromptAdherence: 7 — The output appears structurally usable but should be checked manually.
- Clarity: 7 — The output is readable.
- Feasibility: 7 — The idea appears feasible if scoped tightly.
- AlignmentWithIdea: 8 — The output appears aligned with the submitted idea.

[COMMENT_SUMMARY]
- Gemini judge was temporarily unavailable.
- Review the output manually before approving.
- Check that the output preserves the current stage type.
- For Brainstorm Parallel, preserve both market analysis and creative ideas.
- For Crazy Ideas, include real-world analogies, links, simple MVP versions, and future breakthrough versions.

[ISSUES_AND_FLAGS]
- Model judge failed with: {str(exc)}
- This is a local fallback review.

[IMPROVEMENT_SUGGESTIONS]
- Make the output more specific.
- Add measurable success criteria.
- Clarify MVP boundary.
- Preserve the original target user and problem.
- Add real-world analogy products and links.
- Remove any unrelated ideas.

[REWRITTEN_VERSION]
{stage_content}
""".strip()


async def run_interactive_reviewer_adk(
    *,
    job: Dict[str, Any],
    stage_id: str,
    stage_title: str,
    current_output: str,
    human_feedback: str,
    judge_feedback: str,
    feedback_mode: str,
) -> str:
    session_service = InMemorySessionService()
    user_id = "nextify_reviewer_user"

    input_text = _build_interactive_stage_input(
        job=job,
        stage_id=stage_id,
        stage_title=stage_title,
        current_output=current_output,
        human_feedback=human_feedback,
        judge_feedback=judge_feedback,
        feedback_mode=feedback_mode,
    )

    if human_feedback:
        input_text += f"""

# NON-NEGOTIABLE USER FEEDBACK TO APPLY
{human_feedback}

You must visibly apply this feedback in the revised output.
"""

    try:
        output = await _run_agent_once(
            agent=reviewer_agent,
            input_text=input_text,
            user_id=user_id,
            session_id=f"reviewer_{stage_id}_{uuid.uuid4().hex}",
            session_service=session_service,
        )

        if _looks_like_wrong_stage(stage_id, output):
            return _local_fallback_output(
                stage_id=stage_id,
                stage_title=stage_title,
                job=job,
                current_output=current_output,
                human_feedback=human_feedback,
                judge_feedback=judge_feedback,
                feedback_mode=feedback_mode,
                error_text="Reviewer returned an output that looked like the wrong stage.",
            )

        return output

    except Exception as exc:
        return _local_fallback_output(
            stage_id=stage_id,
            stage_title=stage_title,
            job=job,
            current_output=current_output,
            human_feedback=human_feedback,
            judge_feedback=judge_feedback,
            feedback_mode=feedback_mode,
            error_text=str(exc),
        )


__all__ = [
    "run_interactive_stage_adk",
    "run_interactive_judge_adk",
    "run_interactive_reviewer_adk",
]