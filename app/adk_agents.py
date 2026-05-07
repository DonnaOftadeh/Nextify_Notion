"""
Nextify ADK Interactive Agents

This file powers the interactive HITL workflow:
1. Parse Submission
2. Brainstorm Parallel
3. Idea Cooker
4. Theme & Epic Generator
5. Roadmap Generator
6. Feature Generation
7. Prioritization & RICE
8. OKR Generation
9. Three-Month Planner
10. Report

Rules:
- All agent outputs must be markdown.
- No JSON-only final outputs.
- No code fences unless explicitly requested by the user.
- Reviewer must preserve the current stage type.
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
- Include the exact original form content first.
- Include the idea title.
- Do not omit any submitted field.
- Do not rewrite the original form fields in "Original Submitted Form".
- Preserve the user's original idea.
- Do not invent a different product.
- Do not add MCP unless the user explicitly mentions MCP.
- If the user says "like Notion and Streamlit", treat them as possible tools, not mandatory architecture.
- Explain why you interpreted the idea this way using a short visible rationale.
- Return markdown only.
- Do not output JSON.
- Do not use code fences.

Output exactly:

# Parsed Product Brief

## Original Submitted Form

### Idea Title
<exact idea_title>

### Idea Description
<exact idea_text>

### Target Users
<exact target_users>

### Problem
<exact problem>

### Constraints
<exact constraints>

---

## Agent-Structured Interpretation

### Clean Product Concept
<clear product concept>

### Product Type
<classification>

### Core Job-To-Be-Done
<main job>

### MVP Boundary
<smallest useful MVP>

### What I Used From The Form
- <mapping from exact form to interpretation>

### Inferred Assumptions
- <Explicitly stated / Reasonably inferred / Needs validation>

### What Is Not Assumed
- MCP is not assumed unless explicitly mentioned.
- Full multimodal support is not assumed for MVP unless explicitly prioritized.
- Notion and Streamlit are treated as possible tools unless the form says they are mandatory.

### Agent Reasoning Summary
- <3–5 concise rationale bullets, not hidden chain-of-thought>

### Confidence Level
<High / Medium / Low plus one sentence>

### Key Risks
- <risks>

### Missing Information / Clarifying Questions
- <questions>

### Next Agent Input
<source-of-truth summary for Brainstorm Parallel>
"""


MARKET_ANALYSIS_PROMPT = """
You are the MarketAnalysisAgent for Nextify.

Your job:
- Analyze the market for the accepted parsed brief.
- Include competitors, analogies, and links.
- Apply human feedback if provided.
- Keep output stage-specific.

Return markdown only.
Do not output JSON.
Do not use code fences.

Output exactly:

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

Generate bold, novel, creative, but MVP-buildable concepts.

CRITICAL RULES:
- Apply human feedback if provided.
- Keep creativity and novelty high.
- Every idea must have a simple MVP version.
- Include real-world analogies, inspiration products, and official links.
- Use well-known real products only.
- Do not invent fake company names.
- Do not return parser output.
- Do not include "Parsed Product Brief".
- Return markdown only.
- Do not output JSON.
- Do not use code fences.

Output exactly:

[CRAZY_IDEAS]
## 🎨 Concept Space

### Concept 1 — <name>

- **Summary:** <2–3 sentence description>
- **Real-world analogies mixed:**
  - **<Product name>:** <official URL> — <borrowed element>
  - **<Product name>:** <official URL> — <borrowed element>
  - **<Product name>:** <official URL> — <borrowed element>
- **Why this mix is novel:** <new combination>
- **Why it might win:**
  - <bullet>
  - <bullet>
- **Simple MVP version:** <small version>
- **Future breakthrough version:** <ambitious version>
- **Risks / downsides:**
  - <bullet>
  - <bullet>

Generate 3–5 concepts.

Useful analogy sources:
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
- [MARKET_DATA]
- [CRAZY_IDEAS]
- founder idea form
- accepted prior context
- optional human feedback

Your job:
1. Evaluate concepts from CrazyIdeaAgent.
2. Compare them using a scored tradeoff table.
3. Recommend one winning concept.
4. Explain why it wins.
5. Produce a product snapshot for the chosen concept.
6. Ask the user to approve, pick another concept, combine concepts, or give feedback.

CRITICAL RULES:
- Use the actual concepts from [CRAZY_IDEAS].
- Do not invent unrelated concepts.
- Preserve creativity and analogy links from Brainstorm Parallel.
- If the user selected, approved, or preferred a concept, reflect it inside the product snapshot.
- Never output JSON as the final answer.
- Never wrap output in code fences.
- Never use the heading AGENT_OUTPUT.
- Return markdown only.
- The final output must be readable for product managers.

Output exactly:

[TRADEOFF_TABLE]
## ⚖️ Tradeoff Analysis

### Criteria Used
- Differentiation
- Market Demand
- Technical Feasibility
- Strategic Fit
- Monetization Clarity
- MVP Simplicity

| Concept Name | Differentiation | Market Demand | Feasibility | Strategic Fit | Monetization | MVP Simplicity | Total Score |
|---|---:|---:|---:|---:|---:|---:|---:|

### Scoring Rationale

#### <Concept Name>
- <why it scored this way>
- <tradeoff>
- <real-world analogy relevance>

[TRADEOFF_SUMMARY]
## 🧠 Tradeoff Summary

### Winning Concept
<name>

### Why This Wins
- <reason>
- <reason>
- <reason>

### What To Borrow From Real-World Products
- **<Product>:** <URL> — <borrowed element>

### Tradeoffs To Watch
- <tradeoff>

[PRODUCT_SNAPSHOT_MD]
## 🧾 Product Snapshot: <Winning Concept Name>

### Problem
<problem>

### Target Users
<target users>

### Concept Summary
<summary>

### Core Value Proposition
<value proposition>

### MVP Scope
- <MVP feature>
- <MVP feature>
- <MVP feature>

### Real-World Analogies Used
- **<Product>:** <URL> — <what is borrowed>

### Risks & Mitigations
- **Risk:** <risk>
  - **Mitigation:** <mitigation>

### Example Use Cases
- <use case>

[USER_DECISION_NEEDED]
## ✅ User Decision Needed

Please choose one:
1. Approve this winning concept.
2. Pick another concept.
3. Combine two concepts.
4. Give feedback and rerun the Idea Cooker.
"""


THEME_EPIC_PROMPT = """
You are the ThemeEpicAgent for Nextify.

Return markdown only. Do not output JSON. Do not use code fences.

[THEME_EPIC_MD]
## 🎯 Strategic Themes

## 🧩 Epics
"""


ROADMAP_PROMPT = """
You are the RoadmapAgent for Nextify.

Return markdown only. Do not output JSON. Do not use code fences.

[ROADMAP_GENERATOR_MD]
## 🗺️ Strategic Roadmap

### Phase 1 – Foundation / MVP

### Phase 2 – Validation / Expansion

### Phase 3 – Optimization / Scale

## 🔗 Roadmap Logic
"""


FEATURE_PROMPT = """
You are the FeatureGenerationAgent for Nextify.

Return markdown only. Do not output JSON. Do not use code fences.

[FEATURE_LIST]
## 🧱 Feature List

| id | feature_name | what_it_does | why_it_matters | impact | effort | tags |
|---|---|---|---|---|---|---|

[FEATURE_DETAILS]
## 🔍 Feature Details
"""


PRIORITIZATION_PROMPT = """
You are the PrioritizationAgent for Nextify.

Return markdown only. Do not output JSON. Do not use code fences.

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

Return markdown only. Do not output JSON. Do not use code fences.

[OKR_SUMMARY]

[OBJECTIVES]

[KEY_RESULTS]

[MILESTONES_AND_CHECKPOINTS]

[METRICS_AND_INSTRUMENTATION]
"""


PLANNER_PROMPT = """
You are the Three-Month Planner Agent for Nextify.

Return markdown only. Do not output JSON. Do not use code fences.

[THREE_MONTH_OVERVIEW]

[MONTHLY_BREAKDOWN]

[WEEKLY_PLAN]

[EXPERIMENTS_AND_LEARNING]

[RISKS_AND_DEPENDENCIES]
"""


REPORT_PROMPT = """
You are the Report Writer Agent for Nextify.

Return markdown only. Do not output JSON. Do not use code fences.

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

Evaluate exactly one selected stage output.

Important:
- Evaluate only the current selected stage.
- Do not rewrite the output as a different stage.
- If STAGE_KEY is idea_cooker, rewritten version must preserve [TRADEOFF_TABLE], [TRADEOFF_SUMMARY], [PRODUCT_SNAPSHOT_MD], and [USER_DECISION_NEEDED].
- If STAGE_KEY is brainstorm_parallel, rewritten version must preserve [MARKET_DATA] and [CRAZY_IDEAS].
- If STAGE_KEY is parse_submission, rewritten version must preserve the original submitted form.
- Return markdown only.
- Do not output JSON.
- Do not use code fences.

Output exactly:

[QUALITY_SCORES]
- Overall: <0-10> — <short justification>
- PromptAdherence: <0-10> — <short justification>
- Clarity: <0-10> — <short justification>
- Feasibility: <0-10> — <short justification>
- AlignmentWithIdea: <0-10> — <short justification>

[COMMENT_SUMMARY]
- <3–5 concise bullets>

[ISSUES_AND_FLAGS]
- <issues>

[IMPROVEMENT_SUGGESTIONS]
- <suggestions>

[REWRITTEN_VERSION]
<revised markdown version of the same selected stage, not another stage>
"""


REVIEWER_PROMPT = """
You are the Nextify Reviewer/Rewriter Agent.

You revise exactly one selected stage output using:
- human feedback
- LLM judge feedback
- or both

CRITICAL RULES:
- Apply human feedback visibly.
- Preserve the current stage type.
- Do not return output for a previous stage.
- Use judge comments as critique, not as text to blindly copy.
- Preserve the user's original product idea.
- Return only the improved stage artifact.
- Return markdown only.
- Do not output JSON.
- Do not use code fences.
- Never use the heading AGENT_OUTPUT.

If STAGE_KEY is idea_cooker:
You MUST return Idea Cooker output only.
You MUST include:
[TRADEOFF_TABLE]
[TRADEOFF_SUMMARY]
[PRODUCT_SNAPSHOT_MD]
[USER_DECISION_NEEDED]

You MUST NOT return:
AGENT_OUTPUT
JSON object
code fence
only a short approval summary

If the user approves or prefers a concept:
- Keep the tradeoff table.
- Mark the preferred concept as the winning concept.
- Update the product snapshot around that concept.
- Still output the full Idea Cooker structure.

If STAGE_KEY is brainstorm_parallel:
You MUST include:
[MARKET_DATA]
[CRAZY_IDEAS]

If STAGE_KEY is parse_submission:
You MUST include:
# Parsed Product Brief — Revised Version
## Original Submitted Form
## Agent-Structured Interpretation
"""


STAGE_PROMPTS = {
    "parse_submission": INPUT_PARSER_PROMPT,
    "brainstorm_parallel": MARKET_ANALYSIS_PROMPT + "\n\n---\n\n" + CRAZY_IDEA_PROMPT,
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

input_parser_agent = LlmAgent("InputParserAgent", model=MODEL_NAME, instruction=INPUT_PARSER_PROMPT)
market_agent = LlmAgent("MarketAnalysisAgent", model=MODEL_NAME, instruction=MARKET_ANALYSIS_PROMPT)
crazy_agent = LlmAgent("CrazyIdeaAgent", model=MODEL_NAME, instruction=CRAZY_IDEA_PROMPT)
idea_cooker_agent = LlmAgent("IdeaCookerAgent", model=MODEL_NAME, instruction=IDEA_COOKER_PROMPT)
theme_epic_agent = LlmAgent("ThemeEpicAgent", model=MODEL_NAME, instruction=THEME_EPIC_PROMPT)
roadmap_agent = LlmAgent("RoadmapAgent", model=MODEL_NAME, instruction=ROADMAP_PROMPT)
feature_agent = LlmAgent("FeatureGenerationAgent", model=MODEL_NAME, instruction=FEATURE_PROMPT)
prioritization_agent = LlmAgent("PrioritizationAgent", model=MODEL_NAME, instruction=PRIORITIZATION_PROMPT)
okr_agent = LlmAgent("OKRAgent", model=MODEL_NAME, instruction=OKR_PROMPT)
planner_agent = LlmAgent("PlannerAgent", model=MODEL_NAME, instruction=PLANNER_PROMPT)
report_writer_agent = LlmAgent("ReportWriterAgent", model=MODEL_NAME, instruction=REPORT_PROMPT)
evaluation_agent = LlmAgent("EvaluatorAgent", model=EVAL_MODEL, instruction=EVALUATOR_PROMPT)
reviewer_agent = LlmAgent("ReviewerAgent", model=EVAL_MODEL, instruction=REVIEWER_PROMPT)


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
    return previous_state.get("accepted_output") or previous_state.get("agent_output") or ""


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

    if stage_id == "parse_submission":
        return "[market_data]" in t or "[crazy_ideas]" in t or "[tradeoff_table]" in t

    if stage_id == "brainstorm_parallel":
        return (
            "parsed product brief" in t
            or "original submitted form" in t
            or "[market_data]" not in t
            or "[crazy_ideas]" not in t
        )

    if stage_id == "idea_cooker":
        return (
            "agent_output" in t
            or t.strip().startswith("{")
            or "[tradeoff_table]" not in t
            or "[tradeoff_summary]" not in t
            or "[product_snapshot_md]" not in t
        )

    return False


def _fallback_for_stage(
    *,
    job: Dict[str, Any],
    stage_id: str,
    stage_title: str,
    current_output: str = "",
    human_feedback: str = "",
    judge_feedback: str = "",
    feedback_mode: str = "",
    error_text: str = "",
) -> str:
    payload = job.get("payload", {}) or {}

    if stage_id == "parse_submission":
        return f"""
# Parsed Product Brief — Local Fallback Version

## Original Submitted Form

### Idea Title
{payload.get("idea_title", "Not provided.")}

### Idea Description
{payload.get("idea_text", "Not provided.")}

### Target Users
{payload.get("target_users", "Not provided.")}

### Problem
{payload.get("problem", "Not provided.")}

### Constraints
{payload.get("constraints", "Not provided.")}

---

## Agent-Structured Interpretation

### Clean Product Concept
{payload.get("idea_title", "This product")} is an AI-powered product management workspace that helps PMs transform messy inputs into structured product decisions.

### Product Type
AI-powered product management copilot.

### Core Job-To-Be-Done
Help product managers move from scattered information to clear product decisions.

### MVP Boundary
Text-first input → parsed brief → brainstorm → prioritization → roadmap → human feedback loop.

### What I Used From The Form
- Used the idea title as positioning.
- Used the idea description as the product workflow.
- Used the target users as the first audience.
- Used constraints to keep the MVP narrow.

### Inferred Assumptions
- Explicitly stated: human feedback and agent critique matter.
- Reasonably inferred: trust and versioning matter.
- Needs validation: exact first integration and first output type.

### What Is Not Assumed
- MCP is not assumed unless explicitly mentioned.
- Full multimodal support is not assumed for MVP.
- Notion and Streamlit are not treated as mandatory production architecture unless confirmed.

### Agent Reasoning Summary
- The form emphasizes messy input transformation.
- The target user is product managers and innovators.
- The MVP must be lightweight and achievable.
- The workflow should stay human-in-the-loop.

### Confidence Level
Medium-high.

### Key Risks
- Scope creep.
- Low trust in AI outputs.
- Too much integration complexity.

### Missing Information / Clarifying Questions
- Which first output should be generated?
- Which first input type matters most?
- Is Notion required or optional?

### Next Agent Input
Generate market-grounded and creative product directions for this PM copilot.

## System Note
Fallback used. Last error: {error_text}
""".strip()

    if stage_id == "brainstorm_parallel":
        return f"""
[MARKET_DATA]
## 🌍 Market Overview
The product sits in AI-powered product management, discovery, and product strategy software.

## 📊 TAM / SAM / SOM
- **TAM:** Broad PM, collaboration, and AI productivity software market.
- **SAM:** Product managers, founders, product owners, UX researchers, and innovation teams.
- **SOM:** AI-forward PMs and small product teams seeking faster product decisions.

## 🏁 Competitor Landscape

| Name | Type | URL | What they offer | Strengths | Weaknesses / Gaps |
|---|---|---|---|---|---|
| Cursor | Analogy | https://www.cursor.com/ | AI coding workspace | Strong role-specific AI workflow | Not for PMs |
| Notion AI | Indirect | https://www.notion.com/product/ai | AI inside docs | Strong workspace | Generic |
| Productboard | Adjacent | https://www.productboard.com/ | Feedback and roadmaps | PM-specific | Heavy |
| Jira Product Discovery | Adjacent | https://www.atlassian.com/software/jira/product-discovery | Discovery and prioritization | Strong ecosystem | Less AI-native |
| Dovetail | Adjacent | https://dovetail.com/ | Research synthesis | Strong insights | Not full PM copilot |

## 🔗 Real-World Analogy Map

| Product | URL | What Nextify can learn from it |
|---|---|---|
| Cursor | https://www.cursor.com/ | Deep role-specific AI workspace |
| Perplexity | https://www.perplexity.ai/ | Sourced explanations |
| Miro | https://miro.com/ | Visual thinking |
| Linear | https://linear.app/ | Fast execution UX |
| Gamma | https://gamma.app/ | Structured AI-generated artifacts |

## 🎯 Strategic Insights & Positioning
- Position as Cursor for Product Managers.
- Start with a narrow idea-to-decision workflow.
- Use human and judge feedback as a trust layer.

[CRAZY_IDEAS]
## 🎨 Concept Space

### Concept 1 — Insight Weaver

- **Summary:** Turns messy feedback and notes into product insights, themes, and problem statements.
- **Real-world analogies mixed:**
  - **Dovetail:** https://dovetail.com/ — research synthesis
  - **Perplexity:** https://www.perplexity.ai/ — sourced explanations
  - **Notion AI:** https://www.notion.com/product/ai — editable workspace output
- **Why this mix is novel:** It combines research synthesis, evidence-backed reasoning, and editable PM artifacts.
- **Why it might win:**
  - Solves a painful PM discovery problem.
  - Easy to test with copy-paste inputs.
- **Simple MVP version:** Paste feedback → AI extracts themes, evidence, and problem statements.
- **Future breakthrough version:** Connected insight engine across Slack, support, analytics, and interviews.
- **Risks / downsides:**
  - Needs evidence traceability.
  - Can become generic if outputs lack PM structure.

### Concept 2 — Roadmap Debate Agent

- **Summary:** Generates multiple roadmap options and explains tradeoffs.
- **Real-world analogies mixed:**
  - **ChatGPT:** https://chatgpt.com/ — debate and reasoning
  - **Jira Product Discovery:** https://www.atlassian.com/software/jira/product-discovery — prioritization
  - **Linear:** https://linear.app/ — execution clarity
- **Why this mix is novel:** It makes roadmap planning interactive and explainable.
- **Why it might win:**
  - PMs need to justify roadmap decisions.
  - It turns subjective debate into visible alternatives.
- **Simple MVP version:** Input goals and features → output 3 roadmap options with tradeoffs.
- **Future breakthrough version:** Multi-agent roadmap council.
- **Risks / downsides:**
  - Needs strong scoring criteria.
  - Can become verbose.

### Concept 3 — Decision Catalyst Canvas

- **Summary:** A lightweight canvas that turns product ideas into options, risks, assumptions, and next actions.
- **Real-world analogies mixed:**
  - **Miro:** https://miro.com/ — visual canvas
  - **Gamma:** https://gamma.app/ — structured AI output
  - **Cursor:** https://www.cursor.com/ — role-specific AI assistance
- **Why this mix is novel:** It brings visual product thinking into an AI-guided PM workflow.
- **Why it might win:**
  - Clear and demoable.
  - Helps PMs communicate decisions.
- **Simple MVP version:** Form input → generated decision canvas.
- **Future breakthrough version:** Live strategy map connected to docs and evidence.
- **Risks / downsides:**
  - Needs strong UX.
  - Visual layer can delay MVP if overbuilt.

## Feedback Applied
{human_feedback or "No human feedback provided."}

## System Note
Fallback used. Last error: {error_text}
""".strip()

    if stage_id == "idea_cooker":
        return f"""
[TRADEOFF_TABLE]
## ⚖️ Tradeoff Analysis

### Criteria Used
- Differentiation
- Market Demand
- Technical Feasibility
- Strategic Fit
- Monetization Clarity
- MVP Simplicity

| Concept Name | Differentiation | Market Demand | Feasibility | Strategic Fit | Monetization | MVP Simplicity | Total Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Insight Weaver | 8 | 9 | 8 | 9 | 8 | 9 | 51 |
| Roadmap Debate Agent | 9 | 8 | 7 | 9 | 8 | 7 | 48 |
| Decision Catalyst Canvas | 8 | 8 | 8 | 9 | 7 | 8 | 48 |

### Scoring Rationale

#### Insight Weaver
- Scores highest because it starts with the strongest PM pain: turning messy inputs into usable insights.
- It is feasible as a text-first MVP.
- It borrows from Dovetail, Perplexity, and Notion AI while becoming more PM-specific.

#### Roadmap Debate Agent
- Highly differentiated but slightly harder to make trustworthy.
- Strong for strategy but needs clear scoring criteria.

#### Decision Catalyst Canvas
- Very visual and demoable.
- Could become powerful, but visual UX may add build complexity.

[TRADEOFF_SUMMARY]
## 🧠 Tradeoff Summary

### Winning Concept
Insight Weaver

### Why This Wins
- It has the clearest MVP wedge.
- It directly addresses messy inputs and product decision quality.
- It can later expand into roadmap, OKRs, and strategy canvas workflows.

### What To Borrow From Real-World Products
- **Dovetail:** https://dovetail.com/ — insight synthesis.
- **Perplexity:** https://www.perplexity.ai/ — evidence-backed explanations.
- **Notion AI:** https://www.notion.com/product/ai — editable workspace outputs.

### Tradeoffs To Watch
- It needs evidence traceability to avoid becoming a generic summarizer.
- It should not overbuild integrations too early.

[PRODUCT_SNAPSHOT_MD]
## 🧾 Product Snapshot: Insight Weaver

### Problem
Product managers struggle to turn scattered feedback, research, notes, and stakeholder input into clear product insights and decisions.

### Target Users
Product managers, product owners, founders, UX researchers, and innovation teams.

### Concept Summary
Insight Weaver is an AI-powered PM workspace that converts messy product inputs into structured insights, problem statements, opportunities, and decision-ready outputs.

### Core Value Proposition
Move from scattered context to structured product decisions faster, with evidence, critique, and human feedback.

### MVP Scope
- Paste text-based inputs such as feedback, notes, research, and feature requests.
- Generate themes, problem statements, opportunities, and risks.
- Provide a human feedback loop and revised versions.
- Export or copy structured output.

### Real-World Analogies Used
- **Dovetail:** https://dovetail.com/ — research synthesis.
- **Perplexity:** https://www.perplexity.ai/ — sourced reasoning.
- **Notion AI:** https://www.notion.com/product/ai — editable workspace output.

### Risks & Mitigations
- **Risk:** Generic summaries.
  - **Mitigation:** Force evidence, source snippets, and PM-specific structure.
- **Risk:** Too broad.
  - **Mitigation:** Start with insight-to-decision workflow only.

### Example Use Cases
- Turn customer feedback into opportunity areas.
- Convert meeting notes into product risks and assumptions.
- Generate a first prioritization discussion from messy context.

[USER_DECISION_NEEDED]
## ✅ User Decision Needed

Please choose one:
1. Approve Insight Weaver.
2. Pick another concept.
3. Combine Insight Weaver with another concept.
4. Give feedback and rerun the Idea Cooker.

## System Note
Fallback used. Last error: {error_text}
""".strip()

    return f"""
# {stage_title} — Local Fallback Version

## Draft Output
This stage should build on accepted previous outputs.

## Source Context
{_latest_accepted_output(job, stage_id) or _all_accepted_context(job)}

## System Note
Fallback used. Last error: {error_text}
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

    raise RuntimeError(f"Gemini/ADK call failed after {max_attempts} attempts. Last error: {last_error}")


async def run_interactive_stage_adk(
    *,
    job: Dict[str, Any],
    stage_id: str,
    stage_title: str,
) -> str:
    session_service = InMemorySessionService()
    user_id = "nextify_interactive_user"

    try:
        input_text = _build_interactive_stage_input(
            job=job,
            stage_id=stage_id,
            stage_title=stage_title,
        )

        if stage_id == "parse_submission":
            output = await _run_agent_once(
                agent=input_parser_agent,
                input_text=input_text,
                user_id=user_id,
                session_id=f"parse_{uuid.uuid4().hex}",
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

            output = "\n\n---\n\n".join([market_md, crazy_md]).strip()

        elif stage_id == "idea_cooker":
            output = await _run_agent_once(
                agent=idea_cooker_agent,
                input_text=input_text,
                user_id=user_id,
                session_id=f"cooker_{uuid.uuid4().hex}",
                session_service=session_service,
            )

        else:
            agent_map = {
                "theme_epic_generator": theme_epic_agent,
                "roadmap_generator": roadmap_agent,
                "feature_generation": feature_agent,
                "prioritization_rice": prioritization_agent,
                "okr_generation": okr_agent,
                "three_month_planner": planner_agent,
                "write_report_pdf": report_writer_agent,
            }
            agent = agent_map.get(stage_id)
            if not agent:
                raise ValueError(f"Unknown stage_id: {stage_id}")

            output = await _run_agent_once(
                agent=agent,
                input_text=input_text,
                user_id=user_id,
                session_id=f"{stage_id}_{uuid.uuid4().hex}",
                session_service=session_service,
            )

        if _looks_like_wrong_stage(stage_id, output):
            return _fallback_for_stage(
                job=job,
                stage_id=stage_id,
                stage_title=stage_title,
                error_text="Model returned wrong stage format.",
            )

        return output

    except Exception as exc:
        return _fallback_for_stage(
            job=job,
            stage_id=stage_id,
            stage_title=stage_title,
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
- PromptAdherence: 7 — Check manually.
- Clarity: 7 — The output is readable.
- Feasibility: 7 — The idea appears feasible if scoped tightly.
- AlignmentWithIdea: 8 — The output appears aligned.

[COMMENT_SUMMARY]
- Judge model failed.
- Preserve the selected stage format.
- Avoid JSON-only output.
- For Idea Cooker, preserve tradeoff table, summary, product snapshot, and user decision.

[ISSUES_AND_FLAGS]
- Model judge failed with: {str(exc)}

[IMPROVEMENT_SUGGESTIONS]
- Keep markdown structure.
- Remove JSON/code block output.
- Add clear product rationale.

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
            return _fallback_for_stage(
                job=job,
                stage_id=stage_id,
                stage_title=stage_title,
                current_output=current_output,
                human_feedback=human_feedback,
                judge_feedback=judge_feedback,
                feedback_mode=feedback_mode,
                error_text="Reviewer returned wrong stage format.",
            )

        return output

    except Exception as exc:
        return _fallback_for_stage(
            job=job,
            stage_id=stage_id,
            stage_title=stage_title,
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