"""
Nextify ADK Interactive Agents

This file powers the interactive Human-in-the-Loop workflow:

Raw idea form
→ Input Parser Agent
→ LLM Judge
→ Reviewer Agent applies human / judge / both feedback
→ user accepts output
→ accepted output becomes input to next agent

Every stage supports:
- run stage
- judge stage
- revise stage

Important stability fix:
- Every ADK call retries temporary Gemini errors like 503 high demand and 429 quota.
- Brainstorm Parallel runs MarketAnalysisAgent and CrazyIdeaAgent sequentially for now.
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

Rules:
- Preserve the user's original idea.
- Do not invent a different product.
- Be clear, practical, and useful for the next agent.
- Do not include internal reasoning.
- Return markdown only.

Output exactly this structure:

# Parsed Product Brief

## Clean Product Concept
Explain the product clearly.

## Target Users
List the target users.

## Problem Statement
Clarify the problem.

## Constraints
List constraints.

## Product Type
Classify the product.

## MVP Boundary
Define the smallest useful MVP.

## Key Assumptions
List key assumptions.

## Key Risks
List key risks.

## Next Agent Input
Summarize what the Brainstorming Agent should use as source of truth.
"""

MARKET_ANALYSIS_PROMPT = """
You are the MarketAnalysisAgent for Nextify.

You receive:
- the founder idea form
- the previous accepted stage output
- all accepted context so far

Your job:
1. Identify the relevant market.
2. Estimate TAM / SAM / SOM with plausible ranges.
3. Identify direct and indirect competitors.
4. Explain market gaps and positioning opportunities.
5. Stay faithful to the founder idea and prior accepted output.

Return markdown only.

Output:

[MARKET_DATA]
## 🌍 Market Overview

## 📊 TAM / SAM / SOM

## 🏁 Competitor Landscape

| Name | Type | What they offer | Strengths | Weaknesses / Gaps |
|---|---|---|---|---|

## 🎯 Strategic Insights & Positioning
"""

CRAZY_IDEA_PROMPT = """
You are the CrazyIdeaAgent for Nextify.

You receive:
- the founder idea form
- the previous accepted stage output
- all accepted context so far

Your job:
Generate 3–5 bold but MVP-buildable product concepts.

Rules:
- Stay faithful to the founder idea.
- Do not invent an unrelated product.
- Respect constraints.
- Make the concepts distinct.
- Return markdown only.

Output:

[CRAZY_IDEAS]
## 🎨 Concept Space

For each concept:
- Name
- Summary
- Why it might win
- Risks / downsides
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
Assemble them into a clean final product plan.

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

You receive:
- STAGE_NAME
- STAGE_KEY
- STAGE_CONTENT
- ORIGINAL_PROMPT
- FOUNDER_IDEA_FORM
- PREVIOUS_ACCEPTED_OUTPUT
- ALL_ACCEPTED_CONTEXT

Assess:
1. Prompt adherence
2. Clarity
3. Coherence
4. Feasibility
5. Alignment with the founder idea
6. Alignment with prior accepted context

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
- A revised version of the stage content
"""

REVIEWER_PROMPT = """
You are the Nextify Reviewer/Rewriter Agent.

You revise exactly one selected stage output.

You receive:
- STAGE_NAME
- STAGE_KEY
- CURRENT_STAGE_OUTPUT
- PREVIOUS_ACCEPTED_OUTPUT
- ALL_ACCEPTED_CONTEXT
- FOUNDER_IDEA_FORM_MARKDOWN
- FOUNDER_IDEA_FORM_JSON
- HUMAN_FEEDBACK
- LLM_JUDGE_FEEDBACK
- FEEDBACK_MODE

Your job:
Revise CURRENT_STAGE_OUTPUT using the selected feedback sources.

Rules:
- Preserve the user's original product idea.
- Do not invent an unrelated product.
- Do not include raw judge feedback verbatim unless necessary.
- Do not include raw human feedback verbatim unless necessary.
- Return only the improved stage artifact.
- Keep the same role and purpose of the selected stage.
- Make the output clear, specific, practical, and decision-ready.
- The revised output must be suitable to send to the next agent.

Special rule for parse_submission:
Return:
# Parsed Product Brief — Revised Version
## Clean Product Concept
## Target Users
## Problem Statement
## Constraints
## Product Type
## MVP Boundary
## Key Assumptions
## Key Risks
## Next Agent Input

Special rule for idea_cooker:
- Preserve scored concept comparison.
- If human feedback chooses or combines ideas, reflect that in the recommendation.

Return markdown only.
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
    lines: list[str] = []

    title = idea_form.get("idea_title")
    if title:
        lines.append(f"# {title}")

    for key, value in idea_form.items():
        if key == "idea_title":
            continue
        pretty_key = key.replace("_", " ").title()
        lines.append(f"- **{pretty_key}:** {value}")

    return "\n".join(lines)


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

    if not parts:
        return "No accepted outputs yet."

    return "\n\n---\n\n".join(parts)


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
        parts.extend(["## HUMAN_FEEDBACK", human_feedback])

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


# ============================================================
# PUBLIC FUNCTIONS USED BY FASTAPI
# ============================================================

async def run_interactive_stage_adk(
    *,
    job: Dict[str, Any],
    stage_id: str,
    stage_title: str,
) -> str:
    session_service = InMemorySessionService()
    user_id = "nextify_interactive_user"

    if stage_id == "parse_submission":
        input_text = _build_interactive_stage_input(
            job=job,
            stage_id=stage_id,
            stage_title=stage_title,
        )
        agent = input_parser_agent

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

        return "\n\n---\n\n".join(
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

    return await _run_agent_once(
        agent=agent,
        input_text=input_text,
        user_id=user_id,
        session_id=f"interactive_{stage_id}_{uuid.uuid4().hex}",
        session_service=session_service,
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

    input_text = "\n\n".join([
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
    ])

    return await _run_agent_once(
        agent=evaluation_agent,
        input_text=input_text,
        user_id=user_id,
        session_id=f"judge_{stage_id}_{uuid.uuid4().hex}",
        session_service=session_service,
    )


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

    return await _run_agent_once(
        agent=reviewer_agent,
        input_text=input_text,
        user_id=user_id,
        session_id=f"reviewer_{stage_id}_{uuid.uuid4().hex}",
        session_service=session_service,
    )


__all__ = [
    "run_interactive_stage_adk",
    "run_interactive_judge_adk",
    "run_interactive_reviewer_adk",
]