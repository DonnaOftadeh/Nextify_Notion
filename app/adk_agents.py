"""
Nextify ADK Interactive Agents

This file powers the interactive Human-in-the-Loop workflow.

Stability behavior:
- Tries Gemini/ADK first.
- Retries temporary Gemini errors such as 503 high demand and 429 quota.
- If Gemini still fails, returns a useful local fallback output instead of breaking the UI.
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

Your job:
1. Identify the relevant market.
2. Estimate TAM / SAM / SOM with plausible ranges.
3. Identify direct and indirect competitors.
4. Explain market gaps and positioning opportunities.
5. Stay faithful to the founder idea and prior accepted output.

Return markdown only.
"""

CRAZY_IDEA_PROMPT = """
You are the CrazyIdeaAgent for Nextify.

Your job:
Generate 3–5 bold but MVP-buildable product concepts.

Rules:
- Stay faithful to the founder idea.
- Do not invent an unrelated product.
- Respect constraints.
- Make the concepts distinct.
- Return markdown only.
"""

IDEA_COOKER_PROMPT = """
You are the IdeaCookerAgent for Nextify.

Your job:
1. Evaluate the strongest concepts.
2. Score them with rationale.
3. Recommend one winning concept.
4. Ask the user to approve or give feedback.

Return markdown only.
"""

THEME_EPIC_PROMPT = """
You are the ThemeEpicAgent for Nextify.

Your job:
Create 3–5 strategic themes and epics from the accepted product direction.

Return markdown only.
"""

ROADMAP_PROMPT = """
You are the RoadmapAgent for Nextify.

Your job:
Create a realistic roadmap for an early-stage MVP.

Return markdown only.
"""

FEATURE_PROMPT = """
You are the FeatureGenerationAgent for Nextify.

Your job:
Generate a clean feature list for MVP and near-term roadmap.

Return markdown only.
"""

PRIORITIZATION_PROMPT = """
You are the PrioritizationAgent for Nextify.

Your job:
Apply RICE prioritization and create a short feature roadmap.

Return markdown only.
"""

OKR_PROMPT = """
You are the OKRAgent for Nextify.

Your job:
Generate a small, realistic quarterly OKR set.

Return markdown only.
"""

PLANNER_PROMPT = """
You are the Three-Month Planner Agent for Nextify.

Your job:
Create a practical 3-month execution plan.

Return markdown only.
"""

REPORT_PROMPT = """
You are the Report Writer Agent for Nextify.

Your job:
Assemble all accepted stage outputs into a clean final product plan.

Return markdown only.
"""

EVALUATOR_PROMPT = """
You are the Evaluation & Quality Agent for Nextify.

You evaluate exactly one selected stage output.

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

You revise exactly one selected stage output using:
- human feedback
- LLM judge feedback
- or both

Rules:
- Preserve the user's original product idea.
- Do not invent an unrelated product.
- Return only the improved stage artifact.
- Make the output clearer, more specific, and more decision-ready.
- Return markdown only.
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
    payload = job.get("payload", {}) or {}
    idea_title = payload.get("idea_title", "Untitled idea")
    idea_text = payload.get("idea_text", "")
    target_users = payload.get("target_users", "")
    problem = payload.get("problem", "")
    constraints = payload.get("constraints", "")
    previous_accepted = _latest_accepted_output(job, stage_id)
    all_context = _all_accepted_context(job)

    if stage_id == "parse_submission":
        return f"""
# Parsed Product Brief — Local Fallback Version

## Clean Product Concept
{idea_title}

{idea_text}

## Target Users
{target_users or "Not specified yet."}

## Problem Statement
{problem or "Not specified yet."}

## Constraints
{constraints or "Not specified yet."}

## Product Type
AI-powered product management and product strategy assistant.

## MVP Boundary
Focus the MVP on one strong workflow: turning messy product inputs into one structured, decision-ready output.

## Key Assumptions
- Target users have scattered product information across notes, meetings, customer feedback, and planning tools.
- A focused assistant can create value before becoming a full product operating system.
- The first version should prioritize clarity, speed, and integration with existing workflows.

## Key Risks
- The MVP may become too broad if too many workflows are included.
- Users may not trust generated outputs without transparent review and editing.
- Integrations can slow down delivery if they are added too early.

## Next Agent Input
Use this idea as the source of truth: build a focused AI workspace for product managers that transforms messy inputs into structured product artifacts and supports human feedback loops.

## System Note
Gemini was temporarily unavailable, so this fallback output was generated locally. Last model error: {error_text}
""".strip()

    if stage_id == "brainstorm_parallel":
        return f"""
[MARKET_DATA]
## 🌍 Market Overview
This product sits in the market for AI-powered product management, product operations, and product strategy tools.

## 📊 TAM / SAM / SOM
- **TAM:** Large global market across product management, collaboration, and AI productivity tools.
- **SAM:** Product teams, startup founders, innovation teams, and product owners using tools such as Notion, Jira, Slack, docs, and spreadsheets.
- **SOM:** A realistic first wedge is solo PMs, founders, and small product teams who need faster product documentation and prioritization.

## 🏁 Competitor Landscape
| Name | Type | What they offer | Strengths | Weaknesses / Gaps |
|---|---|---|---|---|
| Notion AI | Indirect | AI inside workspace docs | Strong workspace adoption | Not product-management-specific |
| ChatGPT | Indirect | General AI assistant | Flexible and powerful | Lacks persistent PM workflow structure |
| Jira Product Discovery | Direct/Adjacent | Product discovery and prioritization | Strong Atlassian ecosystem | Less generative and less conversational |
| Productboard | Direct/Adjacent | Feedback and roadmap management | Strong PM workflows | Can be heavy for small teams |

## 🎯 Strategic Insights & Positioning
- Position as “Cursor for Product Managers.”
- Start with one workflow: idea → brainstorm → prioritize → roadmap → feedback loop.
- Avoid replacing existing tools at first; integrate with them.
- Use human and LLM review as a trust-building differentiator.

---

[CRAZY_IDEAS]
## 🎨 Concept Space

### Concept 1: Cursor for PMs
- **Summary:** An AI workspace that helps product managers create product artifacts from messy inputs.
- **Why it might win:** Clear analogy, strong positioning, broad PM pain.
- **Risks:** Could become too broad.

### Concept 2: Product Decision Copilot
- **Summary:** A tool that explains and justifies roadmap and prioritization decisions.
- **Why it might win:** Strong value for teams that need alignment.
- **Risks:** Needs trusted context and good evaluation.

### Concept 3: Notion-native PM Agent
- **Summary:** A Notion-connected assistant that generates PRDs, priorities, and roadmaps inside existing workspaces.
- **Why it might win:** Low migration friction.
- **Risks:** Depends on integration quality.

## System Note
Gemini was temporarily unavailable, so this fallback output was generated locally. Last model error: {error_text}
""".strip()

    if stage_id == "idea_cooker":
        return f"""
[TRADEOFF_TABLE]
## ⚖️ Tradeoff Analysis

| Concept Name | Differentiation | Market Demand | Feasibility | Strategic Fit | Monetization | Total Score |
|---|---:|---:|---:|---:|---:|---:|
| Cursor for PMs | 9 | 8 | 7 | 10 | 8 | 42 |
| Product Decision Copilot | 8 | 8 | 8 | 8 | 8 | 40 |
| Notion-native PM Agent | 7 | 7 | 9 | 8 | 7 | 38 |

[TRADEOFF_SUMMARY]
## 🧠 Tradeoff Summary
The strongest concept is **Cursor for PMs** because it is memorable, directly aligned with the founder vision, and broad enough to grow into a larger platform.

[PRODUCT_SNAPSHOT_MD]
## 🧾 Product Snapshot
- **Problem:** PMs lose time turning scattered information into clear product decisions and artifacts.
- **Target users:** Product managers, founders, product owners, innovation teams, UX researchers.
- **Concept summary:** An AI workspace that helps PMs turn messy inputs into structured product outputs.
- **Core value proposition:** Faster product clarity with human feedback and LLM review.
- **MVP scope:** Start with idea parsing, brainstorming, prioritization, roadmap, and feedback loop.
- **Risks and mitigations:** Keep scope narrow; integrate with existing tools; make outputs editable and reviewable.
- **Example use cases:** PRD drafting, roadmap option generation, feature prioritization, OKR creation.

## User Decision Needed
Approve this direction or give feedback on which concept should be selected.

## System Note
Gemini was temporarily unavailable, so this fallback output was generated locally. Last model error: {error_text}
""".strip()

    if feedback_mode:
        return f"""
# Revised {stage_title} — Local Fallback Version

## Change Summary
The system attempted to revise this stage using **{feedback_mode}**, but Gemini was temporarily unavailable. This local fallback applies the available feedback at a structural level.

## Applied Human Feedback
{human_feedback or "No human feedback was provided for this revision."}

## Applied LLM Judge Feedback
{judge_feedback or "No LLM judge feedback was available or selected for this revision."}

## Revised Output
{current_output or previous_accepted or all_context}

## Practical Improvement Added
- Keep the product direction focused.
- Make the output clearer and easier to approve.
- Preserve the accepted context from previous stages.
- Prepare the content for the next agent.

## System Note
Gemini was temporarily unavailable, so this fallback revision was generated locally. Last model error: {error_text}
""".strip()

    return f"""
# {stage_title} — Local Fallback Version

## Source Context
{previous_accepted or all_context}

## Draft Output
This stage should build on the accepted previous output and create the next product planning artifact.

## Recommended Direction
- Keep the product focused on the approved concept.
- Preserve the target users, problem, constraints, and MVP boundary.
- Make the next output concrete enough for approval and downstream planning.

## System Note
Gemini was temporarily unavailable, so this fallback output was generated locally. Last model error: {error_text}
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
            agent = input_parser_agent

            return await _run_agent_once(
                agent=agent,
                input_text=input_text,
                user_id=user_id,
                session_id=f"interactive_{stage_id}_{uuid.uuid4().hex}",
                session_service=session_service,
            )

        if stage_id == "brainstorm_parallel":
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
- Check that the output preserves the founder idea.
- Add human feedback if the output is too broad or too generic.

[ISSUES_AND_FLAGS]
- Model judge failed with: {str(exc)}
- This is a local fallback review.

[IMPROVEMENT_SUGGESTIONS]
- Make the output more specific.
- Add measurable success criteria.
- Clarify MVP boundary.
- Preserve the original target user and problem.
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

    try:
        return await _run_agent_once(
            agent=reviewer_agent,
            input_text=input_text,
            user_id=user_id,
            session_id=f"reviewer_{stage_id}_{uuid.uuid4().hex}",
            session_service=session_service,
        )
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