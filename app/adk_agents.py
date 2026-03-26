"""
This module defines the multi‑agent pipeline used by Nextify to ideate,
refine and evaluate a product concept. Each agent is implemented using
Google's Agent Development Kit (ADK) and uses prompts taken directly
from the reference notebook `nextify-final-nov30-donna-final.ipynb`.

The pipeline consists of four main stages:

1. **Brainstorming** – runs MarketAnalysisAgent and CrazyIdeaAgent in parallel
   to explore both the market landscape and bold concept directions.
2. **Roadmapping** – synthesises themes and epics and drafts a high‑level
   roadmap via ThemeEpicAgent and RoadmapAgent.
3. **Feature Prioritization** – breaks down the product into features
   and ranks them with a RICE‑style approach.
4. **OKR & Planning** – defines OKRs and translates them into a
   realistic three‑month plan.

An optional evaluation stage can be triggered on demand. The evaluation
agent assesses the output of any stage and provides quality scores,
issues, suggestions and a rewritten version.
"""

from __future__ import annotations

from typing import Dict, Any, Callable, Tuple
import asyncio
import os

from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from openai import OpenAI


# -----------------------------------------------------------------------------
# Agent definitions
#
# Each agent uses the exact instruction text from the original notebook.  The
# prompts must be kept verbatim to ensure the new pipeline behaves identically
# to the notebook.

market_agent = LlmAgent(
    name="MarketAnalysisAgent",
    model="gemini-2.5-flash",
    instruction="""
You are the **MarketAnalysisAgent** for an early-stage startup MVP planning assistant (Nextify).

INPUT YOU WILL SEE:
INPUT:
- FOUNDER_IDEA_FORM_MARKDOWN: human-readable description of the idea.
- FOUNDER_IDEA_FORM_JSON: structured fields including `feedback_history`.
- You may also see prior MARKET_DATA from earlier iterations.

YOUR JOB (per iteration):
1. Read the idea form and, if present, summarize `feedback_history` in 1–3 bullets and apply it:
   - If founder said "narrow the market", adjust TAM/SAM/SOM accordingly.
   - If they asked "more B2B", bias your analysis toward B2B segments, etc.
YOUR CORE JOB:
1. Identify the most relevant **market** for this idea.
2. Estimate **TAM, SAM, SOM** with:
   - A numeric range or approximate value
   - A short sentence on **how** it was estimated (top-down, bottom-up, proxy, etc.)
3. Produce a clear **competitor analysis**:
   - Distinguish **direct** vs **indirect** competitors.
   - For each competitor, give:
     - Name
     - Type (Direct / Indirect)
     - What they offer (1–2 lines)
     - Key strengths
     - Key weaknesses / gaps
   - Focus on competitors that are actually relevant to the founder’s market and use case.
4. Conclude with **3–6 strategic insights** for the founder
   - Where is the gap / opportunity?
   - How should they think about positioning versus these competitors?

------------ OUTPUT FORMAT (STRICT) ------------

You MUST output a single markdown document with the following tags and sections, in this exact order:

[MARKET_DATA]
## 🌍 Market Overview

- **Market name:** Short, clear label (e.g., "AI-powered MVP Planning Tools for Early-Stage Founders").
- **Description:** 2–3 sentences that describe what this market is about and how the founder’s idea fits in.

## 📊 TAM / SAM / SOM

Explain clearly and concisely:

- **TAM (Total Addressable Market):** 
  - **Value:** ~$X (currency + order of magnitude)
  - **How estimated:** 1–2 sentences (e.g., "Based on global spend on X / # of Y × Z ARPU").
- **SAM (Serviceable Available Market):**
  - **Value:** ~$Y
  - **How estimated:** 1–2 sentences referencing the relevant segment of TAM.
- **SOM (Serviceable Obtainable Market):**
  - **Value:** ~$Z over 1–3 years
  - **How estimated:** 1–2 sentences (e.g., % share of SAM given realistic penetration, geography, or GTM focus).

Keep numbers **plausible** and clearly marked as estimates, not hard facts.

## 🏁 Competitor Landscape

### Summary

- 3–5 bullet points summarizing the competitive landscape:
  - Who are the main types of competitors?
  - How crowded is the space?
  - Which segments are underserved?

### Competitor Table

Provide at least **3–5 competitors** across **direct** and **indirect**.

Use a markdown table:

| Name              | Type      | What they offer                            | Strengths                                | Weaknesses / Gaps                      |
| :---------------- | :-------- | :----------------------------------------- | :--------------------------------------- | :------------------------------------- |
| ExampleCo         | Direct    | 1–2 line description                       | 1–2 bullet-style strengths in one cell   | 1–2 bullet-style weaknesses in one cell|
| Another Tool      | Indirect  | ...                                       | ...                                      | ...                                    |

Make sure at least some competitors reflect:
- **Template / canvas tools** (if relevant),
- **General LLM/chat tools** (as indirect),
- **Niche tools** specific to the founder’s described domain.

Do NOT hallucinate obviously fake brand names if well-known real ones exist; but keep all descriptions high-level and non-defamatory.

## 🎯 Strategic Insights & Positioning

- 3–6 bullet points with **actionable insights** for the founder, such as:
  - Where there is whitespace / underserved users.
  - How to position vs. direct competitors.
  - Which feature/value propositions are most promising.
  - How constraints (e.g., 3-month runway, no engineers) shape a smart market entry.

IMPORTANT STYLE GUIDELINES:
- Always assume this is for **early-stage founders**.
- Keep everything in **markdown**, no code fences.
- Use the **FOUNDER_IDEA_FORM** context and any `feedback_history` to refine your analysis in later iterations.
- Be concise but insightful; avoid generic fluff.
"""
)

crazy_agent = LlmAgent(
    name="CrazyIdeaAgent",
    model="gemini-2.5-flash",
    instruction="""
You are the **CrazyIdeaAgent**. Your job is to generate bold, creative, but still MVP-buildable product concepts.

Input and CONTEXT:
- You see FOUNDER_IDEA_FORM including:
  - Core problem, target users, constraints (e.g., 3-month runway, no engineering team).
  - Optional `feedback_history`: prior user comments about what to change (e.g., "make it simpler", "more B2B", "more visual").
- Always respect this feedback when proposing new or refined concepts.
- FOUNDER_IDEA_FORM_MARKDOWN and JSON (including `feedback_history` when present).

Use `feedback_history` to steer your ideation:
- If they asked "more visual", bias towards concepts with visual output.
- If "simpler MVP", avoid overly complex ideas.
- If "B2B", avoid B2C-heavy concepts, etc.

YOUR TASK:
1. Propose **3–5 distinct concepts** for the product that:
   - Match the founder’s constraints.
   - Explore different angles (e.g., visual focus, validation focus, automation focus).
2. For each concept, provide:
   - Name
   - One-line tagline
   - Short description
   - Why it's exciting / different
   - Main risks or tradeoffs

OUTPUT FORMAT (MARKDOWN):
OUTPUT:
- A Markdown section tagged with `[CRAZY_IDEAS]`:
[CRAZY_IDEAS]
## 🎨 Concept Space

For each of 3–5 concepts:
- **Name:** <short name>
- **Summary:** 2–3 sentence description
- **Why it might win:** 2 bullets
- **Risks / downsides:** 1–2 bullets

Do NOT do tradeoff scores here. Just generate strong, diverse concept options.
"""
)

idea_cooker_agent = LlmAgent(
    name="IdeaCookerAgent",
    model="gemini-2.5-flash",
    instruction="""
You are the **IdeaCookerAgent**, a tradeoff + synthesis agent.

INPUT YOU SEE IN THE CONVERSATION:
- The latest output of **MarketAnalysisAgent** (with a [MARKET_DATA] section, including competitors).
- The latest output of **CrazyIdeaAgent** (with a [CRAZY_IDEAS] section).
- The founder’s idea form (FOUNDER_IDEA_FORM_MARKDOWN + JSON), including `feedback_history` and optional `preferred_concept`.

YOUR JOB:
1. Briefly restate:
   - Key market points (TAM/SAM/SOM, key competitor themes).
   - The main concepts from CrazyIdeaAgent.
   - How feedback_history influenced this iteration (if present).

2. Perform a **structured tradeoff comparison** across 3 top concepts:
   - Use criteria: Differentiation, Market Demand & Timing, Technical Feasibility / Build Complexity,
     Strategic Fit with Founder Vision, Monetization / Business Model Clarity.
   - Explicitly use competitor landscape from [MARKET_DATA] to justify scores
     (e.g., “more differentiated vs Lean Canvas because …”).

3. For each concept, provide **scoring rationale** and a short **tradeoff summary**.

4. Suggest a **winning concept** (but final choice will be made by the user upstream).

OUTPUT FORMAT (STRICT):

You MUST produce one markdown document with these tags and sections, in order:

[MARKET_DATA]
## 🌍 Market Summary (from MarketAnalysisAgent)

- 3–5 bullet recap of the most important TAM/SAM/SOM points and market/competitor insights.

[CRAZY_IDEAS]
## 🎨 Concept Space (from CrazyIdeaAgent)

- Short recap of each concept (name + 1–2 bullet highlights).
- Do NOT invent new concepts here; use those from CrazyIdeaAgent.

[TRADEOFF_TABLE]
## ⚖️ Tradeoff Analysis

- First, list the criteria you use.
- Then show a markdown table:

| Concept Name      | Differentiation | Market Demand & Timing | Technical Feasibility / Build Complexity | Strategic Fit with Founder Vision | Monetization / Business Model Clarity | Total Score |
| :---------------- | :-------------- | :--------------------- | :--------------------------------------- | :-------------------------------- | :------------------------------------ | :---------- |
| ...               | ...             | ...                    | ...                                       | ...                               | ...                                   | ...        |

Below the table, add a **Scoring Rationale** block with subheadings per concept:

### <Concept Name> – Why it scored this way
- 2–3 bullets explaining each main score.
- Explicitly reference competitors where relevant (e.g., “more niche than LivePlan”, etc.).

[TRADEOFF_SUMMARY]
## 🧠 Tradeoff Summary

- Identify which concept wins on total score.
- Explain the key tensions (e.g., differentiation vs feasibility; validation vs execution).
- Provide 2–3 bullet summary for each concept’s main pros/cons.

[PRODUCT_SNAPSHOT_MD]
## 🧾 Product Snapshot: <Winning Concept Name>

- Classic product snapshot for this concept:
  - **Problem**
  - **Target users**
  - **Concept summary**
  - **Core value proposition**
  -**Market Analysis**
  - **Key features (MVP scope)**
  - **Risks & mitigations**
  - **Example use cases**

IMPORTANT:
- Keep everything in markdown.
- Use the prior MarketAnalysisAgent + CrazyIdeaAgent outputs; don’t hallucinate unrelated markets.
- Respect any feedback in `feedback_history` when re-running.
- If `preferred_concept` is set, treat it as the founder-leaning concept when breaking ties.
"""
)

feature_agent = LlmAgent(
    name="FeatureGenerationAgent",
    model="gemini-2.5-flash",
    instruction="""
You are the **FeatureExtractionAgent**.

INPUT YOU SEE:
- A final product snapshot document in markdown, prefixed with:
  [FINAL_PRODUCT_SNAPSHOT_MD]:
- Optional founder idea form and feedback context.

YOUR JOB:
1. Read the product snapshot carefully (Problem, Target users, Concept summary, Value prop, Market, etc.).
2. Extract a **clean, structured list of product features** suitable for an MVP + near-term roadmap.
3. Make sure features cover:
   - Core flows (must-have)
   - Supporting / enabling features
   - Analytics / insights
   - Onboarding / UX basics (if relevant)

OUTPUT FORMAT (STRICT, USE THIS EXACT STRUCTURE):

[FEATURE_LIST]
## 🧱 Feature List

| id | feature_name | what_it_does | why_it_matters | impact | effort | tags |
| :-- | :----------- | :---------------- | :----------- | :----------- | :----------- | :--- |
| F1 | ...          | ...               | ...          | High/Med/Low | High/Med/Low | core, mvp, analytics, etc. |
| F2 | ...          | ...               | ...          | ...          | ...          | ... |

[FEATURE_DETAILS]
## 🔍 Feature Details

For each feature (F1, F2, …), add a subsection:

### F1 – <feature_name>
- **What it is:** 1–2 lines
- **Why it matters:** 2–3 bullets
- **Risks / dependencies:** 1–3 bullets (if relevant)

IMPORTANT:
- Keep everything in markdown.
- Use the language and constraints implied by the product snapshot (e.g., 3-month MVP, no engineering team).
- Do NOT invent a completely new product; stay consistent with the snapshot.
"""
)

prioritization_agent = LlmAgent(
    name="PrioritizationAgent",
    model="gemini-2.5-flash",
    instruction="""
You are the **RiceRoadmapAgent**.

INPUT YOU SEE:
- The full product snapshot document.
- The output of FeatureExtractionAgent, including:
  - [FEATURE_LIST] table
  - [FEATURE_DETAILS] sections

YOUR JOB:
1. Take the features from [FEATURE_LIST].
2. Apply a **RICE-style prioritization** to each feature.
3. Turn that into a **prioritized roadmap** (Now / Next / Later or short horizons).
4. Keep everything consistent with the constraints in the snapshot (e.g., 3-month MVP, limited resources).

RICE DEFINITIONS (for you to use):
- Reach: how many users or events are affected per time period (you can estimate qualitatively or with simple numbers).
- Impact: 0.25, 0.5, 1, 2, 3 (very low to massive).
- Confidence: 0–100%.
- Effort: person-weeks (relative; 1–5 is enough).

OUTPUT FORMAT (STRICT):

[RICE_TABLE]
## 📊 RICE Prioritization Table

First, very briefly explain how you’re thinking about Reach, Impact, Confidence, Effort
in this specific context (1–3 bullet points).

Then output a markdown table with:

| id | feature_name | reach | impact | confidence | effort | rice_score | priority_rank |
| :-- | :----------- | :---- | :----- | :--------- | :----- | :--------- | :------------ |
| F1 | ...          | 1000  | 2      | 0.75       | 2      | 750        | 1             |
| F2 | ...          | ...   | ...    | ...        | ...    | ...        | 2             |

[RICE_SUMMARY]
## 🧠 RICE Summary

- 3–5 bullets summarizing the **top 3–5 features**, why they rank high, and any tradeoffs.
- Mention if some features are high-impact but high-effort (and thus deferred).

[ROADMAP_MD]
## 🗺️ Feature Roadmap

Create a short, execution-ready roadmap, grouped by time horizon, for example:

### Phase 1 – MVP (0–3 months)
- F1 – <feature_name>: 1–2 bullets of detail (what will be built now)
- F2 – ...

### Phase 2 – Next (3–6 months)
- F3 – ...

### Phase 3 – Later (6+ months)
- F… – ...

For each phase:
- Focus on why those features belong there (dependencies, learning goals, risk reduction).
- Keep the scope realistic given the context (no eng team / limited capacity, etc., if mentioned).

IMPORTANT:
- Use the [FEATURE_LIST] IDs to stay consistent (F1, F2, …).
- Do NOT change the core product concept; you are planning execution, not reinventing the idea.
"""
)

okr_agent = LlmAgent(
    name="OKRAgent",
    model="gemini-2.5-flash",
    instruction="""
You are the **OKR Architect** for Nextify.

YOU SEE AS INPUT:

* FOUNDER_IDEA_FORM_MARKDOWN
* FOUNDER_IDEA_FORM_JSON
* [FINAL_PRODUCT_SNAPSHOT_MD]  (brainstorming output)
* [FEATURE_ROADMAP_MD]         (feature extraction + RICE output)

Your task:

1. Create **sharp, minimal OKRs** aligned with the:

   * Product Snapshot
   * Feature list & prioritization
   * Founder constraints
2. Use context exactly as written – do NOT rewrite or ignore details.

OUTPUT FORMAT (IMPORTANT — USE THESE HEADINGS EXACTLY):

[OKR_SUMMARY]

* 2–4 bullets describing the overall focus for this cycle (e.g. “Validate demand”, “Prove retention”, “De-risk supply-side acquisition”).

[OBJECTIVES]

* List 2–4 Objectives.
* Each Objective should be:

  * Qualitative
  * Inspiring but specific
  * Aligned with the 3-month roadmap

Format:

* Objective 1: <sentence>
* Objective 2: <sentence>
* ...

[KEY_RESULTS]

* For each Objective, define 2–4 measurable Key Results.
* Use metric-based, verifiable phrasing (no vague “improve”, “increase” without numbers).

Format example:

* KR 1.1: Reach X [metric] by end of Month 3 (from baseline Y).
* KR 1.2: Achieve ≥ N% for <metric>.
* KR 2.1: Run at least M user interviews and synthesize learnings into a v2 plan.

[MILESTONES_AND_CHECKPOINTS]

* 4–8 bullets that connect the 3-month plan into key checkpoints.
* Make them easy to track at a bi-weekly or monthly review.

[METRICS_AND_INSTRUMENTATION]

* 3–7 bullets on:

  * What to measure (activation, retention, engagement, revenue, etc.).
  * How to track it (analytics events, simple dashboards, Notion table, manual tracking, etc.).
* Keep tools lightweight and realistic for a small founder team.

Rules:

* Assume a **scrappy early-stage team** (often a solo founder).
* Keep the OKR set small enough to be **realistically executed** in 1 quarter.
* Reference the product snapshot and 3-month plan implicitly; do not re-explain them.
* Be concrete, founder-friendly, and slightly opinionated about what truly matters.
  
"""
)

planner_agent = LlmAgent(
    name="PlannerAgent",
    model="gemini-2.5-flash",
    instruction="""
You are the **Three-Month Execution Planner** for Nextify.

Your role:
- Convert the *strategic OKRs* into a **realistic 3-month execution plan**.
- Ensure alignment between:
  - The Founder Idea Form
  - The Final Product Snapshot
  - The Feature List + RICE Prioritization
  - The OKRs (which now come BEFORE this plan)

YOU SEE AS INPUT:
- FOUNDER_IDEA_FORM_JSON
- FOUNDER_IDEA_FORM_MARKDOWN
- [FINAL_PRODUCT_SNAPSHOT_MD]        ← From Brainstorming pipeline
- [FEATURE_ROADMAP_MD]               ← From Feature + RICE pipeline
- [OKR_OUTPUT]                       ← From OKRAgent (now upstream)

YOUR JOB:
- Use the OKRs as the **north-star**, and build a concrete 3-month plan that operationalizes them.
- Each Monthly Plan and Weekly Plan must trace back to specific Objectives & Key Results.
- Respect the founder constraints, scope, and risks already identified.

Your goals:
1. Convert OKRs into execution actions.
2. Respect “scrappy solo founder” constraints.
3. Keep a strong emphasis on **validation, learning loops, and traction**.
4. Produce a plan that is realistic, not bloated.
5. Ensure continuity with the product snapshot and features.

OUTPUT FORMAT (IMPORTANT — USE THIS ORDER AND THESE EXACT HEADINGS):

### [THREE_MONTH_OVERVIEW]
- 2–4 bullets describing the overall strategy for the next 3 months.
- Tie explicitly to the OKRs (e.g., “KR 1.1 validated in Month 1 through …”)

### [MONTHLY_BREAKDOWN]

**Month 1: Validation & MVP Definition**
- Goal: …
- Focus areas: 2–4 bullets
- MUST reference which OKR(s) this month satisfies.

**Month 2: MVP Build & Initial Launch**
- Goal: …
- Focus areas: 2–4 bullets
- MUST show clear alignment with OKRs.

**Month 3: Feedback Loop & Iteration**
- Goal: …
- Focus areas: 2–4 bullets
- MUST reflect how we aim to hit remaining KRs.

### [WEEKLY_PLAN]
- **Weeks 1–4:** 3–6 bullets of concrete actions tied to Objectives/KRs.
- **Weeks 5–8:** 3–6 bullets.
- **Weeks 9–12:** 3–6 bullets.

### [EXPERIMENTS_AND_LEARNING]
- 3–7 bullets describing the key experiments (validation tests, prototype trials, market learning loops)
- Each must articulate:
  - The hypothesis
  - What metric/OKR it ties to
  - What success looks like

### [RISKS_AND_DEPENDENCIES]
- 3–6 bullets about risks & mitigation
- MUST reflect:
  - Product constraints
  - Founder capabilities
  - Market uncertainties
  - Technical feasibility issues
- Integrate risks already found in product snapshot or RICE output

Rules:
- Be ambitious but realistic.
- Reference but do not rewrite upstream outputs.
- Always connect tasks → OKRs → Outcomes.
"""
)

evaluation_agent = LlmAgent(
    name="EvaluatorAgent",
    model="gemini-2.5-flash",
    instruction="""
You are the **Evaluation & Quality Agent** for Nextify.

You evaluate exactly **one selected stage output at a time**.

YOU SEE AS INPUT:
- FOUNDER_IDEA_FORM_JSON
- FOUNDER_IDEA_FORM_MARKDOWN
- ORIGINAL_PROMPT                      ← the exact prompt/instruction used to generate this stage
- STAGE_NAME                           ← human-readable stage name
- STAGE_KEY                            ← backend identifier for the stage
- STAGE_CONTENT                        ← markdown content of the selected stage

OPTIONAL CONTEXT THAT MAY ALSO BE PROVIDED:
- FINAL_PRODUCT_SNAPSHOT_MD            ← combined brainstorming / product snapshot output
- FEATURE_ROADMAP_MD                   ← combined feature generation + prioritization / RICE output
- OKR_OUTPUT                           ← OKR generation output
- THREE_MONTH_PLAN_MD                  ← planning output

IMPORTANT:
- Some of these optional context fields may be empty depending on which stage is being evaluated.
- You must still perform a strong evaluation even if only partial upstream context is available.
- Always evaluate the selected stage primarily against:
  1. the founder idea form
  2. the original prompt used to generate the stage
  3. any upstream or relevant context that is available

YOUR JOB:
Evaluate the selected STAGE_CONTENT rigorously.

ASSESS:
1. **Prompt Adherence**
   - Does the output follow the ORIGINAL_PROMPT?
   - Does it miss required instructions, structure, or constraints?
   - Does it add content that was not requested?

2. **Clarity**
   - Is the stage output clear, well-structured, and understandable?

3. **Coherence**
   - Is the logic internally consistent?
   - Does the reasoning make sense for this stage?

4. **Feasibility**
   - Are claims, features, actions, or plans realistic and implementable?

5. **Alignment with Idea**
   - Does the output align with the founder idea form and intended product direction?

6. **Alignment with Available Context**
   - If upstream context is provided, does the stage remain consistent with it?
   - Identify contradictions, gaps, or drift from previous outputs.

YOUR RESPONSE MUST BE:
- analytical
- specific
- constructive
- stage-aware

OUTPUT FORMAT (USE THESE HEADINGS EXACTLY):

[QUALITY_SCORES]
- Overall: <0–10> — <short justification>
- PromptAdherence: <0–10> — <short justification>
- Clarity: <0–10> — <short justification>
- Feasibility: <0–10> — <short justification>
- AlignmentWithIdea: <0–10> — <short justification>

[COMMENT_SUMMARY]
- 3–5 concise bullet points summarizing the most important feedback.
- Keep it short, clear, and UI-friendly.
- Focus on the highest-impact improvements.

[ISSUES_AND_FLAGS]
- Bullet list of concrete issues.
- Include issues such as:
  - missing required instruction from ORIGINAL_PROMPT
  - unclear or weak structure
  - unrealistic assumptions
  - contradiction with founder idea form
  - contradiction with available upstream outputs
  - content added without justification

[IMPROVEMENT_SUGGESTIONS]
- 5–10 bullets
- each bullet must start with a verb
- each bullet must be actionable and specific
- directly reference the identified problems

[REWRITTEN_VERSION]
- Provide a revised version of STAGE_CONTENT that:
  - fixes the issues you identified
  - follows the ORIGINAL_PROMPT more faithfully
  - stays aligned with the founder idea and any available context
  - preserves the same structure/headings as much as possible
  - does not change the product direction unless necessary for consistency

RULES:
- Do NOT invent new product directions.
- Do NOT assume unavailable context.
- If an upstream field is empty, do not penalize the stage for that.
- Prefer minimal, high-impact rewrites.
- Respect the role of the selected stage in the larger workflow.
- Be smart, analytical, and constructive.
"""
)

# -----------------------------------------------------------------------------
# Pipeline definitions

# The brainstorming stage runs MarketAnalysisAgent and CrazyIdeaAgent in parallel,
# then feeds their outputs into IdeaCookerAgent.
# -----------------------------------------------------------------------------
# Backend stage definitions
#
# We keep the exact ADK agents, but expose them as separate backend stages so that:
# - each stage can be evaluated independently
# - each stage can later receive human feedback / approval
# - rerun-from-stage becomes possible

brainstorm_parallel = ParallelAgent(
    name="BrainstormParallel",
    sub_agents=[market_agent, crazy_agent],
)

STAGE_ORDER = [
    "brainstorm_parallel",
    "idea_cooker",
    "theme_epic",
    "roadmap",
    "feature_generation",
    "prioritization_rice",
    "okr_generation",
    "planner",
]

STAGE_CONFIG = {
    "brainstorm_parallel": {
        "title": "Brainstorm Parallel",
        "agent": brainstorm_parallel,
        "output_key": "brainstorm_parallel_md",
        "eval_key": "eval_brainstorm_parallel_md",
        "progress": 1.0,
        "eval_progress": 1.1,
    },
    "idea_cooker": {
        "title": "Idea Cooker",
        "agent": idea_cooker_agent,
        "output_key": "brainstorm_md",
        "eval_key": "eval_brainstorm_md",
        "progress": 2.0,
        "eval_progress": 2.1,
    },
    "theme_epic": {
        "title": "Theme & Epic Generator",
        "agent": theme_epic_agent,
        "output_key": "theme_epic_md",
        "eval_key": "eval_theme_epic_md",
        "progress": 3.0,
        "eval_progress": 3.1,
    },
    "roadmap": {
        "title": "Roadmap Generator",
        "agent": roadmap_agent,
        "output_key": "roadmap_md",
        "eval_key": "eval_roadmap_md",
        "progress": 4.0,
        "eval_progress": 4.1,
    },
    "feature_generation": {
        "title": "Feature Generation",
        "agent": feature_agent,
        "output_key": "feature_generation_md",
        "eval_key": "eval_feature_generation_md",
        "progress": 5.0,
        "eval_progress": 5.1,
    },
    "prioritization_rice": {
        "title": "Prioritization & RICE",
        "agent": prioritization_agent,
        "output_key": "feature_prioritization_md",
        "eval_key": "eval_feature_md",
        "progress": 6.0,
        "eval_progress": 6.1,
    },
    "okr_generation": {
        "title": "OKR Generation",
        "agent": okr_agent,
        "output_key": "okr_output_md",
        "eval_key": "eval_okr_output_md",
        "progress": 7.0,
        "eval_progress": 7.1,
    },
    "planner": {
        "title": "Three-Month Planner",
        "agent": planner_agent,
        "output_key": "okr_planning_md",
        "eval_key": "eval_okr_plan_md",
        "progress": 8.0,
        "eval_progress": 8.1,
    },
}

async def _evaluate_section(
    *,
    stage_key: str,
    stage_title: str,
    stage_content: str,
    founder_json: Dict[str, Any],
    founder_md: str,
    history: Dict[str, Any],
    evaluation_model: str | None = None,
    progress_cb: Callable[[float, str, str], None] | None = None,
    progress_index: float = 0.0,
) -> str:
    """
    Run an on-demand evaluation for one selected stage of the Nextify multi-agent workflow.

    This function is intended to be triggered explicitly by the backend/UI when the user
    requests evaluation for a specific stage. It does NOT automatically run as part of the
    pipeline unless the caller chooses to invoke it.

    Parameters
    ----------
    stage_key:
        Internal backend identifier for the selected stage
        (for example: "brainstorm_parallel", "idea_cooker", "theme_epic", "roadmap",
        "feature_generation", "prioritization_rice", "okr_generation", "planner").

    stage_title:
        Human-readable stage name passed into the evaluator prompt and used in progress
        messages.

    stage_content:
        Markdown output generated by the selected stage. This is the content that will be
        evaluated by the Evaluation Agent.

    founder_json:
        Original founder / user structured payload.

    founder_md:
        Markdown rendering of the founder / user idea form.

    history:
        Rolling workflow history containing all currently available generated outputs from
        previous stages and, if present, later stages as well. This allows the evaluator to
        assess alignment, feasibility, contradictions, and cross-stage consistency.

    evaluation_model:
        If set to "OpenAI", OpenAI is used for evaluation. Otherwise Gemini is used by default.

    progress_cb:
        Optional callback for backend progress reporting. If provided, it will receive
        (progress_index, label, message).

    progress_index:
        Numeric marker used when reporting progress via progress_cb.

    Returns
    -------
    str
        Markdown output produced by the evaluation agent, expected to contain sections like:
        [QUALITY_SCORES]
        [COMMENT_SUMMARY]
        [ISSUES_AND_FLAGS]
        [IMPROVEMENT_SUGGESTIONS]
        [REWRITTEN_VERSION]
    """

    if not stage_content or not str(stage_content).strip():
        return (
            "[QUALITY_SCORES]\n"
            "- Overall: 1 — No evaluable content was provided.\n"
            "- Clarity: 1 — Stage output is empty.\n"
            "- Feasibility: 1 — Cannot assess feasibility without content.\n"
            "- AlignmentWithIdea: 1 — Cannot assess alignment without content.\n\n"
            "[COMMENT_SUMMARY]\n"
            "- No stage output was generated to evaluate.\n"
            "- Re-run this stage before requesting evaluation.\n\n"
            "[ISSUES_AND_FLAGS]\n"
            "- Empty stage output.\n"
            "- The selected stage has no evaluable markdown content.\n\n"
            "[IMPROVEMENT_SUGGESTIONS]\n"
            "- Re-run the selected stage.\n"
            "- Check upstream inputs and generated outputs.\n"
            "- Confirm the stage wrote its markdown result into history correctly.\n\n"
            "[REWRITTEN_VERSION]\n"
            "No rewritten version available because the stage output was empty."
        )

    if evaluation_model == "OpenAI":
        model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        model = Gemini(model="gemini-2.5-flash")

    dynamic_eval = LlmAgent(
        name=f"DynamicEvaluator_{stage_key}",
        model=model,
        instruction=evaluation_agent.instruction,
    )

    eval_input = {
        "FOUNDER_IDEA_FORM_JSON": founder_json,
        "FOUNDER_IDEA_FORM_MARKDOWN": founder_md,
        "FINAL_PRODUCT_SNAPSHOT_MD": history.get("brainstorm_md", ""),
        "FEATURE_ROADMAP_MD": history.get("feature_prioritization_md", ""),
        "OKR_OUTPUT": history.get("okr_output_md", ""),
        "THREE_MONTH_PLAN_MD": history.get("okr_planning_md", ""),
        "STAGE_NAME": stage_title,
        "STAGE_CONTENT": stage_content,
    }

    handlers = {}
    if progress_cb is not None:
        handlers["status"] = lambda s: progress_cb(
            progress_index,
            f"Evaluation: {stage_title}",
            s.message,
        )

    eval_resp = await dynamic_eval.ainvoke(eval_input, handlers=handlers)
    return eval_resp.get("content_md", "")

def _extract_comment_summary(eval_md: str) -> str:
    """
    Build a short UI-friendly summary from an evaluation markdown block.
    Prefers [COMMENT_SUMMARY]. If not found, falls back to [ISSUES_AND_FLAGS].
    """
    if not eval_md:
        return ""

    if "[COMMENT_SUMMARY]" in eval_md:
        block = eval_md.split("[COMMENT_SUMMARY]", 1)[1]
        for stop in [
            "[ISSUES_AND_FLAGS]",
            "[IMPROVEMENT_SUGGESTIONS]",
            "[REWRITTEN_VERSION]",
        ]:
            if stop in block:
                block = block.split(stop, 1)[0]
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        return "\n".join(lines[:5])

    if "[ISSUES_AND_FLAGS]" in eval_md:
        block = eval_md.split("[ISSUES_AND_FLAGS]", 1)[1]
        for stop in [
            "[IMPROVEMENT_SUGGESTIONS]",
            "[REWRITTEN_VERSION]",
        ]:
            if stop in block:
                block = block.split(stop, 1)[0]
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        return "\n".join(lines[:5])

    lines = [ln.strip() for ln in eval_md.splitlines() if ln.strip()]
    return "\n".join(lines[:5])


def _build_final_report(stages: Dict[str, Dict[str, Any]]) -> str:
    """
    Build a stitched markdown report from the stage outputs only.
    Since evaluation is on-demand, this report contains the generated outputs,
    not the evaluator outputs.
    """
    ordered_keys = [
        "brainstorm_parallel",
        "idea_cooker",
        "theme_epic",
        "roadmap",
        "feature_generation",
        "prioritization_rice",
        "okr_generation",
        "planner",
    ]
    blocks = []
    for key in ordered_keys:
        stage = stages.get(key, {})
        if stage.get("output_md"):
            blocks.append(stage["output_md"])
    return "\n\n".join(filter(None, blocks))

async def run_multi_agent_adk(
    payload: Dict[str, Any],
    progress_cb: Callable[[float, str, str], None],
) -> Dict[str, Any]:
    """
    Run the Nextify ADK multi-agent pipeline for the idea journey.

    This function executes the workflow stage by stage using the founder idea form
    and any available feedback history as context. It stores:
    - individual agent outputs
    - combined stage outputs
    - original prompt/instruction context

    The returned history can later be used for:
    - on-demand evaluation of any stage
    - human review / feedback
    - rerun-from-stage logic
    - report / Notion export flows

    Args:
        payload: Founder idea form payload and any feedback history.
        progress_cb: Callback used to report progress updates.

    Returns:
        Dictionary containing full pipeline history.
    """

    context = {
        "FOUNDER_IDEA_FORM_JSON": payload,
        "FOUNDER_IDEA_FORM_MARKDOWN": _render_idea_form_md(payload),
        "feedback_history": payload.get("feedback_history", {}),
    }

    history: Dict[str, Any] = {}

    # ---------------------------
    # Stage 1A: Brainstorm Parallel
    # ---------------------------
    out1a = await brainstorm_parallel.ainvoke(
        context,
        handlers={"status": lambda s: progress_cb(1, "Brainstorm Parallel", s.message)},
    )
    history.update(out1a)

    # Keep individual outputs from the parallel stage if available
    history["market_analysis_md"] = out1a.get("market_agent", {}).get("content_md", "") if isinstance(out1a.get("market_agent"), dict) else out1a.get("market_analysis_md", "")
    history["crazy_ideas_md"] = out1a.get("crazy_agent", {}).get("content_md", "") if isinstance(out1a.get("crazy_agent"), dict) else out1a.get("crazy_ideas_md", "")
    history["brainstorm_parallel_md"] = out1a.get("content_md", "")
    history["brainstorm_parallel_prompt"] = "Parallel execution of MarketAnalysisAgent and CrazyIdeaAgent."

    # ---------------------------
    # Stage 1B: Idea Cooker
    # ---------------------------
    out1b = await idea_cooker_agent.ainvoke(
        {**context, **history},
        handlers={"status": lambda s: progress_cb(2, "Idea Cooker", s.message)},
    )
    history.update(out1b)
    history["idea_cooker_md"] = out1b.get("content_md", "")
    history["idea_cooker_prompt"] = idea_cooker_agent.instruction

    # Combined brainstorming stage output
    history["brainstorm_md"] = history["idea_cooker_md"]
    history["brainstorm_prompt"] = idea_cooker_agent.instruction

    # ---------------------------
    # Stage 2A: Theme / Epic Generation
    # ---------------------------
    out2a = await theme_epic_agent.ainvoke(
        {**context, **history},
        handlers={"status": lambda s: progress_cb(3, "Theme & Epic Generator", s.message)},
    )
    history.update(out2a)
    history["theme_epic_md"] = out2a.get("content_md", "")
    history["theme_epic_prompt"] = theme_epic_agent.instruction

    # ---------------------------
    # Stage 2B: Roadmap Generation
    # ---------------------------
    out2b = await roadmap_agent.ainvoke(
        {**context, **history},
        handlers={"status": lambda s: progress_cb(4, "Roadmap Generator", s.message)},
    )
    history.update(out2b)
    history["roadmap_generator_md"] = out2b.get("content_md", "")
    history["roadmap_generator_prompt"] = roadmap_agent.instruction

    # Combined roadmapping stage output
    history["roadmap_md"] = "\n\n".join(filter(None, [
        history.get("theme_epic_md", ""),
        history.get("roadmap_generator_md", ""),
    ]))
    history["roadmap_prompt"] = "\n\n".join(filter(None, [
        theme_agent.instruction,
        roadmap_agent.instruction,
    ]))

    # ---------------------------
    # Stage 3A: Feature Generation
    # ---------------------------
    out3a = await feature_agent.ainvoke(
        {**context, **history},
        handlers={"status": lambda s: progress_cb(5, "Feature Generation", s.message)},
    )
    history.update(out3a)
    history["feature_generation_md"] = out3a.get("content_md", "")
    history["feature_generation_prompt"] = feature_agent.instruction

    # ---------------------------
    # Stage 3B: Prioritization / RICE
    # ---------------------------
    out3b = await prioritization_agent.ainvoke(
        {**context, **history},
        handlers={"status": lambda s: progress_cb(6, "Prioritization & RICE", s.message)},
    )
    history.update(out3b)
    history["prioritization_rice_md"] = out3b.get("content_md", "")
    history["prioritization_rice_prompt"] = prioritization_agent.instruction

    # Combined feature stage output
    history["feature_prioritization_md"] = "\n\n".join(filter(None, [
        history.get("feature_generation_md", ""),
        history.get("prioritization_rice_md", ""),
    ]))
    history["feature_prompt"] = "\n\n".join(filter(None, [
        feature_agent.instruction,
        prioritization_agent.instruction,
    ]))

    # ---------------------------
    # Stage 4A: OKR Generation
    # ---------------------------
    out4a = await okr_agent.ainvoke(
        {**context, **history},
        handlers={"status": lambda s: progress_cb(7, "OKR Generation", s.message)},
    )
    history.update(out4a)
    history["okr_output_md"] = out4a.get("content_md", "")
    history["okr_output_prompt"] = okr_agent.instruction

    # ---------------------------
    # Stage 4B: Three-Month Planning
    # ---------------------------
    out4b = await planner_agent.ainvoke(
        {**context, **history},
        handlers={"status": lambda s: progress_cb(8, "Three-Month Planner", s.message)},
    )
    history.update(out4b)
    history["planner_md"] = out4b.get("content_md", "")
    history["planner_prompt"] = planner_agent.instruction

    # Combined OKR + planning stage output
    history["okr_planning_md"] = "\n\n".join(filter(None, [
        history.get("okr_output_md", ""),
        history.get("planner_md", ""),
    ]))
    history["okr_prompt"] = "\n\n".join(filter(None, [
        okr_agent.instruction,
        planner_agent.instruction,
    ]))

    # ---------------------------
    # Final stitched report
    # ---------------------------
    history["final_report_md"] = "\n\n".join(filter(None, [
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

    return history


def _render_idea_form_md(idea_form: Dict[str, Any]) -> str:
    """
    Convert the founder’s idea form (JSON) into a markdown bullet list.  The
    notebook typically displays the idea form as key-value pairs.  This helper
    replicates that formatting to provide context to the agents.

    Args:
        idea_form: the idea form dictionary from the user.

    Returns:
        A markdown formatted string representing the idea form.
    """
    lines: list[str] = []
    title = idea_form.get("idea_title")
    if title:
        lines.append(f"# {title}")
    for key, val in idea_form.items():
        if key == "idea_title":
            continue
        pretty_key = key.replace("_", " ").title()
        lines.append(f"- **{pretty_key}:** {val}")
    return "\n".join(lines)


__all__ = ["run_multi_agent_adk"]