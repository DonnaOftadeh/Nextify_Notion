"""
Nextify Interactive Backend

This backend runs the Nextify workflow stage by stage.

Behavior:
- User submits an idea.
- Backend creates an interactive job.
- Each stage can be run manually.
- Agent output appears in the main UI.
- LLM judge feedback is stored separately.
- Human feedback is stored separately.
- Revision uses human feedback, judge feedback, or both.
- Previous versions are stored in sidebar history.
- Approving a stage automatically generates the next stage.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="Nextify Interactive Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STAGES
# ============================================================

STAGES: List[Dict[str, str]] = [
    {
        "id": "parse_submission",
        "title": "Parse Submission",
        "short_title": "Parse",
        "agent": "Input Parser Agent",
        "desc": "Validate and structure the submitted idea.",
    },
    {
        "id": "brainstorm_parallel",
        "title": "Brainstorm Parallel",
        "short_title": "Brainstorm",
        "agent": "Brainstorming Agent: Market Analysis + Crazy Idea Generator",
        "desc": "Generate market-grounded opportunities and unconventional breakthrough ideas.",
    },
    {
        "id": "idea_cooker",
        "title": "Idea Cooker",
        "short_title": "Cooker",
        "agent": "Idea Cooker Agent",
        "desc": "Evaluate every idea, explain the reason behind each score, and help the user choose one.",
    },
    {
        "id": "theme_epic_generator",
        "title": "Theme & Epic Generator",
        "short_title": "Themes",
        "agent": "Theme / Epic Generator",
        "desc": "Turn the chosen idea into strategic themes and epics.",
    },
    {
        "id": "roadmap_generator",
        "title": "Roadmap Generator",
        "short_title": "Roadmap",
        "agent": "Roadmap Generator",
        "desc": "Build a phased product roadmap.",
    },
    {
        "id": "feature_generation",
        "title": "Feature Generation",
        "short_title": "Features",
        "agent": "Feature Generation Agent",
        "desc": "Generate concrete product features.",
    },
    {
        "id": "prioritization_rice",
        "title": "Prioritization & RICE",
        "short_title": "RICE",
        "agent": "Prioritization Agent",
        "desc": "Prioritize features using RICE.",
    },
    {
        "id": "okr_generation",
        "title": "OKR Generation",
        "short_title": "OKRs",
        "agent": "OKR Generator",
        "desc": "Generate objectives and key results.",
    },
    {
        "id": "three_month_planner",
        "title": "Three-Month Planner",
        "short_title": "Planner",
        "agent": "Planner Agent",
        "desc": "Create a practical three-month plan.",
    },
    {
        "id": "write_report_pdf",
        "title": "Write Report (PDF)",
        "short_title": "Report",
        "agent": "Report Writer Agent",
        "desc": "Assemble the approved outputs into a final product plan.",
    },
]

STAGE_IDS = [stage["id"] for stage in STAGES]


# ============================================================
# MODELS
# ============================================================

class Submission(BaseModel):
    journey_type: str = Field(default="idea")
    payload: Dict[str, Any]


class FeedbackRequest(BaseModel):
    feedback: str = ""


class ReviseRequest(BaseModel):
    mode: str = "both"
    feedback: str = ""


# ============================================================
# IN-MEMORY STORAGE
# ============================================================

JOBS: Dict[str, Dict[str, Any]] = {}


# ============================================================
# HELPERS
# ============================================================

def _now_ts() -> float:
    return time.time()


def _get_stage(stage_id: str) -> Dict[str, str]:
    for stage in STAGES:
        if stage["id"] == stage_id:
            return stage
    raise HTTPException(status_code=404, detail=f"Unknown stage: {stage_id}")


def _get_stage_index(stage_id: str) -> int:
    for index, stage in enumerate(STAGES):
        if stage["id"] == stage_id:
            return index
    raise HTTPException(status_code=404, detail=f"Unknown stage: {stage_id}")


def _get_job(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _empty_stage_state(stage_id: str) -> Dict[str, Any]:
    return {
        "stage_id": stage_id,
        "status": "pending",
        "agent_output": "",
        "previous_outputs": [],
        "judge_feedback": "",
        "judge_history": [],
        "user_feedback": "",
        "user_feedback_history": [],
        "applied_feedback_summary": "",
        "revision_count": 0,
        "approved": False,
        "updated_at": None,
    }


def _init_interactive_state() -> Dict[str, Any]:
    return {
        "current_stage_index": 0,
        "stages": {
            stage["id"]: _empty_stage_state(stage["id"])
            for stage in STAGES
        },
    }


def _approved_context(job: Dict[str, Any]) -> str:
    chunks: List[str] = []

    for stage in STAGES:
        state = job["interactive"]["stages"][stage["id"]]
        if state.get("approved") and state.get("agent_output"):
            chunks.append(
                f"## {stage['title']}\n\n"
                f"{state['agent_output']}"
            )

    if not chunks:
        return "No previous stage has been approved yet."

    return "\n\n---\n\n".join(chunks)


def _payload_text(job: Dict[str, Any]) -> str:
    payload = job.get("payload", {})
    lines = []

    for key, value in payload.items():
        pretty = key.replace("_", " ").title()
        lines.append(f"- **{pretty}:** {value}")

    return "\n".join(lines) if lines else "- No payload provided."


def _shorten(text: str, max_chars: int = 900) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n..."


def _payload_value(job: Dict[str, Any], key: str, fallback: str = "") -> str:
    return str(job.get("payload", {}).get(key, fallback) or fallback)


# ============================================================
# AGENT OUTPUT GENERATION
# ============================================================

def _generate_parse_submission(job: Dict[str, Any]) -> str:
    payload = job.get("payload", {})

    idea_title = payload.get("idea_title", "Untitled idea")
    idea_text = payload.get("idea_text", "")
    target_users = payload.get("target_users", "")
    problem = payload.get("problem", "")
    constraints = payload.get("constraints", "")

    return f"""
# Parsed Product Input

## Clean Product Concept
**{idea_title}**

{idea_text}

## Target User Segment
{target_users or "Not specified yet."}

## Problem Statement
{problem or "Not specified yet."}

## Constraints
{constraints or "No constraints provided."}

## Parser Agent Review
The input is understandable and ready for brainstorming.

## Missing Information To Improve Later
- Clearer MVP boundary
- Scientific or technical feasibility assumptions
- First validation experiment
- Success metric for early proof of value

## Recommended Next Step
Proceed to brainstorming, but keep feasibility and solo-founder constraints visible in every later stage.
""".strip()


def _generate_brainstorm(job: Dict[str, Any]) -> str:
    return f"""
# Brainstorming Agent Output

## Source Input
{_payload_text(job)}

## Market-Grounded Ideas

### Idea 1 — Personal Microplastic Exposure Tracker
A consumer-facing product that estimates exposure risk using lifestyle, food, water, and location data.

**Why it matters:** This is more feasible than directly measuring microplastics inside the body at MVP stage.

### Idea 2 — Clinical Risk Screening Dashboard
A tool for researchers or clinics to record suspected exposure sources and symptoms.

**Why it matters:** It creates structured data before expensive biosensor hardware exists.

### Idea 3 — At-Home Sample Collection Kit
A kit that collects biological or environmental samples and sends them to certified labs.

**Why it matters:** It bridges the gap between consumer demand and current lab-based testing capability.

## Crazy Breakthrough Ideas

### Crazy Idea A — Wearable Microplastic Biosensor Patch
A skin patch that detects microplastic-related biomarkers through sweat or interstitial fluid.

### Crazy Idea B — Smart Toilet Health Scanner
A bathroom device that screens waste samples for microplastic particles and related biomarkers.

### Crazy Idea C — Personalized Detox Recommendation Engine
An AI assistant that gives reduction strategies based on measured or estimated exposure.

## Recommendation
Start with a software + lab-partner MVP before trying to build a full biosensor device.
""".strip()


def _generate_idea_cooker(job: Dict[str, Any]) -> str:
    context = _approved_context(job)

    return f"""
# Idea Cooker Agent Output

The Idea Cooker evaluates each brainstormed idea with reasoning, score, and recommendation.

## Approved Context Used
{_shorten(context, 1200)}

## Evaluation Table

| Idea | Feasibility | Impact | Differentiation | Solo-Founder Fit | Score / 10 | Reason |
|---|---:|---:|---:|---:|---:|---|
| Personal Microplastic Exposure Tracker | 8 | 7 | 6 | 9 | 7.5 | Fastest MVP, mostly software, but not direct measurement. |
| Clinical Risk Screening Dashboard | 7 | 7 | 6 | 7 | 6.8 | Useful for research workflows, but harder to access clinics. |
| At-Home Sample Collection Kit | 6 | 8 | 7 | 5 | 6.5 | Strong value, but requires lab partnerships and logistics. |
| Wearable Biosensor Patch | 2 | 10 | 10 | 2 | 5.5 | Inspiring long-term invention, but very hard for a solo founder now. |
| Smart Toilet Scanner | 2 | 9 | 9 | 1 | 4.8 | Technically and commercially complex. |
| Detox Recommendation Engine | 8 | 6 | 5 | 9 | 7.0 | Easy MVP, but needs credibility and strong science backing. |

## Best Direction
Build a **Microplastic Exposure Intelligence Platform** first.

## Why This Wins
It combines:
1. exposure estimation,
2. education,
3. personalized reduction actions,
4. future integration with lab testing or biosensor data.

## Suggested MVP
A web app where users answer lifestyle questions, receive exposure risk insights, and get a personalized reduction plan.

## User Decision Needed
Please choose one direction:

1. Exposure Tracker
2. Lab-Partner Testing Kit
3. Biosensor Patch
4. Detox Recommendation Engine
5. A hybrid idea

You can approve one, or give feedback and ask the Idea Cooker to revise the recommendation.
""".strip()


def _generate_theme_epic(job: Dict[str, Any]) -> str:
    context = _approved_context(job)

    return f"""
# Theme & Epic Generator Output

## Approved Context Used
{_shorten(context, 1200)}

## Product Theme 1 — Exposure Understanding
Help users understand where their microplastic exposure may come from.

### Epics
- Lifestyle exposure questionnaire
- Food and water source risk mapping
- Personal exposure profile

## Product Theme 2 — Risk Reduction
Help users reduce exposure through practical actions.

### Epics
- Personalized reduction checklist
- Alternative product suggestions
- Weekly progress tracking

## Product Theme 3 — Scientific Credibility
Make the product trustworthy.

### Epics
- Evidence library
- Source citations
- Expert review workflow

## Product Theme 4 — Future Measurement Layer
Prepare for lab or biosensor integrations.

### Epics
- Lab result upload
- Biomarker data schema
- Future biosensor API design
""".strip()


def _generate_roadmap(job: Dict[str, Any]) -> str:
    context = _approved_context(job)

    return f"""
# Roadmap Generator Output

## Approved Context Used
{_shorten(context, 1000)}

## Phase 1 — MVP Foundation, Weeks 1–4
- User onboarding
- Exposure questionnaire
- Rule-based exposure scoring
- Personalized recommendations

## Phase 2 — Validation, Weeks 5–8
- User interviews
- Improve scoring logic
- Add evidence sources
- Build habit tracking

## Phase 3 — Pilot, Weeks 9–12
- Launch private beta
- Collect feedback
- Add dashboards
- Explore lab partnerships

## Phase 4 — Expansion
- Lab data import
- Research collaborations
- Biosensor feasibility study
""".strip()


def _generate_features(job: Dict[str, Any]) -> str:
    return """
# Feature Generation Output

## Core MVP Features

1. User profile and onboarding
2. Microplastic exposure questionnaire
3. Exposure risk score
4. Source-by-source exposure breakdown
5. Personalized reduction plan
6. Weekly action tracker
7. Evidence-backed education cards
8. Exportable personal report

## Advanced Features

1. Lab result upload
2. Research dashboard
3. Wearable/biosensor integration placeholder
4. AI coaching assistant
5. Community benchmarking
""".strip()


def _generate_rice(job: Dict[str, Any]) -> str:
    return """
# Prioritization & RICE Output

| Feature | Reach | Impact | Confidence | Effort | RICE |
|---|---:|---:|---:|---:|---:|
| Exposure questionnaire | 9 | 8 | 8 | 3 | 192 |
| Risk score | 9 | 9 | 7 | 4 | 141.75 |
| Personalized reduction plan | 8 | 9 | 7 | 4 | 126 |
| Evidence cards | 7 | 6 | 8 | 3 | 112 |
| Weekly action tracker | 6 | 6 | 7 | 3 | 84 |
| Lab result upload | 3 | 8 | 4 | 7 | 13.7 |

## Recommendation
Build questionnaire, risk score, and reduction plan first.
""".strip()


def _generate_okrs(job: Dict[str, Any]) -> str:
    return """
# OKR Generation Output

## Objective 1 — Validate user demand
- KR1: Interview 25 health-conscious users
- KR2: Get 100 waitlist signups
- KR3: Achieve 40 percent completion rate on exposure questionnaire

## Objective 2 — Build credible MVP
- KR1: Launch risk scoring prototype
- KR2: Add at least 30 evidence-backed recommendations
- KR3: Get review from one scientific advisor or researcher

## Objective 3 — Prepare for pilot
- KR1: Recruit 20 beta users
- KR2: Collect feedback from 80 percent of beta users
- KR3: Identify one lab or research partnership path
""".strip()


def _generate_planner(job: Dict[str, Any]) -> str:
    return """
# Three-Month Planner Output

## Month 1 — Build
- Week 1: Define scoring model
- Week 2: Build onboarding and questionnaire
- Week 3: Build risk score output
- Week 4: Add recommendation engine

## Month 2 — Validate
- Week 5: Interview users
- Week 6: Improve recommendations
- Week 7: Add evidence library
- Week 8: Start private testing

## Month 3 — Pilot
- Week 9: Launch beta
- Week 10: Analyze behavior
- Week 11: Prepare pitch/demo
- Week 12: Decide whether to pursue lab partnership or software-only launch
""".strip()


def _generate_report(job: Dict[str, Any]) -> str:
    context = _approved_context(job)

    return f"""
# Final Nextify Product Plan

## Approved Workflow Summary

{context}

## Final Recommendation
Start with a software-first Microplastic Exposure Intelligence Platform.

## Why
A true biosensor is a long-term invention requiring research, validation, hardware development, and regulatory thinking. A software-first MVP lets a solo founder validate demand, build trust, and collect structured data before moving toward lab testing or biosensor partnerships.

## Next Action
Build the MVP questionnaire, risk score, and personalized reduction plan.
""".strip()


def _generate_stage_output(stage_id: str, job: Dict[str, Any]) -> str:
    generators = {
        "parse_submission": _generate_parse_submission,
        "brainstorm_parallel": _generate_brainstorm,
        "idea_cooker": _generate_idea_cooker,
        "theme_epic_generator": _generate_theme_epic,
        "roadmap_generator": _generate_roadmap,
        "feature_generation": _generate_features,
        "prioritization_rice": _generate_rice,
        "okr_generation": _generate_okrs,
        "three_month_planner": _generate_planner,
        "write_report_pdf": _generate_report,
    }

    generator = generators.get(stage_id)
    if not generator:
        raise HTTPException(status_code=404, detail=f"No generator for stage: {stage_id}")

    return generator(job)


def _judge_stage_output(stage_id: str, agent_output: str, job: Dict[str, Any]) -> str:
    stage = _get_stage(stage_id)

    return f"""
# Judge Review — {stage["title"]}

## Score
7.5 / 10

## What Works
- The output is clear and stage-specific.
- It gives the user something concrete to react to.
- It follows the product workflow.

## What Should Improve
- Add more measurable success criteria.
- Make assumptions and risks more explicit.
- Make the next decision easier for the user.

## Recommended Revision
Revise this stage to be more specific, more practical, and more decision-ready.

## Suggested Feedback To Apply
“Make this output more practical for a solo founder. Add assumptions, risks, measurable success criteria, and a clearer next action.”
""".strip()


def _make_applied_feedback_summary(mode: str, human_feedback: str, judge_feedback: str) -> str:
    used = []

    if mode in ("human_only", "both") and human_feedback.strip():
        used.append("human feedback")

    if mode in ("judge_only", "both") and judge_feedback.strip():
        used.append("LLM judge feedback")

    if not used:
        return "No feedback was available, so the agent refreshed and clarified the current output."

    return f"Applied {' and '.join(used)} to create a revised version."


def _revise_stage_output(stage_id: str, job: Dict[str, Any], mode: str) -> str:
    stage = _get_stage(stage_id)
    state = job["interactive"]["stages"][stage_id]

    human_feedback = state.get("user_feedback", "")
    judge_feedback = state.get("judge_feedback", "")

    use_human = mode in ("human_only", "both") and bool(human_feedback.strip())
    use_judge = mode in ("judge_only", "both") and bool(judge_feedback.strip())

    feedback_instructions = []

    if use_human:
        feedback_instructions.append(f"Human feedback applied:\n{human_feedback}")

    if use_judge:
        feedback_instructions.append(f"LLM judge feedback applied:\n{judge_feedback}")

    if not feedback_instructions:
        feedback_instructions.append(
            "No explicit feedback was available. Improve clarity, specificity, risks, metrics, and next action."
        )

    feedback_text = "\n\n".join(feedback_instructions)

    if stage_id == "parse_submission":
        idea_title = _payload_value(job, "idea_title", "Untitled idea")
        idea_text = _payload_value(job, "idea_text", "No description provided.")
        target_users = _payload_value(job, "target_users", "Not specified.")
        problem = _payload_value(job, "problem", "Not specified.")
        constraints = _payload_value(job, "constraints", "No constraints provided.")

        return f"""
# Reviewed Parsed Idea

## Summary of Changes
The Input Parser Agent revised the parsed form using the selected feedback.

## Feedback Applied
{feedback_text}

## Revised Product Brief

### Idea Title
{idea_title}

### Clearer Idea Description
{idea_text}

### Target Users
{target_users}

### Problem
{problem}

### Constraints
{constraints}

## Improved Interpretation
This idea is being interpreted as an early-stage product concept that needs a realistic MVP path, clear assumptions, and staged validation before moving into heavy technical development.

## MVP Boundary
The first MVP should not try to solve the full invention immediately. It should test the smallest useful version that proves users care about the problem and want guidance, estimation, or measurement.

## Key Assumptions
1. Users are worried enough about this problem to answer questions or track exposure.
2. A software-first version can create value before hardware or biosensor validation.
3. Users will trust recommendations if they are clearly evidence-backed.
4. Future lab or sensor integrations can be added after demand is validated.

## Key Risks
1. Scientific claims may be too strong without research validation.
2. Hardware or biosensor development may be expensive and slow.
3. Users may want direct measurement, not only estimation.
4. Medical or health-related wording may require careful compliance.

## Next Action
Approve this revised parsed idea if it correctly represents the user intent. Then continue to Brainstorm Parallel.
""".strip()

    if stage_id == "idea_cooker":
        return f"""
# Revised Idea Cooker Agent Output

## Summary of Changes
The Idea Cooker revised the recommendation using the selected feedback.

## Feedback Applied
{feedback_text}

## Revised Evaluation

| Idea | Score / 10 | Reason | Recommendation |
|---|---:|---|---|
| Exposure Tracker | 8.2 | Best solo-founder MVP because it is software-first, testable, and does not require lab hardware. | Strongly consider |
| Lab-Partner Testing Kit | 6.8 | Valuable but requires partnerships, logistics, and trust. | Later phase |
| Biosensor Patch | 5.7 | High invention value but scientifically and financially difficult right now. | Long-term vision |
| Detox Recommendation Engine | 7.4 | Easy to build and useful, but needs credible science and differentiation. | Combine with tracker |
| Hybrid Exposure Intelligence Platform | 8.7 | Combines tracking, education, and reduction guidance while preparing for future lab or biosensor data. | Best choice |

## Revised Recommendation
Choose the **Hybrid Exposure Intelligence Platform**.

## Why
It lets the user start now with a realistic MVP, while still keeping the long-term biosensor invention journey alive.

## User Decision
Approve this direction or add another idea/constraint and rerun the Idea Cooker.
""".strip()

    return f"""
# Revised {stage["title"]}

## Summary of Changes
The {stage["agent"]} revised the output using the selected feedback.

## Feedback Applied
{feedback_text}

## Revised Agent Output
This revised version is more specific, more practical, and more decision-ready for the user.

## Updated Recommendations
1. Focus on the smallest useful MVP.
2. Add measurable success criteria.
3. Identify the riskiest assumption.
4. Give the user a clear next decision.
5. Keep solo-founder constraints visible.

## Stage-Specific Result
For **{stage["title"]}**, the output now includes clearer assumptions, risks, success metrics, and next actions.

## Next Decision
Review this revised version. You can approve it, give more feedback, run the judge again, or revise again.
""".strip()


# ============================================================
# API ROUTES
# ============================================================

@app.get("/")
async def root() -> Dict[str, str]:
    return {
        "service": "Nextify Interactive Backend",
        "status": "ok",
    }


@app.get("/api/stages")
async def get_stages() -> Dict[str, Any]:
    return {"stages": STAGES}


@app.post("/api/submit")
async def submit(submission: Submission) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "job_id": job_id,
        "created_at": _now_ts(),
        "status": "interactive",
        "step": "Parse Submission",
        "progress": 0,
        "message": "Interactive job created.",
        "journey_type": submission.journey_type,
        "payload": submission.payload,
        "interactive": _init_interactive_state(),
    }

    return {
        "job_id": job_id,
        "stages": STAGE_IDS,
    }


@app.get("/api/job/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    job = _get_job(job_id)

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
        "stage_config": STAGES,
    }


@app.get("/api/status/{job_id}")
async def status(job_id: str) -> Dict[str, Any]:
    job = _get_job(job_id)
    state = job["interactive"]
    current_index = state["current_stage_index"]
    approved = sum(1 for s in state["stages"].values() if s.get("approved"))

    return {
        "job_id": job_id,
        "status": job["status"],
        "step": STAGES[current_index]["title"],
        "progress": int((approved / len(STAGES)) * 100),
        "message": job.get("message", ""),
        "ready": approved == len(STAGES),
    }


@app.post("/api/stage/{job_id}/{stage_id}/run")
async def run_interactive_stage(job_id: str, stage_id: str) -> Dict[str, Any]:
    job = _get_job(job_id)
    _get_stage(stage_id)

    state = job["interactive"]["stages"][stage_id]

    if state.get("agent_output"):
        state["previous_outputs"].append(
            {
                "version": len(state["previous_outputs"]) + 1,
                "output": state["agent_output"],
                "reason": "Re-run from scratch",
                "created_at": _now_ts(),
            }
        )

    output = _generate_stage_output(stage_id, job)

    state.update(
        {
            "status": "agent_output_ready",
            "agent_output": output,
            "applied_feedback_summary": "Agent generated a fresh output.",
            "updated_at": _now_ts(),
            "approved": False,
        }
    )

    stage_index = _get_stage_index(stage_id)
    job["interactive"]["current_stage_index"] = stage_index
    job["step"] = _get_stage(stage_id)["title"]
    job["progress"] = int((stage_index / len(STAGES)) * 100)
    job["message"] = f"{_get_stage(stage_id)['agent']} finished generating output."

    return state


@app.post("/api/stage/{job_id}/{stage_id}/judge")
async def judge_interactive_stage(job_id: str, stage_id: str) -> Dict[str, Any]:
    job = _get_job(job_id)
    _get_stage(stage_id)

    state = job["interactive"]["stages"][stage_id]

    if not state.get("agent_output"):
        raise HTTPException(status_code=400, detail="Run the agent before judging.")

    if state.get("judge_feedback"):
        state["judge_history"].append(
            {
                "version": len(state["judge_history"]) + 1,
                "feedback": state["judge_feedback"],
                "created_at": _now_ts(),
            }
        )

    feedback = _judge_stage_output(stage_id, state["agent_output"], job)

    state.update(
        {
            "status": "judge_feedback_ready",
            "judge_feedback": feedback,
            "updated_at": _now_ts(),
        }
    )

    job["message"] = f"LLM judge reviewed {_get_stage(stage_id)['title']}."

    return state


@app.post("/api/stage/{job_id}/{stage_id}/feedback")
async def save_interactive_feedback(
    job_id: str,
    stage_id: str,
    req: FeedbackRequest,
) -> Dict[str, Any]:
    job = _get_job(job_id)
    _get_stage(stage_id)

    state = job["interactive"]["stages"][stage_id]
    feedback = req.feedback.strip()

    if feedback:
        state["user_feedback_history"].append(
            {
                "version": len(state["user_feedback_history"]) + 1,
                "feedback": feedback,
                "created_at": _now_ts(),
            }
        )

    state.update(
        {
            "status": "human_feedback_saved",
            "user_feedback": feedback,
            "updated_at": _now_ts(),
        }
    )

    job["message"] = f"Human feedback saved for {_get_stage(stage_id)['title']}."

    return state


@app.post("/api/stage/{job_id}/{stage_id}/revise")
async def revise_interactive_stage(
    job_id: str,
    stage_id: str,
    req: ReviseRequest,
) -> Dict[str, Any]:
    job = _get_job(job_id)
    _get_stage(stage_id)

    state = job["interactive"]["stages"][stage_id]

    if not state.get("agent_output"):
        raise HTTPException(status_code=400, detail="Run the agent before revising.")

    mode = req.mode or "both"

    if mode not in ("human_only", "judge_only", "both"):
        raise HTTPException(status_code=400, detail="Mode must be human_only, judge_only, or both.")

    incoming_feedback = req.feedback.strip()

    if incoming_feedback:
        state["user_feedback"] = incoming_feedback
        state["user_feedback_history"].append(
            {
                "version": len(state["user_feedback_history"]) + 1,
                "feedback": incoming_feedback,
                "created_at": _now_ts(),
            }
        )

    previous_output = state["agent_output"]

    state["previous_outputs"].append(
        {
            "version": len(state["previous_outputs"]) + 1,
            "output": previous_output,
            "reason": f"Before revision using mode: {mode}",
            "created_at": _now_ts(),
        }
    )

    revised_output = _revise_stage_output(stage_id, job, mode)
    summary = _make_applied_feedback_summary(
        mode=mode,
        human_feedback=state.get("user_feedback", ""),
        judge_feedback=state.get("judge_feedback", ""),
    )

    state.update(
        {
            "status": "revised_output_ready",
            "agent_output": revised_output,
            "applied_feedback_summary": summary,
            "revision_count": state.get("revision_count", 0) + 1,
            "approved": False,
            "updated_at": _now_ts(),
        }
    )

    job["message"] = summary

    return state


@app.post("/api/stage/{job_id}/{stage_id}/approve")
async def approve_interactive_stage(job_id: str, stage_id: str) -> Dict[str, Any]:
    job = _get_job(job_id)
    _get_stage(stage_id)

    interactive = job["interactive"]
    state = interactive["stages"][stage_id]

    if not state.get("agent_output"):
        raise HTTPException(status_code=400, detail="Run the agent before approving.")

    state.update(
        {
            "status": "approved",
            "approved": True,
            "updated_at": _now_ts(),
        }
    )

    stage_index = _get_stage_index(stage_id)

    if stage_index < len(STAGES) - 1:
        next_stage = STAGES[stage_index + 1]
        next_stage_id = next_stage["id"]
        interactive["current_stage_index"] = stage_index + 1
        job["step"] = next_stage["title"]
        job["progress"] = int(((stage_index + 1) / len(STAGES)) * 100)
        job["message"] = f"{_get_stage(stage_id)['title']} approved. Next stage is {next_stage['title']}."

        next_state = interactive["stages"][next_stage_id]

        if not next_state.get("agent_output"):
            next_output = _generate_stage_output(next_stage_id, job)
            next_state.update(
                {
                    "status": "agent_output_ready",
                    "agent_output": next_output,
                    "applied_feedback_summary": "Auto-generated after previous stage approval.",
                    "updated_at": _now_ts(),
                }
            )
    else:
        interactive["current_stage_index"] = stage_index
        job["status"] = "done"
        job["step"] = "Complete"
        job["progress"] = 100
        job["message"] = "All stages approved. Final plan is ready."

    return {
        "job_id": job_id,
        "current_stage_index": interactive["current_stage_index"],
        "stages": interactive["stages"],
        "message": job["message"],
    }