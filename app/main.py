"""
Nextify Interactive Backend

This backend supports a Human-in-the-Loop product workflow:

1. Submit an idea.
2. Run each agent stage.
3. Run an LLM-as-judge review.
4. Save human feedback.
5. Revise the stage using human feedback, judge feedback, or both.
6. Approve the stage and automatically prepare the next stage.

This version uses placeholder agent and judge outputs so the workflow works
without Gemini quota/API issues. After this works, the placeholder functions
can be replaced with your real ADK/Gemini agents.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# FastAPI app
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
# Stages
# ============================================================

STAGES: List[Dict[str, str]] = [
    {
        "id": "parse_submission",
        "title": "Parse Submission",
        "short_title": "Parse",
        "agent": "Input Parser Agent",
        "desc": "Validate and structure the submitted idea, including title, description, users, problem and constraints.",
    },
    {
        "id": "brainstorm_parallel",
        "title": "Brainstorm Parallel",
        "short_title": "Brainstorm",
        "agent": "Market Analysis Agent + Crazy Idea Agent",
        "desc": "Generate both market-grounded opportunities and unconventional breakthrough ideas.",
    },
    {
        "id": "idea_cooker",
        "title": "Idea Cooker",
        "short_title": "Cooker",
        "agent": "Idea Cooker Agent",
        "desc": "Synthesize the strongest direction from the brainstorming outputs.",
    },
    {
        "id": "theme_epic_generator",
        "title": "Theme & Epic Generator",
        "short_title": "Themes",
        "agent": "Theme & Epic Agent",
        "desc": "Convert the chosen direction into strategic themes and epics.",
    },
    {
        "id": "roadmap_generator",
        "title": "Roadmap Generator",
        "short_title": "Roadmap",
        "agent": "Roadmap Agent",
        "desc": "Produce a phased roadmap from the approved themes and epics.",
    },
    {
        "id": "feature_generation",
        "title": "Feature Generation",
        "short_title": "Features",
        "agent": "Feature Generation Agent",
        "desc": "Generate concrete product features based on the roadmap.",
    },
    {
        "id": "prioritization_rice",
        "title": "Prioritization & RICE",
        "short_title": "RICE",
        "agent": "Prioritization & RICE Agent",
        "desc": "Prioritize features using the RICE framework.",
    },
    {
        "id": "okr_generation",
        "title": "OKR Generation",
        "short_title": "OKRs",
        "agent": "OKR Agent",
        "desc": "Generate measurable objectives and key results from the approved product direction.",
    },
    {
        "id": "three_month_planner",
        "title": "Three-Month Planner",
        "short_title": "Planner",
        "agent": "Three-Month Planner Agent",
        "desc": "Lay out a practical three-month execution plan.",
    },
    {
        "id": "write_report_pdf",
        "title": "Write Report (PDF)",
        "short_title": "Report",
        "agent": "Report Writer Agent",
        "desc": "Assemble everything into a final report and PDF.",
    },
]

INTERACTIVE_STAGE_IDS = [stage["id"] for stage in STAGES]


# ============================================================
# Request models
# ============================================================

class Submission(BaseModel):
    journey_type: str = Field(default="idea")
    payload: Dict[str, Any]


class FeedbackRequest(BaseModel):
    feedback: str = ""


class ReviseRequest(BaseModel):
    mode: str = "both"


# ============================================================
# In-memory state
# ============================================================

JOBS: Dict[str, Dict[str, Any]] = {}


# ============================================================
# Helpers
# ============================================================

def _now_ts() -> float:
    return time.time()


def _get_stage_index(stage_id: str) -> int:
    for index, stage in enumerate(STAGES):
        if stage["id"] == stage_id:
            return index

    raise HTTPException(status_code=404, detail=f"Unknown stage: {stage_id}")


def _get_stage_config(stage_id: str) -> Dict[str, str]:
    return STAGES[_get_stage_index(stage_id)]


def _empty_stage(stage_id: str) -> Dict[str, Any]:
    return {
        "stage_id": stage_id,
        "status": "pending",
        "agent_output": "",
        "judge_feedback": "",
        "user_feedback": "",
        "change_summary": "",
        "last_revision_mode": "",
        "revision_count": 0,
        "approved": False,
        "updated_at": None,
    }


def _init_interactive_state() -> Dict[str, Any]:
    return {
        "current_stage_index": 0,
        "stages": {
            stage_id: _empty_stage(stage_id)
            for stage_id in INTERACTIVE_STAGE_IDS
        },
    }


def _ensure_job_exists(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


def _build_payload_summary(payload: Dict[str, Any]) -> str:
    if not payload:
        return "- No payload was provided."

    lines = []

    for key, value in payload.items():
        pretty_key = key.replace("_", " ").title()
        lines.append(f"- **{pretty_key}:** {value}")

    return "\n".join(lines)


def _approved_context(job: Dict[str, Any]) -> str:
    parts: List[str] = []

    for stage in STAGES:
        stage_id = stage["id"]
        stage_state = job["interactive"]["stages"][stage_id]

        if stage_state.get("approved") and stage_state.get("agent_output"):
            parts.append(
                f"# {stage['title']}\n\n"
                f"{stage_state['agent_output']}"
            )

    if not parts:
        return "No previous stages approved yet."

    return "\n\n---\n\n".join(parts)


def _make_change_summary(stage_id: str, mode: str, user_feedback: str, judge_feedback: str) -> str:
    stage = _get_stage_config(stage_id)

    human_used = mode in ("human_only", "both") and bool(user_feedback.strip())
    judge_used = mode in ("judge_only", "both") and bool(judge_feedback.strip())

    used_items = []

    if human_used:
        used_items.append("human feedback")

    if judge_used:
        used_items.append("LLM judge feedback")

    if not used_items:
        used = "no saved feedback"
    else:
        used = " and ".join(used_items)

    return f"""
## Change Summary

**Stage:** {stage["title"]}  
**Revision mode:** `{mode}`  
**Feedback applied:** {used}

### What changed

- The stage output was regenerated as a revised version.
- The revision explicitly considered the selected feedback mode.
- The output now includes a feedback application section so you can verify what was used.
- The stage is not automatically approved; you can judge again, add more feedback, revise again, or approve.
"""


def _generate_stage_output(stage_id: str, job: Dict[str, Any]) -> str:
    stage = _get_stage_config(stage_id)
    payload = job.get("payload", {})
    approved_context = _approved_context(job)

    if stage_id == "parse_submission":
        return f"""
# Parse Submission

## Agent
{stage["agent"]}

## Parsed User Input

{_build_payload_summary(payload)}

## Interpretation

The submission has been converted into a structured product-development input.

The next agent should use this input to brainstorm both:
1. market-grounded product opportunities,
2. unconventional breakthrough ideas.

## Next Step

Approve this stage when the parsed input looks correct. After approval, Nextify will automatically prepare the **Brainstorm Parallel** stage.
"""

    return f"""
# {stage["title"]}

## Agent
{stage["agent"]}

## Original User Input

{_build_payload_summary(payload)}

## Approved Previous Context

{approved_context}

## Draft Output

This is the first draft for **{stage["title"]}**.

### Stage Goal

{stage["desc"]}

### Suggested Product Artifact

- Main objective for this stage
- Key assumptions
- User or market insight
- Risks
- Next action

## Next Step

Run the LLM judge or add your own feedback. Then revise or approve this stage.
"""


def _judge_stage_output(stage_id: str, agent_output: str) -> str:
    stage = _get_stage_config(stage_id)

    return f"""
# LLM Judge Review: {stage["title"]}

## Score

**7.5 / 10**

## Strengths

- The output is separated clearly for the **{stage["title"]}** stage.
- It can be reviewed independently before moving to the next stage.
- The workflow supports human approval before downstream agents depend on this output.

## Gaps

- The output should be more specific to the target user and problem.
- It should include sharper assumptions, risks, and success criteria.
- It should be more decision-ready before final approval.

## Recommended Improvements

1. Add measurable success criteria.
2. Clarify the most important user segment.
3. Identify key risks and assumptions.
4. Make the next action concrete.
5. Improve clarity before approving this stage.

## Judge Decision

Revise before approval if this stage will be used as input for downstream agents.
"""


def _revise_stage_output(stage_id: str, job: Dict[str, Any], mode: str) -> str:
    if mode not in ("human_only", "judge_only", "both"):
        mode = "both"

    stage = _get_stage_config(stage_id)
    stage_state = job["interactive"]["stages"][stage_id]

    previous_output = stage_state.get("agent_output", "")
    user_feedback = stage_state.get("user_feedback", "")
    judge_feedback = stage_state.get("judge_feedback", "")
    approved_context = _approved_context(job)

    included_human_feedback = user_feedback if mode in ("human_only", "both") else ""
    included_judge_feedback = judge_feedback if mode in ("judge_only", "both") else ""

    return f"""
# Revised {stage["title"]}

## Revision Mode

**{mode}**

## Feedback Applied

### Human Feedback Applied

{included_human_feedback or "Human feedback was not used in this revision."}

### LLM Judge Feedback Applied

{included_judge_feedback or "Judge feedback was not used in this revision."}

## Approved Previous Context

{approved_context}

---

## Revised Stage Output

This is the revised version of **{stage["title"]}**.

### Improved Direction

The output was regenerated to reflect the selected feedback mode.

### Applied Improvements

- Added clearer structure for the stage.
- Made the stage artifact easier to review before approval.
- Preserved the approved previous context.
- Included the selected feedback source directly in the revision.

### Decision Notes

Review this updated version. You can:
1. run the judge again,
2. add more human feedback,
3. revise again,
4. approve and move to the next stage.

---

## Previous Draft Reference

{previous_output}
"""


def _prepare_next_stage(job: Dict[str, Any], current_stage_id: str) -> None:
    current_index = _get_stage_index(current_stage_id)

    if current_index >= len(STAGES) - 1:
        job["interactive"]["current_stage_index"] = current_index
        return

    next_index = current_index + 1
    next_stage_id = STAGES[next_index]["id"]
    next_stage_state = job["interactive"]["stages"][next_stage_id]

    job["interactive"]["current_stage_index"] = next_index

    if not next_stage_state.get("agent_output"):
        next_stage_state.update(
            {
                "status": "needs_judge",
                "agent_output": _generate_stage_output(next_stage_id, job),
                "approved": False,
                "updated_at": _now_ts(),
            }
        )


# ============================================================
# API endpoints
# ============================================================

@app.get("/")
async def root() -> Dict[str, str]:
    return {
        "service": "Nextify Interactive Backend",
        "status": "ok",
    }


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "Nextify Interactive Backend",
        "stages": INTERACTIVE_STAGE_IDS,
        "active_jobs": len(JOBS),
    }


@app.post("/api/submit")
async def submit(submission: Submission) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "job_id": job_id,
        "created_at": _now_ts(),
        "status": "interactive",
        "step": "Interactive Mode",
        "progress": 0,
        "message": "Interactive job created. Run each stage manually.",
        "pdf_path": None,
        "journey_type": submission.journey_type,
        "payload": submission.payload,
        "raw_report": "",
        "history": {},
        "interactive": _init_interactive_state(),
    }

    return {
        "job_id": job_id,
        "stages": INTERACTIVE_STAGE_IDS,
    }


@app.get("/api/status/{job_id}")
async def status(job_id: str) -> Dict[str, Any]:
    job = _ensure_job_exists(job_id)

    approved_count = sum(
        1
        for stage_state in job["interactive"]["stages"].values()
        if stage_state.get("approved")
    )

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "step": job.get("step"),
        "progress": int((approved_count / len(STAGES)) * 100),
        "message": job.get("message"),
        "ready": approved_count == len(STAGES),
    }


@app.get("/api/job/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    job = _ensure_job_exists(job_id)

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
    }


@app.post("/api/stage/{job_id}/{stage_id}/run")
async def run_interactive_stage(job_id: str, stage_id: str) -> Dict[str, Any]:
    job = _ensure_job_exists(job_id)
    _get_stage_index(stage_id)

    stage_state = job["interactive"]["stages"][stage_id]

    stage_state.update(
        {
            "status": "needs_judge",
            "agent_output": _generate_stage_output(stage_id, job),
            "approved": False,
            "updated_at": _now_ts(),
        }
    )

    return stage_state


@app.post("/api/stage/{job_id}/{stage_id}/judge")
async def judge_interactive_stage(job_id: str, stage_id: str) -> Dict[str, Any]:
    job = _ensure_job_exists(job_id)
    _get_stage_index(stage_id)

    stage_state = job["interactive"]["stages"][stage_id]

    if not stage_state.get("agent_output"):
        raise HTTPException(
            status_code=400,
            detail="Stage must be run before judging.",
        )

    stage_state.update(
        {
            "status": "needs_user_feedback",
            "judge_feedback": _judge_stage_output(stage_id, stage_state["agent_output"]),
            "updated_at": _now_ts(),
        }
    )

    return stage_state


@app.post("/api/stage/{job_id}/{stage_id}/feedback")
async def save_interactive_feedback(
    job_id: str,
    stage_id: str,
    req: FeedbackRequest,
) -> Dict[str, Any]:
    job = _ensure_job_exists(job_id)
    _get_stage_index(stage_id)

    stage_state = job["interactive"]["stages"][stage_id]

    stage_state.update(
        {
            "status": "feedback_saved",
            "user_feedback": req.feedback,
            "updated_at": _now_ts(),
        }
    )

    return stage_state


@app.post("/api/stage/{job_id}/{stage_id}/revise")
async def revise_interactive_stage(
    job_id: str,
    stage_id: str,
    req: ReviseRequest,
) -> Dict[str, Any]:
    job = _ensure_job_exists(job_id)
    _get_stage_index(stage_id)

    stage_state = job["interactive"]["stages"][stage_id]

    if not stage_state.get("agent_output"):
        raise HTTPException(
            status_code=400,
            detail="Stage must be run before revising.",
        )

    mode = req.mode

    if mode not in ("human_only", "judge_only", "both"):
        mode = "both"

    user_feedback = stage_state.get("user_feedback", "")
    judge_feedback = stage_state.get("judge_feedback", "")

    stage_state.update(
        {
            "status": f"revised_{mode}",
            "agent_output": _revise_stage_output(stage_id, job, mode),
            "change_summary": _make_change_summary(
                stage_id=stage_id,
                mode=mode,
                user_feedback=user_feedback,
                judge_feedback=judge_feedback,
            ),
            "last_revision_mode": mode,
            "revision_count": stage_state.get("revision_count", 0) + 1,
            "approved": False,
            "updated_at": _now_ts(),
        }
    )

    return stage_state


@app.post("/api/stage/{job_id}/{stage_id}/approve")
async def approve_interactive_stage(job_id: str, stage_id: str) -> Dict[str, Any]:
    job = _ensure_job_exists(job_id)
    _get_stage_index(stage_id)

    stage_state = job["interactive"]["stages"][stage_id]

    if not stage_state.get("agent_output"):
        raise HTTPException(
            status_code=400,
            detail="Stage must be run before approval.",
        )

    stage_state.update(
        {
            "status": "approved",
            "approved": True,
            "updated_at": _now_ts(),
        }
    )

    _prepare_next_stage(job, stage_id)

    return {
        "job_id": job_id,
        "current_stage_index": job["interactive"]["current_stage_index"],
        "stages": job["interactive"]["stages"],
    }