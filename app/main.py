"""
Nextify Interactive Backend

This FastAPI application provides interactive endpoints to run agents stage by stage,
collect judge feedback and human feedback, revise outputs, and progress through a
multi‑stage product development workflow.  It disables automatic long pipeline
execution and instead lets the user manually control each stage.  Upon approval
of a stage, the next stage is automatically run.

The core endpoints are:
  POST /api/submit            – start a new interactive job (no long pipeline).
  GET  /api/job/{job_id}      – fetch job state including all stages.
  POST /api/stage/{job_id}/{stage_id}/run       – run an agent for the stage.
  POST /api/stage/{job_id}/{stage_id}/judge     – run an LLM judge on stage output.
  POST /api/stage/{job_id}/{stage_id}/feedback  – save human feedback for a stage.
  POST /api/stage/{job_id}/{stage_id}/revise    – revise using selected feedback mode.
  POST /api/stage/{job_id}/{stage_id}/approve   – approve a stage and run the next one.

This file does not implement a long‑running ADK pipeline.  Instead, the user
controls each stage individually and only uses LLM calls when needed, which
helps avoid exceeding API quotas during interactive sessions.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Stage configuration
#
# Each stage represents a step in the product development workflow.  The
# ``id`` field is used in API paths, ``title`` is displayed to users, and
# ``short_title`` is used in tab labels.  ``agent`` names the agent to run,
# while ``desc`` describes what the stage should accomplish.
# ---------------------------------------------------------------------------

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
        "agent": "Market Analysis + Crazy Idea",
        "desc": "Generate both market‑grounded opportunities and unconventional breakthrough ideas.",
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
        "agent": "RICE Agent",
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
        "title": "Three‑Month Planner",
        "short_title": "Planner",
        "agent": "Planner Agent",
        "desc": "Lay out a practical three‑month execution plan.",
    },
    {
        "id": "write_report_pdf",
        "title": "Write Report (PDF)",
        "short_title": "Report",
        "agent": "Report Writer Agent",
        "desc": "Assemble everything into a final report and PDF.",
    },
]

# Helper to locate stage index by id
def _get_stage_index(stage_id: str) -> int:
    for i, s in enumerate(STAGES):
        if s["id"] == stage_id:
            return i
    raise HTTPException(status_code=404, detail=f"Unknown stage: {stage_id}")


# ---------------------------------------------------------------------------
# Interactive job state and helper functions
# ---------------------------------------------------------------------------

def _now_ts() -> float:
    return time.time()


def _init_stage_state(stage_id: str) -> Dict[str, Any]:
    return {
        "stage_id": stage_id,
        "status": "pending",
        "agent_output": "",
        "judge_feedback": "",
        "user_feedback": "",
        "revision_count": 0,
        "approved": False,
        "updated_at": None,
    }


def _init_interactive_state() -> Dict[str, Any]:
    # Each interactive job stores current_stage_index and a dictionary of stage states
    return {
        "current_stage_index": 0,
        "stages": {stage["id"]: _init_stage_state(stage["id"]) for stage in STAGES},
    }


# Global in‑memory job store.  In a real application this would likely be persisted
# in a database.
JOBS: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# FastAPI application setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Nextify Interactive Backend")


# Models for incoming feedback request
class FeedbackRequest(BaseModel):
    feedback: str


@app.get("/")
async def root() -> Dict[str, str]:
    return {"service": "Nextify Interactive Backend", "status": "ok"}


@app.post("/api/submit")
async def submit(submission: BaseModel) -> Dict[str, Any]:
    """
    Start a new interactive job.  The submission payload should include the
    journey type and the user input fields (title, description, etc.).
    The job is stored in the global JOBS dictionary and returns its id.
    No long pipeline is run automatically.
    """
    job_id = str(uuid.uuid4())
    # Store full submission payload for context in stage prompts
    JOBS[job_id] = {
        "job_id": job_id,
        "created_at": _now_ts(),
        "status": "interactive",
        "step": "Interactive Mode",
        "progress": 0,
        "message": "Interactive job created. Run each stage manually.",
        "pdf_path": None,
        "journey_type": submission.dict().get("journey_type", "idea"),
        "payload": submission.dict().get("payload", {}),
        "raw_report": "",
        "history": {},
        "interactive": _init_interactive_state(),
    }
    # Do not call any long pipeline automatically here.  Users must run stages manually.
    return {"job_id": job_id, "stages": [s["id"] for s in STAGES]}


def _ensure_job_exists(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _approved_context(job: Dict[str, Any]) -> str:
    """Concatenate approved agent outputs as context for subsequent stages."""
    outputs: List[str] = []
    for stage in STAGES:
        s_state = job["interactive"]["stages"][stage["id"]]
        if s_state["approved"] and s_state["agent_output"]:
            outputs.append(f"# {stage['title']}\n\n{s_state['agent_output']}")
    return "\n\n---\n\n".join(outputs) if outputs else "No previous stages approved yet."


def _generate_stage_output(stage_id: str, job: Dict[str, Any]) -> str:
    """
    Generate a placeholder output for the given stage.  In a production system
    this would call the appropriate agent and return its output.  Here we
    synthesise a simple markdown section using the job payload and approved
    context for demonstration purposes.
    """
    payload = job.get("payload", {})
    approved_context = _approved_context(job)
    stage = STAGES[_get_stage_index(stage_id)]
    # Build a Markdown string summarising the input and context
    lines: List[str] = [f"# {stage['title']}", "", "## Inputs"]
    for key, value in payload.items():
        pretty = key.replace("_", " ").title()
        lines.append(f"- **{pretty}:** {value}")
    lines.extend([
        "",
        "## Prior Approved Context",
        approved_context,
        "",
        "## Draft Output",
        f"This is a placeholder draft for the **{stage['title']}** stage.",
        "You can use the judge or provide feedback and revise this output.",
    ])
    return "\n".join(lines)


def _judge_stage_output(stage_id: str, agent_output: str) -> str:
    """
    Produce placeholder judge feedback for demonstration.  A real implementation
    would call an evaluator model to critique the output and suggest
    improvements.  The feedback is Markdown and kept separate from the main
    workflow UI.
    """
    stage = STAGES[_get_stage_index(stage_id)]
    return (
        f"# LLM Judge Review: {stage['title']}\n\n"
        "## Score\n\n7.5 / 10\n\n"
        "## Strengths\n\n"
        "- Clearly identifies inputs and context.\n"
        "- Provides a draft output section.\n\n"
        "## Weaknesses\n\n"
        "- Lacks specific guidance for this stage.\n"
        "- Needs more actionable metrics and next steps.\n\n"
        "## Recommended Improvements\n\n"
        "1. Add metrics and success criteria.\n"
        "2. Tailor output to the target users.\n"
        "3. Identify key risks or unknowns.\n"
    )


# ---------------------------------------------------------------------------
# Revision helper
#
# The `_revise_stage_output` function constructs a revised draft for a stage
# based on the selected feedback mode.  When revising, the caller can choose
# to incorporate just the human feedback (``mode="human_only"``), just the
# judge feedback (``mode="judge_only"``), or both (``mode="both"``).  The
# function builds a Markdown document that summarises the inputs, shows the
# selected feedback, and provides a placeholder revised output.  In a
# production system, this would call a revision agent.  Here we simply
# structure the information for demonstration purposes.

def _revise_stage_output(stage_id: str, job: Dict[str, Any], mode: str) -> str:
    """Generate a revised draft using the specified feedback mode."""
    stage_state = job["interactive"]["stages"][stage_id]
    # Select which feedback to include
    human_feedback = stage_state.get("user_feedback", "") if mode in ("human_only", "both") else ""
    judge_feedback = stage_state.get("judge_feedback", "") if mode in ("judge_only", "both") else ""
    approved_context = _approved_context(job)
    previous_output = stage_state.get("agent_output", "")
    stage = STAGES[_get_stage_index(stage_id)]
    return (
        f"# Revised {stage['title']}\n\n"
        "## Revision Inputs\n\n"
        "### Human Feedback\n"
        f"{human_feedback or 'No human feedback.'}\n\n"
        "### Judge Feedback\n"
        f"{judge_feedback or 'No judge feedback.'}\n\n"
        "### Approved Context\n"
        f"{approved_context}\n\n"
        "---\n\n"
        "## Revised Output\n\n"
        "This draft improves the previous output using the selected feedback.\n\n"
        "---\n\n"
        "## Previous Draft\n"
        f"{previous_output}\n"
    )


@app.get("/api/job/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    """Return the full state of an interactive job."""
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
    """
    Run the agent for a given stage.  Stores the generated output in the
    job state and sets the status to ``needs_judge``.  A real implementation
    would call the appropriate agent; here we produce a placeholder.
    """
    job = _ensure_job_exists(job_id)
    stage_state = job["interactive"]["stages"][stage_id]
    output = _generate_stage_output(stage_id, job)
    stage_state.update({
        "status": "needs_judge",
        "agent_output": output,
        "updated_at": _now_ts(),
    })
    return stage_state


@app.post("/api/stage/{job_id}/{stage_id}/judge")
async def judge_interactive_stage(job_id: str, stage_id: str) -> Dict[str, Any]:
    job = _ensure_job_exists(job_id)
    stage_state = job["interactive"]["stages"][stage_id]
    if not stage_state.get("agent_output"):
        raise HTTPException(status_code=400, detail="Stage must be run before judging.")
    feedback = _judge_stage_output(stage_id, stage_state["agent_output"])
    stage_state.update({
        "status": "needs_user_feedback",
        "judge_feedback": feedback,
        "updated_at": _now_ts(),
    })
    return stage_state


@app.post("/api/stage/{job_id}/{stage_id}/feedback")
async def save_interactive_feedback(job_id: str, stage_id: str, req: FeedbackRequest) -> Dict[str, Any]:
    job = _ensure_job_exists(job_id)
    stage_state = job["interactive"]["stages"][stage_id]
    stage_state.update({
        "status": "feedback_saved",
        "user_feedback": req.feedback,
        "updated_at": _now_ts(),
    })
    return stage_state


@app.post("/api/stage/{job_id}/{stage_id}/revise")
async def revise_interactive_stage(job_id: str, stage_id: str, mode: str = "both") -> Dict[str, Any]:
    """
    Revise a stage using the selected feedback mode.  ``mode`` can be
    ``human_only``, ``judge_only`` or ``both`` (default).  The revised draft
    becomes the new ``agent_output`` and the stage returns to ``needs_judge``.
    """
    job = _ensure_job_exists(job_id)
    stage_state = job["interactive"]["stages"][stage_id]
    if not stage_state.get("agent_output"):
        raise HTTPException(status_code=400, detail="Stage must be run before revising.")
    revised_output = _revise_stage_output(stage_id, job, mode)
    stage_state.update({
        "status": "needs_judge",
        "agent_output": revised_output,
        "revision_count": stage_state.get("revision_count", 0) + 1,
        "updated_at": _now_ts(),
    })
    return stage_state


@app.post("/api/stage/{job_id}/{stage_id}/approve")
async def approve_interactive_stage(job_id: str, stage_id: str) -> Dict[str, Any]:
    """
    Approve a stage.  Marks it as approved and, if there is a subsequent
    stage, automatically runs the next stage.  This advances the current
    stage index and pre‑populates the next stage output so the user can
    immediately review it.
    """
    job = _ensure_job_exists(job_id)
    state = job["interactive"]
    stage_state = state["stages"][stage_id]
    if not stage_state.get("agent_output"):
        raise HTTPException(status_code=400, detail="Stage must be run before approval.")
    stage_state.update({
        "status": "approved",
        "approved": True,
        "updated_at": _now_ts(),
    })
    # Move to next stage if available
    idx = _get_stage_index(stage_id)
    if idx < len(STAGES) - 1:
        next_id = STAGES[idx + 1]["id"]
        state["current_stage_index"] = idx + 1
        # Run the next stage to generate its initial output
        run_interactive_stage(job_id, next_id)
    return {
        "job_id": job_id,
        "current_stage_index": state["current_stage_index"],
        "stages": state["stages"],
    }
