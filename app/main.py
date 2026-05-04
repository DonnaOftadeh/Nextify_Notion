"""
Nextify Interactive ADK Backend

FastAPI orchestration layer for:
- running each ADK agent stage-by-stage
- running LLM judge on each stage
- applying human feedback, judge feedback, or both through ReviewerAgent
- accepting output and sending it to the next agent
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

from .adk_agents import (
    run_interactive_stage_adk,
    run_interactive_judge_adk,
    run_interactive_reviewer_adk,
)


STAGES = [
    {"id": "parse_submission", "title": "Parse Submission", "short_title": "Parse", "agent": "Input Parser Agent", "desc": "Turn the raw idea form into a clean product brief."},
    {"id": "brainstorm_parallel", "title": "Brainstorm Parallel", "short_title": "Brainstorm", "agent": "MarketAnalysisAgent + CrazyIdeaAgent", "desc": "Generate market-grounded and breakthrough ideas from the accepted parsed brief."},
    {"id": "idea_cooker", "title": "Idea Cooker", "short_title": "Cooker", "agent": "IdeaCookerAgent", "desc": "Score, explain, and recommend the best product direction."},
    {"id": "theme_epic_generator", "title": "Theme & Epic Generator", "short_title": "Themes", "agent": "ThemeEpicAgent", "desc": "Create themes and epics from the accepted idea direction."},
    {"id": "roadmap_generator", "title": "Roadmap Generator", "short_title": "Roadmap", "agent": "RoadmapAgent", "desc": "Create a phased roadmap."},
    {"id": "feature_generation", "title": "Feature Generation", "short_title": "Features", "agent": "FeatureGenerationAgent", "desc": "Generate MVP and future features."},
    {"id": "prioritization_rice", "title": "Prioritization & RICE", "short_title": "RICE", "agent": "PrioritizationAgent", "desc": "Prioritize features using RICE."},
    {"id": "okr_generation", "title": "OKR Generation", "short_title": "OKRs", "agent": "OKRAgent", "desc": "Generate measurable OKRs."},
    {"id": "three_month_planner", "title": "Three-Month Planner", "short_title": "Planner", "agent": "PlannerAgent", "desc": "Create the 3-month execution plan."},
    {"id": "write_report_pdf", "title": "Write Report", "short_title": "Report", "agent": "ReportWriterAgent", "desc": "Assemble the final product plan."},
]

STAGE_IDS = [stage["id"] for stage in STAGES]


app = FastAPI(title="Nextify Interactive ADK Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Submission(BaseModel):
    journey_type: str = Field(default="idea")
    payload: Dict[str, Any]


class ReviseRequest(BaseModel):
    mode: str = "both"
    feedback: str = ""


JOBS: Dict[str, Dict[str, Any]] = {}


def now_ts() -> float:
    return time.time()


def get_stage(stage_id: str) -> Dict[str, str]:
    for stage in STAGES:
        if stage["id"] == stage_id:
            return stage
    raise HTTPException(status_code=404, detail=f"Unknown stage: {stage_id}")


def get_stage_index(stage_id: str) -> int:
    for index, stage in enumerate(STAGES):
        if stage["id"] == stage_id:
            return index
    raise HTTPException(status_code=404, detail=f"Unknown stage: {stage_id}")


def get_job_or_404(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def empty_stage_state(stage_id: str) -> Dict[str, Any]:
    return {
        "stage_id": stage_id,
        "status": "pending",
        "agent_output": "",
        "accepted_output": "",
        "previous_outputs": [],
        "judge_feedback": "",
        "judge_history": [],
        "user_feedback": "",
        "user_feedback_history": [],
        "applied_feedback_summary": "",
        "revision_count": 0,
        "approved": False,
        "updated_at": None,
        "error": "",
    }


def init_interactive_state() -> Dict[str, Any]:
    return {
        "current_stage_index": 0,
        "stages": {
            stage_id: empty_stage_state(stage_id)
            for stage_id in STAGE_IDS
        },
    }


def save_previous_version(state: Dict[str, Any], reason: str) -> None:
    if not state.get("agent_output"):
        return

    state["previous_outputs"].append(
        {
            "version": len(state["previous_outputs"]) + 1,
            "output": state["agent_output"],
            "reason": reason,
            "created_at": now_ts(),
        }
    )


def feedback_summary(mode: str, human_feedback: str, judge_feedback: str) -> str:
    used = []

    if mode in ("human_only", "both") and human_feedback.strip():
        used.append("human feedback")

    if mode in ("judge_only", "both") and judge_feedback.strip():
        used.append("LLM judge feedback")

    if not used:
        return "Reviewer Agent regenerated the output with general quality improvements."

    return f"Reviewer Agent applied {' and '.join(used)} and created a revised version."


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "service": "Nextify Interactive ADK Backend",
        "status": "ok",
        "gemini_key_loaded": bool(os.getenv("GOOGLE_API_KEY")),
    }


@app.get("/api/stages")
async def get_stages() -> Dict[str, Any]:
    return {"stages": STAGES}


@app.post("/api/submit")
async def submit(submission: Submission) -> Dict[str, Any]:
    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "job_id": job_id,
        "created_at": now_ts(),
        "status": "interactive",
        "step": "Parse Submission",
        "progress": 0,
        "message": "Interactive ADK job created.",
        "journey_type": submission.journey_type,
        "payload": submission.payload,
        "interactive": init_interactive_state(),
    }

    return {"job_id": job_id, "stages": STAGE_IDS}


@app.get("/api/job/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    job = get_job_or_404(job_id)

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


@app.post("/api/stage/{job_id}/{stage_id}/run")
async def run_stage(job_id: str, stage_id: str) -> Dict[str, Any]:
    job = get_job_or_404(job_id)
    stage = get_stage(stage_id)
    state = job["interactive"]["stages"][stage_id]

    save_previous_version(state, "Re-run from scratch")

    state.update(
        {
            "status": "running_agent",
            "error": "",
            "updated_at": now_ts(),
        }
    )

    try:
        output = await run_interactive_stage_adk(
            job=job,
            stage_id=stage_id,
            stage_title=stage["title"],
        )

        state.update(
            {
                "status": "agent_output_ready",
                "agent_output": output,
                "applied_feedback_summary": "ADK agent generated a fresh output.",
                "approved": False,
                "updated_at": now_ts(),
                "error": "",
            }
        )

        index = get_stage_index(stage_id)
        job["interactive"]["current_stage_index"] = index
        job["step"] = stage["title"]
        job["progress"] = int((index / len(STAGES)) * 100)
        job["message"] = f"{stage['agent']} finished."

        return state

    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "error": str(exc),
                "updated_at": now_ts(),
            }
        )
        job["message"] = f"Agent error: {exc}"
        return state


@app.post("/api/stage/{job_id}/{stage_id}/judge")
async def judge_stage(job_id: str, stage_id: str) -> Dict[str, Any]:
    job = get_job_or_404(job_id)
    stage = get_stage(stage_id)
    state = job["interactive"]["stages"][stage_id]

    if not state.get("agent_output"):
        raise HTTPException(status_code=400, detail="Run the agent before judging.")

    if state.get("judge_feedback"):
        state["judge_history"].append(
            {
                "version": len(state["judge_history"]) + 1,
                "feedback": state["judge_feedback"],
                "created_at": now_ts(),
            }
        )

    state.update(
        {
            "status": "running_judge",
            "error": "",
            "updated_at": now_ts(),
        }
    )

    try:
        feedback = await run_interactive_judge_adk(
            job=job,
            stage_id=stage_id,
            stage_title=stage["title"],
            stage_content=state["agent_output"],
        )

        state.update(
            {
                "status": "judge_feedback_ready",
                "judge_feedback": feedback,
                "updated_at": now_ts(),
                "error": "",
            }
        )

        job["message"] = f"LLM judge reviewed {stage['title']}."
        return state

    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "error": str(exc),
                "updated_at": now_ts(),
            }
        )
        job["message"] = f"Judge error: {exc}"
        return state


@app.post("/api/stage/{job_id}/{stage_id}/revise")
async def revise_stage(job_id: str, stage_id: str, req: ReviseRequest) -> Dict[str, Any]:
    job = get_job_or_404(job_id)
    stage = get_stage(stage_id)
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
                "created_at": now_ts(),
            }
        )

    human_feedback = state.get("user_feedback", "") if mode in ("human_only", "both") else ""
    judge_feedback = state.get("judge_feedback", "") if mode in ("judge_only", "both") else ""

    save_previous_version(state, f"Before Reviewer Agent revision using mode: {mode}")

    state.update(
        {
            "status": "running_reviewer",
            "error": "",
            "updated_at": now_ts(),
        }
    )

    try:
        revised_output = await run_interactive_reviewer_adk(
            job=job,
            stage_id=stage_id,
            stage_title=stage["title"],
            current_output=state["agent_output"],
            human_feedback=human_feedback,
            judge_feedback=judge_feedback,
            feedback_mode=mode,
        )

        summary = feedback_summary(mode, human_feedback, judge_feedback)

        state.update(
            {
                "status": "revised_output_ready",
                "agent_output": revised_output,
                "applied_feedback_summary": summary,
                "revision_count": state.get("revision_count", 0) + 1,
                "approved": False,
                "updated_at": now_ts(),
                "error": "",
            }
        )

        job["message"] = summary
        return state

    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "error": str(exc),
                "updated_at": now_ts(),
            }
        )
        job["message"] = f"Reviewer error: {exc}"
        return state


@app.post("/api/stage/{job_id}/{stage_id}/approve")
async def approve_stage(job_id: str, stage_id: str) -> Dict[str, Any]:
    job = get_job_or_404(job_id)
    current_stage = get_stage(stage_id)
    state = job["interactive"]["stages"][stage_id]

    if not state.get("agent_output"):
        raise HTTPException(status_code=400, detail="Run the agent before approving.")

    state.update(
        {
            "status": "approved",
            "approved": True,
            "accepted_output": state["agent_output"],
            "updated_at": now_ts(),
        }
    )

    index = get_stage_index(stage_id)

    if index < len(STAGES) - 1:
        next_stage = STAGES[index + 1]
        next_id = next_stage["id"]

        job["interactive"]["current_stage_index"] = index + 1
        job["step"] = next_stage["title"]
        job["progress"] = int(((index + 1) / len(STAGES)) * 100)
        job["message"] = f"{current_stage['title']} accepted. Next agent received this output."

        next_state = job["interactive"]["stages"][next_id]

        if not next_state.get("agent_output"):
            next_state.update(
                {
                    "status": "running_agent",
                    "error": "",
                    "updated_at": now_ts(),
                }
            )

            try:
                next_output = await run_interactive_stage_adk(
                    job=job,
                    stage_id=next_id,
                    stage_title=next_stage["title"],
                )

                next_state.update(
                    {
                        "status": "agent_output_ready",
                        "agent_output": next_output,
                        "applied_feedback_summary": "Auto-generated from previous accepted agent output.",
                        "approved": False,
                        "updated_at": now_ts(),
                        "error": "",
                    }
                )

            except Exception as exc:
                next_state.update(
                    {
                        "status": "failed",
                        "error": str(exc),
                        "updated_at": now_ts(),
                    }
                )
                job["message"] = f"Next agent error: {exc}"

    else:
        job["interactive"]["current_stage_index"] = index
        job["status"] = "done"
        job["step"] = "Complete"
        job["progress"] = 100
        job["message"] = "All stages accepted. Final plan is ready."

    return {
        "job_id": job_id,
        "current_stage_index": job["interactive"]["current_stage_index"],
        "stages": job["interactive"]["stages"],
        "message": job["message"],
    }