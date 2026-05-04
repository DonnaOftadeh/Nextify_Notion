"""
Nextify Interactive UI

Modern LLM-style workflow UI:
- Main area shows current final output only.
- Human feedback, LLM judge feedback, and combined revision are explicit.
- Final Accept & Continue button is at the bottom.
- Sidebar shows workflow, versions, judge feedback, and feedback history.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests
import streamlit as st


API_BASE = "http://127.0.0.1:8000"


STAGES = [
    {"id": "parse_submission", "title": "Parse Submission", "short_title": "Parse", "agent": "Input Parser Agent", "desc": "Turn the raw idea form into a clean product brief."},
    {"id": "brainstorm_parallel", "title": "Brainstorm Parallel", "short_title": "Brainstorm", "agent": "MarketAnalysisAgent + CrazyIdeaAgent", "desc": "Generate market-grounded and breakthrough ideas from the accepted parsed brief."},
    {"id": "idea_cooker", "title": "Idea Cooker", "short_title": "Cooker", "agent": "IdeaCookerAgent", "desc": "Score, explain, and recommend the best direction."},
    {"id": "theme_epic_generator", "title": "Theme & Epic Generator", "short_title": "Themes", "agent": "ThemeEpicAgent", "desc": "Create themes and epics from the accepted idea direction."},
    {"id": "roadmap_generator", "title": "Roadmap Generator", "short_title": "Roadmap", "agent": "RoadmapAgent", "desc": "Create a phased roadmap."},
    {"id": "feature_generation", "title": "Feature Generation", "short_title": "Features", "agent": "FeatureGenerationAgent", "desc": "Generate MVP and future features."},
    {"id": "prioritization_rice", "title": "Prioritization & RICE", "short_title": "RICE", "agent": "PrioritizationAgent", "desc": "Prioritize features using RICE."},
    {"id": "okr_generation", "title": "OKR Generation", "short_title": "OKRs", "agent": "OKRAgent", "desc": "Generate measurable OKRs."},
    {"id": "three_month_planner", "title": "Three-Month Planner", "short_title": "Planner", "agent": "PlannerAgent", "desc": "Create the 3-month execution plan."},
    {"id": "write_report_pdf", "title": "Write Report", "short_title": "Report", "agent": "ReportWriterAgent", "desc": "Assemble the final product plan."},
]


st.set_page_config(page_title="Nextify Interactive", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; max-width: 1180px; }
    .hero {
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 20px 22px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 8px 28px rgba(15,23,42,.06);
        margin-bottom: 18px;
    }
    .stage-chip {
        display: inline-block;
        padding: 7px 11px;
        border-radius: 999px;
        margin: 4px 5px 4px 0;
        font-size: 13px;
        border: 1px solid #ddd;
    }
    .approved { background: #ecfdf5; border-color: #a7f3d0; }
    .current { background: #fff7ed; border-color: #fdba74; font-weight: 700; }
    .ready { background: #eff6ff; border-color: #bfdbfe; }
    .pending { background: #f8fafc; color: #64748b; }

    .output-card {
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 28px;
        background: white;
        box-shadow: 0 8px 28px rgba(15,23,42,.05);
        margin-bottom: 20px;
    }
    .summary-card {
        border-left: 5px solid #2563eb;
        background: #eff6ff;
        padding: 14px 16px;
        border-radius: 14px;
        margin: 14px 0;
    }
    .review-card {
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 20px;
        background: #fbfdff;
        margin-top: 18px;
    }
    .accept-card {
        border: 1px solid #bbf7d0;
        border-radius: 22px;
        padding: 20px;
        background: #f0fdf4;
        margin-top: 18px;
    }
    .muted { color: #64748b; font-size: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str) -> Optional[Dict[str, Any]]:
    try:
        res = requests.get(f"{API_BASE}{path}", timeout=180)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not connect to backend: {exc}")
        return None

    if res.status_code >= 400:
        st.error(f"Backend error {res.status_code}: {res.text}")
        return None

    return res.json()


def api_post(path: str, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        res = requests.post(f"{API_BASE}{path}", json=data or {}, timeout=300)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not connect to backend: {exc}")
        return None

    if res.status_code >= 400:
        st.error(f"Backend error {res.status_code}: {res.text}")
        return None

    return res.json()


def rerun_app() -> None:
    st.rerun()


def get_stage(stage_id: str) -> Dict[str, str]:
    for stage in STAGES:
        if stage["id"] == stage_id:
            return stage
    return STAGES[0]


def get_job() -> Optional[Dict[str, Any]]:
    job_id = st.session_state.get("job_id")
    if not job_id:
        return None
    return api_get(f"/api/job/{job_id}")


def reset() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    rerun_app()


def submit_new_job() -> None:
    idea_title = st.session_state.get("idea_title", "").strip()
    idea_text = st.session_state.get("idea_text", "").strip()

    if not idea_title and not idea_text:
        st.warning("Please add at least an idea title or idea description.")
        return

    result = api_post(
        "/api/submit",
        {
            "journey_type": "idea",
            "payload": {
                "idea_title": st.session_state.get("idea_title", ""),
                "idea_text": st.session_state.get("idea_text", ""),
                "target_users": st.session_state.get("target_users", ""),
                "problem": st.session_state.get("problem", ""),
                "constraints": st.session_state.get("constraints", ""),
            },
        },
    )

    if not result:
        return

    job_id = result["job_id"]
    st.session_state.job_id = job_id
    st.session_state.selected_stage_id = "parse_submission"

    with st.spinner("Running Input Parser Agent..."):
        api_post(f"/api/stage/{job_id}/parse_submission/run")

    rerun_app()


def progress(job: Dict[str, Any]) -> Dict[str, Any]:
    approved = sum(1 for s in job["stages"].values() if s.get("approved"))
    current_index = max(0, min(job.get("current_stage_index", 0), len(STAGES) - 1))
    return {"approved": approved, "current_index": current_index, "value": approved / len(STAGES)}


def render_sidebar(job: Optional[Dict[str, Any]]) -> None:
    with st.sidebar:
        st.header("🧭 Workflow")

        if not job:
            st.info("Submit an idea to start.")
            return

        selected_stage_id = st.session_state.get("selected_stage_id", STAGES[job["current_stage_index"]]["id"])
        selected = get_stage(selected_stage_id)
        selected_state = job["stages"][selected_stage_id]

        st.markdown(f"**Selected:** {selected['title']}")
        st.caption(selected["agent"])

        st.divider()

        for stage in STAGES:
            sid = stage["id"]
            state = job["stages"][sid]
            if state.get("approved"):
                icon = "✅"
            elif state.get("agent_output"):
                icon = "🔵"
            else:
                icon = "⚪"

            if st.button(f"{icon} {stage['short_title']}", key=f"side_{sid}", use_container_width=True):
                st.session_state.selected_stage_id = sid
                rerun_app()

        st.divider()

        st.markdown("### Selected stage memory")

        with st.expander("Current output", expanded=False):
            if selected_state.get("agent_output"):
                st.markdown(selected_state["agent_output"])
            else:
                st.info("No output yet.")

        with st.expander("Accepted output", expanded=False):
            if selected_state.get("accepted_output"):
                st.markdown(selected_state["accepted_output"])
            else:
                st.info("Not accepted yet.")

        with st.expander("Previous versions", expanded=True):
            versions = selected_state.get("previous_outputs", [])
            if not versions:
                st.info("No previous versions yet.")
            else:
                for item in reversed(versions):
                    with st.expander(f"Version {item.get('version')} — {item.get('reason')}", expanded=False):
                        st.markdown(item.get("output", ""))

        with st.expander("LLM judge feedback", expanded=False):
            if selected_state.get("judge_feedback"):
                st.markdown(selected_state["judge_feedback"])
            else:
                st.info("No judge feedback yet.")

        with st.expander("Human feedback history", expanded=False):
            history = selected_state.get("user_feedback_history", [])
            if not history:
                st.info("No human feedback yet.")
            else:
                for item in reversed(history):
                    st.markdown(f"**Feedback {item.get('version')}**")
                    st.write(item.get("feedback", ""))

        st.divider()
        if st.button("Start new run", use_container_width=True):
            reset()


def render_header(job: Optional[Dict[str, Any]]) -> None:
    if not job:
        st.markdown("## 🧠 Nextify Interactive AI")
        st.caption("An agentic product workflow with human feedback, LLM critique, and versioned decisions.")
        return

    p = progress(job)
    current = STAGES[p["current_index"]]

    st.markdown('<div class="hero">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2.4, 1, 1])
    with c1:
        st.markdown("## 🧠 Nextify Interactive AI")
        st.markdown(
            f"""
            <div class="muted">
            Current workflow stage: <strong>{current['title']}</strong><br>
            Active agent: <strong>{current['agent']}</strong><br>
            Job: <code>{job['job_id']}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("Accepted", f"{p['approved']} / {len(STAGES)}")
    with c3:
        st.metric("Step", f"{p['current_index'] + 1} / {len(STAGES)}")

    st.progress(p["value"])

    chips = ""
    for idx, stage in enumerate(STAGES):
        sid = stage["id"]
        state = job["stages"][sid]
        if state.get("approved"):
            cls, icon = "stage-chip approved", "✅"
        elif idx == p["current_index"]:
            cls, icon = "stage-chip current", "🟡"
        elif state.get("agent_output"):
            cls, icon = "stage-chip ready", "🔵"
        else:
            cls, icon = "stage-chip pending", "⚪"
        chips += f'<span class="{cls}">{icon} {stage["short_title"]}</span>'

    st.markdown(chips, unsafe_allow_html=True)
    if job.get("message"):
        st.caption(job["message"])

    st.markdown("</div>", unsafe_allow_html=True)


def render_form() -> None:
    with st.container(border=True):
        st.markdown("### ✨ Start with your product idea")
        st.text_input("Idea title", key="idea_title")
        st.text_area("Idea description", key="idea_text", height=160)
        st.text_input("Target users", key="target_users")
        st.text_area("Problem", key="problem", height=120)
        st.text_input("Constraints", key="constraints")

        st.button("🚀 Submit idea to Parse Agent", type="primary", on_click=submit_new_job, use_container_width=True)


def render_workspace(job: Dict[str, Any]) -> None:
    if "selected_stage_id" not in st.session_state:
        st.session_state.selected_stage_id = STAGES[job["current_stage_index"]]["id"]

    stage_id = st.session_state.selected_stage_id
    stage = get_stage(stage_id)
    state = job["stages"][stage_id]

    stage_index = [s["id"] for s in STAGES].index(stage_id)
    next_stage = STAGES[stage_index + 1] if stage_index < len(STAGES) - 1 else None

    st.markdown(f"## {stage['title']}")
    st.caption(f"{stage['agent']} · {stage['desc']}")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Status", state.get("status", "pending"))
    with m2:
        st.metric("Revisions", state.get("revision_count", 0))
    with m3:
        st.metric("Accepted", "Yes" if state.get("approved") else "No")

    if state.get("error"):
        st.error(state["error"])

    if state.get("applied_feedback_summary"):
        st.markdown(
            f"""
            <div class="summary-card">
            <strong>Latest change</strong><br>
            {state["applied_feedback_summary"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Current final output")
    st.markdown('<div class="output-card">', unsafe_allow_html=True)
    if state.get("agent_output"):
        st.markdown(state["agent_output"])
    else:
        st.info("No output yet. Run this agent.")
    st.markdown("</div>", unsafe_allow_html=True)

    q1, q2 = st.columns(2)

    with q1:
        if st.button("Run / regenerate this agent", use_container_width=True):
            with st.spinner(f"Running {stage['agent']}..."):
                api_post(f"/api/stage/{job['job_id']}/{stage_id}/run")
            rerun_app()

    with q2:
        if st.button("Run LLM judge", use_container_width=True, disabled=not bool(state.get("agent_output"))):
            with st.spinner("Running LLM judge..."):
                api_post(f"/api/stage/{job['job_id']}/{stage_id}/judge")
            rerun_app()

    st.markdown('<div class="review-card">', unsafe_allow_html=True)
    st.markdown("### Review and revise")
    st.caption("Revise with human feedback, LLM judge feedback, or both. Old versions move to the sidebar.")

    with st.form(key=f"review_form_{stage_id}", clear_on_submit=True):
        human_feedback = st.text_area(
            "Human feedback",
            height=120,
            placeholder="Example: Make this more practical, preserve the product idea, add risks and assumptions, or choose idea 2.",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            apply_human = st.form_submit_button("Apply human feedback", use_container_width=True, disabled=not bool(state.get("agent_output")))
        with c2:
            apply_judge = st.form_submit_button("Apply LLM judge feedback", use_container_width=True, disabled=not bool(state.get("agent_output") and state.get("judge_feedback")))
        with c3:
            apply_both = st.form_submit_button("Apply both", use_container_width=True, disabled=not bool(state.get("agent_output")))

    if apply_human:
        with st.spinner("Reviewer Agent applying human feedback..."):
            api_post(f"/api/stage/{job['job_id']}/{stage_id}/revise", {"mode": "human_only", "feedback": human_feedback})
        rerun_app()

    if apply_judge:
        with st.spinner("Reviewer Agent applying LLM judge feedback..."):
            api_post(f"/api/stage/{job['job_id']}/{stage_id}/revise", {"mode": "judge_only", "feedback": ""})
        rerun_app()

    if apply_both:
        with st.spinner("Reviewer Agent applying both feedback sources..."):
            api_post(f"/api/stage/{job['job_id']}/{stage_id}/revise", {"mode": "both", "feedback": human_feedback})
        rerun_app()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="accept-card">', unsafe_allow_html=True)
    st.markdown("### Final decision for this stage")
    st.caption("Accepting this output sends it as context to the next agent.")

    if next_stage:
        label = f"Accept this output and send to {next_stage['title']}"
    else:
        label = "Accept final report"

    if st.button(label, type="primary", use_container_width=True, disabled=not bool(state.get("agent_output"))):
        with st.spinner("Accepting output and preparing next agent..."):
            api_post(f"/api/stage/{job['job_id']}/{stage_id}/approve")
        if next_stage:
            st.session_state.selected_stage_id = next_stage["id"]
        rerun_app()

    st.markdown("</div>", unsafe_allow_html=True)


job = get_job()
render_sidebar(job)
render_header(job)

if not job:
    render_form()
else:
    with st.expander("Submitted idea", expanded=False):
        payload = job.get("payload", {})
        st.markdown(f"**Idea title:** {payload.get('idea_title', '')}")
        st.markdown(f"**Idea description:** {payload.get('idea_text', '')}")
        st.markdown(f"**Target users:** {payload.get('target_users', '')}")
        st.markdown(f"**Problem:** {payload.get('problem', '')}")
        st.markdown(f"**Constraints:** {payload.get('constraints', '')}")

    render_workspace(job)