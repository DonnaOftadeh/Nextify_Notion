"""
Nextify Interactive UI

Main page:
- Shows only the current final agent output.
- Shows latest change summary above the output.
- Does not show judge output in the main area.
- Does not show human feedback history in the main area.

Sidebar:
- Shows workflow history.
- Shows previous versions.
- Shows LLM judge feedback.
- Shows human feedback history.

Streamlit fix:
- Feedback input uses st.form(clear_on_submit=True)
- This prevents StreamlitAPIException from changing session_state after widget creation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests
import streamlit as st


API_BASE = "http://127.0.0.1:8000"


STAGES = [
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


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(page_title="Nextify Interactive", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    .top-card {
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px;
        background: white;
        box-shadow: 0 2px 16px rgba(0,0,0,0.04);
        margin-bottom: 18px;
    }

    .stage-pill {
        display: inline-block;
        padding: 7px 11px;
        border-radius: 999px;
        margin: 4px 5px 4px 0;
        font-size: 13px;
        border: 1px solid #dddddd;
    }

    .stage-approved {
        background: #e9f8ef;
        border-color: #b7e4c7;
    }

    .stage-current {
        background: #fff4d6;
        border-color: #ffd166;
        font-weight: 700;
    }

    .stage-ready {
        background: #eaf2ff;
        border-color: #b6d4fe;
    }

    .stage-pending {
        background: #f7f7f7;
        color: #666;
    }

    .output-box {
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 24px;
        background: #ffffff;
        margin-top: 12px;
        margin-bottom: 18px;
    }

    .summary-box {
        border-left: 5px solid #2563eb;
        background: #eff6ff;
        padding: 14px 16px;
        border-radius: 10px;
        margin-bottom: 16px;
    }

    .main-hint-box {
        border-left: 5px solid #10b981;
        background: #ecfdf5;
        padding: 14px 16px;
        border-radius: 10px;
        margin-bottom: 16px;
    }

    .muted {
        color: #6b7280;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API HELPERS
# ============================================================

def api_get(path: str) -> Optional[Dict[str, Any]]:
    try:
        res = requests.get(f"{API_BASE}{path}", timeout=120)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not connect to backend: {exc}")
        return None

    if res.status_code >= 400:
        st.error(f"Backend error {res.status_code}: {res.text}")
        return None

    return res.json()


def api_post(path: str, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        res = requests.post(f"{API_BASE}{path}", json=data or {}, timeout=120)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not connect to backend: {exc}")
        return None

    if res.status_code >= 400:
        st.error(f"Backend error {res.status_code}: {res.text}")
        return None

    return res.json()


def rerun_app() -> None:
    st.rerun()


# ============================================================
# STATE HELPERS
# ============================================================

def get_stage_config(stage_id: str) -> Dict[str, str]:
    for stage in STAGES:
        if stage["id"] == stage_id:
            return stage
    return STAGES[0]


def get_current_job() -> Optional[Dict[str, Any]]:
    job_id = st.session_state.get("job_id")
    if not job_id:
        return None
    return api_get(f"/api/job/{job_id}")


def start_new_run() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    rerun_app()


def submit_new_job() -> None:
    idea_title = st.session_state.get("idea_title", "").strip()
    idea_text = st.session_state.get("idea_text", "").strip()

    if not idea_title and not idea_text:
        st.warning("Please add at least an idea title or idea description.")
        return

    payload = {
        "journey_type": "idea",
        "payload": {
            "idea_title": st.session_state.get("idea_title", ""),
            "idea_text": st.session_state.get("idea_text", ""),
            "target_users": st.session_state.get("target_users", ""),
            "problem": st.session_state.get("problem", ""),
            "constraints": st.session_state.get("constraints", ""),
        },
    }

    result = api_post("/api/submit", payload)
    if not result:
        return

    job_id = result["job_id"]
    st.session_state.job_id = job_id
    st.session_state.selected_stage_id = "parse_submission"

    api_post(f"/api/stage/{job_id}/parse_submission/run")

    rerun_app()


def calculate_progress(job: Dict[str, Any]) -> Dict[str, Any]:
    stages = job["stages"]
    approved_count = sum(1 for s in stages.values() if s.get("approved"))

    current_index = job.get("current_stage_index", 0)
    current_index = max(0, min(current_index, len(STAGES) - 1))

    return {
        "approved_count": approved_count,
        "current_index": current_index,
        "current_stage_id": STAGES[current_index]["id"],
        "progress_value": approved_count / len(STAGES),
    }


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar(job: Optional[Dict[str, Any]]) -> None:
    with st.sidebar:
        st.header("🧭 Nextify Control Center")

        if not job:
            st.info("Submit an idea to start.")
            return

        selected_stage_id = st.session_state.get(
            "selected_stage_id",
            STAGES[job["current_stage_index"]]["id"],
        )
        selected_stage = get_stage_config(selected_stage_id)
        selected_state = job["stages"][selected_stage_id]

        st.markdown("### Current Stage")
        st.markdown(f"**{selected_stage['title']}**")
        st.caption(selected_stage["agent"])

        st.divider()

        st.markdown("### Workflow History")

        for stage in STAGES:
            sid = stage["id"]
            state = job["stages"][sid]

            if state.get("approved"):
                icon = "✅"
            elif state.get("agent_output"):
                icon = "🔵"
            else:
                icon = "⚪"

            if st.button(
                f"{icon} {stage['short_title']}",
                key=f"sidebar_select_{sid}",
                use_container_width=True,
            ):
                st.session_state.selected_stage_id = sid
                rerun_app()

        st.divider()

        st.markdown("### Selected Stage Details")
        st.markdown(f"**{selected_stage['title']}**")
        st.caption(f"Status: {selected_state.get('status', 'pending')}")
        st.caption(f"Revisions: {selected_state.get('revision_count', 0)}")

        with st.expander("🤖 Current final agent output", expanded=False):
            if selected_state.get("agent_output"):
                st.markdown(selected_state["agent_output"])
            else:
                st.info("No agent output yet.")

        with st.expander("🕘 Previous versions", expanded=True):
            previous_outputs = selected_state.get("previous_outputs", [])
            if not previous_outputs:
                st.info("No previous versions yet.")
            else:
                for item in reversed(previous_outputs):
                    version = item.get("version", "?")
                    reason = item.get("reason", "Previous version")
                    with st.expander(f"Version {version} — {reason}", expanded=False):
                        st.markdown(item.get("output", ""))

        with st.expander("⚖️ LLM judge feedback", expanded=False):
            if selected_state.get("judge_feedback"):
                st.markdown(selected_state["judge_feedback"])
            else:
                st.info("No judge feedback yet.")

        with st.expander("💬 Human feedback history", expanded=False):
            history = selected_state.get("user_feedback_history", [])
            if not history:
                st.info("No human feedback saved yet.")
            else:
                for item in reversed(history):
                    version = item.get("version", "?")
                    st.markdown(f"**Feedback {version}**")
                    st.write(item.get("feedback", ""))


# ============================================================
# TOP PROGRESS
# ============================================================

def render_progress_header(job: Optional[Dict[str, Any]]) -> None:
    if not job:
        st.markdown("## 🧠 Nextify Interactive AI")
        st.caption("Submit your idea to start the agentic workflow.")
        return

    progress = calculate_progress(job)
    current_stage = STAGES[progress["current_index"]]

    st.markdown('<div class="top-card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2.2, 1, 1])

    with col1:
        st.markdown("## 🧠 Nextify Interactive AI")
        st.markdown(
            f"""
            <div class="muted">
            Current stage: <strong>{current_stage["title"]}</strong><br>
            Agent: <strong>{current_stage["agent"]}</strong><br>
            Job: <code>{job["job_id"]}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.metric("Approved", f"{progress['approved_count']} / {len(STAGES)}")

    with col3:
        st.metric("Stage", f"{progress['current_index'] + 1} / {len(STAGES)}")

    st.progress(progress["progress_value"])

    pill_html = ""

    for index, stage in enumerate(STAGES):
        sid = stage["id"]
        state = job["stages"][sid]

        if state.get("approved"):
            css = "stage-pill stage-approved"
            icon = "✅"
        elif index == progress["current_index"]:
            css = "stage-pill stage-current"
            icon = "🟡"
        elif state.get("agent_output"):
            css = "stage-pill stage-ready"
            icon = "🔵"
        else:
            css = "stage-pill stage-pending"
            icon = "⚪"

        pill_html += f'<span class="{css}">{icon} {stage["short_title"]}</span>'

    st.markdown(pill_html, unsafe_allow_html=True)

    if job.get("message"):
        st.caption(job["message"])

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SUBMISSION FORM
# ============================================================

def render_submission_form() -> None:
    with st.container(border=True):
        st.markdown("## ✨ Submit Your Idea")

        st.text_input("Idea Title", key="idea_title")
        st.text_area("Idea Description", key="idea_text", height=140)
        st.text_input("Target Users", key="target_users")
        st.text_area("Problem", key="problem", height=120)
        st.text_input("Constraints", key="constraints")

        st.button(
            "🚀 Submit & Start Workflow",
            type="primary",
            on_click=submit_new_job,
            use_container_width=True,
        )


# ============================================================
# MAIN WORKSPACE
# ============================================================

def render_stage_workspace(job: Dict[str, Any]) -> None:
    if "selected_stage_id" not in st.session_state:
        st.session_state.selected_stage_id = STAGES[job["current_stage_index"]]["id"]

    selected_stage_id = st.session_state.selected_stage_id
    stage = get_stage_config(selected_stage_id)
    state = job["stages"][selected_stage_id]

    stage_index = [s["id"] for s in STAGES].index(selected_stage_id)
    next_stage = STAGES[stage_index + 1] if stage_index < len(STAGES) - 1 else None

    st.markdown(f"## {stage['title']}")
    st.caption(f"Agent: {stage['agent']}")
    st.write(stage["desc"])

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        st.metric("Status", state.get("status", "pending"))

    with status_col2:
        st.metric("Revisions", state.get("revision_count", 0))

    with status_col3:
        st.metric("Approved", "Yes" if state.get("approved") else "No")

    if state.get("applied_feedback_summary"):
        st.markdown(
            f"""
            <div class="summary-box">
            <strong>Summary of latest change:</strong><br>
            {state["applied_feedback_summary"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # -----------------------------
    # FINAL OUTPUT ONLY IN MAIN WINDOW
    # -----------------------------
    st.markdown("### ✅ Current Final Output")

    if selected_stage_id == "parse_submission":
        st.markdown(
            """
            <div class="main-hint-box">
            This stage shows the reviewed and revised parsed idea. If you apply human or judge feedback,
            this final parsed idea will be regenerated and the previous version will move to the sidebar.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if state.get("agent_output"):
        st.markdown('<div class="output-box">', unsafe_allow_html=True)
        st.markdown(state["agent_output"])
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No output yet. Run this agent to generate the first version.")

    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        if st.button("▶️ Run / Fresh Draft", type="primary", use_container_width=True):
            api_post(f"/api/stage/{job['job_id']}/{selected_stage_id}/run")
            rerun_app()

    with action_col2:
        if st.button(
            "⚖️ Run LLM Judge",
            use_container_width=True,
            disabled=not bool(state.get("agent_output")),
        ):
            api_post(f"/api/stage/{job['job_id']}/{selected_stage_id}/judge")
            rerun_app()

    with action_col3:
        if st.button(
            "✅ Approve & Continue",
            use_container_width=True,
            disabled=not bool(state.get("agent_output")),
        ):
            api_post(f"/api/stage/{job['job_id']}/{selected_stage_id}/approve")

            if next_stage:
                st.session_state.selected_stage_id = next_stage["id"]

            rerun_app()

    st.divider()

    # -----------------------------
    # FEEDBACK FORM
    # -----------------------------
    st.markdown("### 💬 Revise Current Output")

    st.caption(
        "Write feedback, then choose how to apply it. The box clears after submit. "
        "Previous versions and feedback history stay in the sidebar."
    )

    with st.form(
        key=f"feedback_form_{selected_stage_id}",
        clear_on_submit=True,
    ):
        feedback_text = st.text_area(
            "Your feedback",
            height=120,
            placeholder=(
                "Example: Revise the parsed form to be clearer, more practical, "
                "and more specific. Add assumptions, risks, MVP boundary, and next step."
            ),
        )

        form_col1, form_col2, form_col3 = st.columns(3)

        with form_col1:
            apply_human = st.form_submit_button(
                "🔁 Apply my feedback",
                use_container_width=True,
                disabled=not bool(state.get("agent_output")),
            )

        with form_col2:
            apply_judge = st.form_submit_button(
                "⚖️ Apply judge feedback",
                use_container_width=True,
                disabled=not bool(state.get("agent_output") and state.get("judge_feedback")),
            )

        with form_col3:
            apply_both = st.form_submit_button(
                "🧠 Apply both",
                use_container_width=True,
                disabled=not bool(state.get("agent_output")),
            )

    if apply_human:
        api_post(
            f"/api/stage/{job['job_id']}/{selected_stage_id}/revise",
            {
                "mode": "human_only",
                "feedback": feedback_text,
            },
        )
        rerun_app()

    if apply_judge:
        api_post(
            f"/api/stage/{job['job_id']}/{selected_stage_id}/revise",
            {
                "mode": "judge_only",
                "feedback": "",
            },
        )
        rerun_app()

    if apply_both:
        api_post(
            f"/api/stage/{job['job_id']}/{selected_stage_id}/revise",
            {
                "mode": "both",
                "feedback": feedback_text,
            },
        )
        rerun_app()

    st.caption("Only the final revised output appears above. Versions and feedback are kept in the sidebar.")

    st.divider()

    if next_stage:
        st.info(f"Next stage after approval: **{next_stage['title']}** — {next_stage['agent']}")
    else:
        st.success("This is the final stage. Approve it when the final report looks ready.")


# ============================================================
# APP ENTRY
# ============================================================

job = get_current_job()
render_sidebar(job)
render_progress_header(job)

if not job:
    render_submission_form()
else:
    with st.expander("✨ Submitted Idea", expanded=False):
        payload = job.get("payload", {})
        st.markdown(f"**Idea Title:** {payload.get('idea_title', '')}")
        st.markdown(f"**Idea Description:** {payload.get('idea_text', '')}")
        st.markdown(f"**Target Users:** {payload.get('target_users', '')}")
        st.markdown(f"**Problem:** {payload.get('problem', '')}")
        st.markdown(f"**Constraints:** {payload.get('constraints', '')}")

        st.button("🔄 Start New Run", on_click=start_new_run)

    render_stage_workspace(job)