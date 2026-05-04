"""
Nextify Interactive UI

This Streamlit app provides a clear Human-in-the-Loop workflow.

Main area:
- Shows the current agent output.
- Lets the user run, revise, and approve stages.

Sidebar:
- Shows human feedback.
- Shows LLM judge feedback.
- Shows change summary from the latest revision.

Judge feedback is not displayed in the main interaction panel.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests
import streamlit as st


# ============================================================
# Config
# ============================================================

API_BASE = "http://127.0.0.1:8000"

STAGES = [
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


# ============================================================
# Page setup
# ============================================================

st.set_page_config(page_title="Nextify Interactive", layout="wide")

st.markdown(
    """
    <style>
    .nextify-topbar {
        position: sticky;
        top: 0;
        z-index: 999;
        background: white;
        border: 1px solid #e6e6e6;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .stage-chip {
        display: inline-block;
        padding: 6px 10px;
        margin: 4px 4px 4px 0;
        border-radius: 999px;
        border: 1px solid #ddd;
        font-size: 13px;
    }
    .stage-chip-current {
        background: #fff4d6;
        border-color: #ffd166;
        font-weight: 700;
    }
    .stage-chip-approved {
        background: #eaf8ee;
        border-color: #b7e4c7;
    }
    .stage-chip-ready {
        background: #eaf2ff;
        border-color: #b6d4fe;
    }
    .stage-chip-pending {
        background: #f6f6f6;
        color: #777;
    }
    .nextify-card {
        border: 1px solid #e6e6e6;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 16px;
        background: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Nextify Interactive AI")
st.caption("Agent-by-agent product development with LLM judge review and human feedback.")


# ============================================================
# API helpers
# ============================================================

def safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def api_get(path: str) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=60)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not connect to backend: {exc}")
        return None

    if response.status_code >= 400:
        st.error(f"API error {response.status_code}: {response.text}")
        return None

    return response.json()


def api_post(path: str, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        response = requests.post(f"{API_BASE}{path}", json=data or {}, timeout=60)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not connect to backend: {exc}")
        return None

    if response.status_code >= 400:
        st.error(f"API error {response.status_code}: {response.text}")
        return None

    return response.json()


# ============================================================
# Stage helpers
# ============================================================

def stage_ids() -> list[str]:
    return [stage["id"] for stage in STAGES]


def get_stage_config(stage_id: str) -> Dict[str, str]:
    for stage in STAGES:
        if stage["id"] == stage_id:
            return stage
    return STAGES[0]


def get_next_stage_id(stage_id: str) -> Optional[str]:
    ids = stage_ids()
    if stage_id not in ids:
        return None

    index = ids.index(stage_id)
    if index >= len(ids) - 1:
        return None

    return ids[index + 1]


def get_current_job() -> Optional[Dict[str, Any]]:
    job_id = st.session_state.get("job_id")
    if not job_id:
        return None
    return api_get(f"/api/job/{job_id}")


def render_progress_header(job: Optional[Dict[str, Any]]) -> None:
    if not job:
        current_stage_index = 0
        approved_count = 0
        current_stage = STAGES[0]
        job_id = "No active job"
    else:
        current_stage_index = job.get("current_stage_index", 0)
        approved_count = sum(
            1 for stage_state in job["stages"].values()
            if stage_state.get("approved")
        )
        current_stage = STAGES[current_stage_index]
        job_id = job.get("job_id", "Unknown")

    st.markdown('<div class="nextify-topbar">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2.4, 1, 1])

    with col1:
        st.markdown("### 🧭 Nextify Progress")
        st.markdown(
            f"""
            **Current stage:** {current_stage["title"]}  
            **Agent:** {current_stage["agent"]}  
            **Job:** `{job_id}`
            """
        )

    with col2:
        st.metric("Approved", f"{approved_count} / {len(STAGES)}")

    with col3:
        st.metric("Step", f"{current_stage_index + 1} / {len(STAGES)}")

    st.progress(approved_count / len(STAGES))

    chip_html = ""

    for index, stage in enumerate(STAGES):
        state = job["stages"][stage["id"]] if job else {}

        if state.get("approved"):
            css_class = "stage-chip stage-chip-approved"
            icon = "✅"
        elif index == current_stage_index:
            css_class = "stage-chip stage-chip-current"
            icon = "🟡"
        elif state.get("agent_output"):
            css_class = "stage-chip stage-chip-ready"
            icon = "🔵"
        else:
            css_class = "stage-chip stage-chip-pending"
            icon = "⚪"

        chip_html += f'<span class="{css_class}">{icon} {stage["short_title"]}</span>'

    st.markdown(chip_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Submit/new run
# ============================================================

def submit_new_job() -> None:
    idea_title = st.session_state.get("idea_title", "")
    idea_text = st.session_state.get("idea_text", "")

    if not idea_title.strip() and not idea_text.strip():
        st.warning("Please provide at least an idea title or description.")
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

    api_post(f"/api/stage/{job_id}/parse_submission/run")
    safe_rerun()


def start_new_run() -> None:
    for key in [
        "job_id",
        "idea_title",
        "idea_text",
        "target_users",
        "problem",
        "constraints",
    ]:
        if key in st.session_state:
            del st.session_state[key]

    safe_rerun()


# ============================================================
# Main
# ============================================================

job = get_current_job()
render_progress_header(job)

if "job_id" not in st.session_state:
    with st.container(border=True):
        st.markdown("## ✨ Submit Your Idea")

        st.text_input("Idea Title", key="idea_title")
        st.text_area("Idea Description", key="idea_text", height=140)
        st.text_input("Target Users", key="target_users")
        st.text_area("Problem", key="problem", height=120)
        st.text_input("Constraints (e.g. solo founder, 3 months)", key="constraints")

        submitted = st.button(
            "🚀 Submit & Start Interactive Run",
            type="primary",
        )

        if submitted:
            submit_new_job()

    st.stop()


job = get_current_job()

if not job:
    st.error("Unable to load job state. Make sure FastAPI is running on http://127.0.0.1:8000.")
    st.stop()


with st.expander("✨ Submitted Idea", expanded=False):
    payload = job.get("payload", {})
    st.markdown(f"**Idea Title:** {payload.get('idea_title', '')}")
    st.markdown(f"**Idea Description:** {payload.get('idea_text', '')}")
    st.markdown(f"**Target Users:** {payload.get('target_users', '')}")
    st.markdown(f"**Problem:** {payload.get('problem', '')}")
    st.markdown(f"**Constraints:** {payload.get('constraints', '')}")
    st.button("🔄 Start New Run", on_click=start_new_run)


# ============================================================
# Sidebar: feedback and judge
# ============================================================

with st.sidebar:
    st.header("🧾 Review Panel")
    st.caption("Human feedback, judge feedback, and applied changes live here.")

    for stage in STAGES:
        stage_id = stage["id"]
        state = job["stages"][stage_id]

        has_any_sidebar_content = (
            bool(state.get("user_feedback"))
            or bool(state.get("judge_feedback"))
            or bool(state.get("change_summary"))
        )

        if has_any_sidebar_content:
            with st.expander(stage["title"], expanded=False):
                if state.get("user_feedback"):
                    st.markdown("### 🧑 Human Feedback")
                    st.markdown(state["user_feedback"])

                if state.get("judge_feedback"):
                    st.markdown("### ⚖️ LLM Judge Feedback")
                    st.markdown(state["judge_feedback"])

                if state.get("change_summary"):
                    st.markdown("### ✅ Applied Change Summary")
                    st.markdown(state["change_summary"])

    if not any(
        job["stages"][stage["id"]].get("user_feedback")
        or job["stages"][stage["id"]].get("judge_feedback")
        or job["stages"][stage["id"]].get("change_summary")
        for stage in STAGES
    ):
        st.info("No feedback or judge reviews yet.")


# ============================================================
# Stage workspace
# ============================================================

st.markdown("## 🧩 Stage-by-Stage Workspace")

tab_labels = []

for stage in STAGES:
    state = job["stages"][stage["id"]]

    if state.get("approved"):
        prefix = "✅"
    elif state.get("agent_output"):
        prefix = "🔵"
    else:
        prefix = "⚪"

    tab_labels.append(f"{prefix} {stage['short_title']}")

tabs = st.tabs(tab_labels)

for index, stage in enumerate(STAGES):
    stage_id = stage["id"]
    state = job["stages"][stage_id]
    next_stage_id = get_next_stage_id(stage_id)

    with tabs[index]:
        st.markdown('<div class="nextify-card">', unsafe_allow_html=True)

        st.subheader(stage["title"])
        st.caption(f"Agent: {stage['agent']}")
        st.write(stage["desc"])

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Status", state.get("status", "pending"))

        with col_b:
            st.metric("Approved", "Yes" if state.get("approved") else "No")

        with col_c:
            st.metric("Revisions", state.get("revision_count", 0))

        st.divider()

        st.markdown("### 1️⃣ Agent Output")

        if not state.get("agent_output"):
            st.info("This agent has not generated output yet.")

            if st.button(f"▶️ Run {stage['title']}", key=f"run_{stage_id}", type="primary"):
                api_post(f"/api/stage/{job['job_id']}/{stage_id}/run")
                safe_rerun()
        else:
            st.success("Agent output is ready.")

            with st.expander("🤖 View / hide agent output", expanded=True):
                st.markdown(state["agent_output"])

            if st.button("🔄 Rerun this agent from scratch", key=f"rerun_{stage_id}"):
                api_post(f"/api/stage/{job['job_id']}/{stage_id}/run")
                safe_rerun()

        st.divider()

        st.markdown("### 2️⃣ Feedback & Judge")

        st.caption("Judge feedback and your saved feedback will appear in the sidebar, not here.")

        feedback_key = f"feedback_{stage_id}"

        feedback = st.text_area(
            "Write or update your feedback for this stage.",
            value=state.get("user_feedback", ""),
            key=feedback_key,
            height=120,
            placeholder="Example: Make this more practical for a solo founder. Add MVP scope, risks, metrics, and next steps.",
        )

        fb_col_1, fb_col_2 = st.columns(2)

        with fb_col_1:
            if st.button("💾 Save my feedback", key=f"save_feedback_{stage_id}"):
                api_post(
                    f"/api/stage/{job['job_id']}/{stage_id}/feedback",
                    {"feedback": feedback},
                )
                safe_rerun()

        with fb_col_2:
            if st.button(
                "⚖️ Run LLM judge",
                key=f"judge_{stage_id}",
                disabled=not bool(state.get("agent_output")),
            ):
                api_post(f"/api/stage/{job['job_id']}/{stage_id}/judge")
                safe_rerun()

        if state.get("user_feedback") or state.get("judge_feedback"):
            st.info("Saved feedback and judge review are available in the sidebar.")

        st.divider()

        st.markdown("### 3️⃣ Apply Feedback / Revise")

        has_agent_output = bool(state.get("agent_output"))
        has_judge = bool(state.get("judge_feedback"))
        has_human = bool(feedback.strip() or state.get("user_feedback", "").strip())

        if not has_agent_output:
            st.warning("Run the agent first before applying feedback.")
        else:
            col_1, col_2, col_3 = st.columns(3)

            with col_1:
                if st.button(
                    "🔁 Apply my feedback",
                    key=f"apply_human_{stage_id}",
                    disabled=not has_human,
                ):
                    api_post(
                        f"/api/stage/{job['job_id']}/{stage_id}/feedback",
                        {"feedback": feedback},
                    )
                    api_post(
                        f"/api/stage/{job['job_id']}/{stage_id}/revise",
                        {"mode": "human_only"},
                    )
                    safe_rerun()

            with col_2:
                if st.button(
                    "⚖️ Apply judge feedback",
                    key=f"apply_judge_{stage_id}",
                    disabled=not has_judge,
                ):
                    api_post(
                        f"/api/stage/{job['job_id']}/{stage_id}/revise",
                        {"mode": "judge_only"},
                    )
                    safe_rerun()

            with col_3:
                if st.button(
                    "🧠 Apply both",
                    key=f"apply_both_{stage_id}",
                    disabled=not (has_human or has_judge),
                    type="primary",
                ):
                    api_post(
                        f"/api/stage/{job['job_id']}/{stage_id}/feedback",
                        {"feedback": feedback},
                    )
                    api_post(
                        f"/api/stage/{job['job_id']}/{stage_id}/revise",
                        {"mode": "both"},
                    )
                    safe_rerun()

            if state.get("change_summary"):
                st.success("Latest applied change summary is available in the sidebar.")

        st.divider()

        st.markdown("### 4️⃣ Approve and Continue")

        if not state.get("agent_output"):
            st.warning("Run the agent first before approving.")
        else:
            approve_label = "✅ Approve final stage" if not next_stage_id else "✅ Approve & prepare next agent"

            if st.button(approve_label, key=f"approve_{stage_id}", type="primary"):
                api_post(f"/api/stage/{job['job_id']}/{stage_id}/approve")
                safe_rerun()

            if next_stage_id:
                next_stage = get_stage_config(next_stage_id)
                st.info(
                    f"After approval, Nextify prepares: **{next_stage['title']}** "
                    f"({next_stage['agent']})."
                )
            else:
                st.info("This is the final stage.")

        st.markdown("</div>", unsafe_allow_html=True)