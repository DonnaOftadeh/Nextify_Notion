"""
Nextify Interactive UI

This Streamlit application provides a user interface for the interactive
Nextify workflow.  Each stage of the process is displayed in its own tab
with a simple wizard‑style sequence:

1. Generate the agent output for the stage.
2. Run the LLM judge to critique the output (feedback appears in the sidebar).
3. Provide your own feedback and save it.
4. Revise the output using human feedback, judge feedback, or both.
5. Approve the stage and automatically continue to the next one.

Judge feedback is only shown in the sidebar.  The main panel remains
uncluttered and focuses on the core workflow actions.
"""

import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

# Define the stages so the UI can iterate through them.  This must match the
# backend configuration in app/main.py.
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
        "agent": "Market Analysis + Crazy Idea",
        "desc": "Generate market‑grounded opportunities and unconventional breakthrough ideas.",
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


def api_get(path: str):
    try:
        res = requests.get(f"{API_BASE}{path}", timeout=60)
        if res.status_code < 400:
            return res.json()
    except Exception:
        return None
    return None


def api_post(path: str, data: Optional[dict] = None):
    try:
        res = requests.post(f"{API_BASE}{path}", json=data or {}, timeout=60)
        if res.status_code < 400:
            return res.json()
    except Exception:
        return None
    return None


st.set_page_config(page_title="Nextify Interactive", layout="wide")
st.title("🧠 Nextify Interactive AI")
st.caption("Agent‑by‑agent product development with LLM judge review and human feedback.")


def submit_new_job():
    """Handle submission of a new interactive job."""
    if not st.session_state.idea_title.strip() and not st.session_state.idea_text.strip():
        st.warning("Please provide at least a title or description.")
        return
    payload = {
        "journey_type": "idea",
        "payload": {
            "idea_title": st.session_state.idea_title,
            "idea_text": st.session_state.idea_text,
            "target_users": st.session_state.target_users,
            "problem": st.session_state.problem,
            "constraints": st.session_state.constraints,
        },
    }
    res = api_post("/api/submit", payload)
    if res:
        st.session_state.job_id = res.get("job_id")
        # Auto‑run first stage (Parse Submission)
        api_post(f"/api/stage/{st.session_state.job_id}/parse_submission/run")
        st.experimental_rerun()


if "job_id" not in st.session_state:
    # Show submission form
    st.header("Submit Your Idea")
    st.text_input("Idea Title", key="idea_title")
    st.text_area("Idea Description", key="idea_text")
    st.text_input("Target Users", key="target_users")
    st.text_area("Problem", key="problem")
    st.text_input("Constraints (e.g. solo founder, 3 months)", key="constraints")
    st.button("🚀 Submit & Start", on_click=submit_new_job)
else:
    # Display interactive workflow
    job = api_get(f"/api/job/{st.session_state.job_id}")
    if not job:
        st.error("Unable to load job state.")
    else:
        # Progress info
        current_stage_index = job["current_stage_index"]
        approved_count = sum(1 for s in job["stages"].values() if s["approved"])
        st.info(f"Current stage: **{STAGES[current_stage_index]['title']}** | Approved: {approved_count} / {len(STAGES)}")
        # Sidebar judge feedback
        with st.sidebar:
            st.header("⚖️ Judge Reviews")
            judge_options = [s["id"] for s in STAGES if job["stages"][s["id"]]["judge_feedback"]]
            selected_review = st.selectbox(
                "Select a stage", judge_options,
                format_func=lambda x: STAGES[[s["id"] for s in STAGES].index(x)]["title"]
            ) if judge_options else None
            if selected_review:
                st.markdown(job["stages"][selected_review]["judge_feedback"])

        # Build tab labels with indicators
        tab_labels = []
        for stage in STAGES:
            state = job["stages"][stage["id"]]
            if state["approved"]:
                prefix = "✅"
            elif state["agent_output"]:
                prefix = "🔵"
            else:
                prefix = "⚪"
            tab_labels.append(f"{prefix} {stage['short_title']}")
        tabs = st.tabs(tab_labels)
        # Loop through each stage tab
        for idx, stage in enumerate(STAGES):
            sid = stage["id"]
            state = job["stages"][sid]
            next_sid = STAGES[idx + 1]["id"] if idx < len(STAGES) - 1 else None
            with tabs[idx]:
                st.subheader(stage["title"])
                st.caption(f"Agent: {stage['agent']}")
                st.write(stage["desc"])

                # Step 1: Generate or view output
                st.markdown("### 1. Generate output")
                if not state["agent_output"]:
                    if st.button("▶️ Run agent", key=f"run_{sid}"):
                        api_post(f"/api/stage/{job['job_id']}/{sid}/run")
                        st.experimental_rerun()
                else:
                    st.success("Output ready.")
                    with st.expander("View agent output"):
                        st.markdown(state["agent_output"])
                    if st.button("🔄 Re‑run agent", key=f"rerun_{sid}"):
                        api_post(f"/api/stage/{job['job_id']}/{sid}/run")
                        st.experimental_rerun()

                # Step 2: Judge
                st.markdown("### 2. Ask LLM judge")
                if state["agent_output"]:
                    if st.button("⚖️ Run judge", key=f"judge_{sid}"):
                        api_post(f"/api/stage/{job['job_id']}/{sid}/judge")
                        st.experimental_rerun()
                    if state["judge_feedback"]:
                        st.caption("Judge feedback available in sidebar.")

                # Step 3: User feedback
                st.markdown("### 3. Provide feedback")
                feedback_val = st.text_area(
                    "What should change?", value=state.get("user_feedback", ""), key=f"fb_{sid}"
                )
                if st.button("💾 Save feedback", key=f"save_{sid}"):
                    api_post(f"/api/stage/{job['job_id']}/{sid}/feedback", {"feedback": feedback_val})
                    st.success("Feedback saved.")
                    st.experimental_rerun()

                # Step 4: Revise
                st.markdown("### 4. Revise")
                if state["agent_output"]:
                    has_judge = bool(state["judge_feedback"])
                    has_human = bool(feedback_val.strip())
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.button(
                            "🔁 Human only", key=f"rev_h_{sid}", disabled=not has_human,
                            on_click=lambda sid=sid, feedback=feedback_val: (api_post(f"/api/stage/{job['job_id']}/{sid}/revise", {"mode": "human_only"}) or st.experimental_rerun())
                        )
                    with col2:
                        st.button(
                            "⚖️ Judge only", key=f"rev_j_{sid}", disabled=not has_judge,
                            on_click=lambda sid=sid: (api_post(f"/api/stage/{job['job_id']}/{sid}/revise", {"mode": "judge_only"}) or st.experimental_rerun())
                        )
                    with col3:
                        st.button(
                            "🧠 Both", key=f"rev_b_{sid}", disabled=not (has_judge or has_human),
                            on_click=lambda sid=sid, feedback=feedback_val: (api_post(f"/api/stage/{job['job_id']}/{sid}/revise", {"mode": "both"}) or st.experimental_rerun())
                        )
                    st.caption("After revising, you may judge again, add more feedback, revise further, or approve.")

                # Step 5: Approve and continue
                st.markdown("### 5. Approve & continue")
                if state["agent_output"]:
                    label = "✅ Approve final" if next_sid is None else "✅ Approve & next"
                    def do_approve(stage_id=sid):
                        api_post(f"/api/stage/{job['job_id']}/{stage_id}/approve")
                        st.success("Stage approved.")
                        st.experimental_rerun()
                    st.button(label, key=f"approve_{sid}", on_click=do_approve)