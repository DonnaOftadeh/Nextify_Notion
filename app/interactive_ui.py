import streamlit as st
import requests

# ============================================================
# CONFIG
# ============================================================

API_BASE = "http://127.0.0.1:8000"

STAGES = [
    {
        "id": "parse_submission",
        "title": "Parse Submission",
        "short_title": "Parse",
        "agent": "Input Parser Agent",
        "description": "Validate and structure the user's submitted idea, product, company, or industry context.",
    },
    {
        "id": "brainstorm_parallel",
        "title": "Brainstorm Parallel",
        "short_title": "Brainstorm",
        "agent": "Market Analysis Agent + Crazy Idea Agent",
        "description": "Generate both market-grounded opportunities and unconventional breakthrough ideas.",
    },
    {
        "id": "idea_cooker",
        "title": "Idea Cooker",
        "short_title": "Cooker",
        "agent": "Idea Cooker Agent",
        "description": "Synthesize the strongest strategic product direction from the brainstorming outputs.",
    },
    {
        "id": "theme_epic_generator",
        "title": "Theme & Epic Generator",
        "short_title": "Themes",
        "agent": "Theme & Epic Agent",
        "description": "Convert the selected direction into strategic themes and product epics.",
    },
    {
        "id": "roadmap_generator",
        "title": "Roadmap Generator",
        "short_title": "Roadmap",
        "agent": "Roadmap Agent",
        "description": "Turn themes and epics into a phased roadmap.",
    },
    {
        "id": "feature_generation",
        "title": "Feature Generation",
        "short_title": "Features",
        "agent": "Feature Generation Agent",
        "description": "Generate concrete product features based on the approved strategy and roadmap.",
    },
    {
        "id": "prioritization_rice",
        "title": "Prioritization & RICE",
        "short_title": "RICE",
        "agent": "Prioritization & RICE Agent",
        "description": "Score and prioritize features using the RICE framework.",
    },
    {
        "id": "okr_generation",
        "title": "OKR Generation",
        "short_title": "OKRs",
        "agent": "OKR Agent",
        "description": "Generate measurable objectives and key results from the approved product direction.",
    },
    {
        "id": "three_month_planner",
        "title": "Three-Month Planner",
        "short_title": "Planner",
        "agent": "Three-Month Planner Agent",
        "description": "Create a practical 3-month execution plan.",
    },
    {
        "id": "write_report_pdf",
        "title": "Write Report (PDF)",
        "short_title": "Report",
        "agent": "Report Writer Agent",
        "description": "Assemble the approved outputs into a final report and PDF-ready narrative.",
    },
]

# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(page_title="Nextify Interactive", layout="wide")

st.markdown(
    """
    <style>
    .nextify-progress-box {
        position: sticky;
        top: 0;
        z-index: 999;
        background: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }

    .stage-pill {
        display: inline-block;
        padding: 7px 10px;
        border-radius: 999px;
        margin: 4px 4px 4px 0;
        font-size: 13px;
        border: 1px solid #dddddd;
    }

    .stage-approved {
        background: #eaf8ee;
        border-color: #b7e4c7;
    }

    .stage-current {
        background: #fff4d6;
        border-color: #ffd166;
        font-weight: 700;
    }

    .stage-pending {
        background: #f6f6f6;
        color: #666666;
    }

    .stage-started {
        background: #eaf2ff;
        border-color: #b6d4fe;
    }

    .nextify-card {
        border: 1px solid #e6e6e6;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 16px;
        background: #ffffff;
    }

    .small-muted {
        color: #777777;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Nextify Interactive AI")
st.caption(
    "Agent-by-agent product development with LLM judge review and human-in-the-loop feedback."
)

# ============================================================
# API HELPERS
# ============================================================

def api_post(path, json_payload=None):
    try:
        response = requests.post(f"{API_BASE}{path}", json=json_payload, timeout=180)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not connect to backend: {exc}")
        return None

    if response.status_code >= 400:
        st.error(f"API error {response.status_code}: {response.text}")
        return None

    return response.json()


def api_get(path):
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=180)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not connect to backend: {exc}")
        return None

    if response.status_code >= 400:
        st.error(f"API error {response.status_code}: {response.text}")
        return None

    return response.json()

# ============================================================
# STATE HELPERS
# ============================================================

def get_stage_config(stage_id):
    for stage in STAGES:
        if stage["id"] == stage_id:
            return stage
    return STAGES[0]


def get_current_job():
    if "job_id" not in st.session_state:
        return None
    return api_get(f"/api/job/{st.session_state.job_id}")


def submit_and_start_interactive_run():
    if not st.session_state.idea_title.strip() and not st.session_state.idea_text.strip():
        st.warning("Please add at least an idea title or idea description.")
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

    submit_data = api_post("/api/submit", payload)

    if not submit_data:
        return

    job_id = submit_data["job_id"]
    st.session_state.job_id = job_id
    st.session_state.selected_stage_id = "parse_submission"

    # IMPORTANT FIX:
    # Immediately run the first stage so the user sees parsing output right away.
    api_post(f"/api/stage/{job_id}/parse_submission/run")

    st.success(f"✅ Interactive Nextify run started: {job_id}")
    st.rerun()


def start_new_run():
    keys_to_clear = [
        "job_id",
        "selected_stage_id",
        "idea_title",
        "idea_text",
        "target_users",
        "problem",
        "constraints",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()


def calculate_progress(job):
    if not job:
        return {
            "approved_count": 0,
            "total_count": len(STAGES),
            "progress_value": 0,
            "current_stage_id": STAGES[0]["id"],
            "current_stage_index": 0,
        }

    approved_count = 0

    for stage_config in STAGES:
        stage_id = stage_config["id"]
        stage_state = job["stages"].get(stage_id, {})

        if stage_state.get("approved"):
            approved_count += 1

    current_stage_index = job.get("current_stage_index", 0)

    if current_stage_index < 0:
        current_stage_index = 0

    if current_stage_index >= len(STAGES):
        current_stage_index = len(STAGES) - 1

    current_stage_id = STAGES[current_stage_index]["id"]
    progress_value = approved_count / len(STAGES)

    return {
        "approved_count": approved_count,
        "total_count": len(STAGES),
        "progress_value": progress_value,
        "current_stage_id": current_stage_id,
        "current_stage_index": current_stage_index,
    }


def render_progress_header(job):
    progress = calculate_progress(job)
    current_stage = get_stage_config(progress["current_stage_id"])

    if job:
        job_id_text = job.get("job_id", "Unknown")
        backend_status = job.get("status", "unknown")
        backend_progress = job.get("progress", 0)
        backend_message = job.get("message", "")
        backend_step = job.get("step", "")
    else:
        job_id_text = "No active job yet"
        backend_status = "not started"
        backend_progress = 0
        backend_message = ""
        backend_step = ""

    st.markdown('<div class="nextify-progress-box">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([2.4, 1.1, 1.1, 1.2])

    with col1:
        st.markdown("### 🧭 Nextify Progress")
        st.markdown(
            f"""
            <div class="small-muted">
            Current HITL stage: <strong>{current_stage["title"]}</strong><br>
            Agent: <strong>{current_stage["agent"]}</strong><br>
            Backend step: <strong>{backend_step}</strong><br>
            Job: <code>{job_id_text}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.metric(
            "Approved",
            f"{progress['approved_count']} / {progress['total_count']}",
        )

    with col3:
        st.metric(
            "HITL step",
            f"{progress['current_stage_index'] + 1} / {len(STAGES)}",
        )

    with col4:
        st.metric(
            "Backend",
            f"{backend_progress}%",
            backend_status,
        )

    st.progress(progress["progress_value"])

    if backend_message:
        st.caption(f"Backend pipeline message: {backend_message}")

    pill_html = ""

    for stage_config in STAGES:
        stage_id = stage_config["id"]

        if job:
            stage_state = job["stages"].get(stage_id, {})
            is_approved = stage_state.get("approved", False)
            has_output = bool(stage_state.get("agent_output"))
        else:
            is_approved = False
            has_output = False

        if is_approved:
            css_class = "stage-pill stage-approved"
            icon = "✅"
        elif stage_id == progress["current_stage_id"]:
            css_class = "stage-pill stage-current"
            icon = "🟡"
        elif has_output:
            css_class = "stage-pill stage-started"
            icon = "🔵"
        else:
            css_class = "stage-pill stage-pending"
            icon = "⚪"

        pill_html += (
            f'<span class="{css_class}">'
            f'{icon} {stage_config["short_title"]}'
            f"</span>"
        )

    st.markdown(pill_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TOP PROGRESS AREA
# ============================================================

job = get_current_job()
render_progress_header(job)

# ============================================================
# IDEA FORM
# ============================================================

if "job_id" not in st.session_state:
    with st.container(border=True):
        st.markdown("## ✨ Submit Your Idea")

        st.text_input("Idea Title", key="idea_title")
        st.text_area("Idea Description", key="idea_text")
        st.text_input("Target Users", key="target_users")
        st.text_area("Problem", key="problem")
        st.text_input("Constraints (e.g. solo founder, 3 months)", key="constraints")

        st.button(
            "🚀 Submit & Start Interactive Nextify Run",
            on_click=submit_and_start_interactive_run,
            type="primary",
        )
else:
    with st.expander("✨ Submitted Idea", expanded=False):
        current_job = get_current_job()

        if current_job:
            payload = current_job.get("payload", {})
            st.markdown(f"**Idea Title:** {payload.get('idea_title', '')}")
            st.markdown(f"**Idea Description:** {payload.get('idea_text', '')}")
            st.markdown(f"**Target Users:** {payload.get('target_users', '')}")
            st.markdown(f"**Problem:** {payload.get('problem', '')}")
            st.markdown(f"**Constraints:** {payload.get('constraints', '')}")

        st.button("🔄 Start New Run", on_click=start_new_run)

job = get_current_job()

if not job:
    st.info("Submit an idea to start the interactive agent workflow.")
    st.stop()

# ============================================================
# SIDEBAR JUDGE PANEL
# ============================================================

if "selected_stage_id" not in st.session_state:
    st.session_state.selected_stage_id = STAGES[0]["id"]

selected_stage_id = st.session_state.selected_stage_id
selected_stage_state = job["stages"].get(selected_stage_id, {})
selected_stage_config = get_stage_config(selected_stage_id)

with st.sidebar:
    st.header("⚖️ LLM Judge Panel")
    st.caption("Judge comments for the currently selected stage.")

    st.markdown(f"### {selected_stage_config['title']}")
    st.markdown(f"**Agent:** {selected_stage_config['agent']}")

    if selected_stage_state.get("judge_feedback"):
        st.markdown(selected_stage_state["judge_feedback"])
    else:
        st.info("No judge feedback yet. Run the stage, then click **Judge**.")

# ============================================================
# STAGE TABS — WIZARD STYLE
# ============================================================

st.markdown("## 🧩 Stage-by-Stage Workspace")

tab_titles = []

for stage_config in STAGES:
    stage_id = stage_config["id"]
    stage_state = job["stages"].get(stage_id, {})

    if stage_state.get("approved"):
        icon = "✅"
    elif stage_state.get("agent_output"):
        icon = "🔵"
    else:
        icon = "⚪"

    tab_titles.append(f"{icon} {stage_config['short_title']}")

stage_tabs = st.tabs(tab_titles)


def get_next_stage_id(current_stage_id):
    stage_ids = [stage["id"] for stage in STAGES]

    if current_stage_id not in stage_ids:
        return None

    current_index = stage_ids.index(current_stage_id)

    if current_index >= len(stage_ids) - 1:
        return None

    return stage_ids[current_index + 1]


def run_stage(job_id, stage_id):
    return api_post(f"/api/stage/{job_id}/{stage_id}/run")


def judge_stage(job_id, stage_id):
    return api_post(f"/api/stage/{job_id}/{stage_id}/judge")


def save_feedback(job_id, stage_id, feedback):
    return api_post(
        f"/api/stage/{job_id}/{stage_id}/feedback",
        {"feedback": feedback},
    )


def revise_stage(job_id, stage_id, feedback=None):
    if feedback is not None:
        save_feedback(job_id, stage_id, feedback)

    return api_post(f"/api/stage/{job_id}/{stage_id}/revise")


def approve_stage(job_id, stage_id):
    return api_post(f"/api/stage/{job_id}/{stage_id}/approve")


for index, stage_config in enumerate(STAGES):
    stage_id = stage_config["id"]
    stage_state = job["stages"][stage_id]
    next_stage_id = get_next_stage_id(stage_id)

    with stage_tabs[index]:
        st.session_state.selected_stage_id = stage_id

        st.markdown("### Current Agent")
        st.subheader(stage_config["title"])
        st.caption(f"Agent: {stage_config['agent']}")
        st.write(stage_config["description"])

        status = stage_state.get("status", "pending")
        approved = stage_state.get("approved", False)
        revision_count = stage_state.get("revision_count", 0)

        status_col_1, status_col_2, status_col_3 = st.columns(3)

        with status_col_1:
            st.metric("Stage status", status)

        with status_col_2:
            st.metric("Approved", "Yes" if approved else "No")

        with status_col_3:
            st.metric("Revisions", revision_count)

        st.divider()

        # ----------------------------------------------------
        # STEP 1 — GENERATE AGENT OUTPUT
        # ----------------------------------------------------
        st.markdown("## 1️⃣ Generate this agent output")

        if not stage_state.get("agent_output"):
            st.info("This agent has not generated an output yet.")

            if st.button(
                f"▶️ Generate {stage_config['title']}",
                key=f"generate_{stage_id}",
                type="primary",
            ):
                run_stage(job["job_id"], stage_id)
                st.rerun()
        else:
            st.success("Agent output is ready.")

            with st.expander("🤖 View / hide agent output", expanded=True):
                st.markdown(stage_state["agent_output"])

            rerun_col_1, rerun_col_2 = st.columns([1, 3])

            with rerun_col_1:
                if st.button(
                    "🔄 Rerun agent from scratch",
                    key=f"rerun_from_scratch_{stage_id}",
                ):
                    run_stage(job["job_id"], stage_id)
                    st.rerun()

            with rerun_col_2:
                st.caption(
                    "Use this if you want a fresh draft for this stage. "
                    "It does not automatically approve the output."
                )

        st.divider()

        # ----------------------------------------------------
        # STEP 2 — JUDGE OUTPUT
        # Judge output is NOT shown here. It only appears in sidebar.
        # ----------------------------------------------------
        st.markdown("## 2️⃣ Ask the LLM judge")

        if not stage_state.get("agent_output"):
            st.warning("Generate the agent output first before asking the judge.")
        else:
            judge_col_1, judge_col_2, judge_col_3 = st.columns([1.2, 1.2, 2])

            with judge_col_1:
                if st.button(
                    "⚖️ Run judge",
                    key=f"run_judge_{stage_id}",
                ):
                    judge_stage(job["job_id"], stage_id)
                    st.session_state.selected_stage_id = stage_id
                    st.rerun()

            with judge_col_2:
                if stage_state.get("judge_feedback"):
                    if st.button(
                        "🙈 Ignore judge",
                        key=f"ignore_judge_{stage_id}",
                    ):
                        # We do not delete it from backend; we just mark preference in UI state.
                        st.session_state[f"ignore_judge_{stage_id}"] = True
                        st.success("Judge feedback ignored for your next revision.")
                else:
                    st.button(
                        "🙈 Ignore judge",
                        key=f"ignore_judge_disabled_{stage_id}",
                        disabled=True,
                    )

            with judge_col_3:
                if stage_state.get("judge_feedback"):
                    st.info(
                        "Judge feedback is available in the left sidebar only. "
                        "Use it manually, ignore it, or revise with it."
                    )
                else:
                    st.caption("No judge feedback yet.")

        st.divider()

        # ----------------------------------------------------
        # STEP 3 — HUMAN FEEDBACK
        # ----------------------------------------------------
        st.markdown("## 3️⃣ Add your own feedback")

        feedback_key = f"feedback_{stage_id}"

        user_feedback = st.text_area(
            "Write what you want the agent to change.",
            key=feedback_key,
            value=stage_state.get("user_feedback", ""),
            height=140,
            placeholder=(
                "Example: Make it more practical for a solo founder. "
                "Add risks, scientific feasibility, MVP scope, and clearer next steps."
            ),
        )

        feedback_col_1, feedback_col_2 = st.columns([1, 3])

        with feedback_col_1:
            if st.button("💾 Save my feedback", key=f"save_user_feedback_{stage_id}"):
                save_feedback(job["job_id"], stage_id, user_feedback)
                st.rerun()

        with feedback_col_2:
            if stage_state.get("user_feedback"):
                st.success("Human feedback is saved.")
            else:
                st.caption("You can revise with your feedback, judge feedback, or both.")

        st.divider()

        # ----------------------------------------------------
        # STEP 4 — REVISION MODE
        # ----------------------------------------------------
        st.markdown("## 4️⃣ Revise this stage")

        if not stage_state.get("agent_output"):
            st.warning("Generate the agent output first before revising.")
        else:
            has_judge = bool(stage_state.get("judge_feedback"))
            has_human = bool(user_feedback.strip() or stage_state.get("user_feedback"))

            rev_col_1, rev_col_2, rev_col_3 = st.columns(3)

            with rev_col_1:
                if st.button(
                    "🔁 Revise with my feedback only",
                    key=f"revise_human_only_{stage_id}",
                    disabled=not has_human,
                ):
                    save_feedback(job["job_id"], stage_id, user_feedback)
                    # Temporarily ignore judge in UI state
                    st.session_state[f"revision_mode_{stage_id}"] = "human_only"
                    revise_stage(job["job_id"], stage_id, user_feedback)
                    st.rerun()

            with rev_col_2:
                if st.button(
                    "⚖️ Revise with judge only",
                    key=f"revise_judge_only_{stage_id}",
                    disabled=not has_judge,
                ):
                    st.session_state[f"revision_mode_{stage_id}"] = "judge_only"
                    revise_stage(job["job_id"], stage_id, stage_state.get("user_feedback", ""))
                    st.rerun()

            with rev_col_3:
                if st.button(
                    "🧠 Revise with both",
                    key=f"revise_both_{stage_id}",
                    disabled=not (has_judge or has_human),
                    type="primary",
                ):
                    save_feedback(job["job_id"], stage_id, user_feedback)
                    st.session_state[f"revision_mode_{stage_id}"] = "both"
                    revise_stage(job["job_id"], stage_id, user_feedback)
                    st.rerun()

            st.caption(
                "After revision, review the new output above. "
                "You can judge it again, add more feedback, revise again, or approve."
            )

        st.divider()

        # ----------------------------------------------------
        # STEP 5 — APPROVE AND CONTINUE
        # ----------------------------------------------------
        st.markdown("## 5️⃣ Decision")

        decision_col_1, decision_col_2 = st.columns([1.4, 2.6])

        with decision_col_1:
            if next_stage_id:
                approve_label = "✅ Approve & go to next agent"
            else:
                approve_label = "✅ Approve final stage"

            if st.button(
                approve_label,
                key=f"approve_and_continue_{stage_id}",
                disabled=not bool(stage_state.get("agent_output")),
                type="primary",
            ):
                approve_stage(job["job_id"], stage_id)

                if next_stage_id:
                    # Automatically prepare next stage so the user understands what comes next.
                    st.session_state.selected_stage_id = next_stage_id
                    st.success("Stage approved. Move to the next tab/agent.")
                else:
                    st.success("Final stage approved.")

                st.rerun()

        with decision_col_2:
            if next_stage_id:
                next_stage_config = get_stage_config(next_stage_id)
                st.info(
                    f"After approval, continue with: "
                    f"**{next_stage_config['title']}** "
                    f"({next_stage_config['agent']})."
                )
            else:
                st.info("This is the final stage. After approval, the workflow is complete.")