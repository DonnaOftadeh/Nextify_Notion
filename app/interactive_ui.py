import streamlit as st
import requests
import time

st.set_page_config(page_title="Nextify Interactive", layout="wide")

st.title("🧠 Nextify Interactive AI")

# -------------------------
# IDEA FORM
# -------------------------
st.markdown("## ✨ Submit Your Idea")

idea_title = st.text_input("Idea Title")
idea_text = st.text_area("Idea Description")
target_users = st.text_input("Target Users")
problem = st.text_area("Problem")
constraints = st.text_input("Constraints (e.g. solo founder, 3 months)")

# -------------------------
# SUBMIT BUTTON
# -------------------------
if st.button("🚀 Run Brainstorming"):

    payload = {
        "journey_type": "idea",
        "payload": {
            "idea_title": idea_title,
            "idea_text": idea_text,
            "target_users": target_users,
            "problem": problem,
            "constraints": constraints
        }
    }

    res = requests.post("http://127.0.0.1:8000/api/submit", json=payload)

    if res.status_code != 200:
        st.error("❌ Failed to submit")
    else:
        job_id = res.json()["job_id"]
        st.session_state["job_id"] = job_id
        st.success(f"✅ Job started: {job_id}")

# -------------------------
# POLLING STATUS
# -------------------------
if "job_id" in st.session_state:

    st.markdown("## ⚙️ Agent Progress")

    status_placeholder = st.empty()

    while True:
        status = requests.get(
            f"http://127.0.0.1:8000/api/status/{st.session_state['job_id']}"
        ).json()

        status_placeholder.markdown(f"""
        **Status:** {status['status']}  
        **Step:** {status['step']}  
        **Progress:** {status['progress']}%  
        **Message:** {status['message']}
        """)

        if status["ready"]:
            break

        time.sleep(2)

    # -------------------------
    # GET RAW OUTPUT
    # -------------------------
    st.markdown("## 🧠 Brainstorm Output")

    raw = requests.get(
        f"http://127.0.0.1:8000/api/debug/{st.session_state['job_id']}/raw"
    ).text

    st.markdown(raw)