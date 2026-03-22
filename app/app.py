# app/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


# === Config ===
st.set_page_config(
    page_title="Nextify Dashboard",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# === Load Data ===
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/all_experiment_view.csv")
        return df
    except:
        return pd.DataFrame()

df = load_data()

# === Sidebar Filters ===
st.sidebar.title("🌟 Nextify Control Panel")
companies = st.sidebar.multiselect("Select Companies", df["Company"].unique() if not df.empty else [])
strategies = st.sidebar.multiselect("Select Strategies", df["Strategy"].unique() if not df.empty else [])
prompt_tags = st.sidebar.multiselect("Select Prompt Tags", df["Prompt Tag"].unique() if not df.empty else [])

# === Apply Filters ===
filtered_df = df.copy()
if companies:
    filtered_df = filtered_df[filtered_df["Company"].isin(companies)]
if strategies:
    filtered_df = filtered_df[filtered_df["Strategy"].isin(strategies)]
if prompt_tags:
    filtered_df = filtered_df[filtered_df["Prompt Tag"].isin(prompt_tags)]

# === Tabs ===
tabs = st.tabs(["Overview", "Scores & Trends", "Prompt Table", "Multi-Agent System", "Embeddings & RAG"])

# === Tab 1: Overview ===
with tabs[0]:
    st.markdown("## 📊 Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Run Count", f"{filtered_df['Run'].nunique() if not df.empty else 0}")
    col2.metric("Average LLM Score", f"{filtered_df['LLM Score'].mean():.2f}" if not df.empty else "N/A")
    col3.metric("Average Human Score", f"{filtered_df['Human Score'].mean():.2f}" if not df.empty else "N/A")

    st.markdown("### 🧾 Evaluation Table")
    st.dataframe(filtered_df, use_container_width=True, height=400)

# === Tab 2: Scores & Trends ===
with tabs[1]:
    st.markdown("## 📊 Scores & Trends")

    if not filtered_df.empty:
        st.markdown(f"**Selected Prompt Tags:** {', '.join(filtered_df['Prompt Tag'].unique())}")

        # 🟦 Heatmap by Metric and Prompt
        pivot_metric = filtered_df.pivot_table(
            values=["LLM Score", "Human Score"],
            index="Section",
            columns="Prompt Tag",
            aggfunc="mean"
        )

        st.markdown("### 🔥 LLM Score Heatmap")
        fig_llm = px.imshow(pivot_metric["LLM Score"].fillna(0), text_auto=True, color_continuous_scale="blues")
        st.plotly_chart(fig_llm, use_container_width=True)

        st.markdown("### 🧠 Human Score Heatmap")
        fig_human = px.imshow(pivot_metric["Human Score"].fillna(0), text_auto=True, color_continuous_scale="greens")
        st.plotly_chart(fig_human, use_container_width=True)

        # 🎯 Combined Score per Section
        st.markdown("### 📊 Combined Score by Section (LLM + Human Average)")
        filtered_df["Combined Score"] = filtered_df[["LLM Score", "Human Score"]].mean(axis=1)
        avg_combined = filtered_df.groupby(["Section", "Prompt Tag"])["Combined Score"].mean().reset_index()
        fig_comb = px.bar(avg_combined, x="Section", y="Combined Score", color="Prompt Tag", barmode="group")
        st.plotly_chart(fig_comb, use_container_width=True)

# === Tab 3: Prompt Table ===
with tabs[2]:
    st.markdown("## 📋 Full Prompt Evaluation")
    selected_prompt = st.selectbox("Choose a Prompt Tag", df["Prompt Tag"].unique() if not df.empty else [])
    if selected_prompt:
        prompt_rows = df[df["Prompt Tag"] == selected_prompt]
        for _, row in prompt_rows.iterrows():
            with st.expander(f"🔹 {row['Section']}" if pd.notna(row['Section']) else "🔹 Section"):
                st.markdown(f"**Metric**: {row['Metric']}")
                st.markdown(f"**LLM Score**: {row['LLM Score']}")
                st.markdown(f"**Human Score**: {row['Human Score']}")
                st.markdown(f"**Feedback**: {row['Feedback']}")
                st.markdown(f"**Lesson**: {row['Lesson']}")
                st.markdown("---")
                st.markdown("**LLM Output Section**:")
                st.markdown(row["LLM Output Section"])

# === Tab 4: Multi-Agent (Preview) ===
with tabs[3]:
    st.markdown("## 🤖 Multi-Agent System (Preview)")
    st.info("This section will contain agent logs, recommendations, and real-time chaining.")
    st.markdown("**Planned agents:**")
    st.markdown("- Feature Ideator\n- Roadmap Generator\n- OKR Builder\n- Competitive Analyst")
    st.markdown("**Coming soon: Upload documents, run retrieval, view multi-agent flow.**")

# === Tab 5: Embeddings & RAG (Future) ===
with tabs[4]:
    st.markdown("## 🧠 Embeddings + Retrieval Augmented Generation")
    st.info("Upload files, view document embeddings, and test similarity-based prompting")
    uploaded_file = st.file_uploader("Upload Document (PDF, TXT)", type=["pdf", "txt"])
    if uploaded_file:
        st.success(f"Uploaded {uploaded_file.name}. Embedding + RAG view coming soon.")
