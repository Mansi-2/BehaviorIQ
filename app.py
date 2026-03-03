import streamlit as st
from data_loader import load_data
from session_builder import build_sessions
from model import train_model

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title="Digital Productivity Intelligence",
    page_icon="📊",
    layout="centered"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
}

section[data-testid="stSidebar"] {
    background: #0f172a;
}

.sidebar-title {
    color: white;
    font-size: 17px;
    font-weight: 600;
    margin-top: 25px;
    margin-bottom: 10px;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    padding: 10px;
    font-size: 15px;
    font-weight: 600;
    border: 2px solid #3b82f6;
    background-color: #1e293b;
    color: white;
}

.stButton>button:hover {
    background-color: #3b82f6;
    color: white;
}

h1 {
    font-size: 34px !important;
    font-weight: 800 !important;
}

h2 {
    font-size: 22px !important;
    font-weight: 700 !important;
}

h3 {
    font-size: 18px !important;
}

p {
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)

home_btn = st.sidebar.button("🏠 Overview")
model_btn = st.sidebar.button("🧠 Model Insights")
analysis_btn = st.sidebar.button("📊 Interactive Analysis")

if "page" not in st.session_state:
    st.session_state.page = "home"

if home_btn:
    st.session_state.page = "home"
if model_btn:
    st.session_state.page = "model"
if analysis_btn:
    st.session_state.page = "analysis"

# ---------------- LOAD DATA ----------------

df = load_data("data/usage_events_cleaned.csv")
sessions = build_sessions(df)

if len(sessions) > 0:
    sessions["duration_minutes"] = sessions["duration"] / 60

# ---------------- OVERVIEW ----------------

if st.session_state.page == "home":

    st.title("Digital Productivity Intelligence")

    st.write("""
    This system analyzes behavioral patterns in your device usage
    and identifies distraction tendencies and focus stability.
    """)

    if len(sessions) > 0:

        col1, col2, col3 = st.columns(3)

        col1.metric("Sessions", len(sessions))
        col2.metric("Events", len(df))
        col3.metric("Dopamine Ratio", f"{sessions['dopamine'].mean():.2f}")

        st.markdown("### Usage Composition")

        donut_df = pd.DataFrame({
            "Type": ["Dopamine", "Productive"],
            "Ratio": [
                sessions["dopamine"].mean(),
                1 - sessions["dopamine"].mean()
            ]
        })

        fig = px.pie(donut_df, values="Ratio", names="Type", hole=0.6)
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Insight:**  
        A higher dopamine ratio indicates that a large portion of sessions
        are entertainment-driven rather than task-driven.
        """)

# ---------------- MODEL ----------------

elif st.session_state.page == "model":

    st.title("Behavioral Model Insights")

    accuracy, cm, feature_importance = train_model(df)

    if accuracy is None:
        st.error("Not enough session data to train the model.")
    else:

        col1, col2 = st.columns(2)
        col1.metric("Model Accuracy", f"{accuracy:.2f}")
        col2.metric("Total Sessions", len(sessions))

        st.markdown("### Feature Influence")

        fig_bar, ax = plt.subplots(figsize=(6,3))
        sns.barplot(
            x="Coefficient",
            y="Feature",
            data=feature_importance,
            ax=ax
        )
        plt.tight_layout()
        st.pyplot(fig_bar)

        st.markdown("""
        **Insight:**  
        The feature with the larger absolute coefficient has stronger
        influence on predicting dopamine-heavy sessions.
        """)

        st.markdown("### Confusion Matrix")

        fig_cm, ax_cm = plt.subplots(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        plt.tight_layout()
        st.pyplot(fig_cm)

        st.markdown("""
        **Insight:**  
        This matrix shows how well the model distinguishes productive
        vs dopamine sessions. Balanced diagonal values indicate better performance.
        """)

# ---------------- ANALYSIS ----------------

elif st.session_state.page == "analysis":

    st.title("Interactive Behavioral Analysis")

    time_scale = st.selectbox("Time Scale", ["Hourly", "Weekly", "Monthly"])
    feature = st.selectbox(
        "Metric",
        ["Average Session Duration", "Total Usage", "Dopamine Ratio", "Session Count"]
    )

    if time_scale == "Hourly":
        group_col = "hour"
    elif time_scale == "Weekly":
        group_col = "weekday"
        ordered_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        sessions["weekday"] = pd.Categorical(
            sessions["weekday"],
            categories=ordered_days,
            ordered=True
        )
    else:
        group_col = "month"

    if feature == "Average Session Duration":
        data = sessions.groupby(group_col)["duration_minutes"].mean()
    elif feature == "Total Usage":
        data = sessions.groupby(group_col)["duration_minutes"].sum()
    elif feature == "Dopamine Ratio":
        data = sessions.groupby(group_col)["dopamine"].mean()
    else:
        data = sessions.groupby(group_col).size()

    data = data.reset_index()

    fig, ax = plt.subplots(figsize=(6,3))
    sns.barplot(x=group_col, y=data.columns[1], data=data, ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Behavioral Insight")

    if feature == "Dopamine Ratio":
        st.write("Higher peaks suggest time blocks dominated by entertainment usage.")
    elif feature == "Session Count":
        st.write("Frequent short sessions indicate fragmented attention.")
    elif feature == "Average Session Duration":
        st.write("Longer sessions may represent deep work or prolonged distraction.")
    else:
        st.write("Usage peaks reveal your highest engagement intensity periods.")