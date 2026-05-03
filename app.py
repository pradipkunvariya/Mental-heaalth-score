import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Mental Health",
    page_icon="🧠",
    layout="wide",
)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("student_mental_health_v3.csv")
    df["sleep_hours"]  = df["sleep_hours"].fillna(df["sleep_hours"].median())
    df["screen_time"]  = df["screen_time"].fillna(df["screen_time"].median())
    df["study_hours"]  = df["study_hours"].fillna(df["study_hours"].median())
    return df

df = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.title("🧠 Mental Health App")
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

gender_options = sorted(df["gender"].unique().tolist())
gender_sel = st.sidebar.multiselect("Gender", gender_options, default=gender_options)

edu_options = sorted(df["education"].unique().tolist())
edu_sel = st.sidebar.multiselect("Education", edu_options, default=edu_options)

age_min, age_max = int(df["age"].min()), int(df["age"].max())
age_sel = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

score_sel = st.sidebar.slider("Mental Health Score", 1, 10, (1, 10))

# ── Apply filters ──────────────────────────────────────────────────────────────
filtered = df[
    df["gender"].isin(gender_sel) &
    df["education"].isin(edu_sel) &
    df["age"].between(age_sel[0], age_sel[1]) &
    df["mental_health_score"].between(score_sel[0], score_sel[1])
].copy()

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("🧠 Student Mental Health Dashboard")
st.caption(f"Showing **{len(filtered)}** of **{len(df)}** students")
st.markdown("---")

# ── KPI Cards ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Avg MH Score",    f"{filtered['mental_health_score'].mean():.1f} / 10")
k2.metric("Avg Sleep",       f"{filtered['sleep_hours'].mean():.1f} hrs")
k3.metric("Avg Screen Time", f"{filtered['screen_time'].mean():.1f} hrs")
k4.metric("Avg Study Hours", f"{filtered['study_hours'].mean():.1f} hrs")
k5.metric("Exercise Rate",   f"{(filtered['exercise']=='Yes').mean()*100:.0f}%")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "👥 Demographics",
    "😰 Stress & Anxiety",
    "💤 Lifestyle",
    "📋 Data Explorer",
])

COLORS = px.colors.qualitative.Vivid
FREQ_ORDER = ["Never", "Rarely", "Sometimes", "Often", "Always"]

# ══════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Score Distribution")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            filtered, x="mental_health_score", nbins=8,
            title="Mental Health Score Distribution",
            labels={"mental_health_score": "Score", "count": "Count"},
            color_discrete_sequence=["#7c83fd"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(
            filtered, x="education", y="mental_health_score",
            color="education", color_discrete_sequence=COLORS,
            title="Score by Education Level",
            labels={"mental_health_score": "Score", "education": ""},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Score vs Sleep & Screen Time")
    c3, c4 = st.columns(2)

    with c3:
        fig = px.scatter(
            filtered, x="sleep_hours", y="mental_health_score",
            color="gender", color_discrete_sequence=COLORS,
            title="Sleep Hours vs Mental Health Score",
            labels={"sleep_hours": "Sleep Hours", "mental_health_score": "Score"},
        )
        x_vals = filtered["sleep_hours"].values
        y_vals = filtered["mental_health_score"].values
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode="lines", name="Trend",
            line=dict(color="white", width=2, dash="dash"),
        ))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.scatter(
            filtered, x="screen_time", y="mental_health_score",
            color="diet", color_discrete_sequence=COLORS,
            title="Screen Time vs Mental Health Score",
            labels={"screen_time": "Screen Time (hrs)", "mental_health_score": "Score"},
        )
        x_vals = filtered["screen_time"].values
        y_vals = filtered["mental_health_score"].values
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode="lines", name="Trend",
            line=dict(color="white", width=2, dash="dash"),
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    num_cols = ["age", "sleep_hours", "screen_time", "study_hours", "mental_health_score"]
    corr = filtered[num_cols].corr().round(2)
    fig = px.imshow(
        corr, text_auto=True,
        color_continuous_scale="RdBu",
        title="Numeric Feature Correlations",
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 2 – DEMOGRAPHICS
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Population Breakdown")
    c1, c2, c3 = st.columns(3)

    with c1:
        counts = filtered["gender"].value_counts()
        fig = px.pie(values=counts.values, names=counts.index,
                     title="Gender", hole=0.4,
                     color_discrete_sequence=COLORS)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        counts = filtered["education"].value_counts()
        fig = px.pie(values=counts.values, names=counts.index,
                     title="Education Level", hole=0.4,
                     color_discrete_sequence=COLORS)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        counts = filtered["diet"].value_counts()
        fig = px.pie(values=counts.values, names=counts.index,
                     title="Diet Quality", hole=0.4,
                     color_discrete_sequence=COLORS)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Score by Group")
    c4, c5 = st.columns(2)

    with c4:
        avg = filtered.groupby("gender")["mental_health_score"].mean().reset_index()
        fig = px.bar(avg, x="gender", y="mental_health_score",
                     color="gender", color_discrete_sequence=COLORS,
                     title="Avg Score by Gender",
                     labels={"mental_health_score": "Avg Score", "gender": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        avg = filtered.groupby("education")["mental_health_score"].mean().reset_index()
        fig = px.bar(avg, x="education", y="mental_health_score",
                     color="education", color_discrete_sequence=COLORS,
                     title="Avg Score by Education",
                     labels={"mental_health_score": "Avg Score", "education": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Age Distribution")
    fig = px.histogram(
        filtered, x="age", color="education",
        nbins=10, barmode="overlay", opacity=0.75,
        color_discrete_sequence=COLORS,
        title="Age Distribution by Education Level",
        labels={"age": "Age"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 3 – STRESS & ANXIETY
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("Frequency Distributions")
    c1, c2 = st.columns(2)

    with c1:
        counts = (
            filtered["stress_frequency"]
            .value_counts()
            .reindex(FREQ_ORDER, fill_value=0)
            .reset_index()
        )
        counts.columns = ["Frequency", "Count"]
        fig = px.bar(counts, x="Frequency", y="Count",
                     color="Frequency", color_discrete_sequence=COLORS,
                     title="Stress Frequency")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        counts = (
            filtered["anxiety_frequency"]
            .value_counts()
            .reindex(FREQ_ORDER, fill_value=0)
            .reset_index()
        )
        counts.columns = ["Frequency", "Count"]
        fig = px.bar(counts, x="Frequency", y="Count",
                     color="Frequency", color_discrete_sequence=COLORS,
                     title="Anxiety Frequency")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Impact on Mental Health Score")
    c3, c4 = st.columns(2)

    with c3:
        avg = (
            filtered.groupby("stress_frequency")["mental_health_score"]
            .mean()
            .reindex(FREQ_ORDER)
            .reset_index()
        )
        avg.columns = ["Stress Frequency", "Avg Score"]
        fig = px.line(avg, x="Stress Frequency", y="Avg Score",
                      markers=True, color_discrete_sequence=["#7c83fd"],
                      title="MH Score vs Stress Frequency")
        fig.update_traces(line_width=3, marker_size=10)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        avg = (
            filtered.groupby("anxiety_frequency")["mental_health_score"]
            .mean()
            .reindex(FREQ_ORDER)
            .reset_index()
        )
        avg.columns = ["Anxiety Frequency", "Avg Score"]
        fig = px.line(avg, x="Anxiety Frequency", y="Avg Score",
                      markers=True, color_discrete_sequence=["#f97316"],
                      title="MH Score vs Anxiety Frequency")
        fig.update_traces(line_width=3, marker_size=10)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Stress × Anxiety Heatmap")
    pivot = (
        filtered.groupby(["stress_frequency", "anxiety_frequency"])["mental_health_score"]
        .mean()
        .unstack(fill_value=0)
        .reindex(index=FREQ_ORDER, columns=FREQ_ORDER, fill_value=0)
    )
    fig = px.imshow(
        pivot.values,
        x=FREQ_ORDER, y=FREQ_ORDER,
        text_auto=".1f",
        color_continuous_scale="Viridis",
        title="Avg MH Score by Stress & Anxiety Frequency",
        labels={"x": "Anxiety Frequency", "y": "Stress Frequency", "color": "Avg Score"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 4 – LIFESTYLE
# ══════════════════════════════════════════════════════════
with tab4:
    st.subheader("Exercise & Diet Impact")
    c1, c2 = st.columns(2)

    with c1:
        ex_avg = filtered.groupby("exercise")["mental_health_score"].mean().reset_index()
        fig = px.bar(ex_avg, x="exercise", y="mental_health_score",
                     color="exercise",
                     color_discrete_map={"Yes": "#5ce6a0", "No": "#f97316"},
                     title="Avg MH Score: Exercise vs No Exercise",
                     labels={"mental_health_score": "Avg Score", "exercise": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        diet_avg = filtered.groupby("diet")["mental_health_score"].mean().reset_index()
        fig = px.bar(diet_avg, x="diet", y="mental_health_score",
                     color="diet", color_discrete_sequence=COLORS,
                     title="Avg MH Score by Diet Quality",
                     labels={"mental_health_score": "Avg Score", "diet": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sleep & Study Patterns")
    c3, c4 = st.columns(2)

    with c3:
        fig = px.violin(
            filtered, y="sleep_hours", x="diet",
            color="diet", box=True, points="all",
            color_discrete_sequence=COLORS,
            title="Sleep Hours by Diet Quality",
            labels={"sleep_hours": "Sleep (hrs)", "diet": ""},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.scatter(
            filtered, x="study_hours", y="mental_health_score",
            color="education", color_discrete_sequence=COLORS,
            title="Study Hours vs MH Score",
            labels={"study_hours": "Study Hours/day", "mental_health_score": "Score"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Hours Comparison")
    melted = filtered[["screen_time", "study_hours", "sleep_hours", "gender"]].melt(
        id_vars="gender", var_name="Activity", value_name="Hours"
    )
    avg_act = melted.groupby(["gender", "Activity"])["Hours"].mean().reset_index()
    fig = px.bar(
        avg_act, x="Activity", y="Hours", color="gender",
        barmode="group", color_discrete_sequence=COLORS,
        title="Avg Daily Hours by Activity & Gender",
        labels={"Hours": "Avg Hours"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 5 – DATA EXPLORER
# ══════════════════════════════════════════════════════════
with tab5:
    st.subheader("Raw Data Table")

    c1, c2, c3 = st.columns(3)
    sort_col = c1.selectbox("Sort by", filtered.columns.tolist(), index=10)
    sort_asc = c2.radio("Order", ["Ascending", "Descending"], horizontal=True) == "Ascending"
    rows = c3.slider("Rows to show", 10, max(10, len(filtered)), min(50, max(10, len(filtered))))

    display_df = filtered.sort_values(sort_col, ascending=sort_asc).head(rows)
    st.dataframe(display_df, use_container_width=True, height=400)

    st.subheader("Summary Statistics")
    st.dataframe(filtered.describe().round(2), use_container_width=True)

    st.subheader("Download Data")
    c1, c2 = st.columns(2)
    with c1:
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name="filtered_mental_health.csv",
            mime="text/csv",
        )
    with c2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            filtered.to_excel(writer, index=False, sheet_name="Data")
            filtered.describe().to_excel(writer, sheet_name="Summary Stats")
        st.download_button(
            label="⬇️ Download Excel",
            data=buf.getvalue(),
            file_name="filtered_mental_health.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.subheader("Quick Insights")
    i1, i2, i3 = st.columns(3)
    with i1:
        best_diet = filtered.groupby("diet")["mental_health_score"].mean().idxmax()
        st.info(f"🥗 **Best diet for MH:** {best_diet}")
        ex_yes = filtered[filtered["exercise"] == "Yes"]["mental_health_score"].mean()
        ex_no  = filtered[filtered["exercise"] == "No"]["mental_health_score"].mean()
        st.info(f"🏃 **Exercise benefit:** +{ex_yes - ex_no:.2f} pts")
    with i2:
        best_edu = filtered.groupby("education")["mental_health_score"].mean().idxmax()
        st.info(f"🎓 **Best MH by education:** {best_edu}")
        sleep_corr = filtered["sleep_hours"].corr(filtered["mental_health_score"])
        st.info(f"💤 **Sleep × MH correlation:** {sleep_corr:.3f}")
    with i3:
        high_s = filtered[filtered["screen_time"] > filtered["screen_time"].median()]["mental_health_score"].mean()
        low_s  = filtered[filtered["screen_time"] <= filtered["screen_time"].median()]["mental_health_score"].mean()
        st.info(f"📱 **High vs low screen time:** {high_s:.2f} vs {low_s:.2f}")
        top_gender = filtered.groupby("gender")["mental_health_score"].mean().idxmax()
        st.info(f"👤 **Best MH by gender:** {top_gender}")
