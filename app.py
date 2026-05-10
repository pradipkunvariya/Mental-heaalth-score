import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Clustering Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():

    df = pd.read_csv("student_mental_health_v3.csv")

    # Fill null values
    df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].median())
    df["screen_time"] = df["screen_time"].fillna(df["screen_time"].median())
    df["study_hours"] = df["study_hours"].fillna(df["study_hours"].median())

    return df

df = load_data()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.title("🧠 Mental Health Clustering")

st.sidebar.markdown("---")

st.sidebar.subheader("Filters")

gender_options = sorted(df["gender"].unique())
gender_sel = st.sidebar.multiselect(
    "Gender",
    gender_options,
    default=gender_options
)

edu_options = sorted(df["education"].unique())
edu_sel = st.sidebar.multiselect(
    "Education",
    edu_options,
    default=edu_options
)

age_min = int(df["age"].min())
age_max = int(df["age"].max())

age_sel = st.sidebar.slider(
    "Age Range",
    age_min,
    age_max,
    (age_min, age_max)
)

# Number of clusters
n_clusters = st.sidebar.slider(
    "Number of Clusters (K)",
    2,
    8,
    4
)

# ─────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────
filtered = df[
    (df["gender"].isin(gender_sel)) &
    (df["education"].isin(edu_sel)) &
    (df["age"].between(age_sel[0], age_sel[1]))
].copy()

# ─────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────
st.title("🧠 Student Mental Health Clustering Dashboard")

st.markdown("""
This dashboard uses **Unsupervised Machine Learning**
to discover hidden mental health patterns among students.
""")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
freq_map = {
    "Never": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
    "Always": 5
}

filtered["stress_frequency"] = filtered["stress_frequency"].map(freq_map)
filtered["anxiety_frequency"] = filtered["anxiety_frequency"].map(freq_map)

# Encode categorical columns
cat_cols = ["gender", "education", "diet", "exercise"]

encoder = LabelEncoder()

for col in cat_cols:
    filtered[col] = encoder.fit_transform(filtered[col])

# Features for clustering
features = [
    "age",
    "sleep_hours",
    "screen_time",
    "study_hours",
    "stress_frequency",
    "anxiety_frequency",
    "diet",
    "exercise"
]

X = filtered[features]

# Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────────────────────
# ELBOW METHOD
# ─────────────────────────────────────────────────────────────
inertia = []

for i in range(1, 11):

    km = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    km.fit(X_scaled)

    inertia.append(km.inertia_)

# ─────────────────────────────────────────────────────────────
# FINAL KMEANS
# ─────────────────────────────────────────────────────────────
kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

filtered["Cluster"] = kmeans.fit_predict(X_scaled)

# ─────────────────────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────────────────────
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

filtered["PCA1"] = X_pca[:, 0]
filtered["PCA2"] = X_pca[:, 1]

# ─────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Students", len(filtered))
k2.metric("Clusters Found", n_clusters)
k3.metric("Avg Sleep", f"{filtered['sleep_hours'].mean():.1f} hrs")
k4.metric("Avg Screen Time", f"{filtered['screen_time'].mean():.1f} hrs")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🧩 Cluster Analysis",
    "📉 PCA Visualization",
    "📋 Data Explorer",
    "💡 Insights"
])

COLORS = px.colors.qualitative.Vivid

# ════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
with tab1:

    st.subheader("Mental Health Feature Distributions")

    c1, c2 = st.columns(2)

    with c1:

        fig = px.histogram(
            filtered,
            x="sleep_hours",
            nbins=20,
            color_discrete_sequence=["#7c83fd"],
            title="Sleep Hours Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:

        fig = px.histogram(
            filtered,
            x="screen_time",
            nbins=20,
            color_discrete_sequence=["#f97316"],
            title="Screen Time Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Correlation Heatmap")

    corr = filtered[features].corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        title="Correlation Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — CLUSTER ANALYSIS
# ════════════════════════════════════════════════════════════
with tab2:

    st.subheader("Cluster Distribution")

    cluster_counts = (
        filtered["Cluster"]
        .value_counts()
        .reset_index()
    )

    cluster_counts.columns = ["Cluster", "Count"]

    fig = px.bar(
        cluster_counts,
        x="Cluster",
        y="Count",
        color="Cluster",
        color_discrete_sequence=COLORS,
        title="Students in Each Cluster"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Feature Comparison")

    cluster_summary = (
        filtered
        .groupby("Cluster")[features]
        .mean()
        .round(2)
    )

    st.dataframe(cluster_summary, use_container_width=True)

    st.subheader("Average Lifestyle Habits per Cluster")

    melted = (
        cluster_summary[
            ["sleep_hours", "screen_time", "study_hours"]
        ]
        .reset_index()
        .melt(
            id_vars="Cluster",
            var_name="Feature",
            value_name="Value"
        )
    )

    fig = px.bar(
        melted,
        x="Feature",
        y="Value",
        color="Cluster",
        barmode="group",
        title="Lifestyle Comparison by Cluster"
    )

    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — PCA VISUALIZATION
# ════════════════════════════════════════════════════════════
with tab3:

    st.subheader("PCA Cluster Visualization")

    fig = px.scatter(
        filtered,
        x="PCA1",
        y="PCA2",
        color=filtered["Cluster"].astype(str),
        hover_data=[
            "sleep_hours",
            "screen_time",
            "study_hours"
        ],
        title="Student Mental Health Clusters (PCA)",
        color_discrete_sequence=COLORS
    )

    fig.update_traces(marker=dict(size=10))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Elbow Method")

    elbow_df = pd.DataFrame({
        "K": range(1, 11),
        "Inertia": inertia
    })

    fig = px.line(
        elbow_df,
        x="K",
        y="Inertia",
        markers=True,
        title="Optimal Cluster Selection"
    )

    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 4 — DATA EXPLORER
# ════════════════════════════════════════════════════════════
with tab4:

    st.subheader("Clustered Dataset")

    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(filtered["Cluster"].unique())
    )

    cluster_df = filtered[
        filtered["Cluster"] == selected_cluster
    ]

    st.dataframe(
        cluster_df,
        use_container_width=True,
        height=500
    )

    st.subheader("Download Clustered Data")

    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name="mental_health_clusters.csv",
        mime="text/csv"
    )

# ════════════════════════════════════════════════════════════
# TAB 5 — INSIGHTS
# ════════════════════════════════════════════════════════════
with tab5:

    st.subheader("AI-Based Cluster Insights")

    for cluster in sorted(filtered["Cluster"].unique()):

        temp = filtered[filtered["Cluster"] == cluster]

        avg_sleep = temp["sleep_hours"].mean()
        avg_screen = temp["screen_time"].mean()
        avg_stress = temp["stress_frequency"].mean()

        st.markdown(f"## 🧩 Cluster {cluster}")

        if avg_sleep < 6:
            st.warning(
                "😴 Students in this cluster show low sleep patterns."
            )

        if avg_screen > 6:
            st.warning(
                "📱 High screen time detected in this cluster."
            )

        if avg_stress > 3.5:
            st.warning(
                "😰 Stress levels are significantly high."
            )

        if avg_sleep >= 7 and avg_stress < 3:
            st.success(
                "✅ Students in this cluster appear balanced and healthy."
            )

        st.info(
            f"""
            Average Sleep: {avg_sleep:.1f} hrs

            Average Screen Time: {avg_screen:.1f} hrs

            Average Stress Level: {avg_stress:.1f}
            """
        )

        st.markdown("---")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
---
### 📌 Technologies Used
- Streamlit
- Scikit-learn
- K-Means Clustering
- PCA
- Plotly
- Pandas
""")
