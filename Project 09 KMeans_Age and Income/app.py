import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="K-Means Clustering Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1 {
    color:#1f77b4;
    text-align:center;
}

.metric {
    background-color:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 0px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -------------------- Title --------------------
st.title("📊 K-Means Clustering Dashboard")
st.markdown("### Customer Segmentation using **Age** and **Income**")

# -------------------- Sidebar --------------------
st.sidebar.header("⚙️ Settings")

k = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=8,
    value=3
)

uploaded_file = st.sidebar.file_uploader(
    "Upload income.csv",
    type=["csv"]
)

# -------------------- Load Dataset --------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("income.csv")

# -------------------- Metrics --------------------
c1, c2, c3 = st.columns(3)

c1.metric("Rows", len(df))
c2.metric("Columns", len(df.columns))
c3.metric("Clusters", k)

st.divider()

# -------------------- Tabs --------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Dataset",
    "📍 Clustering",
    "📉 Elbow Method",
    "📈 Scaled Data"
])

# ====================================================
# TAB 1
# ====================================================
with tab1:

    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(df["Age"], df["Income($)"], color="dodgerblue", s=80)
    ax.set_xlabel("Age")
    ax.set_ylabel("Income")
    ax.set_title("Original Data")

    st.pyplot(fig)

# ====================================================
# TAB 2
# ====================================================
with tab2:

    scaler = MinMaxScaler()

    scaled = df.copy()
    scaled["Age"] = scaler.fit_transform(df[["Age"]])

    scaler2 = MinMaxScaler()
    scaled["Income($)"] = scaler2.fit_transform(df[["Income($)"]])

    km = KMeans(n_clusters=k, random_state=42)

    scaled["Cluster"] = km.fit_predict(
        scaled[["Age", "Income($)"]]
    )

    colors = [
        "red",
        "green",
        "blue",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray"
    ]

    fig, ax = plt.subplots(figsize=(8,6))

    for i in range(k):
        d = scaled[scaled.Cluster == i]

        ax.scatter(
            d["Age"],
            d["Income($)"],
            s=80,
            color=colors[i],
            label=f"Cluster {i+1}"
        )

    centers = km.cluster_centers_

    ax.scatter(
        centers[:,0],
        centers[:,1],
        marker="*",
        s=350,
        color="black",
        label="Centroids"
    )

    ax.set_title("K-Means Clustering")
    ax.set_xlabel("Scaled Age")
    ax.set_ylabel("Scaled Income")
    ax.legend()

    st.pyplot(fig)

# ====================================================
# TAB 3
# ====================================================
with tab3:

    sse=[]

    for i in range(1,11):
        model=KMeans(n_clusters=i, random_state=42)
        model.fit(scaled[["Age","Income($)"]])
        sse.append(model.inertia_)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(range(1,11), sse,
            marker='o',
            linewidth=3)

    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("SSE")
    ax.set_title("Elbow Method")

    st.pyplot(fig)

# ====================================================
# TAB 4
# ====================================================
with tab4:

    st.subheader("Scaled Dataset")

    st.dataframe(scaled, use_container_width=True)

    st.write("### Cluster Centers")

    st.dataframe(
        pd.DataFrame(
            km.cluster_centers_,
            columns=["Scaled Age","Scaled Income"]
        )
    )

st.success("✅ Dashboard Loaded Successfully")