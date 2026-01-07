import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import autoprepml

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AutoPrepML",
    page_icon="💻",
    layout="wide"
)

sns.set_theme(style="white")
plt.rcParams["figure.dpi"] = 110

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: 800;
}
.subtitle {
    font-size: 18px;
    color: #6b7280;
}
.card {
    padding: 22px;
    border-radius: 14px;
    background-color: #ffffff;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="big-title">👨🏻‍💻 AutoPrepML</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Raw data → Clean → ML-ready in seconds</div>', unsafe_allow_html=True)
st.write("")

# ================= UPLOAD =================
uploaded_file = st.file_uploader("📤 Upload Dataset (CSV / Excel)", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)

    # ================= TABS =================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Raw Data",
        "📊 Auto EDA",
        "⚙️ Preprocessing",
        "🧪 ML-Ready Data",
        "⬇️ Download"
    ])

    # ================= TAB 1: RAW DATA =================
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Raw Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ================= TAB 2: AUTO EDA =================
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Auto EDA")

        target_eda = st.selectbox(
            "🎯 Select target column for EDA",
            df.columns
        )

        col1, col2 = st.columns(2)

        # ---- Missing Values ----
        with col1:
            st.markdown("**Missing Values (%)**")

            missing = (df.isnull().mean() * 100).round(2)
            missing = missing[missing > 0]

            if not missing.empty:
                fig, ax = plt.subplots(figsize=(5, 3))
                missing.sort_values().plot(kind="barh", ax=ax)
                ax.set_xlabel("Percentage")
                st.pyplot(fig)
            else:
                st.success("No missing values found")

            if df[target_eda].isnull().sum() > 0:
                st.warning(
                    f"⚠ Target column `{target_eda}` contains missing values "
                    f"({df[target_eda].isnull().sum()} rows)"
                )
            else:
                st.success(f"Target column `{target_eda}` has no missing values")

        # ---- Correlation ----
        with col2:
            st.markdown("**Correlation (Numeric Features)**")
            num_df = df.select_dtypes(include=["int64", "float64"])

            if target_eda in num_df.columns:
                corr = num_df.corr()[[target_eda]].sort_values(by=target_eda, ascending=False)

                fig, ax = plt.subplots(figsize=(5, 3))
                corr.drop(target_eda).head(10).plot(kind="barh", ax=ax)
                ax.set_title(f"Correlation with {target_eda}")
                st.pyplot(fig)
            else:
                st.info("Target is categorical – correlation not applicable")

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= TAB 3: PREPROCESS =================
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Run Preprocessing")

        target = st.selectbox("🎯 Select target column", df.columns)

        if st.button("🚀 Run AutoPrepML"):
            with st.spinner("Cleaning & transforming dataset..."):
                result = autoprepml(df, target)

            st.session_state["result"] = result

            st.success("Preprocessing completed")

            st.metric("Data Quality Score", f"{result['quality_score']} / 100")

            for check in result["quality_checks"]:
                st.success(check)

            comp = pd.DataFrame({
                "Raw": [result["outliers_before"], result["raw_features"]],
                "Processed": [result["outliers_after"], result["processed_features"]]
            }, index=["Outliers", "Features"])

            st.dataframe(comp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ================= TAB 4: ML READY =================
    with tab4:
        if "result" in st.session_state:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("ML-Ready Dataset")
            st.caption("Encoded, scaled & ready for training")
            st.dataframe(
                st.session_state["result"]["X_train"].head(10),
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Run preprocessing first")

    # ================= TAB 5: DOWNLOAD =================
    with tab5:
     if "result" in st.session_state:
        r = st.session_state["result"]

        st.subheader("⬇ Download Files")

        # X_train always exists
        st.download_button(
            "⬇ Download X_train.csv",
            r["X_train"].to_csv(index=False),
            "X_train.csv",
            mime="text/csv"
        )

        # X_test only in supervised mode
        if r["X_test"] is not None:
            st.download_button(
                "⬇ Download X_test.csv",
                r["X_test"].to_csv(index=False),
                "X_test.csv",
                mime="text/csv"
            )

        # y_train only in supervised mode
        if r["y_train"] is not None:
            st.download_button(
                "⬇ Download y_train.csv",
                r["y_train"].to_csv(index=False),
                "y_train.csv",
                mime="text/csv"
            )

        # y_test only in supervised mode
        if r["y_test"] is not None:
            st.download_button(
                "⬇ Download y_test.csv",
                r["y_test"].to_csv(index=False),
                "y_test.csv",
                mime="text/csv"
            )

        # Mode info (VERY IMPORTANT UX)
        st.info(f"🧠 Preprocessing Mode: **{r['mode']}**")
