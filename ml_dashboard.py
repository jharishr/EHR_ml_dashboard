# TEAM 25 - Assignment 2 Dashboard 
# J. Harish Rajeshwaran - 2023A3PS0393H 
# Divija - 2023A4PS0765H 
# Sanjana - 2023A4PS0662H 
# Utkarsh Anand - 2023AAPS0195H

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Automated Machine Learning Pipeline and Dashboard for Clinical Prediction under Temporal Shift in EHR Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# DATA
# -----------------------------
overview = {
    "Historical encounters": "(27191, 15)",
    "Current encounters": "(22594, 15)",
    "Target class 0": "2716 (96.21%)",
    "Target class 1": "107 (3.79%)",
    "Model df shape": "(2823, 307)",
    "Final train shape": "(2102, 249)",
    "Final test shape": "(1956, 249)",
    "Train balance": "0: 95.29%, 1: 4.71%",
    "Test balance": "0: 94.99%, 1: 5.01%",
    "Numeric columns": "249",
    "Categorical columns": "0",
}

model_metrics = pd.DataFrame([
    ["Decision Tree", "Historical Test", 0.997625, 1.000000, 0.95, 0.974359],
    ["Decision Tree", "Current Test", 1.000000, 1.000000, 1.00, 1.000000],
    ["SVM", "Historical Test", 0.926366, 0.333333, 0.55, 0.415094],
    ["SVM", "Current Test", 0.943878, 0.472222, 0.85, 0.607143],
    ["MLP", "Historical Test", 0.952494, 0.500000, 0.05, 0.090909],
    ["MLP", "Current Test", 0.979592, 1.000000, 0.60, 0.750000],
], columns=["Model", "Dataset", "Accuracy", "Precision", "Recall", "F1-score"])

train_test_metrics = pd.DataFrame([
    ["Decision Tree", 1.000000, 1.000000, 1.000000, 1.000000, 0.997625, 1.000000, 0.95, 0.974359],
    ["SVM", 0.955979, 0.516340, 1.000000, 0.681034, 0.926366, 0.333333, 0.55, 0.415094],
    ["MLP", 0.994051, 1.000000, 0.873418, 0.932432, 0.952494, 0.500000, 0.05, 0.090909],
], columns=[
    "Model", "Train Accuracy", "Train Precision", "Train Recall", "Train F1",
    "Test Accuracy", "Test Precision", "Test Recall", "Test F1"
])

conf_mats = {
    ("Decision Tree", "Historical Test"): [[401, 0], [1, 19]],
    ("Decision Tree", "Current Test"): [[372, 0], [0, 20]],
    ("SVM", "Historical Test"): [[379, 22], [9, 11]],
    ("SVM", "Current Test"): [[353, 19], [3, 17]],
    ("MLP", "Historical Test"): [[400, 1], [19, 1]],
    ("MLP", "Current Test"): [[372, 0], [8, 12]],
    ("Fine-tuned MLP", "Current Test"): [[368, 4], [18, 2]],
}

before_after = pd.DataFrame([
    ["Accuracy", 0.9796, 0.9464],
    ["Precision", 1.0000, 0.4828],
    ["Recall", 0.6000, 0.7000],
    ["F1-score", 0.7500, 0.5714],
], columns=["Metric", "Before CL", "After CL + Threshold Tuning"])

threshold_df = pd.DataFrame([
    [0.1, 0.923469, 0.386364, 0.85, 0.531250, 345, 27, 3, 17],
    [0.2, 0.946429, 0.482759, 0.70, 0.571429, 357, 15, 6, 14],
    [0.3, 0.948980, 0.500000, 0.50, 0.500000, 362, 10, 10, 10],
    [0.4, 0.951531, 0.538462, 0.35, 0.424242, 366, 6, 13, 7],
    [0.5, 0.943878, 0.333333, 0.10, 0.153846, 368, 4, 18, 2],
    [0.6, 0.941327, 0.200000, 0.05, 0.080000, 368, 4, 19, 1],
    [0.7, 0.948980, 0.500000, 0.05, 0.090909, 371, 1, 19, 1],
    [0.8, 0.948980, 0.000000, 0.00, 0.000000, 372, 0, 20, 0],
    [0.9, 0.948980, 0.000000, 0.00, 0.000000, 372, 0, 20, 0],
], columns=["Threshold", "Accuracy", "Precision", "Recall", "F1-score", "TN", "FP", "FN", "TP"])

feature_importance = pd.DataFrame([
    ["Dominant feature", 1.000],
    ["Feature 2", 0.000],
    ["Feature 3", 0.000],
    ["Feature 4", 0.000],
    ["Feature 5", 0.000],
], columns=["Feature", "Importance"])

# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: #0e1117;
        color: #f5f5f5;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .title-main {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .title-sub {
        font-size: 1.0rem;
        color: #c8c8c8;
        margin-bottom: 1rem;
    }
    .card {
        background: #161b22;
        border: 1px solid #2b313c;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.20);
        margin-bottom: 12px;
    }
    .card h3, .card h4 {
        margin: 0 0 8px 0;
        color: #ffffff;
    }
    .card p {
        margin: 0;
        color: #d0d0d0;
        line-height: 1.5;
    }
    section[data-testid="stSidebar"] {
        background: #0b0f14;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Dashboard Navigation")
section = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Data Summary",
        "Model Comparison",
        "Confusion Matrices",
        "Interpretability",
        "Continual Learning",
        "Threshold Tuning",
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Dark theme • Polished demo style • Results-first")

# -----------------------------
# OVERVIEW
# -----------------------------
if section == "Overview":
    st.markdown('<div class="title-main">Automated Machine Learning Pipeline and Dashboard for Clinical Prediction under Temporal Shift in EHR Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="title-sub">Interactive summary of the pipeline, model performance, and interpretability insights.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="card"><h4>Historical encounters</h4><p>{overview["Historical encounters"]}</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><h4>Current encounters</h4><p>{overview["Current encounters"]}</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card"><h4>Target imbalance</h4><p>{overview["Target class 0"]}<br>{overview["Target class 1"]}</p></div>', unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(f'<div class="card"><h4>Train / Test</h4><p>{overview["Final train shape"]}<br>{overview["Final test shape"]}</p></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="card"><h4>Model df shape</h4><p>{overview["Model df shape"]}</p></div>', unsafe_allow_html=True)
    with c6:
        st.markdown(f'<div class="card"><h4>Feature space</h4><p>Numeric: {overview["Numeric columns"]}<br>Categorical: {overview["Categorical columns"]}</p></div>', unsafe_allow_html=True)

    st.info(
        "The dataset is strongly imbalanced, so recall and F1-score are especially important for judging whether the models "
        "detect the minority class reliably."
    )

    st.success(
        "Final takeaway: Continual learning alone degraded minority-class performance. Threshold tuning significantly improved "
        "recall and F1-score, demonstrating the importance of post-training calibration in imbalanced classification tasks."
    )

# -----------------------------
# DATA SUMMARY
# -----------------------------
elif section == "Data Summary":
    st.markdown("## Data Summary")

    summary_df = pd.DataFrame(list(overview.items()), columns=["Item", "Value"])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    imbalance_fig = px.pie(
        names=["Class 0", "Class 1"],
        values=[96.209706, 3.790294],
        hole=0.5,
        template="plotly_dark",
        title="Target Distribution"
    )
    imbalance_fig.update_traces(textinfo="percent+label")
    st.plotly_chart(imbalance_fig, use_container_width=True)

    st.info(
        "The target distribution is highly skewed toward class 0, which makes this a challenging classification problem "
        "where accuracy alone can be misleading."
    )

# -----------------------------
# MODEL COMPARISON
# -----------------------------
elif section == "Model Comparison":
    st.markdown("## Model Comparison")

    metric = st.selectbox("Select metric", ["Accuracy", "Precision", "Recall", "F1-score"])
    fig = px.bar(
        model_metrics,
        x="Model",
        y=metric,
        color="Dataset",
        barmode="group",
        template="plotly_dark",
        title=f"{metric} Across Models and Datasets"
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Decision Tree shows near-perfect performance, which is impressive but may reflect overfitting. "
        "SVM offers a more balanced tradeoff, while MLP struggles with recall on Historical Test."
    )

    st.markdown("### Detailed Table")
    st.dataframe(model_metrics, use_container_width=True, hide_index=True)

    st.markdown("### Train vs Test")
    metric_choice = st.selectbox("Select train/test metric", ["Accuracy", "Precision", "Recall", "F1"], key="train_test_metric")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Train", x=train_test_metrics["Model"], y=train_test_metrics[f"Train {metric_choice}"]))
    fig2.add_trace(go.Bar(name="Test", x=train_test_metrics["Model"], y=train_test_metrics[f"Test {metric_choice}"]))
    fig2.update_layout(
        template="plotly_dark",
        barmode="group",
        height=450,
        title=f"Train vs Test {metric_choice}"
    )
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# CONFUSION MATRICES
# -----------------------------
elif section == "Confusion Matrices":
    st.markdown("## Confusion Matrices")

    model = st.selectbox("Model", ["Decision Tree", "SVM", "MLP", "Fine-tuned MLP"])
    dataset = st.selectbox("Dataset", ["Historical Test", "Current Test"], key="cm_dataset")

    key = (model, dataset)
    if key in conf_mats:
        cm = conf_mats[key]
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            template="plotly_dark",
            title=f"{model} - {dataset}"
        )
        cm_fig.update_layout(height=520)
        st.plotly_chart(cm_fig, use_container_width=True)

        tn, fp = cm[0]
        fn, tp = cm[1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TN", tn)
        c2.metric("FP", fp)
        c3.metric("FN", fn)
        c4.metric("TP", tp)

        st.info(
            "The confusion matrices show that the Decision Tree is strongest overall, while the MLP improves on Current Test "
            "after adaptation but still has limited minority-class robustness."
        )
    else:
        st.warning("No confusion matrix found for the selected model and dataset.")

# -----------------------------
# INTERPRETABILITY
# -----------------------------
elif section == "Interpretability":
    st.markdown("## Interpretability")

    st.write(
        "This section emphasizes how the model reaches its predictions, not just how well it scores. "
        "Interpretability is especially important when one feature appears to dominate the explanation."
    )

    fig = px.bar(
        feature_importance.sort_values("Importance", ascending=True),
        x="Importance",
        y="Feature",
        orientation="h",
        template="plotly_dark",
        title="Top Feature Importance"
    )
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The model relies heavily on a dominant feature, which may indicate strong predictive power or potential overfitting "
        "depending on how stable the result is across splits."
    )

    st.markdown("### Notes")
    st.write("- The feature importance output suggests that one feature contributes far more than the others.")
    st.write("- That pattern should be interpreted cautiously because very concentrated importance can sometimes reflect instability.")

# -----------------------------
# CONTINUAL LEARNING
# -----------------------------
elif section == "Continual Learning":
    st.markdown("## Continual Learning")

    st.write("Comparison of MLP performance before and after fine-tuning on Current Test.")

    fig = px.bar(
        before_after,
        x="Metric",
        y=["Before CL", "After CL + Threshold Tuning"],
        barmode="group",
        template="plotly_dark",
        title="Before vs After Continual Learning"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(before_after, use_container_width=True, hide_index=True)

    st.info(
        "Fine-tuning alone did not improve the minority class enough; the model became more conservative after adaptation, "
        "which reduced precision and F1-score."
    )

    st.markdown("### Fine-tuned MLP Result")
    st.write("Accuracy: 0.9438775510204082")
    st.write("Precision: 0.3333333333333333")
    st.write("Recall: 0.1")
    st.write("F1-score: 0.15384615384615385")
    st.write("Confusion Matrix: [[368, 4], [18, 2]]")

# -----------------------------
# THRESHOLD TUNING
# -----------------------------
elif section == "Threshold Tuning":
    st.markdown("## Threshold Tuning")

    threshold = st.selectbox("Select threshold", threshold_df["Threshold"].tolist())
    row = threshold_df[threshold_df["Threshold"] == threshold].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f'{row["Accuracy"]:.4f}')
    c2.metric("Precision", f'{row["Precision"]:.4f}')
    c3.metric("Recall", f'{row["Recall"]:.4f}')
    c4.metric("F1-score", f'{row["F1-score"]:.4f}')

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("TN", int(row["TN"]))
    c6.metric("FP", int(row["FP"]))
    c7.metric("FN", int(row["FN"]))
    c8.metric("TP", int(row["TP"]))

    fig = px.line(
        threshold_df,
        x="Threshold",
        y=["Accuracy", "Precision", "Recall", "F1-score"],
        markers=True,
        template="plotly_dark",
        title="Threshold vs Performance"
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Threshold 0.2 provides a better recall balance than the default 0.5 setting, which is more suitable when missing "
        "positive cases is costly."
    )

    st.markdown("### Selected Threshold Table")
    st.dataframe(threshold_df, use_container_width=True, hide_index=True)
