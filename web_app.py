# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_score, recall_score, accuracy_score,
    mean_absolute_error, mean_squared_error, brier_score_loss, log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="EVOQUE Suitability Prediction", layout="centered")

# Edwards branding colors
COLOR_SUITABLE = "#007A33"       # green
COLOR_UNSUITABLE = "#C8102E"     # red
COLOR_INDETERMINATE = "#F39200"  # orange
COLOR_CONFIDENCE = "#666666"     # neutral gray

# ----------------------------
# Load and Train Model (Cached)
# ----------------------------
@st.cache_resource
def load_and_train_model():
    # Load data
    data_path = os.getenv("DATA_PATH", "Echo_Annular_Measurements.xlsx")
    df = pd.read_excel(data_path, engine="openpyxl")

    # Rename columns
    df = df.rename(columns={
        "S-L dimension (mm)": "SL",
        "A-P Dimension (mm)": "AP",
        "Decision": "Decision"
    })

    # Derived features
    df["Ratio"] = df["SL"] / df["AP"]
    df["SL_z"] = (df["SL"] - df["SL"].mean()) / df["SL"].std()
    df["AP_z"] = (df["AP"] - df["AP"].mean()) / df["AP"].std()

    # Encode target
    df["Target"] = df["Decision"].apply(lambda x: 1 if x == "Suitable" else 0)

    # Features and target
    features = ["SL", "AP", "Ratio", "SL_z", "AP_z"]
    X = df[features]
    y = df["Target"]

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train Random Forest with fixed hyperparameters
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        criterion="log_loss",
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Calibrate with Platt scaling (sigmoid), 3-fold CV
    cal_rf = CalibratedClassifierCV(rf, method="sigmoid", cv=3)
    cal_rf.fit(X_train, y_train)

    # Predefined thresholds and corridor
    thresholds = {"t_low": 0.40, "t_high": 0.60}
    corridor = {"SL_cutoff": 55.0, "AP_cutoff": 49.0}

    return cal_rf, thresholds, corridor, features, df

# ----------------------------
# Prediction Mapping
# ----------------------------
def map_prediction(prob, SL, AP, thresholds, corridor):
    if prob >= thresholds["t_high"]:
        return "Suitable Annular Dimensions", COLOR_SUITABLE
    elif prob < thresholds["t_low"]:
        # Guardrail: downgrade to CT Recommended if in mixed corridor
        if SL >= corridor["SL_cutoff"] and AP <= corridor["AP_cutoff"]:
            return "Indeterminate Annular Dimensions", COLOR_INDETERMINATE
        else:
            return "Unsuitable Annular Dimensions", COLOR_UNSUITABLE
    else:
        return "Indeterminate Annular Dimensions", COLOR_INDETERMINATE

# ----------------------------
# Load Model and Data
# ----------------------------
cal_rf, thresholds, corridor, features, df_full = load_and_train_model()

# ----------------------------
# UI Layout
# ----------------------------
st.title("EVOQUE Suitability Prediction")
st.subheader("Echo measurements")

SL_input = st.number_input("S-L Diameter (mm)", min_value=30.0, max_value=70.0, step=1.0, key="sl_input")
AP_input = st.number_input("A-P Diameter (mm)", min_value=30.0, max_value=70.0, step=1.0, key="ap_input")

if st.button("Predict"):
    # Derived features
    ratio = SL_input / AP_input
    SL_z = (SL_input - df_full["SL"].mean()) / df_full["SL"].std()
    AP_z = (AP_input - df_full["AP"].mean()) / df_full["AP"].std()

    input_df = pd.DataFrame([[SL_input, AP_input, ratio, SL_z, AP_z]], columns=features)
    prob = cal_rf.predict_proba(input_df)[0][1]
    category, color = map_prediction(prob, SL_input, AP_input, thresholds, corridor)

    # Display result
    st.markdown(f"<h2 style='color:{color};text-align:center'>{category}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{COLOR_CONFIDENCE};text-align:center'><em>Confidence: {prob*100:.1f}%</em></p>", unsafe_allow_html=True)

    # Tooltip
    if category == "Suitable Annular Dimensions":
        tooltip = "S‑L and A‑P echo diameters fall within typical annular ranges for EVOQUE. Final therapy decision requires complete imaging and clinical assessment."
    elif category == "Unsuitable Annular Dimensions":
        tooltip = "S‑L and A‑P echo diameters are outside typical annular ranges for EVOQUE."
    else:
        tooltip = "S‑L and A‑P echo diameters are near decision boundaries or conflicting; proceed with CT and full assessment for final suitability decision and sizing."

    st.markdown(f"<p style='text-align:center'>{tooltip}</p>", unsafe_allow_html=True)

# ----------------------------
# Metrics and Downloads
# ----------------------------
st.markdown("---")
if st.button("Build metrics & predictions workbook"):
    X_full = df_full[features]
    df_full["p_Suitable"] = cal_rf.predict_proba(X_full)[:, 1]
    df_full["Predicted Category"] = df_full.apply(
        lambda row: map_prediction(row["p_Suitable"], row["SL"], row["AP"], thresholds, corridor)[0],
        axis=1
    )

    # Metrics
    y_true = df_full["Target"]
    y_pred = df_full["Predicted Category"].map(lambda x: 1 if x == "Suitable Annular Dimensions" else 0)
    cm = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, df_full["p_Suitable"])
    pr_auc = average_precision_score(y_true, df_full["p_Suitable"])
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, df_full["p_Suitable"])
    mse = mean_squared_error(y_true, df_full["p_Suitable"])
    brier = brier_score_loss(y_true, df_full["p_Suitable"])
    logloss = log_loss(y_true, df_full["p_Suitable"])

    # Save to Excel
    with pd.ExcelWriter("metrics_predictions.xlsx", engine="openpyxl") as writer:
        df_full.to_excel(writer, sheet_name="Predictions", index=False)
        metrics_df = pd.DataFrame({
            "Metric": ["ROC-AUC", "PR-AUC", "Precision", "Recall", "Accuracy", "MAE", "MSE", "Brier Score", "Log Loss"],
            "Value": [roc_auc, pr_auc, precision, recall, accuracy, mae, mse, brier, logloss]
        })
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

    st.success("Workbook generated.")
    with open("metrics_predictions.xlsx", "rb") as f:
        st.download_button("Download workbook", f, file_name="metrics_predictions.xlsx")

if st.button("Plot download"):
    fig, ax = plt.subplots()
    sns.histplot(df_full["p_Suitable"], bins=30, kde=True, ax=ax)
    ax.set_title("Probability Distribution (p(Suitable))")
    plt.tight_layout()
    fig.savefig("probability_distribution.png")
    with open("probability_distribution.png", "rb") as f:
        st.download_button("Download probability plot", f, file_name="probability_distribution.png")

if st.button("SHAP"):
    explainer = shap.Explainer(cal_rf)
    shap_values = explainer(df_full[features])
    fig = shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    fig.savefig("shap_plot.png")
    with open("shap_plot.png", "rb") as f:
        st.download_button("Download SHAP plot", f, file_name="shap_plot.png")
