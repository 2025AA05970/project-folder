import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from model.logistic_model import train_predict as logistic_run
from model.decision_tree_model import train_predict as dt_run
from model.knn_model import train_predict as knn_run
from model.naive_bayes_model import train_predict as nb_run
from model.random_forest_model import train_predict as rf_run
from model.xgboost_model import train_predict as xgb_run

from model.utils import compute_metrics


st.set_page_config(page_title="Adult Income Classification", layout="wide")

st.title("Adult Income Prediction - ML Models Comparison")

uploaded_file = st.file_uploader("Upload Adult Income CSV File", type="csv")

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    target = "income"

    X = data.drop(target, axis=1)
    y = data[target]

    encoder = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = encoder.fit_transform(X[col])

    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.subheader("Select Model")

    model_choice = st.selectbox(
        "",
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )

    if st.button("Run Selected Model"):

        if model_choice == "Logistic Regression":
            y_pred, y_prob = logistic_run(X_train, X_test, y_train)

        elif model_choice == "Decision Tree":
            y_pred, y_prob = dt_run(X_train, X_test, y_train)

        elif model_choice == "KNN":
            y_pred, y_prob = knn_run(X_train, X_test, y_train)

        elif model_choice == "Naive Bayes":
            y_pred, y_prob = nb_run(X_train, X_test, y_train)

        elif model_choice == "Random Forest":
            y_pred, y_prob = rf_run(X_train, X_test, y_train)

        elif model_choice == "XGBoost":
            y_pred, y_prob = xgb_run(X_train, X_test, y_train)

        metrics = compute_metrics(y_test, y_pred, y_prob)

        st.subheader("Evaluation Metrics")
        st.write(metrics)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    if st.button("Generate Comparison Table"):

        from pandas import DataFrame

        results = {}

        y_pred, y_prob = logistic_run(X_train, X_test, y_train)
        results["Logistic Regression"] = compute_metrics(y_test, y_pred, y_prob)

        y_pred, y_prob = dt_run(X_train, X_test, y_train)
        results["Decision Tree"] = compute_metrics(y_test, y_pred, y_prob)

        y_pred, y_prob = knn_run(X_train, X_test, y_train)
        results["KNN"] = compute_metrics(y_test, y_pred, y_prob)

        y_pred, y_prob = nb_run(X_train, X_test, y_train)
        results["Naive Bayes"] = compute_metrics(y_test, y_pred, y_prob)

        y_pred, y_prob = rf_run(X_train, X_test, y_train)
        results["Random Forest"] = compute_metrics(y_test, y_pred, y_prob)

        y_pred, y_prob = xgb_run(X_train, X_test, y_train)
        results["XGBoost"] = compute_metrics(y_test, y_pred, y_prob)

        comparison_df = DataFrame(results).T

        st.subheader("Model Comparison Table")
        st.dataframe(comparison_df)
