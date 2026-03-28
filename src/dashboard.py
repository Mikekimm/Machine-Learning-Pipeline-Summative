from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_URL = st.sidebar.text_input("API URL", "http://127.0.0.1:8000").rstrip("/")


def _safe_json(resp: requests.Response) -> dict:
    try:
        return resp.json()
    except Exception:
        return {"status_code": resp.status_code, "text": resp.text[:500]}


def _backend_routes(base_url: str) -> set[str]:
    try:
        resp = requests.get(f"{base_url}/openapi.json", timeout=15)
        data = _safe_json(resp)
        paths = data.get("paths", {}) if isinstance(data, dict) else {}
        return set(paths.keys())
    except Exception:
        return set()

st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")
st.title("Machine Learning Pipeline Dashboard")

routes = _backend_routes(API_URL)
if routes:
    supports_bulk_predict = "/predict-bulk" in routes
else:
    supports_bulk_predict = True

if routes and not supports_bulk_predict:
    st.warning("Connected API does not expose /predict-bulk. Bulk prediction will use a safe single-file fallback.")


col1, col2 = st.columns(2)
with col1:
    if st.button("Check Model Uptime"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=30)
            st.json(response.json())
        except Exception as exc:
            st.error(str(exc))

with col2:
    if st.button("Trigger Retraining"):
        try:
            response = requests.post(f"{API_URL}/retrain", timeout=300)
            st.json(response.json())
        except Exception as exc:
            st.error(str(exc))

st.subheader("Single Image Prediction")
predict_file = st.file_uploader("Upload one image", type=["jpg", "jpeg", "png", "bmp", "webp"])
if st.button("Predict") and predict_file is not None:
    try:
        files = {"file": (predict_file.name, predict_file.getvalue(), predict_file.type)}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=90)
        st.json(_safe_json(response))
    except Exception as exc:
        st.error(str(exc))

st.subheader("Bulk Image Prediction")
bulk_predict_files = st.file_uploader(
    "Upload multiple images for prediction",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
    key="bulk_predict_files",
)
if st.button("Predict Bulk") and bulk_predict_files:
    try:
        files = [("files", (f.name, f.getvalue(), f.type)) for f in bulk_predict_files]

        if supports_bulk_predict:
            response = requests.post(f"{API_URL}/predict-bulk", files=files, timeout=180)
            if response.status_code == 200:
                st.json(_safe_json(response))
            elif response.status_code == 404:
                supports_bulk_predict = False
            else:
                st.error(f"Bulk prediction failed: HTTP {response.status_code}")
                st.json(_safe_json(response))

        if not supports_bulk_predict:
            fallback_results = []
            for f in bulk_predict_files:
                single_file = {"file": (f.name, f.getvalue(), f.type)}
                single_resp = requests.post(f"{API_URL}/predict", files=single_file, timeout=90)
                payload = _safe_json(single_resp)
                if single_resp.status_code == 200:
                    fallback_results.append({"filename": f.name, "prediction": payload})
                else:
                    fallback_results.append(
                        {
                            "filename": f.name,
                            "error": f"HTTP {single_resp.status_code}",
                            "detail": payload,
                        }
                    )

            st.json({"count": len(fallback_results), "results": fallback_results, "mode": "fallback_single_predict"})
    except Exception as exc:
        st.error(str(exc))

st.subheader("Bulk Upload For Retraining")
bulk_files = st.file_uploader(
    "Upload multiple images",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
)
if st.button("Upload Bulk Data") and bulk_files:
    try:
        files = [("files", (f.name, f.getvalue(), f.type)) for f in bulk_files]
        response = requests.post(f"{API_URL}/upload-bulk", files=files, timeout=180)
        st.json(response.json())
    except Exception as exc:
        st.error(str(exc))

st.subheader("Model Evaluation")
metrics_path = PROJECT_ROOT / "results" / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    st.metric("Precision (weighted)", f"{metrics.get('precision_weighted', 0.0):.4f}")
    st.metric("Recall (weighted)", f"{metrics.get('recall_weighted', 0.0):.4f}")
    st.metric("F1 (weighted)", f"{metrics.get('f1_weighted', 0.0):.4f}")

    classes = metrics["classes"]
    matrix = pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes)
    st.dataframe(matrix)

    fig, ax = plt.subplots(figsize=(8, 4))
    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
    per_class = report_df.loc[classes, ["precision", "recall", "f1-score"]]
    per_class.plot(kind="bar", ax=ax)
    ax.set_title("Per-Class Metrics")
    ax.set_ylabel("Score")
    st.pyplot(fig)
else:
    st.info("Train the model first to populate metrics.")

st.subheader("Connected Backend")
st.write(f"Current API URL: {API_URL}")
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=20)
    st.write(f"Health check status: {health_resp.status_code}")
    metrics_resp = requests.get(f"{API_URL}/metrics", timeout=20)
    if metrics_resp.status_code == 200:
        live_metrics = _safe_json(metrics_resp)
        if isinstance(live_metrics, dict) and "accuracy" in live_metrics:
            st.write(f"Live backend accuracy: {live_metrics['accuracy']:.4f}")
except Exception as exc:
    st.info(f"Could not read backend health/metrics: {exc}")

st.subheader("Feature Story (3+ Features)")
st.write("Features tracked: brightness, blue_ratio, green_ratio, texture_strength.")
feature_path = PROJECT_ROOT / "results" / "feature_story.json"
if feature_path.exists():
    df = pd.read_json(feature_path)
    st.dataframe(df.head(20))

    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    df.groupby("class")["brightness"].mean().plot(kind="bar", ax=axes[0], title="Avg Brightness")
    df.groupby("class")["blue_ratio"].mean().plot(kind="bar", ax=axes[1], title="Avg Blue Ratio")
    df.groupby("class")["texture_strength"].mean().plot(kind="bar", ax=axes[2], title="Avg Texture")
    st.pyplot(fig2)

    st.markdown(
        "Story: sea and glacier classes show stronger blue channels, forest has stronger green ratio, and street/mountain often carry higher texture patterns."
    )
else:
    st.info("Feature story appears after training.")
