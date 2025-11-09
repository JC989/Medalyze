import streamlit as st
import requests
import io
import base64
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# ==============================
# Load environment variables
# ==============================
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")

# ==============================
# Derived API endpoints
# ==============================
API_UPLOAD_DOCUMENT = f"{API_BASE_URL}/exploreupload"
API_ANALYSIS_STATUS = f"{API_BASE_URL}/get_analysis"
API_UPLOAD_HEATMAP = f"{API_BASE_URL}/upload_heatmap"

HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# ==============================
# Page configuration
# ==============================
st.set_page_config(
    page_title="Medalyze - Medical CallChat Analysis",
    layout="wide"
)

# ==============================
# Header
# ==============================
st.title("💝 Medalyze")
st.subheader("Medical CallChat Analysis")
st.write("Analyze your medical call transcripts with ease!")

# ==============================
# Tabs
# ==============================
tab1, tab2 = st.tabs(["Upload Transcripts", "Data Visualization"])

# ==============================
# Tab 1: Upload Multiple Transcripts
# ==============================
with tab1:
    st.header("Upload Your Medical Call Transcripts")
    upload_transcripts = st.file_uploader(
        "Choose one or more transcripts",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload multiple medical call transcripts at once."
    )

    if upload_transcripts:
        if st.button("Upload & Process All Transcripts"):
            all_analysis_results = []

            for doc in upload_transcripts:
                with st.spinner(f"Uploading {doc.name}..."):
                    files = {"file": (doc.name, doc, doc.type)}
                    try:
                        response = requests.post(API_UPLOAD_DOCUMENT, files=files, headers=HEADERS)
                        response.raise_for_status()
                        result = response.json()
                        result["file_name"] = doc.name  # store filename for row labels
                        all_analysis_results.append(result)
                    except Exception as e:
                        st.error(f"Failed to upload {doc.name}: {e}")

            st.session_state["all_analysis_results"] = all_analysis_results
            st.success(f"✅ {len(all_analysis_results)} transcripts uploaded and processed!")

# ==============================
# Tab 2: Data Visualization
# ==============================
with tab2:
    st.header("Rubric Heatmap for All Transcripts")

    if "all_analysis_results" not in st.session_state or not st.session_state["all_analysis_results"]:
        st.info("Upload and process transcripts in Tab 1 first.")
        st.stop()

    # Combine all matrices into one
    all_matrices = []
    row_labels = []

    for result in st.session_state["all_analysis_results"]:
        matrix = np.array(result["matrix"])
        all_matrices.append(matrix)
        row_labels.extend(result.get("row_labels", [f"{result['file_name']} {i+1}" for i in range(matrix.shape[0])]))

    combined_matrix = np.vstack(all_matrices)
    n_criteria = combined_matrix.shape[1]
    overall_scores = combined_matrix.mean(axis=1)

    # Column labels
    col_labels = ["Criterion A", "Criterion B", "Criterion C", "Criterion D"]
    if n_criteria > 4:
        col_labels = [f"Criterion {i+1}" for i in range(n_criteria)]

    # DataFrame for visualization
    df_scores = pd.DataFrame(combined_matrix, columns=col_labels, index=row_labels)
    df_scores["Overall Score"] = overall_scores
    st.dataframe(df_scores)

    # Generate heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, len(row_labels) * 0.3)))
    sns.heatmap(combined_matrix, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=col_labels, yticklabels=row_labels, ax=ax)
    ax.set_title("Rubric Heatmap for All Transcripts")
    st.pyplot(fig)

    # Send heatmap to NeuralSeek/mAIstral for mass email
    if st.button("Send Heatmap to NeuralSeek for Email"):
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        heatmap_base64 = base64.b64encode(buf.read()).decode("utf-8")

        payload = {
            "filename": "heatmap.png",
            "image_base64": heatmap_base64
        }

        try:
            upload_resp = requests.post(API_UPLOAD_HEATMAP, json=payload, headers=HEADERS)
            if upload_resp.status_code == 200:
                st.success("✅ Heatmap successfully uploaded for mass email!")
            else:
                st.error(f"❌ Upload failed with status {upload_resp.status_code}")
                st.write(upload_resp.text)
        except Exception as e:
            st.error(f"❌ Could not send heatmap: {e}")
