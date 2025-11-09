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
st.write("Upload transcripts, visualize rubric scores, and send heatmap to NeuralSeek.")

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
                    # Read file and encode as base64
                    file_bytes = doc.read()
                    file_b64 = base64.b64encode(file_bytes).decode("utf-8")

                    payload = {
                        "ntl": "",
                        "agent": "Agent-1",  # replace with your agent name
                        "params": [
                            {"name": "file_name", "value": doc.name},
                            {"name": "file_content_base64", "value": file_b64}
                        ],
                        "options": {
                            "timeout": 600000,
                            "streaming": False
                        }
                    }

                    try:
                        response = requests.post(API_BASE_URL, json=payload, headers=HEADERS)
                        response.raise_for_status()
                        result = response.json()
                        result["file_name"] = doc.name
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
        # Fetch analysis for each transcript via agent
        analysis_id = result.get("analysis_id", "")
        if not analysis_id:
            st.warning(f"No analysis ID for {result['file_name']}")
            continue

        st.info(f"Fetching analysis for {result['file_name']}...")
        payload = {
            "ntl": "",
            "agent": "Agent-3",  # replace with your agent name
            "params": [
                {"name": "analysis_id", "value": analysis_id}
            ],
            "options": {
                "timeout": 600000,
                "streaming": False
            }
        }

        try:
            analysis_response = requests.post(API_BASE_URL, json=payload, headers=HEADERS)
            analysis_response.raise_for_status()
            analyzed_data = analysis_response.json()
            matrix = np.array(analyzed_data.get("matrix", []))
            if matrix.size == 0:
                st.warning(f"No matrix returned for {result['file_name']}")
                continue
            all_matrices.append(matrix)
            row_labels.extend(analyzed_data.get("row_labels", [f"{result['file_name']} {i+1}" for i in range(matrix.shape[0])]))
        except Exception as e:
            st.error(f"Failed to fetch analysis for {result['file_name']}: {e}")

    if not all_matrices:
        st.error("No valid analysis data available to plot.")
        st.stop()

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

    # Send heatmap to NeuralSeek via mAIstral agent
    if st.button("Send Heatmap to NeuralSeek for Email"):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        heatmap_b64 = base64.b64encode(buf.read()).decode("utf-8")

        payload = {
            "ntl": "",
            "agent": "Agent-4",  # replace with your agent name
            "params": [
                {"name": "file_name", "value": "heatmap.png"},
                {"name": "file_content_base64", "value": heatmap_b64}
            ],
            "options": {
                "timeout": 600000,
                "streaming": False
            }
        }

        try:
            upload_resp = requests.post(API_BASE_URL, json=payload, headers=HEADERS)
            upload_resp.raise_for_status()
            st.success("✅ Heatmap successfully sent to NeuralSeek via agent!")
        except Exception as e:
            st.error(f"❌ Could not send heatmap: {e}")
