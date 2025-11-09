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
import json 
import re 

# ======= Load environment variables =======

load_dotenv()
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")

HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# ======= Helper function for JSON cleaning
def clean_and_parse_json_string(json_string):
    """Cleans a dirty JSON string and returns the parsed object."""
    try:
        if not json_string:
            return None
        
        # 1. Replace escaped newlines and single quotes
        cleaned_string = json_string.strip()
        cleaned_string = cleaned_string.replace('\\n', '').replace("'", '"')
        
        # 2. Use regex to remove literal newlines and carriage returns (Handles dirty output)
        cleaned_string = re.sub(r'[\r\n]+', '', cleaned_string)
        
        # 3. Aggressive closure attempt (Needed because your backend output is truncated/incomplete)
        if cleaned_string.startswith('{') and not cleaned_string.endswith('}'):
            # Check and add closing brackets/braces if the string looks incomplete
            if cleaned_string.count('[') > cleaned_string.count(']'):
                 cleaned_string += ']' * (cleaned_string.count('[') - cleaned_string.count(']'))
            
            if cleaned_string.count('{') > cleaned_string.count('}'):
                 cleaned_string += '}' * (cleaned_string.count('{') - cleaned_string.count('}'))
            
            # Final object closure
            if not cleaned_string.endswith('}'):
                 cleaned_string += '}'

        # 4. Attempt to load the JSON
        return json.loads(cleaned_string)
    except json.JSONDecodeError as e:
        return None
    except Exception:
        return None

# ====== Page configuration ====== 
st.set_page_config(
    page_title="Medalyze - Medical CallChat Analysis",
    layout="wide"
)

# ======= Header ======= 
st.title("💝 Medalyze")
st.subheader("Medical CallChat Analysis")
st.write("Upload transcripts, visualize rubric scores, and send heatmap to NeuralSeek.")

# ======= Tabs =======
tab1, tab2 = st.tabs(["Upload Transcripts", "Data Visualization"])

# ====== Tab 1: Upload Multiple Transcripts (CLEANED) ======
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
            
            # st.subheader("Debugging: Agent-1 Response for Analysis ID") <-- REMOVED
            
            for doc in upload_transcripts:
                with st.spinner(f"Uploading and processing **{doc.name}**..."):
                    file_bytes = doc.read()
                    file_b64 = base64.b64encode(file_bytes).decode("utf-8")

                    payload = {
                        "ntl": "",
                        "agent": "Agent-1",
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
                        
                        # --- FIX FOR AGENT-1: Use robust parser ---
                        json_string = result.get('answer', '')
                        inner_data = clean_and_parse_json_string(json_string)
                        
                        if inner_data:
                            result['analysis_id'] = inner_data.get('analysis_id', '')
                            # if result['analysis_id']:
                            #     st.success(f"Extracted ID: {result['analysis_id']}") <-- REMOVED
                            # else:
                            #     st.warning("Parsed response but 'analysis_id' key missing.") <-- REMOVED
                        # else:
                        #     st.error("Could not parse inner JSON from 'answer'.") <-- REMOVED
                        # --- END FIX FOR AGENT-1 ---
                        
                        result["file_name"] = doc.name 
                        all_analysis_results.append(result)
                        
                    except Exception as e:
                        # This handles the recent 'Connection aborted' error due to large file/timeout
                        st.error(f"❌ Failed to upload {doc.name}: {e}") 

            st.session_state["all_analysis_results"] = all_analysis_results
            st.success(f"✅ **{len(all_analysis_results)}** transcripts uploaded and processed! Navigate to **Data Visualization**.")

# ======= Tab 2: Data Visualization =====
with tab2:
    st.header("Rubric Heatmap for All Transcripts")

    if "all_analysis_results" not in st.session_state or not st.session_state["all_analysis_results"]:
        st.info("Upload and process transcripts in Tab 1 first.")
    else:
        all_matrices = []
        row_labels = []
        col_labels = []
        
        # Use a single progress bar for fetching 
        progress_bar = st.progress(0)
        total_files = len(st.session_state["all_analysis_results"])
        
        st.markdown(f"Fetching analysis for {total_files} files...")

        for i, result in enumerate(st.session_state["all_analysis_results"]):
            analysis_id = result.get("analysis_id", "")
            
            if not analysis_id:

                continue

            
            payload = {
                "ntl": "",
                "agent": "Agent-3",
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
                
                json_string = analyzed_data.get('answer', '')
                final_analysis_data = clean_and_parse_json_string(json_string)
                
                matrix = np.array([])
                
                if final_analysis_data:
                    evals = final_analysis_data.get("evaluations_0", []) 
                    
                    if evals and isinstance(evals, list) and len(evals) > 0:
                        df = pd.DataFrame(evals)
                        matrix = df.values 
                        
                        if not col_labels:
                            col_labels = df.columns.tolist() 
                        
                        file_row_labels = [f"{result['file_name']} (Eval {i+1})" for i in range(matrix.shape[0])]
                        row_labels.extend(file_row_labels)
                        
                
                if matrix.size == 0 or matrix.ndim != 2:
                 
                    continue
                    
                all_matrices.append(matrix)
                
            except Exception as e:
                st.error(f"❌ Failed to fetch analysis for {result['file_name']}: {e}")
            

            progress_bar.progress((i + 1) / total_files)

        progress_bar.empty() 

        st.markdown("---")
        st.subheader("Rubric Heatmap Results")
        
        if not all_matrices:
            st.error("❌ No valid analysis data to plot.")
        else:
            combined_matrix = np.vstack(all_matrices)
            overall_scores = combined_matrix.mean(axis=1)

            if not col_labels or len(col_labels) != combined_matrix.shape[1]:
                 col_labels = [f"Criterion {i+1}" for i in range(combined_matrix.shape[1])]

            df_scores = pd.DataFrame(combined_matrix, columns=col_labels, index=row_labels)
            df_scores["Overall Score"] = overall_scores
            
            st.markdown("#### Score Summary Table")
            st.dataframe(df_scores)

            fig, ax = plt.subplots(figsize=(10, max(6, len(row_labels) * 0.3)))
            sns.heatmap(combined_matrix, annot=True, fmt=".2f", cmap="viridis",
                        xticklabels=col_labels, yticklabels=row_labels, ax=ax)
            ax.set_title("Rubric Heatmap for All Transcripts")
            st.markdown("#### Visualization")
            st.pyplot(fig)

            if st.button("Send Heatmap to NeuralSeek for Email"):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                heatmap_b64 = base64.b64encode(buf.read()).decode("utf-8")

                payload = {
                    "ntl": "",
                    "agent": "Agent-4",
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