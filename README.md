# Medalyze
** App for Call Transcript analysis of medical-provider/patient conversations in order to gather insights on quality of provider/patient conversations. **

*Develeoped for Stony Brook University(SBU) 2025 Hackathon*

AI-powered analysis of call transcripts via the use of agentic NeuralSeek to summarize the call transcripts while removing Personal Identifiying Information(PII). In addition, the transcripts are graded by internal Large Language Models(LLM), NeuralSeek Managed ChatGPT-4o mini, to then convert to json format. It will then relay to the frontend, where a data visualization of a heatmap will be generated. The created png file will then be sent back to Neuroseek, in order to send a mass email to relevant parties abouit the insights gathered. 

This could be relevant to Heads of Hospital Departments and Board of Directors of Hospitals to provide detailed evealtuions of their networks/daprtment doctors and useuful evaluations via data analysis.

## 👥 Team & Contributions

### SBUHacks 2025
| Role | Responsibilities | Team Member |
|------|------------------|-------------|
| ** Backend NueralSeek ** | AI Agent 1 | ** Keneth Exposito **
| ** Backend NeuralSeek ** | AI Agent 2 | ** Cherie Millan **
| ** Backend NueralSeek ** | AI Agent 3 | ** Jean Carlo Chajon **
| ** Backend NeuralSeek ** | AI Agent 4 | ** Fiona Vladi **
| ** Frontend ** | Streamlit UI | Jean Carlo Chajon

### Key Integrations
-- **NeuralSeek**: Advanced AI Agent Platform
-- **Streamlit Frontend**: Modern user interface

## 🏗️ Architecture

```
``
ResumeBeaver/
├── README.md                # Documentation
├── .gitignore               # Git ignore rules
├── frontend/                # React frontend
│   ├── streamlit_app.py     # Streamlit demo UI
├── backend/
    ├── neuralseek_agent_1_ntl.txt
    ├── neuralseek_agent_2_ntl.txt
    ├── neuralseek_agent_3_ntl.txt
    ├── neuralseek_agent_4_ntl.txt

```
