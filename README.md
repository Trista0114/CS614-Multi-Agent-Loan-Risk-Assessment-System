# Intelligent Risk Assessment System for Loan Applications

## Project Overview

### Core Positioning
This is a multi-agent AI system based on **LangGraph** , designed to automate loan risk assessment by providing comprehensive credit, fraud, and regulatory compliance analysis, resulting in highly consistent decision recommendations.

**[📚 Click to view the detailed Slides (PDF)]**(./Group7%20Slides.pdf)

### System Architecture Workflow
```
User Input Loan Application 
   → Data Preprocessing 
      → 3 Expert Agents Parallel Analysis 
         → Comprehensive Decision Agent Integration (Agent4)
            → Generate Risk Assessment Report
```

### Technical Core
- **Framework:** LangGraph + LangChain  
- **Model Strategy:** Open-source fine-tuning 
- **Knowledge Enhancement:** RAG retrieval system (Micmic banking internal policy based on Fannie Mae)  
- **Reasoning:** Chain-of-Thought interpretable reasoning
- **Evaluation of Reasoning:** using LLM Judge Qwen2.5-14B-Instruct to automatically validate the reasoning Faithfulness and Relevance of all agents.

### Four-Agent Design Architecture
| Agent ID | Role | Core Model | Key Techniques |
| :---- | :--- | :--- | :--- |
| **Agent 1** | Credit Risk Analysis | Ensemble ML (XGBoost, LightGBM, CatBoost) + Llama-3.1-8B (LoRA) | Predicts default probability (Threshold > 0.45) and generates LLM explanations. |
| **Agent 2** | Fraud Risk Detection | Llama-3.1-8B-Instruct (LoRA) | Classifies risk based on multiple fraud signals (Threshold > 0.50). |
| **Agent 3** | Regulatory Compliance | Llama-3.1-8B-Instruct (4-bit) | Embeds **Deterministic Logic** for numerical checks (DTI, Age) and utilizes **Hybrid Graph RAG**. |
| **Agent 4** | Decision Coordinator | Llama-3.1-8B-Instruct (4-bit) | Integrates results using a **Weighted Score (Credit * 0.6 + Fraud * 0.4)** and synthesizes the final report. |



### System Highlights
- **100% Decision Consistency**: Routing logic in LangGraph ensures the final decision and LLM-generated report are completely aligned with underlying rule-based risk scores3.
- **Memory Optimization**: Utilizing 4-bit Quantization and a Shared Model Architecture, we integrated Agents 2, 3, and 4 onto a single Llama-3.1-8B-Instruct instance, reducing total memory usage from $\sim 27$ GB to $\sim 16$ GB (a 41% saving).
- **Hybrid RAG**: Agent 3 uses Hybrid Graph RAG 6combining a Knowledge Graph with Vector Search (ChromaDB)  for highly accurate regulatory retrieval.
- **LLM Judge Evaluation**: Implemented an LLM-as-a-Judge pipeline using Qwen2.5-14B-Instruct to automatically validate the reasoning Faithfulness and Relevance of all agents.
---

## Data Resource Configuration

- **Credit Risk Benchmark (Kaggle):** Credit risk model training  
- **Bank Loan Fraud Detection (Kaggle):** Fraud detection model training  
- **FINRA Regulatory Rules (Official):** RAG knowledge base construction  

**Data Preprocessing:**  
- Standardization → JSON format  
- Feature Engineering → Extract risk indicators & ratios  
- Data Cleaning → Handle missing/outliers  
- Train/Test Split → 80/20  

---

Environment Setup: Clone this repository and install the dependencies from requirements.txt.
Configuration: Set your Hugging Face Token in the environment variable HF_TOKEN.
Model Access: Please note that the LoRA adapter weights are not included in this repository.
Run Demo: Execute the Streamlit application to launch the interactive UI and the full LangGraph pipeline.

