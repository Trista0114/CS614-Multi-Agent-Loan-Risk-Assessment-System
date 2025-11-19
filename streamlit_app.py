import streamlit as st
import sys
import os
import json
import torch

# Add paths
PROJECT_ROOT = "./"
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "agents"))

from Agent_4 import (
    setup_environment,
    load_agent_modules,
    initialize_agent1,
    initialize_agent2,
    initialize_agent3,
    initialize_agent4,
    build_workflow,
    CONFIG
)

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="Loan Decision System", page_icon="💰", layout="wide")

# -------------------------------------------------------------
# CACHED MODEL INITIALIZATION (KEY FIX!)
# -------------------------------------------------------------
@st.cache_resource(show_spinner="🔄 Initializing models... This will only happen once.")
def initialize_models_once():
    """
    Initialize all models once and cache them.
    This function will only run once per Streamlit session.
    """
    print("=" * 70)
    print("INITIALIZING MODELS (This should only print once!)")
    print("=" * 70)
    
    # Step 1: Setup environment
    setup_environment()
    
    # Step 2: Load agent modules
    load_agent_modules()
    
    # Step 3-6: Initialize all agents
    analyzer_agent1, _, _ = initialize_agent1()
    
    # Load shared model
    master_llm_model, master_llm_tokenizer = initialize_agent4(shared_llm_model=None)
    
    # Initialize Agent 2 with shared model
    agent2_loaded = initialize_agent2(master_llm_model)
    
    # Initialize Agent 3 with shared model
    agent3_resources = initialize_agent3(master_llm_model)
    
    # Agent 4 uses shared model
    if agent3_resources and 'llm_model' in agent3_resources:
        agent4_model = agent3_resources['llm_model']
        agent4_tokenizer = agent3_resources['llm_tokenizer']
    else:
        agent4_model = master_llm_model
        agent4_tokenizer = master_llm_tokenizer
    
    # Step 7: Build workflow
    app = build_workflow(
        analyzer_agent1,
        agent3_resources,
        agent4_model,
        agent4_tokenizer
    )
    
    print("=" * 70)
    print("✅ MODELS INITIALIZED AND CACHED")
    print("=" * 70)
    
    return app

# -------------------------------------------------------------
# LOAD MODELS (Cached - only runs once)
# -------------------------------------------------------------
try:
    app = initialize_models_once()
    st.success("✅ Models ready!")
except Exception as e:
    st.error(f"❌ Model initialization failed: {e}")
    st.stop()

# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
st.title("💰 Multi-Agent Loan Application Decision System")
st.markdown("All four agents (Credit Risk · Fraud · Compliance · LLM Summary) run locally.")

# Show submission counter
if 'submission_count' not in st.session_state:
    st.session_state.submission_count = 0

if st.session_state.submission_count > 0:
    st.info(f"📊 Applications processed this session: {st.session_state.submission_count}")

# -------------------------------------------------------------
# INPUT FORM
# -------------------------------------------------------------
with st.form("applicant_form"):
    st.markdown("## 📝 Applicant Information")

    # ===== 1. Applicant Profile =====
    st.markdown("### 👤 Applicant Profile")
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=35)
        monthly_inc = st.number_input("Monthly Income ($)", min_value=0.0, max_value=500000.0, value=6000.0)
    with col2:
        EmploymentVerification = st.selectbox("Employment Verification", ["Verified", "Not Verified"])
        debt_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.25)

    st.divider()

    # ===== 2. Credit Profile =====
    st.markdown("### 💳 Credit Profile")
    col3, col4 = st.columns(2)
    with col3:
        open_credit = st.number_input("Open Credit Lines", min_value=0, max_value=20, value=6)
        late_90 = st.number_input("90+ Days Late Payments", min_value=0, max_value=10, value=0)
    with col4:
        CreditScore = st.number_input("Credit Score", min_value=300, max_value=850, value=720)
        rev_util = st.number_input("Revolving Credit Utilization", min_value=0.0, max_value=1.0, value=0.35)

    st.divider()

    # ===== 3. Loan & Account Information =====
    st.markdown("### 💰 Loan & Account Information")
    col5, col6 = st.columns(2)
    with col5:
        LoanAmountRequested = st.number_input("Loan Amount Requested ($)", min_value=0.0, max_value=1000000.0, value=15000.0)
        LoanTerm = st.number_input("Loan Term (years)", min_value=1, max_value=30, value=3)
        PurposeoftheLoan = st.selectbox("Purpose of Loan", ["Auto", "Education", "Home", "Medical", "Personal", "Travel"])
        PreviousLoans = st.number_input("Previous Loans", min_value=0, max_value=10, value=1)
    with col6:
        ApplicationBehavior = st.selectbox("Application Behavior", ["Normal", "Rapid"])
        LocationofApplication = st.selectbox("Application Location", ["Local", "Unusual"])
        AccountActivity = st.selectbox("Account Activity", ["Normal", "Unusual"])
        Blacklists = st.selectbox("Blacklist Status", ["No", "Yes"])
        PastFinancialMalpractices = st.selectbox("Past Financial Malpractices", ["No", "Yes"])
        ConsistencyinData = st.selectbox("Data Consistency", ["Consistent", "Inconsistent"])

    submitted = st.form_submit_button("🚀 Run Decision Pipeline", type="primary", use_container_width=True)

# -------------------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------------------
if submitted:
    # Increment counter
    st.session_state.submission_count += 1
    
    with st.spinner("⏳ Evaluating application..."):
        # Prepare applicant data
        applicant_data_dict = {
            "id": f"streamlit_user_{st.session_state.submission_count}",
            "Age": Age,
            "age": Age,
            "rev_util": rev_util,
            "debt_ratio": debt_ratio,
            "monthly_inc": monthly_inc,
            "open_credit": open_credit,
            "late_90": late_90,
            "CreditScore": CreditScore,
            "LoanAmountRequested": LoanAmountRequested,
            "LoanTerm": LoanTerm,
            "PurposeoftheLoan": PurposeoftheLoan,
            "EmploymentVerification": EmploymentVerification,
            "PreviousLoans": PreviousLoans,
            "ApplicationBehavior": ApplicationBehavior,
            "LocationofApplication": LocationofApplication,
            "AccountActivity": AccountActivity,
            "Blacklists": Blacklists,
            "PastFinancialMalpractices": PastFinancialMalpractices,
            "ConsistencyinData": ConsistencyinData,
        }
        
        # Create initial state
        initial_state = {
            "applicant_data": applicant_data_dict,
            "agent1_result": None,
            "agent2_result": None,
            "agent3_result": None,
            "final_decision": None,
            "reasoning_trace": None,
            "application_summary": None,
            "recommendation": None,
        }
        
        try:
            # Execute workflow
            final_state = app.invoke(
                initial_state, 
                config={"recursion_limit": 10}
            )
            
            # Clear GPU cache after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Extract results
            final_decision = final_state.get("final_decision", "Error")
            decision_reason = final_state.get("decision_reason", "No reason provided")
            weighted_score = final_state.get("weighted_score", 0.0)
            recommendation = final_state.get("recommendation", "")
            reasoning_trace = final_state.get("reasoning_trace", [])
            
            # Display Results
            st.markdown("---")
            st.subheader("📊 Final Decision")
            
            # Decision Badge
            if final_decision == "Approved":
                st.success(f"### ✅ {final_decision}")
            else:
                st.error(f"### ❌ {final_decision}")
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Weighted Risk Score", f"{weighted_score:.4f}")
            with col2:
                threshold = CONFIG["FINAL_APPROVAL_THRESHOLD"]
                st.metric("Approval Threshold", f"{threshold:.4f}")
            
            # Recommendation
            st.markdown("### 💡 Recommendation")
            st.info(recommendation)
            
            # Reasoning Trace
            with st.expander("🔍 Detailed Reasoning Trace"):
                for trace in reasoning_trace:
                    st.write(f"- {trace}")
            
            # Agent Details
            with st.expander("🤖 Agent-by-Agent Analysis"):
                agent1_result = final_state.get("agent1_result")
                agent2_result = final_state.get("agent2_result")
                agent3_result = final_state.get("agent3_result")
                
                if agent3_result:
                    st.markdown("#### Agent 3: Compliance Check")
                    st.json(agent3_result)
                
                if agent1_result:
                    st.markdown("#### Agent 1: Credit Risk Analysis")
                    st.json(agent1_result)
                
                if agent2_result:
                    st.markdown("#### Agent 2: Fraud Detection")
                    st.json(agent2_result)
            
            # Download Report
            report_json = {
                "applicant_data": applicant_data_dict,
                "agent1_result": agent1_result,
                "agent2_result": agent2_result,
                "agent3_result": agent3_result,
                "agent4_final_report": {
                    "application_summary": final_state.get("application_summary"),
                    "final_decision": final_decision,
                    "recommendation": recommendation,
                    "reasoning_trace": reasoning_trace
                }
            }
            
            st.download_button(
                label="📥 Download Full Report (JSON)",
                data=json.dumps(report_json, indent=2),
                file_name=f"loan_report_{applicant_data_dict['id']}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# -------------------------------------------------------------
# SIDEBAR: System Info
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔧 System Information")
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        gpu_mem_max = torch.cuda.max_memory_allocated() / 1024**3
        st.metric("GPU Memory Used", f"{gpu_mem:.2f} GB")
        st.metric("GPU Peak Memory", f"{gpu_mem_max:.2f} GB")
    
    st.markdown("### 📋 Model Status")
    st.success("Agent 1: ✅ Loaded")
    st.success("Agent 2: ✅ Loaded")
    st.success("Agent 3: ✅ Loaded")
    st.success("Agent 4: ✅ Loaded")
    
    if st.button("🔄 Clear Cache & Reload Models"):
        st.cache_resource.clear()
        st.rerun()
