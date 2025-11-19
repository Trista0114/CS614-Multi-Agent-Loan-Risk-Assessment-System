"""
=============================================================================
Agent 4 - Loan Application Decision System
=============================================================================
Before running, please ensure:
1. All dependencies are installed 
2. All paths are properly configured (see CONFIGURATION section below)
3. Agent modules (Agent_1.py, Agent_2.py, Agent_3.py) exist in the specified paths
4. Test dataset (Test Data.csv) exists in the specified path
5. You are logged into Hugging Face (requires HF_TOKEN)


=============================================================================
Memory Optimization Notes:
=============================================================================
Two Model Groups:
1. Agent 1: Independent Llama-3.1-8B Base (~7.5 GB)
   - Includes Agent 1 LoRA + ensemble models
   
2. Agents 2-4: Shared Llama-3.1-8B-Instruct (~8.5 GB total)
   - Base model loaded once (~6 GB)
   - Agent 2 adds LoRA adapters (~0.5 GB)
   - Agents 3-4 use base model directly

Total Memory: ~16 GB (vs ~27 GB without sharing)
Memory Savings: 41%

=============================================================================
Usage Example:
=============================================================================

# Method 1: Using default config
python Agent_4.py

# Method 2: Providing custom config externally (overrides only specified keys)
from Agent_4 import main

custom_config = {
    "TEST_DATA_PATH": "/path/to/your/test.csv",
    "OUTPUT_DIR": "/path/to/output/",
    "NUM_CASES_TO_PROCESS": 10
}

results = main(custom_config=custom_config)

# Unspecified keys will use the defaults defined inside this file.

=============================================================================
"""

import sys
import os
import json
import traceback
import importlib
from typing import TypedDict, Optional, List, Dict

import pandas as pd
import numpy as np
import torch

# LangGraph
from langgraph.graph import StateGraph, END

# Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =========================================================================
# CONFIGURATION SECTION
# =========================================================================

# Default CONFIG (internal)
DEFAULT_CONFIG = {
    # --- Hugging Face Token ---
    "HF_TOKEN": "YOUR_HF_TOKEN_HERE",
    
    "PROJECT_ROOT": "./",

    "AGENT1_MODULE_DIR": "./agents/",
    "AGENT2_MODULE_DIR": "./agents/",
    "AGENT3_MODULE_DIR": "./agents/",

    "AGENT1_MODEL_DIR": "./models/Agent1/",
    "AGENT1_LORA_DIR": "./models/Agent1/llama_lora_adapter/",
    "AGENT2_LORA_DIR": "./models/Agent2/lora_adapater_fraud/",

    "AGENT3_POLICY_FILE": "./data/Internal Policy.docx",

    "AGENT4_MODEL_ID": "meta-llama/Llama-3.1-8B-Instruct",

    "TEST_DATA_PATH": "./data/Test.csv",

    "OUTPUT_DIR": "./reports/",

    # --- Decision Thresholds ---
    "CREDIT_RISK_REJECT_THRESHOLD": 0.45,
    "FRAUD_RISK_REJECT_THRESHOLD": 0.5,
    "WEIGHT_CREDIT": 0.6,
    "WEIGHT_FRAUD": 0.4,
    "FINAL_APPROVAL_THRESHOLD": 0.4,
    
    # --- Number of Cases to Process ---
    "NUM_CASES_TO_PROCESS": 5,
}

# Initialize CONFIG (can be overridden in main)
CONFIG = DEFAULT_CONFIG.copy()

# Validate required CONFIG keys
def validate_config():
    """Validate required CONFIG keys"""
    required_keys = [
        "HF_TOKEN", "TEST_DATA_PATH", "OUTPUT_DIR", 
        "NUM_CASES_TO_PROCESS", "AGENT4_MODEL_ID"
    ]
    missing_keys = [key for key in required_keys if key not in CONFIG]
    if missing_keys:
        raise ValueError(f"Missing required CONFIG keys: {missing_keys}")

# =========================================================================
# STEP 1: Environment Setup
# =========================================================================

def setup_environment():
    """Set up environment: add paths and log in to Hugging Face"""
    print("=" * 70)
    print("STEP 1: Environment Setup")
    print("=" * 70)
    
    paths_to_add = [
        CONFIG["AGENT1_MODULE_DIR"],
        CONFIG["AGENT2_MODULE_DIR"],
        CONFIG["AGENT3_MODULE_DIR"],
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.append(path)
            print(f"✓ Added to sys.path: {path}")
    
    try:
        from huggingface_hub import login
        hf_token = os.environ.get('HF_TOKEN', CONFIG["HF_TOKEN"])
        if hf_token and hf_token != "YOUR_HF_TOKEN_HERE":
            login(token=hf_token)
            print("✓ Logged in to Hugging Face")
        else:
            print("⚠ HF_TOKEN not set")
    except Exception as e:
        print(f"⚠ Hugging Face login failed: {e}")
    
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    print(f"✓ Output directory ready: {CONFIG['OUTPUT_DIR']}")
    print("✓ Environment setup complete\n")

# =========================================================================
# STEP 2: Load Agent Modules
# =========================================================================

def load_agent_modules():
    """Load all Agent modules"""
    print("=" * 70)
    print("STEP 2: Load Agent Modules")
    print("=" * 70)
    
    global Agent_1, Agent_2, Agent_3
    
    try:
        import Agent_1
        print("✓ Agent_1 module loaded")
    except ImportError as e:
        print(f"✗ Failed to import Agent_1: {e}")
        raise
    
    try:
        import Agent_2
        importlib.reload(Agent_2)
        print("✓ Agent_2 module loaded")
    except ImportError as e:
        print(f"✗ Failed to import Agent_2: {e}")
        raise
    
    try:
        import Agent_3
        importlib.reload(Agent_3)
        print("✓ Agent_3 module loaded")
    except ImportError as e:
        print(f"✗ Failed to import Agent_3: {e}")
        raise
    
    print("✓ All agent modules loaded\n")
    return Agent_1, Agent_2, Agent_3

# =========================================================================
# STEP 3–6: Initialize Agents (retain original logic)
# =========================================================================

def initialize_agent1():
    """Initialize Agent 1 model"""
    print("=" * 70)
    print("STEP 3: Initialize Agent 1 (Credit Risk Analyzer)")
    print("=" * 70)
    
    try:
        from Agent_1 import CreditRiskAnalyzer
        
        print("Loading Agent 1 models...")
        analyzer = CreditRiskAnalyzer(
            model_dir=CONFIG["AGENT1_MODEL_DIR"],
            lora_dir=CONFIG["AGENT1_LORA_DIR"]
        )
        print("✓ Agent 1 initialized successfully")
        
        # Check if Agent 1 includes LLM (usually Llama)
        llm_model = None
        llm_tokenizer = None
        
        if hasattr(analyzer, 'llm') and hasattr(analyzer, 'tokenizer'):
            llm_model = analyzer.llm
            llm_tokenizer = analyzer.tokenizer
            print("✓ Agent 1 LLM available for sharing")
        else:
            print("⚠ Agent 1 LLM not accessible (will load separately for Agent 3)")
        
        print()
        return analyzer, llm_model, llm_tokenizer
    
    except Exception as e:
        print(f"✗ Agent 1 initialization failed: {e}")
        traceback.print_exc()
        return None, None, None

def initialize_agent2(shared_model=None):
    """
    Initialize Agent 2 model (Fraud Detection with LoRA)

    Notes:
    - Agent 2 uses the shared Llama-3.1-8B-Instruct base model + LoRA adapters.
    - Saves GPU memory by loading the base model only once and applying adapters on demand.
    """
    print("=" * 70)
    print("STEP 4: Initialize Agent 2 (Fraud Detection with LoRA)")
    print("=" * 70)

    try:
        import Agent_2
        from Agent_2 import agent2_predict_from_raw

        original_dir = os.getcwd()
        agent2_dir = CONFIG["AGENT2_MODULE_DIR"]

        print(f"Switching to Agent 2 directory: {agent2_dir}")
        os.chdir(agent2_dir)

        if shared_model is not None:
            print("✓ Using shared Llama-3.1-8B-Instruct base model")
            print("⚠ Note: Agent 2 will apply LoRA adapters during inference")

            base_model = shared_model
            
            # --- MODIFICATION START ---
            
            print("Loading Agent 2 tokenizer...")
          
            tokenizer = AutoTokenizer.from_pretrained(CONFIG["AGENT4_MODEL_ID"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # --- MODIFICATION END ---
            
        else:
            print("Loading Agent 2 LLM with LoRA adapters…")
            base_model, tokenizer = Agent_2.initialize_model()

        Agent_2._AGENT2_MODEL = base_model
        Agent_2._AGENT2_TOKENIZER = tokenizer

        os.chdir(original_dir)
        print("✓ Agent 2 initialized with shared base model + LoRA adapters\n")
        return True

    except Exception as e:
        # ... (error handling) ...
        return False

    except Exception as e:
        print(f"✗ Agent 2 initialization failed: {e}")
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os.chdir(original_dir)
        return False




def initialize_agent3(shared_llm_model=None):
    """Initialize Agent 3 (Compliance Check) — optionally reuse shared LLM"""
    print("=" * 70)
    print("STEP 5: Initialize Agent 3 (Compliance Check)")
    print("=" * 70)

    try:
        import Agent_3
        policy_file = CONFIG["AGENT3_POLICY_FILE"]
        print(f"Loading policy document: {policy_file}")
        internal_policy_text = Agent_3.load_policy_from_docx(policy_file)
        print("✓ Policy document loaded")

        print("Building knowledge base and graph…")
        G, embedding_fn, kb_data = Agent_3.build_and_index_knowledge_base(internal_policy_text)
        print("✓ Knowledge base built")

        # --- MODIFICATION START ---
        
        # 1. Condition changed (no longer checks for shared_llm_tokenizer)
        if shared_llm_model is not None:
            print("✓ Reusing shared LLM for memory optimization")
            llm_model = shared_llm_model
            
            # 2. Load tokenizer manually instead of receiving it
            print("Loading Agent 3 tokenizer...")
            # (Assuming Agent_3 imports AutoTokenizer, based on initialize_agent4)
            llm_tokenizer = Agent_3.AutoTokenizer.from_pretrained(CONFIG["AGENT4_MODEL_ID"])
            if llm_tokenizer.pad_token is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
            print("✓ Agent 3 tokenizer loaded")

        # --- MODIFICATION END ---
        
        else:
            # This "else" block is unchanged
            print("Loading Agent 3 LLM (Llama-3.1-8B-Instruct)…")
            llm_model, llm_tokenizer = Agent_3.load_llama3_instruct_model()
            print("✓ Agent 3 LLM loaded")

        def llm_invoke_fn(prompt: str, max_new_tokens: int = 2000) -> str:
            inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
            with torch.no_grad():
                outputs = llm_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    pad_token_id=llm_tokenizer.eos_token_id
                )
            generated = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_only = generated
            if generated.startswith(prompt):
                response_only = generated[len(prompt):].strip()
            else:
                json_start = generated.find('{')
                if json_start != -1:
                    response_only = generated[json_start:].strip()
            json_str = Agent_3.extract_first_json_object(response_only)
            return json_str if json_str else response_only

        print("✓ Agent 3 initialized successfully\n")
        return {
            'G': G,
            'embedding_fn': embedding_fn,
            'kb_data': kb_data,
            'llm_invoke_fn': llm_invoke_fn,
            'llm_model': llm_model,
            'llm_tokenizer': llm_tokenizer
        }

    except Exception as e:
        print(f"✗ Agent 3 initialization failed: {e}")
        traceback.print_exc()
        return None


def initialize_agent4(shared_llm_model=None):
    """
    Initialize Agent 4 LLM (Fallback mode only if Agent 3 fails)
    """
    print("=" * 70)
    print("STEP 6: Initialize Agent 4 (Final Decision LLM) – Fallback Mode")
    print("=" * 70)

    try:
        # --- MODIFICATION START ---
        
        # 1. Condition changed (no longer checks for shared_llm_tokenizer)
        if shared_llm_model is not None:
            print("✓ Reusing shared LLM (fallback mode)")
            
            # 2. Load tokenizer manually instead of receiving it
            import Agent_3 # Need this to get AutoTokenizer
            model_id = CONFIG["AGENT4_MODEL_ID"]
            print(f"Loading tokenizer from {model_id} (fallback mode)…")
            tokenizer = Agent_3.AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print("✓ Agent 4 LLM initialized successfully\n")
            # Return shared model, but newly loaded tokenizer
            return shared_llm_model, tokenizer

        # --- MODIFICATION END ---
            
        # This 'else' block (when shared_llm_model is None) is unchanged
        # It loads both the model and tokenizer from scratch

        import Agent_3
        model_id = CONFIG["AGENT4_MODEL_ID"]

        bnb_config = Agent_3.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        print(f"Loading tokenizer from {model_id}…")
        tokenizer = Agent_3.AutoTokenizer.from_pretrained(model_id)
        print(f"Loading model from {model_id} (4-bit quantized)…")
        model = Agent_3.AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("✓ Agent 4 LLM loaded successfully\n")
        return model, tokenizer

    except Exception as e:
        print(f"✗ Agent 4 initialization failed: {e}")
        traceback.print_exc()
        return None, None

# =========================================================================
# LangGraph State Definition
# =========================================================================

class AgentState(TypedDict):
    applicant_data: dict
    agent1_result: Optional[dict]
    agent2_result: Optional[dict]
    agent3_result: Optional[dict]
    final_decision: Optional[str]
    reasoning_trace: Optional[List[str]]
    application_summary: Optional[str]
    recommendation: Optional[str]
    weighted_score: Optional[float]
    decision_reason: Optional[str]
    applicant_id: Optional[str]

# =========================================================================
# STEP 8: Define All Node Functions
# =========================================================================

def create_agent1_node(analyzer):
    """Create Agent 1 node"""
    def call_agent1_node(state: AgentState) -> dict:
        print("--- [REAL] Calling Agent 1: Credit Risk ---")
        
        if analyzer is None:
            return {
                "agent1_result": {
                    "agent_id": "agent_1",
                    "agent_name": "Agent 1 - Credit Risk Analyst",
                    "analysis": {
                        "risk_level": "ERROR",
                        "reasoning": "Agent 1 not loaded.",
                        "default_probability": 1.0
                    }
                }
            }
        
        try:
            app_data = state.get("applicant_data", {})
            
            def to_float(val, default=0.0):
                try:
                    if val in ["", None, "None", "nan"]:
                        return default
                    return float(val)
                except (ValueError, TypeError):
                    return default
            
            def to_int(val, default=0):
                try:
                    if val in ["", None, "None", "nan"]:
                        return default
                    return int(float(val))
                except (ValueError, TypeError):
                    return default
            
            applicant_data = {
                "rev_util": to_float(app_data.get("rev_util")),
                "age": to_int(app_data.get("age")),
                "debt_ratio": to_float(app_data.get("debt_ratio")),
                "monthly_inc": to_float(app_data.get("monthly_inc")),
                "open_credit": to_int(app_data.get("open_credit")),
                "late_90": to_int(app_data.get("late_90")),
                "applicant_id": app_data.get("id", "unknown")
            }
            
            result = analyzer.analyze(applicant_data)
            
            print(f"✓ Agent 1: {result['analysis']['risk_level']}, "
                  f"Prob: {result['analysis']['default_probability']:.4f}")
            
            return {"agent1_result": result}
        
        except Exception as e:
            print(f"!!! Agent 1 Error: {e} !!!")
            traceback.print_exc()
            return {
                "agent1_result": {
                    "agent_id": "agent_1",
                    "agent_name": "Agent 1 - Credit Risk Analyst",
                    "analysis": {
                        "risk_level": "ERROR",
                        "reasoning": str(e),
                        "default_probability": 1.0,
                    }
                }
            }
    
    return call_agent1_node
    
    return call_agent1_node

def create_agent2_node():
    """Create Agent 2 node"""
    def call_agent2_node(state: AgentState) -> dict:
        print("--- [REAL] Calling Agent 2: Fraud Risk ---")
        
        try:
            from Agent_2 import agent2_predict_from_raw
            
            app_data = state.get('applicant_data', {})
            analysis_result = agent2_predict_from_raw(app_data)
            
            fraud_prob = analysis_result.get("fraud_probability")
            
            if isinstance(fraud_prob, str):
                try:
                    fraud_prob = float(fraud_prob)
                except ValueError:
                    fraud_prob = None
            
            if fraud_prob is None:
                fraud_prob = 100.0
            
            if fraud_prob > 1.0:
                fraud_prob /= 100.0
            
            analysis_result["fraud_probability"] = round(fraud_prob, 4)
            
            print(f"✓ Agent 2: Fraud Prob: {fraud_prob:.4f}")
            return {"agent2_result": analysis_result}
        
        except Exception as e:
            print(f"!!! Agent 2 Error: {e} !!!")
            traceback.print_exc()
            return {
                "agent2_result": {
                    "risk_level": "ERROR",
                    "reasoning": str(e),
                    "fraud_probability": 1.0
                }
            }
    
    return call_agent2_node

def create_agent3_node(agent3_resources):
    """Create Agent 3 node"""
    def call_agent3_node(state: AgentState) -> dict:
        print("--- Calling Agent 3: Compliance Check ---")
        
        if agent3_resources is None:
            return {
                "agent3_result": {
                    "Overall_compliance_status": "SETUP_ERROR",
                    "error_message": "Agent 3 not loaded"
                }
            }
        
        try:
            import Agent_3
            
            app_data = state.get('applicant_data', {})
            
            debt_ratio = app_data.get("debt_ratio")
            dti_percentage = None
            
            if debt_ratio is not None:
                try:
                    dti_percentage = float(debt_ratio) * 100.0
                except (ValueError, TypeError):
                    dti_percentage = None
            
            applicant_info = {
                "Age": app_data.get("Age"),
                "LoanTerm": app_data.get("LoanTerm"),
                "DTI": dti_percentage,
                "CreditScore": app_data.get("CreditScore"),
                "EmploymentVerification": app_data.get("EmploymentVerification"),
                "PurposeoftheLoan": app_data.get("PurposeoftheLoan")
            }
            
            report = Agent_3.run_compliance_graphrag(
                applicant_info=applicant_info,
                kb_data=agent3_resources['kb_data'],
                embedding_fn=agent3_resources['embedding_fn'],
                llm_invoke_fn=agent3_resources['llm_invoke_fn'],
                top_k=2
            )
            
            status = report.get("Overall_compliance_status", "N/A")
            print(f"✓ Agent 3: {status}")
            return {"agent3_result": report}
        
        except Exception as e:
            print(f"!!! Agent 3 Error: {e} !!!")
            traceback.print_exc()
            return {
                "agent3_result": {
                    "Overall_compliance_status": "ERROR",
                    "error_message": str(e)
                }
            }
    
    return call_agent3_node

def calculate_weighted_score(state: AgentState) -> dict:
    """Calculate weighted score"""
    print("--- Node: Calculating Weighted Score ---")
    
    agent1_result = state.get('agent1_result', {})
    agent1_analysis = agent1_result.get('analysis', {})
    default_prob = agent1_analysis.get("default_probability", 0.0)
    
    agent2_result = state.get('agent2_result', {})
    fraud_prob = agent2_result.get("fraud_probability", 0.0)
    
    # Read weights from CONFIG
    weight_credit = CONFIG["WEIGHT_CREDIT"]
    weight_fraud = CONFIG["WEIGHT_FRAUD"]
    
    score = (default_prob * weight_credit) + (fraud_prob * weight_fraud)
    
    print(f"✓ Weighted Score calculated: {score:.4f}")
    print(f"  - Credit Risk (prob={default_prob:.4f}, weight={weight_credit})")
    print(f"  - Fraud Risk (prob={fraud_prob:.4f}, weight={weight_fraud})")
    
    return {"weighted_score": score}

def set_decision_approve(state: AgentState) -> dict:
    score = state.get("weighted_score", 0.0)
    return {
        "final_decision": "Approved",
        "decision_reason": f"Weighted score ({score:.4f}) below threshold."
    }

def set_decision_fail_compliance(state: AgentState) -> dict:
    return {
        "final_decision": "Disapproved",
        "decision_reason": "Failed compliance checks."
    }

def set_decision_fail_credit(state: AgentState) -> dict:
    agent1_result = state.get('agent1_result', {})
    agent1_analysis = agent1_result.get('analysis', {})
    prob = agent1_analysis.get("default_probability", 0.0)
    
    return {
        "final_decision": "Disapproved",
        "decision_reason": f"High credit risk (prob: {prob:.4f})."
    }

def set_decision_fail_fraud(state: AgentState) -> dict:
    prob = state.get('agent2_result', {}).get("fraud_probability", 0.0)
    return {
        "final_decision": "Disapproved",
        "decision_reason": f"High fraud risk (prob: {prob:.4f})."
    }

def set_decision_fail_weighted(state: AgentState) -> dict:
    score = state.get("weighted_score", 0.0)
    return {
        "final_decision": "Disapproved",
        "decision_reason": f"Weighted score ({score:.4f}) exceeded threshold."
    }

def create_report_generator(agent4_model, agent4_tokenizer, call_agent1_fn, call_agent2_fn):
    """Create report generation node"""
    
    def llm_invoke_fn_agent4(prompt: str, max_new_tokens: int = 2000) -> str:
        messages = [{"role": "user", "content": prompt}]
        text_prompt = agent4_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = agent4_tokenizer(text_prompt, return_tensors="pt").to(agent4_model.device)
        
        with torch.no_grad():
            outputs = agent4_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=agent4_tokenizer.eos_token_id
            )
        
        response_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated = agent4_tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return generated.strip()
    
    def generate_report(state: AgentState) -> dict:
        print("--- Node: Generating Final Report ---")
        
        app_data = state.get('applicant_data', {})
        applicant_id = app_data.get('Test Case ID', 
                                    app_data.get('Test_Case_ID',
                                                app_data.get('id', 'N/A')))
        
        final_decision = state.get("final_decision", "Error")
        rule_based_reason = state.get("decision_reason", "No reason")
        
        a1_report = state.get('agent1_result')
        a2_report = state.get('agent2_result')
        a3_report = state.get('agent3_result') or {}
        
        if final_decision == "Disapproved" and a1_report is None:
            print("--- Running A1 & A2 for feedback ---")
            try:
                a1_report = call_agent1_fn(state).get('agent1_result') or {}
            except Exception as e:
                a1_report = {"reasoning": f"Failed: {e}", "default_probability": 0.0}
            
            try:
                a2_report = call_agent2_fn(state).get('agent2_result') or {}
            except Exception as e:
                a2_report = {"reasoning": f"Failed: {e}", "fraud_probability": 0.0}
        
        elif final_decision == "Disapproved" and a2_report is None:
            print("--- Running A2 for feedback ---")
            try:
                a2_report = call_agent2_fn(state).get('agent2_result') or {}
            except Exception as e:
                a2_report = {"reasoning": f"Failed: {e}", "fraud_probability": 0.0}
        
        a1_report = a1_report or {}
        a2_report = a2_report or {}
        
        applicant_summary = (
            f"Application ID: {applicant_id}, "
            f"Age: {app_data.get('Age')}, "
            f"Purpose: {app_data.get('PurposeoftheLoan')}, "
            f"Credit Score: {app_data.get('CreditScore')}, "
            f"Debt Ratio: {float(app_data.get('debt_ratio', 0)):.2f}"
        )
        
        trace_parts = []
        a3_status = a3_report.get("Overall_compliance_status", "N/A")
        trace_parts.append(f"1. Compliance: {a3_status}")
        
        a1_threshold = CONFIG["CREDIT_RISK_REJECT_THRESHOLD"]
        if a1_report:
    # ✅ 正確：從 analysis 中取值
            a1_analysis = a1_report.get("analysis", {})
            a1_prob = a1_analysis.get("default_probability", 0.0)
            a1_status = "FAIL" if a1_prob > CONFIG["CREDIT_RISK_REJECT_THRESHOLD"] else "PASS"
            trace_parts.append(
        f"2. Credit Risk: {a1_status} (prob: {a1_prob:.4f}, threshold: {a1_threshold})"
    )
    
        a2_threshold = CONFIG["FRAUD_RISK_REJECT_THRESHOLD"]
        if a2_report:
            a2_prob = a2_report.get("fraud_probability", 0.0)
            a2_status = "FAIL" if a2_prob > CONFIG["FRAUD_RISK_REJECT_THRESHOLD"] else "PASS"
            trace_parts.append(
        f"3. Fraud Risk: {a2_status} (prob: {a2_prob:.4f} threshold: {a2_threshold})"
    )
        ws_threshold = CONFIG["FINAL_APPROVAL_THRESHOLD"]
        weighted_score = state.get('weighted_score')
        if weighted_score is not None:
            ws_status = "FAIL" if weighted_score > CONFIG["FINAL_APPROVAL_THRESHOLD"] else "PASS"
            trace_parts.append(
        f"4. Weighted Score: {ws_status} ({weighted_score:.4f} threshold: {ws_threshold})"
    )
        
        trace_parts.append(f"5. Final Decision: {final_decision}")
        reasoning_trace_list = trace_parts
        
        final_recommendation_text = f"{final_decision}. {rule_based_reason}"
        
        if agent4_model is not None:
            print("Calling LLM for summary...")
            
            a1_analysis = a1_report.get("analysis", {})
            a1_reason = a1_analysis.get("reasoning", "No credit reasoning")
            a1_prob_val = a1_analysis.get("default_probability", 0.0)
            a1_status = "FAIL" if a1_prob_val > CONFIG["CREDIT_RISK_REJECT_THRESHOLD"] else "PASS"

            a2_reason = a2_report.get("reasoning", "No fraud reasoning")
            a2_prob_val = a2_report.get("fraud_probability", 0.0)
            a2_status = "FAIL" if a2_prob_val > CONFIG["FRAUD_RISK_REJECT_THRESHOLD"] else "PASS"
            
            a3_fail_details = ""
            if a3_status != "PASS":
                try:
                    fail_reasons = [
                        f"- {check.get('check')}: {check.get('justification')}"
                        for check in a3_report.get("check_details", [])
                        if check.get("result", "").upper() != "PASS"
                    ]
                    a3_fail_details = "\n".join(fail_reasons) if fail_reasons else "No details"
                except Exception:
                    a3_fail_details = "Error parsing"

            failed_agents = []
            if a1_status == "FAIL":
                failed_agents.append("Agent 1 (Credit Risk)")
            if a2_status == "FAIL":
                failed_agents.append("Agent 2 (Fraud Risk)")
            if a3_status == "FAIL" or a3_status != "PASS":
                failed_agents.append("Agent 3 (Compliance)")
            
            failed_agents_list = " and ".join(failed_agents) if failed_agents else "None"
            
            prompt = f"""You are a senior loan officer writing a final "Reasoning" summary.
Your task is to synthesize all provided information into a **single, coherent paragraph**.

**Information to Synthesize:**
1.  **Final Decision:** {final_decision}
2.  **Primary Reason (from rules):** {rule_based_reason}

**Agent Assessment Results:**
3.  **Agent 1 (Credit Risk)**
    - Status: {a1_status}
    - Default Probability: {a1_prob_val:.4f}
    - Reasoning: {a1_reason}

4.  **Agent 2 (Fraud Risk)**
    - Status: {a2_status}
    - Fraud Probability: {a2_prob_val:.4f}
    - Reasoning: {a2_reason}

5.  **Agent 3 (Compliance)**
    - Status: {a3_status}
    - Failure Details: {a3_fail_details if a3_fail_details else "N/A"}


*CRITICAL INSTRUCTIONS - READ CAREFULLY:**

1. Start with: "{final_decision}"
2. **Do not say** any words like "Agent5", there is no Agent5!! We have ONLY Agent1 (Credit Risk),Agent 2 (Fraud Risk) and Agent 3 (Compliance). 
3. If {final_decision} = Approve, **DO NOT** mention any word "Disapprove" and "Disapproval"!! Don't use any transition word like "However"!
4. If all {a3_status} {a2_status} {a1_status} is Fail, Don't use any transition word like "However" and "Despite"!
5. If the probability and weighted score are lOWER than threshold, it means the agent is pass. 

6. **CRITICAL - The agents that FAILED are listed above as: {failed_agents_list}**
   If the {final_decision} = Disapproved, follow above structure
   **In your "Disapproved because of..." statement:**
   - ✅ CORRECT: Use EXACTLY the failed agents listed above: "{failed_agents_list}"
   - ❌ WRONG: Do NOT add any agents not in the failed agents list
   - ❌ WRONG: Do NOT mention agents with Status = "PASS" as reasons for disapproval
   
   **Your opening sentence MUST be:**
   - "Disapproved because of {failed_agents_list}."
   
   **DO NOT say:**
   - "Disapproved because of Agent X and Agent Y" if Agent X or Y is not in the failed agents list
   - **Do not say** any words like "Agent5", there is no agent5 in this entire project!!
   - If {final_decision} = Approve, **do not say any word "Disapprove"!!**  Please briefly explain why all agents passed and the application was strong**

7. **After the "because of..." statement:**
   - First, explain WHY each failed agent failed (based on their Status and details below)
   - Then, you can mention ALL passing agents of why they pass. 
   - Passing agents provide positive context but are NOT failure reasons

8. **Structure:**
   - Line 1: "Disapproved because of {failed_agents_list}."
   - Lines 2-N: Explain why each failed agent failed
   - Final lines: Briefly mention what each passing agent found.

9. If the {final_decision} = Approved, briefly explain why all agents passed and the application was strong**


**EXAMPLES:**

**❌ WRONG Example - DO NOT DO THIS:**
If Agent 1 Status = PASS, Agent 2 Status = PASS, Agent 3 Status = FAIL:
"Disapproved because of Agent 3 (Compliance) and Agent 1 (Credit Risk)..." ← WRONG! Agent 1 is PASS!

If Agent 1 Status = PASS, Agent 2 Status = PASS, Agent 3 Status = PASS:
"Approved. Disapproved because of None."

**✅ CORRECT EXAMPLES:**

**Example 1 - Only Agent 3 Failed (Agent 1 and 2 Passed):**
"Disapproved because of Agent 3 (Compliance)."

**Example 2 - Agent 1 and Agent 3 Failed (Agent 2 Passed):**
"Disapproved because of Agent 1 (Credit Risk) and Agent 3 (Compliance). "

**Example 3 - Only Agent 1 Failed (Agent 2 and 3 Passed):**
"Disapproved because of Agent 1 (Credit Risk). "

**Example 4 - All Agents Passed (Approved):**
"Approved."The application was strong due to...

**Respond with ONLY this single, combined paragraph.**
"""

            try:
                llm_summary = llm_invoke_fn_agent4(prompt)
                final_recommendation_text = llm_summary.strip().strip('\"').strip()
                print("✓ LLM summary generated")
            except Exception as e:
                print(f"⚠ LLM summary failed: {e}")
        
        return {
            "applicant_id": applicant_id,
            "final_decision": final_decision,
            "application_summary": applicant_summary,
            "decision_reason": rule_based_reason,
            "recommendation": final_recommendation_text,
            "reasoning_trace": reasoning_trace_list,
            "agent1_result": a1_report,
            "agent2_result": a2_report,
            "agent3_result": a3_report,
        }
    
    return generate_report


# =========================================================================
# STEP 9: Define Routing Functions
# =========================================================================

def route_after_agent_3(state: AgentState) -> str:
    """
    Routing function: Conditional routing after Agent 3
    
    Determines the workflow path based on the compliance check result:
    - PASS: Continue to Agent 1 (Credit Risk Assessment)
    - FAIL: Directly disapprove (fail_compliance)
    """
    status = state.get('agent3_result', {}).get("Overall_compliance_status")
    if status == "PASS":
        return "continue_to_agent_1"
    else:
        return "fail_compliance"

def route_after_agent_1(state: AgentState) -> str:
    """Routing: Determines whether to continue or reject after Agent 1 (based on credit risk)"""
    agent1_result = state.get('agent1_result', {})
    agent1_analysis = agent1_result.get('analysis', {})
    default_prob = agent1_analysis.get("default_probability", 1.0)
  
    threshold = CONFIG["CREDIT_RISK_REJECT_THRESHOLD"]
    if default_prob > threshold:
        return "fail_credit"
    else:
        return "continue_to_agent_2"

def route_after_agent_2(state: AgentState) -> str:
    """Routing: Determines whether to continue or reject after Agent 2 (based on fraud risk)"""
    # Safely read the value from CONFIG
    threshold = CONFIG.get("FRAUD_RISK_REJECT_THRESHOLD")
    fraud_prob = state.get('agent2_result', {}).get("fraud_probability", 1.0)
    threshold = CONFIG["FRAUD_RISK_REJECT_THRESHOLD"]
    if fraud_prob > threshold:
        return "fail_fraud"
    else:
        return "continue_to_calculation"

def route_after_calculation(state: AgentState) -> str:
    """Routing: Decide approval or rejection after weighted score calculation"""
    score = state.get("weighted_score")
    threshold = CONFIG["FINAL_APPROVAL_THRESHOLD"]

    if score is None:
        print("⚠ WARNING: weighted_score is None! Using default 1.0")
        score = 1.0

    if score < threshold:
        return "approve_weighted"
    else:
        return "fail_weighted"

# =========================================================================
# STEP 10: Build LangGraph Workflow
# =========================================================================

def build_workflow(analyzer_agent1, agent3_resources, agent4_model, agent4_tokenizer):
    """Build the complete LangGraph workflow"""
    print("=" * 70)
    print("STEP 7: Build LangGraph Workflow")
    print("=" * 70)
    
    # Create node functions
    call_agent1_fn = create_agent1_node(analyzer_agent1)
    call_agent2_fn = create_agent2_node()
    call_agent3_fn = create_agent3_node(agent3_resources)
    generate_report_fn = create_report_generator(
        agent4_model, 
        agent4_tokenizer,
        call_agent1_fn,
        call_agent2_fn
    )
    
    # Initialize StateGraph
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    print("Adding nodes...")
    workflow.add_node("call_agent_3", call_agent3_fn)
    workflow.add_node("call_agent_1", call_agent1_fn)
    workflow.add_node("call_agent_2", call_agent2_fn)
    workflow.add_node("calculate_weighted_score", calculate_weighted_score)
    workflow.add_node("set_decision_approve", set_decision_approve)
    workflow.add_node("set_decision_fail_compliance", set_decision_fail_compliance)
    workflow.add_node("set_decision_fail_credit", set_decision_fail_credit)
    workflow.add_node("set_decision_fail_fraud", set_decision_fail_fraud)
    workflow.add_node("set_decision_fail_weighted", set_decision_fail_weighted)
    workflow.add_node("generate_report", generate_report_fn)
    print("✓ All nodes added")
    
    # Set entry point
    workflow.set_entry_point("call_agent_3")
    
    # Add conditional edges
    print("Adding conditional edges...")
    workflow.add_conditional_edges(
        "call_agent_3",
        route_after_agent_3,
        {
            "continue_to_agent_1": "call_agent_1",
            "fail_compliance": "set_decision_fail_compliance"
        }
    )
    
    workflow.add_conditional_edges(
        "call_agent_1",
        route_after_agent_1,
        {
            "continue_to_agent_2": "call_agent_2",
            "fail_credit": "set_decision_fail_credit"
        }
    )
    
    workflow.add_conditional_edges(
        "call_agent_2",
        route_after_agent_2,
        {
            "continue_to_calculation": "calculate_weighted_score",
            "fail_fraud": "set_decision_fail_fraud"
        }
    )
    
    workflow.add_conditional_edges(
        "calculate_weighted_score",
        route_after_calculation,
        {
            "approve_weighted": "set_decision_approve",
            "fail_weighted": "set_decision_fail_weighted"
        }
    )
    print("✓ All conditional edges added")
    
    # Add edges to report node
    print("Adding edges to report generation...")
    workflow.add_edge("set_decision_approve", "generate_report")
    workflow.add_edge("set_decision_fail_compliance", "generate_report")
    workflow.add_edge("set_decision_fail_credit", "generate_report")
    workflow.add_edge("set_decision_fail_fraud", "generate_report")
    workflow.add_edge("set_decision_fail_weighted", "generate_report")
    
    # Define end of workflow
    workflow.add_edge("generate_report", END)
    print("✓ All edges connected")
    
    # Compile
    print("Compiling workflow...")
    app = workflow.compile()
    print("✓ Workflow compiled successfully\n")
    
    return app

# =========================================================================
# STEP 11: Execute Test Cases
# =========================================================================

def run_test_cases(df_test, app):
    """Execute test cases"""
    print("=" * 70)
    print("STEP 8: Load test data and execute")
    print("=" * 70)
    
    all_final_reports = []
    
    # Safely get the number of cases to process
    max_cases = CONFIG.get("NUM_CASES_TO_PROCESS", 5)  # Default is 5
    num_cases = min(max_cases, len(df_test))
    rows_to_process = df_test.head(num_cases)
    
    print(f"Processing the first {num_cases} cases\n")
    
    for index, row in rows_to_process.iterrows():
        applicant_data_dict = row.to_dict()
        
        # Use ID from df_test
        applicant_id = row.get('Test Case ID', 
                               row.get('Test_Case_ID',
                                      row.get('ID',
                                             row.get('id', index))))
        
        applicant_data_dict['id'] = applicant_id
        
        print("=" * 56)
        print(f"--- EXECUTING LANGGRAPH FOR APPLICANT ID: {applicant_id} ---")
        print("=" * 56)
        
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
            # Execute graph
            final_state = app.invoke(initial_state, config={"recursion_limit": 10})
            
            print("\n--- EXECUTION COMPLETE ---")
            
            # Reformat structure
            structured_report = {
                "applicant_data": final_state.get("applicant_data"),
                "agent1_result": final_state.get("agent1_result"),
                "agent2_result": final_state.get("agent2_result"),
                "agent3_result": final_state.get("agent3_result"),
                "agent4_final_report": {
                    "application_summary": final_state.get("application_summary"),
                    "final_decision": final_state.get("final_decision"),
                    "recommendation": final_state.get("recommendation"),
                    "reasoning_trace": final_state.get("reasoning_trace")
                }
            }
            
            # Print JSON
            print("\n--- FINAL OUTPUT (JSON) ---")
            print(json.dumps(structured_report, indent=2, ensure_ascii=False))
            
            # Save to file
            file_name = os.path.join(CONFIG["OUTPUT_DIR"], f"report_applicant_{applicant_id}.json")
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(structured_report, f, indent=2, ensure_ascii=False)
                print(f"\n✓ Successfully saved report to: {file_name}")
            except Exception as save_e:
                print(f"\n⚠ Error saving report: {save_e}")
            
            all_final_reports.append(structured_report)
        
        except Exception as e:
            print(f"\n!!! EXECUTION FAILED for ID {applicant_id}: {e} !!!")
            traceback.print_exc()
            all_final_reports.append({
                "error": str(e), 
                "applicant_id": applicant_id
            })
    
    # Clear GPU cache after batch processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n✓ GPU cache cleared after batch processing")
    
    return all_final_reports


# =========================================================================
# MAIN EXECUTION
# =========================================================================

def main(custom_config=None):
    """
    Main execution function
    ...
    """
    # If a custom config is provided, merge it into CONFIG
    if custom_config is not None:
        # Update module-level CONFIG
        CONFIG.update(custom_config)
        print("✓ Using custom configuration merged with default values")
    
    print("\n" + "=" * 70)
    print("Agent 4 - Loan Application Decision System")
    print("=" * 70 + "\n")
    
    try:
        # Step 0: Validate CONFIG
        validate_config()
        
        # Step 1: Environment setup
        setup_environment()
        
        # Step 2: Load agent modules
        Agent_1, Agent_2, Agent_3 = load_agent_modules()
        
        # --- MODIFICATION START ---
        
        # Step 3: Initialize Agent 1 (Standalone)
        # We call Agent 1, but we DO NOT use its returned model for sharing.
        analyzer_agent1, _, _ = initialize_agent1()
        
        # Step 4: Load the SHARED model for Agents 2, 3, and 4
        # We call initialize_agent4() with shared_llm_model=None
        # This forces it to load the model from CONFIG["AGENT4_MODEL_ID"]
        print("=" * 70)
        print("Loading shared LLM for Agents 2, 3, 4 (Llama-3.1-8B-Instruct)...")
        print("=" * 70)
        master_llm_model, master_llm_tokenizer = initialize_agent4(shared_llm_model=None)
        
        if master_llm_model is None:
             print("\n✗ FATAL: Failed to load the shared model for Agents 2-4.")
             # Stop execution if the shared model fails to load
             raise RuntimeError("Failed to initialize shared LLM for Agents 2-4")
        
        # Step 5: Initialize Agent 2 (shares the new master model)
        agent2_loaded = initialize_agent2(master_llm_model)
        
        # Step 5: Unified management (Updated print statements)
        print("=" * 70)
        print("📌 Memory Optimization: Dual Model Groups")
        print("=" * 70)
        print("✓ Group 1: Agent 1 (Llama-3.1-8B Base + LoRA)")
        print("✓ Group 2: Shared Llama-3.1-8B-Instruct (4-bit)")
        print("  ✓ Agent 2: Uses Group 2 Model + LoRA adapters")
        print("  ✓ Agent 3: Uses Group 2 Model (shared)")
        print("  ✓ Agent 4: Uses Group 2 Model (shared)")
        print("✓ This architecture correctly matches models to tasks.\n")
        
        
        # Step 5 (cont.): Initialize Agent 3 (shares the new master model)
        agent3_resources = initialize_agent3(master_llm_model)
        
        # Step 6: Initialize Agent 4 (uses the shared model)
        print("=" * 70)
        print("STEP 6: Initialize Agent 4 (Final Decision LLM)")
        print("=" * 70)
        
        agent4_model = None
        agent4_tokenizer = None

        if agent3_resources and 'llm_model' in agent3_resources:
            # A3 Succeeded: Use its model (which is the shared one) 
            # and its tokenizer (which it loaded itself)
            agent4_model = agent3_resources['llm_model']
            agent4_tokenizer = agent3_resources['llm_tokenizer']
            print("✓ Agent 4 using shared Group 2 model")
            print("✓ Agent 4 using Agent 3's loaded tokenizer")
            print("✓ Agent 4 LLM initialized successfully\n")
        else:
            # A3 Failed: Use the master model and tokenizer loaded in Step 4
            agent4_model = master_llm_model
            agent4_tokenizer = master_llm_tokenizer
            print("⚠ Agent 3 failed. Agent 4 using fallback shared model and tokenizer.")
            print("✓ Agent 4 LLM initialized successfully\n")
            
        # --- MODIFICATION END ---
        
        # Verify all agents are successfully initialized
        if analyzer_agent1 is None:
            print("\n⚠ Warning: Agent 1 failed to initialize")
        if not agent2_loaded:
            print("\n⚠ Warning: Agent 2 failed to initialize")
        if agent3_resources is None:
            print("\n⚠ Warning: Agent 3 failed to initialize")
            
        if agent4_model is None or agent4_tokenizer is None:
            print("\n⚠ Warning: Agent 4 failed to initialize (model or tokenizer is missing)")
        
        # ... (GPU memory report - this will now show memory for TWO models) ...
        if torch.cuda.is_available():
            print("\n" + "=" * 70)
            print("GPU Memory Status (Two models loaded)")
            # ... (rest of GPU report) ...
        
        # Step 7: Build the Workflow
        app = build_workflow(
            analyzer_agent1,
            agent3_resources,
            agent4_model,
            agent4_tokenizer
        )
        
        # ... (Rest of the file (Step 8, 9, summary) remains the same) ...
        
        print("=" * 70)
        print("Loading test data...")
        print("=" * 70)
        
        df_test = pd.read_csv(CONFIG["TEST_DATA_PATH"])
        print(f"✓ Successfully loaded {len(df_test)} test cases\n")
        
        # Step 9: Run test cases
        results = run_test_cases(df_test, app)

        # Step 10: Save all results to a single JSON file
        if results:
            print("\n" + "=" * 70)
            print("Saving All Results to JSON File")
            print("=" * 70)
            
            # Create final results structure
            final_results = {
                "metadata": {
                    "total_cases": len(results),
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "test_data_path": CONFIG["TEST_DATA_PATH"],
                    "num_cases_processed": CONFIG.get("NUM_CASES_TO_PROCESS", 5)
                },
                "results": results
            }
            
            # Save to final_results.json
            final_results_path = os.path.join(CONFIG["OUTPUT_DIR"], "final_results.json")
            try:
                with open(final_results_path, 'w', encoding='utf-8') as f:
                    json.dump(final_results, f, indent=2, ensure_ascii=False)
                print(f"✓ All results saved to: {final_results_path}")
                print(f"✓ Total cases processed: {len(results)}")
            except Exception as e:
                print(f"✗ Error saving final results: {e}")
        
        # Final summary
        print("\n" + "=" * 70)
        print("Execution Summary")
        print("=" * 70)
        if results:
            approved = sum(1 for r in results if r.get("agent4_final_report", {}).get("final_decision") == "Approved")
            disapproved = sum(1 for r in results if r.get("agent4_final_report", {}).get("final_decision") == "Disapproved")
            errors = sum(1 for r in results if "error" in r)
            
            print(f"Total Cases Processed: {len(results)}")
            print(f"  ✓ Approved: {approved}")
            print(f"  ✗ Disapproved: {disapproved}")
            if errors > 0:
                print(f"  ⚠ Errors: {errors}")
            print(f"\nFinal results saved to: {final_results_path}")
        print("=" * 70)

        return results
        
       

    except Exception as e:
        print(f"\n✗ Execution Failed: {e}")
        traceback.print_exc()
        return None


# =========================================================================
# Entry Point
# =========================================================================

if __name__ == "__main__":
    results = main()

