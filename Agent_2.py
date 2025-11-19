#!pip install transformers accelerate bitsandbytes
#!pip install -q transformers peft bitsandbytes accelerate datasets einops

# Imports#
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import json
import re

BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
CHESS_LORA_ID = "mkopecki/chess-lora-adapter-llama-3.1-8b"
LORA_ADAPTER_ID = "./lora_adapter_fraud"

# GLOBAL VARIABLES to hold the loaded model and tokenizer
_AGENT2_MODEL = None
_AGENT2_TOKENIZER = None

required_keys = [
    "ApplicationBehavior", "LocationofApplication", "AccountActivity",
    "Blacklists", "PastFinancialMalpractices", "ConsistencyinData", "PreviousLoans",
]

# Initialise the model
def initialize_model():
    global _AGENT2_MODEL, _AGENT2_TOKENIZER
    """Loads the base model and attaches the LoRA adapter."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    

    # 3. Chess LoRA
    print(f"Loading first adapter (Chess): {CHESS_LORA_ID}...")
    model_with_chess = PeftModel.from_pretrained(
        base_model,
        CHESS_LORA_ID,
        adapter_name="chess", 
        is_trainable=False 
    )
    print("Chess adapter loaded.")

    print(f"Loading second adapter (Fraud): {LORA_ADAPTER_ID}...")
    model_with_fraud = PeftModel.from_pretrained(
        model_with_chess, 
        LORA_ADAPTER_ID,
        adapter_name="fraud",
        is_trainable=False 
    )
    print("Fraud adapter loaded.")


    model_llama = model_with_fraud 
    model_llama.eval() 


    _AGENT2_MODEL = model_llama
    _AGENT2_TOKENIZER = tokenizer

    print("Agent 2 Model Initialization Complete (Base + Chess + Fraud).")
    return model_llama, tokenizer
    

# Feature Conversion Functions
def describe_prev_loans(n):
    if n <= 0:    return "no previous loans", "first-time borrower profile"
    if n == 1:    return "1 previous loan", "limited prior borrowing"
    if n <= 2:    return f"{n} previous loans", "moderate borrowing history"
    if n <= 4:    return f"{n} previous loans", "active borrowing pattern"
    return f"{n} previous loans", "frequent borrowing pattern"

def phrase_binary(name, value, yes_desc, no_desc):
    text = f"{name}: {value}"
    expl = yes_desc if str(value) == "Yes" else no_desc
    return text, expl

def phrase_two_level(name, value, low_label, low_desc, high_label, high_desc):
    text = f"{name}: {value}"
    expl = low_desc if str(value) == low_label else high_desc
    return text, expl

def convert_to_text(row):
    app_beh   = row["ApplicationBehavior"]
    loc_app   = row["LocationofApplication"]
    act       = row["AccountActivity"]
    bl        = row["Blacklists"]
    malpr     = row["PastFinancialMalpractices"]
    consist   = row["ConsistencyinData"]
    prev_n    = int(row["PreviousLoans"])

    prev_txt, prev_expl = describe_prev_loans(prev_n)
    app_txt, app_expl = phrase_two_level("Application Behavior", app_beh, low_label="Normal", low_desc="typical application speed", high_label="Rapid", high_desc="accelerated submission pattern")
    loc_txt, loc_expl = phrase_two_level("Location of Application", loc_app, low_label="Local", low_desc="usual geographic context", high_label="Unusual", high_desc="atypical or high-risk location context")
    act_txt, act_expl = phrase_two_level("Account Activity", act, low_label="Normal", low_desc="no irregularities observed", high_label="Unusual", high_desc="detected activity anomalies")
    bl_txt, bl_expl = phrase_binary("Blacklist Match", bl, yes_desc="appears on a blacklist indicator", no_desc="no blacklist indicator")
    pfm_txt, pfm_expl = phrase_binary("Past Financial Malpractices", malpr, yes_desc="recorded prior malpractices", no_desc="no recorded malpractices")
    cons_txt, cons_expl = phrase_two_level("Data Consistency", consist, low_label="Consistent", low_desc="internally consistent data", high_label="Inconsistent", high_desc="information inconsistencies detected")

    text = f"""Fraud-Screening Applicant Profile

Signals:
- {app_txt} ({app_expl})
- {loc_txt} ({loc_expl})
- {act_txt} ({act_expl})
- {bl_txt} ({bl_expl})
- {pfm_txt} ({pfm_expl})
- {cons_txt} ({cons_expl})
- Previous loans: {prev_txt} ({prev_expl})

Interpret these categorical fraud signals holistically, giving extra attention to blacklist hits, past malpractices, unusual activity/location, and data inconsistencies."""
    return text

# Parsing Functions
_RISK_PATTERN = re.compile(r"RISK\s*LEVEL\s*:\s*(LOW_RISK|HIGH_RISK|MODERATE_RISK)", re.IGNORECASE)
_PROB_PATTERN = re.compile(r"(FRAUD\s*PROB(ABILITY)?|DEFAULT\s*PROB)\s*:\s*(\d{1,3})(\.\d+)?\s*%", re.IGNORECASE)

def _extract_label(text: str) -> str:
    m = _RISK_PATTERN.search(text)
    if m:
        return m.group(1).upper()
    t = text.upper()
    for k in ("HIGH_RISK", "LOW_RISK", "MODERATE_RISK"):
        if k in t:
            return k
    return "UNKNOWN"

def _extract_prob(text: str):
    m = _PROB_PATTERN.search(text)
    if not m:
        return None
    try:
        return float(m.group(3) + (m.group(4) or ""))
    except Exception:
        return None

def extract_reasoning(response: str):
    """
    Extract only the reasoning portion from a model response that contains 'Reasoning:'.
    Returns the reasoning text as a clean string or None if not found.
    """
    if not response:
        return None

    match = re.search(r"Reasoning\s*:\s*(.*)", response, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    reasoning = match.group(1).strip()
    reasoning = reasoning.rstrip(' "\'')

    return reasoning

# Prompt
FRAUD_SYSTEM_PROMPT = (
    "You are a Fraud Risk Analyst for a bank. Assess whether the application is fraudulent "
    "based on categorical screening signals.\n\n"
    "Respond with:\n"
    "- Risk Level: LOW_RISK or HIGH_RISK\n"
    "- Fraud Probability: Percentage estimate\n"
    "- Key Red Flags: 2–3 short bullets\n"
    "- Recommendation: APPROVE / REVIEW_MANUALLY / REJECT\n"
    "- Reasoning: 2–3 concise sentences"
)

# Inference function
def agent2_predict(applicant_data, model, tokenizer):
    """
    Agent 2 — Fraud risk standard inference interface with robust text parsing.

    Input:
      applicant_data (dict): must contain 'applicant_description'
    Output:
      dict with standardized fraud assessment
    """
    desc = applicant_data.get("applicant_description", "").strip()
    if not desc:
        raise ValueError("applicant_data must include 'applicant_description'")

    # Build messages (system + user), aligned with training
    messages = [
        {"role": "system", "content": FRAUD_SYSTEM_PROMPT},
        {"role": "user", "content": desc + "\n\nProvide your fraud risk assessment."},
    ]

    # Format and Generate 
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    # 1. Parse all known fields
    raw_label = _extract_label(response)
    prob = _extract_prob(response)

    # 2. Robust Reasoning Extraction 
    parsed_reasoning = extract_reasoning(response)

    if parsed_reasoning is None:
        # Fallback 1: Try to find the Recommendation label 
        rec_match = re.search(r"Recommendation\s*:\s*(.*)", response, flags=re.IGNORECASE | re.DOTALL)
        if rec_match:
            rec_text = rec_match.group(1)
            try:
                # Find the start index of the Recommendation text found by the model
                start_index = response.lower().find("recommendation")
                # Find the end of the Recommendation text, and take what follows as reasoning

                risk_level_match = re.search(r"RISK\s*LEVEL\s*:\s*(.*)", response, flags=re.IGNORECASE | re.DOTALL)
                if risk_level_match:
                    # The reasoning is the text that follows, stripped of the Risk Level text.
                    # This captures the unformatted block: "Signals are reassuring..."
                    parsed_reasoning = response[risk_level_match.end():].strip()

                    # Clean up: remove any stray newlines or initial bullets/dashes
                    if parsed_reasoning and parsed_reasoning.startswith('-'):
                         parsed_reasoning = parsed_reasoning.lstrip('-').strip()

            except Exception:
                pass # Stick with null if extraction fails

    # Mapping and Confidence Calculation 

    if raw_label == "MODERATE_RISK":
        risk_level = "HIGH_RISK"
    elif raw_label in ("HIGH_RISK", "LOW_RISK"):
        risk_level = raw_label
    else:
        risk_level = "UNKNOWN"

    if prob is not None:
        confidence = min(max(prob / 100.0, 0.0), 1.0)
        # Consistency nudge
        if risk_level == "LOW_RISK" and prob > 50:
            risk_level = "HIGH_RISK"
        if risk_level == "HIGH_RISK" and prob < 50:
            risk_level = "LOW_RISK"
    else:
        # Heuristic fallback if probability is NULL
        t = response.lower()
        strong = any(w in t for w in ["strong", "clear", "highly", "significant", "elevated"])
        confidence = 0.75 if strong else 0.6 # Original heuristic

    return {
        "agent": "Agent 2 - Fraud Risk Analyst",
        "risk_level": risk_level,
        "raw_label": raw_label,
        "fraud_probability": prob,
        "confidence": round(confidence, 3),
        "reasoning": parsed_reasoning,
        "model_version": "llama-3.1-8b-lora-finetuned-fraud",
    }


def agent2_predict_from_raw(raw_data: dict): 
    """
    Agent 2 main entry point. It loads the model once (if not already loaded) 
    and predicts the fraud risk for the provided raw data.
    """
    global _AGENT2_MODEL, _AGENT2_TOKENIZER
    global required_keys 

    # Check if the model is loaded; if not, load it 
    if _AGENT2_MODEL is None:
        print("Model not yet loaded. Initializing Agent 2 (Llama-3.1-8B LoRA)...")
        _AGENT2_MODEL, _AGENT2_TOKENIZER = initialize_model()
        print("Initialization complete. Model ready for predictions.")
    
    # 1. Validate and Prepare Input (Data Cleaning)
    
    # Coerces all others to string and strips whitespace:
    input_row = {k: str(raw_data.get(k, 'Unknown')).strip() for k in required_keys} # Using global required_keys
    
    # Clean and clip PreviousLoans
    try:
        prev_loans_val = int(float(raw_data.get("PreviousLoans", 0)))
        input_row["PreviousLoans"] = max(0, min(4, prev_loans_val))
    except (ValueError, TypeError):
        input_row["PreviousLoans"] = 0

    # 2. Convert Raw Features into the Narrative Text
    applicant_description_text = convert_to_text(input_row)

    # 3. Call Core Agent 2 Inference 
    prediction_input = {"applicant_description": applicant_description_text}
    
    # 4. CALL agent2_predict with the global model and tokenizer
    result = agent2_predict(
        prediction_input, 
        _AGENT2_MODEL, 
        _AGENT2_TOKENIZER # Using the global objects
    ) 
    
    return result