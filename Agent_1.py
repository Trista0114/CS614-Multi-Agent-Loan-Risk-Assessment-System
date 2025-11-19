import numpy as np
import pandas as pd
import pickle
import torch
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


class CreditRiskAnalyzer:
    def __init__(self, model_dir, lora_dir):
        """
        Initialize Credit Risk Analyzer
        
        Args:
            model_dir: Directory containing xgboost, lightgbm, catboost models and data_splits.pkl
            lora_dir: Directory path of the Llama LoRA adapter
        """
        # Load three ensemble models
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(f'{model_dir}/xgboost_model.json')
        self.lgb_model = lgb.Booster(model_file=f'{model_dir}/lightgbm_model.txt')
        self.cat_model = CatBoostClassifier()
        self.cat_model.load_model(f'{model_dir}/catboost_model.cbm')
        
        # Load data splits to obtain feature names
        with open(f'{model_dir}/data_splits.pkl', 'rb') as f:
            splits = pickle.load(f)
            self.feature_names = splits['X_train'].columns.tolist()
        
        # Fixed threshold 0.45
        self.chosen_threshold = 0.45
        
        # Configure 4-bit quantization for Llama
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load base model (assuming already downloaded locally)
        # If loading from Hugging Face, use: "meta-llama/Llama-3.1-8B"
        base_model_path = "meta-llama/Llama-3.1-8B"  # or local path
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load LoRA adapter
        self.llm = PeftModel.from_pretrained(base_model, lora_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set model to evaluation mode
        self.llm.eval()
        
    def predict_ensemble(self, features):
        """
        Use ensemble of three models to predict default probability
        
        Args:
            features: DataFrame with columns ['rev_util', 'age', 'debt_ratio', 'monthly_inc', 'open_credit', 'late_90']
        
        Returns:
            ensemble_probs: Averaged default probability
        """
        dmatrix = xgb.DMatrix(features)
        xgb_probs = self.xgb_model.predict(dmatrix)
        lgb_probs = self.lgb_model.predict(features.values)
        cat_probs = self.cat_model.predict_proba(features.values)[:, 1]
        
        ensemble_probs = (xgb_probs + lgb_probs + cat_probs) / 3
        return ensemble_probs
    
    def generate_reasoning(self, features_dict, default_prob, risk_level):
        """
        Generate reasoning using fine-tuned Llama model
        Implemented exactly according to the notebook Cell 23 'predict_reasoning' function
        
        Args:
            features_dict: dict with keys ['rev_util', 'age', 'debt_ratio', 'monthly_inc', 'open_credit', 'late_90']
            default_prob: float, default probability
            risk_level: str, "HIGH_RISK" or "LOW_RISK"
        
        Returns:
            reasoning: str, generated credit risk analysis text
        """
        # Follow the prompt format from the notebook
        instruction = f"""Analyze this loan application and provide a risk assessment.

Applicant Profile:
- Credit Utilization: {features_dict['rev_util']:.1%}
- Age: {features_dict['age']} years
- Debt-to-Income Ratio: {features_dict['debt_ratio']:.1%}
- Monthly Income: ${features_dict['monthly_inc']:,.0f}
- Open Credit Lines: {features_dict['open_credit']}
- 90+ Days Late Payments: {features_dict['late_90']}

Default Probability: {default_prob:.4f}
Risk Classification: {risk_level}

Provide your professional credit risk analysis."""

        # Use the notebook's prompt format (including ### Instruction and ### Response)
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract reasoning (remove prompt part)
        reasoning = response.split("### Response:")[-1].strip()
        
        return reasoning
    
    def analyze(self, applicant_data):
        """
        Full analysis workflow: ensemble prediction + reasoning generation
        
        Args:
            applicant_data: dict with keys:
                - 'rev_util': float
                - 'age': int
                - 'debt_ratio': float
                - 'monthly_inc': float
                - 'open_credit': int
                - 'late_90': int
                - 'applicant_id': str (optional)
        
        Returns:
            result: dict containing the complete analysis result
        """
        # Prepare features DataFrame (must follow the same column order as training)
        features = pd.DataFrame([{
            'rev_util': applicant_data['rev_util'],
            'age': applicant_data['age'],
            'debt_ratio': applicant_data['debt_ratio'],
            'monthly_inc': applicant_data['monthly_inc'],
            'open_credit': applicant_data['open_credit'],
            'late_90': applicant_data['late_90']
        }])
        
        # 1. Ensemble prediction
        probability = self.predict_ensemble(features)[0]
        
        # 2. Determine risk level
        risk_level = "HIGH_RISK" if probability > self.chosen_threshold else "LOW_RISK"
        
        # 3. Generate reasoning
        reasoning = self.generate_reasoning(
            applicant_data, 
            probability, 
            risk_level
        )
        
        # 4. Calculate confidence (based on logic from notebook Cell 26)
        confidence = abs(probability - self.chosen_threshold) / self.chosen_threshold
        confidence = min(max(confidence, 0.5), 0.95)  # clip between 0.5 and 0.95
        
        # 5. Assemble output format
        return {
            "agent_id": "agent_1",
            "agent_name": "Agent 1 - Credit Risk Analyst",
            "applicant_id": applicant_data.get("applicant_id", "unknown"),
            "analysis": {
                "risk_level": risk_level,
                "default_probability": round(probability, 4),
                "confidence": round(confidence, 2),
                "reasoning": reasoning
            },
            "metadata": {
                "model_version": "ensemble-xgb-lgb-cat + llama-3.1-8b-lora",
                "threshold": self.chosen_threshold
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CreditRiskAnalyzer(
        model_dir="/path/to/Agent1_V2",  # Directory containing xgboost, lightgbm, catboost models
        lora_dir="/path/to/Agent1_V2/llama_lora_adapter"  # Directory of LoRA adapter
    )
    
    # Test data
    test_applicant = {
        'rev_util': 0.45,      # 45% credit utilization
        'age': 35,
        'debt_ratio': 0.30,    # 30% debt-to-income ratio
        'monthly_inc': 5000.0,
        'open_credit': 8,
        'late_90': 0,
        'applicant_id': 'test_0001'
    }
    
    # Run analysis
    result = analyzer.analyze(test_applicant)
    
    # Output result
    import json
    print(json.dumps(result, indent=2))
