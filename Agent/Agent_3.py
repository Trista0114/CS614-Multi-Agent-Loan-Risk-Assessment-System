# Installation Requirements
#!pip install chromadb sentence-transformers python-docx
#!pip install -q -U accelerate bitsandbytes peft transformers sentence-transformers networkx

# Import Requirements
import re
from typing import List, Dict, Any, Tuple, TypedDict, Callable, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings
from docx import Document
from chromadb.utils import embedding_functions
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from langchain_core.prompts import ChatPromptTemplate
import json
import math
import networkx as nx
import numpy as np
import pandas as pd
import torch
from contextlib import redirect_stdout
from difflib import SequenceMatcher
from huggingface_hub import login
import os
from dotenv import load_dotenv
load_dotenv()

import random
import itertools
import os
import time


# ---------1. Functions for Knowledge Base COnstruction-------
# load internal policy document
def load_policy_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():  # skip empty paragraphs
            full_text.append(para.text.strip())
    return "\n".join(full_text)


# Defining Class
Triple = Tuple[str, str, str, int]  # (head, relation, tail, provenance_chunk_id)

class ChunkMetadata(TypedDict, total=False):
    section: str
    header: str
    check_type: str

class ChunkData(TypedDict):
    id: int
    text: str
    metadata: ChunkMetadata

class KnowledgeBaseData(TypedDict):
    chunks: List[ChunkData]
    triples: List[Triple]
    summary_nodes: List[str]
    relations: List[str]

# Knowledge Base Construction (Textual) & Knowledge Graph Construction
def create_knowledge_base_chunks_with_graph(policy_text: str) -> KnowledgeBaseData:
    """
    Chunk policy by numbered headers and extract triples with provenance.
    Returns KnowledgeBaseData with chunks, triples, and helper lists.
    """
    header_regex = re.compile(r'(?m)^\s*(\d+[\d\.]*)\s*(?:[.)-]?\s*)?(.*)$')
    matches = list(header_regex.finditer(policy_text))
    chunks: List[ChunkData] = []
    triples: List[Triple] = []

    # Chunk policy by numbered headers
    if not matches:
        chunks.append({"id": 0, "text": policy_text.strip(),
                       "metadata": {"section": "0.0", "check_type": "GENERAL_POLICY"}})
    else:
        first_start = matches[0].start()
        preamble = policy_text[:first_start].strip()
        chunk_id = 0
        if preamble:
            chunks.append({"id": chunk_id, "text": preamble,
                           "metadata": {"section": "0.0", "check_type": "GENERAL_POLICY"}})
            chunk_id += 1

        # Iterate sections
        for i, m in enumerate(matches):
            section_no = m.group(1).strip()
            header = m.group(2).strip()
            start = m.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(policy_text)
            content = policy_text[start:end].strip()
            full_text = (header + "\n" + content).strip()

            # Classify by major section number
            major = int(section_no.split('.')[0])
            check_type = (
                "SCOPE_POLICY" if major == 1 else
                "LOAN_TYPES" if major == 2 else
                "AGE_CHECKS" if major == 3 else
                "DTI_CREDIT_SCORE_RULES" if major == 4 else
                "INCOME_VERIFICATION_POLICY" if major == 5 else
                "OPERATION_GUIDELINES" if major == 6 else
                "FEES_AND_PRICING_POLICY" if major == 7 else
                "GENERAL_POLICY"
            )

            chunks.append({
                "id": chunk_id,
                "text": full_text,
                "metadata": {"section": section_no, "header": header, "check_type": check_type}
            })

            # Extract Knowledge graph Triples
            lower = full_text.lower()
            # Section 3: AGE_CHECKS ---
            if check_type == "AGE_CHECKS":
                if (mmin := re.search(r'minimum[^.]*?(\d{1,2})\s*years', lower)):
                    triples.append(("Applicant", "REQUIRES_MIN_AGE", mmin.group(1), chunk_id))
                if (mm := re.search(r'exceed[^.]*?(\d{2,3})\s*years', lower)):
                    triples.append(("Loan", "MAX_AGE_AT_MATURITY_TRIGGER", mm.group(1), chunk_id))
                if (madvisor := re.search(r'applicants aged\s*(\d{2,3})\s*years or older', lower)):
                    triples.append(("Applicant", "REQUIRES_ADVISORY_AGE", madvisor.group(1), chunk_id))

            # Section 4: DTI_CREDIT_SCORE_RULES
            elif check_type == "DTI_CREDIT_SCORE_RULES":
                loan_type_match = re.match(r'([A-Za-z]+)', header)
                loan_type = loan_type_match.group(1).capitalize() if loan_type_match else "General"

            # Max DTI
                if (mdti := re.search(r'maximum\s*dti[^.]*?(\d+)\s*%', full_text, re.IGNORECASE)):
                    triples.append((loan_type, "MAX_DTI_PERCENT", mdti.group(1), chunk_id))

             # Tiered credit-score rules. Create subnodes for each range
             # Tier 1: DTI ≤ 36%
                if (m_tier1 := re.search(r'less than or equal to\s*36%.*?minimum credit score required is (\d{3})', lower)):
                    node_id = f"{loan_type}_DTI_LE_36"
                    triples.append((loan_type, "HAS_DTI_RULE", node_id, chunk_id))
                    triples.append((node_id, "MAX_DTI_PERCENT", "36", chunk_id))
                    triples.append((node_id, "MIN_CREDIT_SCORE_REQUIRED", m_tier1.group(1), chunk_id))

              # Tier 2: 36 < DTI ≤ X%
                if (m_tier2 := re.search(
                    r'more than\s*36%.*?less than or equal to\s*(\d+)%.*?minimum credit score required is (\d{3})',
                    lower
                  )):
                    upper = m_tier2.group(1)
                    score = m_tier2.group(2)
                    node_id = f"{loan_type}_DTI_36_TO_{upper}"
                    triples.append((loan_type, "HAS_DTI_RULE", node_id, chunk_id))
                    triples.append((node_id, "MIN_DTI_PERCENT", "36", chunk_id))
                    triples.append((node_id, "MAX_DTI_PERCENT", upper, chunk_id))
                    triples.append((node_id, "MIN_CREDIT_SCORE_REQUIRED", score, chunk_id))

             # Credit report validity
                if (m_validity := re.search(r'within\s*(\d+)\s*calendar days prior to final approval', lower)):
                    triples.append(("Credit Report", "VALIDITY_DAYS_PRIOR_APPROVAL", m_validity.group(1), chunk_id))

            # Multiple reports
                if "average score shall be used" in lower:
                    triples.append(("Multiple Credit Reports", "CALCULATION_METHOD", "AVERAGE_SCORE", chunk_id))
                if (m_disc := re.search(r'discrepancies \(exceeding\s*(\d+)\s*points\)', lower)):
                    triples.append(("Multiple Credit Reports", "DISCREPANCY_THRESHOLD_POINTS", m_disc.group(1), chunk_id))

            # Section 5: Income Verification
            elif check_type == "INCOME_VERIFICATION_POLICY":
                if "verification" in lower or "mandates the verification" in lower:
                    triples.append(("Income", "REQUIRES_VERIFICATION", "TRUE", chunk_id))
                if (m_timeliness := re.search(r'dated no earlier than\s*(\d+)\s*days before the application date', lower)):
                    triples.append(("Income Documentation", "MAX_AGE_DAYS_PRIOR_APPLICATION", m_timeliness.group(1), chunk_id))

            # Section 6: Operation Guidelines
            elif check_type == "OPERATION_GUIDELINES":
                if "manual review triggers" in lower:
                    if (m_score_trigger := re.search(r'(\d+)\s*points below the minimum required threshold', lower)):
                        triples.append(("Credit Score", "MANUAL_REVIEW_TRIGGER_BELOW_POINTS", m_score_trigger.group(1), chunk_id))
                    if (m_size_trigger := re.search(r'exceeds the average amount[^.]*?by more than\s*(\d+)\s*%', lower)):
                        triples.append(("Loan Amount", "MANUAL_REVIEW_TRIGGER_EXCEED_AVG_PERCENT", m_size_trigger.group(1), chunk_id))

            # Section 7: Fees & Pricing
            elif check_type == "FEES_AND_PRICING_POLICY":
                if (m_grace := re.search(r'within the established\s*(\d+)-day grace period', lower)):
                    triples.append(("Payment", "GRACE_PERIOD_DAYS", m_grace.group(1), chunk_id))
                if (m_fee_cap := re.search(r'maximum late fee charged shall not exceed\s*\$(\d+)', lower)):
                    triples.append(("Late Fee", "MAX_CAP_USD", m_fee_cap.group(1), chunk_id))
                if (m_fee_perc := re.search(r'or\s*(\d+)%\s*of the past due payment amount', lower)):
                    triples.append(("Late Fee", "MAX_CAP_PERCENT", m_fee_perc.group(1), chunk_id))
                if "periodic review" in header.lower() and (m_period := re.search(r'review by on a\s*(\w+)\s*basis', lower)):
                    triples.append(("Loan Pricing Matrix", "REVIEW_FREQUENCY", m_period.group(1).upper(), chunk_id))

            chunk_id += 1

    # Deduplicate triples
    seen = set()
    unique_triples: List[Triple] = []
    for h, r, t, p in triples:
        key = (h, r, t, p)
        if key not in seen:
            seen.add(key)
            unique_triples.append((h, r, t, p))

    summary_nodes = list({t[0] for t in unique_triples})
    relations = list({t[1] for t in unique_triples})

    return {
        "chunks": chunks,
        "triples": unique_triples,
        "summary_nodes": summary_nodes,
        "relations": relations
    }


# Build Graph (NetworkX) with provenance
def build_graph_from_triples(triples: List[Triple]) -> nx.DiGraph:
    G = nx.DiGraph()
    for head, rel, tail, prov in triples:
        G.add_node(head, label=head)
        G.add_node(tail, label=tail)
        if G.has_edge(head, tail):
            G.edges[head, tail]["relations"].append({"rel": rel, "prov": prov})
        else:
            G.add_edge(head, tail, relations=[{"rel": rel, "prov": prov}])
    return G

#EMBEDDING
encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def embedding_fn(texts: List[str]) -> np.ndarray:
    embs = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms

def build_and_index_knowledge_base(policy_text: str):
    kb_data = create_knowledge_base_chunks_with_graph(policy_text)
    G = build_graph_from_triples(kb_data["triples"])
    return G, embedding_fn, kb_data

# --------------2. Load Model----------------------
def load_llama3_instruct_model(base_model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    print("Llama-3.1-8B-Instruct loaded successfully")
    return model, tokenizer

# ----------------3. Functions for Rule Retrieval---------------------------
# Compliance check query construction for rule retrieval
def construct_query_text_for_check(check_name: str, applicant_info: dict):
    """Generate semantically rich query text for each check type."""
    if check_name == "DTI_CHECK":
        return (
            f"Debt-to-Income (DTI) rules for {applicant_info.get('PurposeoftheLoan')} loan. "
            f"Applicant has DTI = {applicant_info.get('DTI')}%. "
            "Retrieve thresholds limits for DTI"
        )
    elif check_name == "CREDIT_SCORE_CHECK":
        return (
            f"Minimum credit score policy for {applicant_info.get('PurposeoftheLoan')} loan. "
            f"Applicant has DTI = {applicant_info.get('DTI')}% and credit score = {applicant_info.get('CreditScore')}. "
            "Find the minimum credit score allowed based on applicant DTI tier."
        )
    elif check_name == "INCOME_EMPLOYMENT_CHECK":
        return (
            f"Income verification and employment verification requirements for {applicant_info.get('PurposeoftheLoan')} loan. "
            f"Employment verification status = {applicant_info.get('EmploymentVerification')}."
        )
    elif check_name == "MIN_AGE_CHECK":
        return "Minimum legal age requirement for loan eligibility."
    elif check_name == "MAX_AGE_AT_MATURITY_CHECK":
        return (
            f"Find the maximum age allowed at loan maturity: Applicant age = {applicant_info.get('Age')}, "
            f"loan term = {applicant_info.get('LoanTerm')} years."
        )
    else:
        return ""
    
# Textual Chunk Retrieval
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid_retrieve_rules(applicant_info: dict, kb_data: dict, embedding_fn, top_k: int = 2, rerank_top_n: int = 5):
    """
    Retrieve rule chunks using 2-stage retrieval:
    (1) Semantic match to metadata check type
    (2) Semantic ranking within chunks of that check type
    (3) Cross-encoder re-ranking within top candidates
    Returns only top_k chunks per check.
    """
    # Precompute embeddings for chunks
    chunk_texts = [c["text"] for c in kb_data["chunks"]]
    chunk_embs = embedding_fn(chunk_texts)

    # Unique metadata check types
    unique_check_types = list({c["metadata"]["check_type"] for c in kb_data["chunks"]})
    check_type_embs = embedding_fn(unique_check_types)

    retrieved = {}

    for check_name in ["DTI_CHECK", "CREDIT_SCORE_CHECK", "INCOME_EMPLOYMENT_CHECK",
                       "MIN_AGE_CHECK", "MAX_AGE_AT_MATURITY_CHECK"]:
        query = construct_query_text_for_check(check_name, applicant_info)
        if not query:
            retrieved[check_name] = []
            continue

        # Stage 1: semantic match on metadata check type
        q_emb = embedding_fn([query])[0]
        sims_check_type = np.dot(check_type_embs, q_emb) / (
            np.linalg.norm(check_type_embs, axis=1) * np.linalg.norm(q_emb) + 1e-12
        )
        best_check_type = unique_check_types[np.argmax(sims_check_type)]

        # Stage 2: semantic similarity within filtered chunks
        filtered_idx = [i for i, c in enumerate(kb_data["chunks"])
                        if c["metadata"]["check_type"] == best_check_type]

        sims = []
        for i in filtered_idx:
            sim = float(np.dot(q_emb, chunk_embs[i]) /
                        (np.linalg.norm(q_emb) * np.linalg.norm(chunk_embs[i]) + 1e-12))
            sims.append((sim, i, chunk_texts[i]))
        sims.sort(reverse=True, key=lambda x: x[0])

        # Stage 3: Cross-Encoder re-ranking
        top_candidates = sims[:rerank_top_n]
        if len(top_candidates) > 1:
            pairs = [(query, text) for _, _, text in top_candidates]
            cross_scores = cross_encoder.predict(pairs)
            reranked = sorted(zip(cross_scores, top_candidates),
                              reverse=True, key=lambda x: x[0])
            top = [(i, text) for (_, (_, i, text)) in reranked[:top_k]]
        else:
            top = [(i, text) for _, i, text in top_candidates[:top_k]]

        retrieved[check_name] = top

    return retrieved

# Knowledge Graph Rule Retrieval
# Function to retrieve the graph facts based on the retrieved textual rule chunks
def collect_graph_provenance_facts_only(G: nx.DiGraph, graph_triples: List[Triple], selected_chunk_ids: List[int]):
    """
    returns graph triples directly tied to selected chunks (provenance).
    Neighboring nodes are explicitly EXCLUDED.
    Deduplicates by (head, rel, tail).
    """
    collected = []
    seen = set()

    # Triples directly tied to selected chunks (provenance)
    for h, r, t, prov in graph_triples:
        if prov in selected_chunk_ids:
            key = (h, r, t)
            if key not in seen:
                seen.add(key)
                collected.append((h, r, t, prov)) # prov is the chunk ID

    return collected

# Function to obtain thresholds for deterministic python checks
def get_thresholds_from_graph(G: nx.DiGraph, loan_type: str) -> Dict[str, Any]:
    """
    Deterministically read numeric thresholds for a loan type from the graph.
    Falls back to safe defaults if missing.
    """
    defaults = {"MAX_DTI": 100, "MIN_AGE": 0, "MAX_AGE_AT_MATURITY": 100}
    res = defaults.copy()
    # MAX_DTI: look for edge loan_type
    if G.has_node(loan_type):
        for nbr in G.successors(loan_type):
            for entry in G.edges[loan_type, nbr]["relations"]:
                if entry["rel"] == "MAX_DTI_PERCENT":
                    try:
                        res["MAX_DTI"] = int(nbr)
                    except Exception:
                        pass
    # MIN_AGE and MAX_AGE_AT_MATURITY
    if G.has_node("Applicant"):
        for nbr in G.successors("Applicant"):
            for entry in G.edges["Applicant", nbr]["relations"]:
                if entry["rel"] == "REQUIRES_MIN_AGE":
                    try:
                        res["MIN_AGE"] = int(nbr)
                    except:
                        pass
    if G.has_node("Loan"):
        for nbr in G.successors("Loan"):
            for entry in G.edges["Loan", nbr]["relations"]:
                if entry["rel"] == "MAX_AGE_AT_MATURITY_TRIGGER":
                    try:
                        res["MAX_AGE_AT_MATURITY"] = int(nbr)
                    except:
                        pass
    return res


def get_min_credit_score_required_from_graph(G: nx.DiGraph, applicant_info: Dict[str, Any]) -> Tuple[int, str, str]:
    """
    Deterministically traverses the graph to find the minimum required credit score
    based on the applicant's Loan Type and DTI.

    Returns: (min_score: int, max_dti_in_tier: str, rule_node_id: str)
    """
    loan_type = applicant_info.get("PurposeoftheLoan", "General").capitalize()
    try:
        applicant_dti = float(applicant_info.get("DTI", 101)) # Default to high DTI to hit highest tier
    except (ValueError, TypeError):
        applicant_dti = 101

    required_score = 999
    max_dti_in_tier = "N/A"
    rule_node_id = "N/A_RULE"

    if not G.has_node(loan_type):
        return required_score, max_dti_in_tier, rule_node_id

    # 1. Collect all DTI tier rule nodes for the loan type
    dti_rule_nodes = []
    for nbr in G.successors(loan_type):
        for entry in G.edges[loan_type, nbr]["relations"]:
            if entry["rel"] == "HAS_DTI_RULE":
                dti_rule_nodes.append(nbr)

    # 2. Extract DTI tiers (Min DTI, Max DTI, Min CS) from rule nodes
    tiers_data = []
    for node in dti_rule_nodes:
        # Defaults for a rule node
        min_dti = 0
        max_dti = 100
        min_cs = 999

        for nbr in G.successors(node):
            for entry in G.edges[node, nbr]["relations"]:
                try:
                    value = int(nbr)
                except (ValueError, TypeError):
                    continue

                if entry["rel"] == "MIN_DTI_PERCENT":
                    min_dti = value
                elif entry["rel"] == "MAX_DTI_PERCENT":
                    max_dti = value
                elif entry["rel"] == "MIN_CREDIT_SCORE_REQUIRED":
                    min_cs = value

        # Store all required data for tier evaluation
        tiers_data.append({
            'min_dti': min_dti,
            'max_dti': max_dti,
            'min_cs': min_cs,
            'node_id': node
        })

    # Sort the tiers by max_dti (ascending) to ensure the first match is the correct tier
    tiers_data.sort(key=lambda x: x['max_dti'])

    # 3. Find the matching tier
    for tier in tiers_data:
        # Check if DTI falls within the tier's range (exclusive min, inclusive max)
        if tier['min_dti'] < applicant_dti <= tier['max_dti']:
            return tier['min_cs'], str(tier['max_dti']), tier['node_id']
    return required_score, max_dti_in_tier, rule_node_id

# ---------------------------4. Prompt Construction-----------------------------
def extract_first_json_object(text: str) -> Optional[str]:
    """
    Robustly tries to find and return the first JSON object string from a text block.
    """
    # 1. Find the first opening brace '{'
    start_match = re.search(r'\{', text)
    if not start_match:
        return None
    start_index = start_match.start()

    potential_json_str = text[start_index:]

    # 2. Attempt Fault-Tolerant Load (Primary Repair)
    try:
        # Try loading the content from the first '{' onwards.
        # This catches if the only error is trailing garbage.
        _ = json.loads(potential_json_str)
        return potential_json_str
    except json.JSONDecodeError:
        pass # Go to the next repair step

    # 3. Manual Recursive Brace Matching (Fallback Repair)
    brace_count = 0
    end_index = -1
    in_string = False

    # Iterate from the first found '{'
    for i, char in enumerate(potential_json_str):
        if char == '"' and (i == 0 or potential_json_str[i-1] != '\\'):
            # Toggle string flag, ignoring escaped quotes
            in_string = not in_string

        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    end_index = i + 1
                    break

    # 4. Return Result
    if end_index != -1:
        # We found a balanced JSON structure
        json_candidate = potential_json_str[:end_index]

        # FINAL CHECK: Ensure the extracted chunk is valid JSON
        try:
            _ = json.loads(json_candidate)
            return json_candidate
        except json.JSONDecodeError:
            # The structure was balanced but still invalid (e.g., missing comma, invalid characters)
            return None

    return None


def extract_last_json_from_raw(raw_output: str) -> Optional[Dict[str, Any]]:
    """
    Specifically extracts the final, clean JSON object from the raw_llm_output
    when the LLM has added multiple intermediate outputs (like the code block).
    """
    # Look for the pattern: "The final answer is:\n\n```\n{...}"
    final_json_marker = "The final answer is:"

    if final_json_marker in raw_output:
        # Split on the final answer marker
        parts = raw_output.rsplit(final_json_marker, 1)

        # Look for the last markdown code block (```) starting after the marker
        potential_json_block = parts[1]

        # Regex to find the content inside the final ```...``` block
        # that looks like JSON (starts with { and ends with }).
        match = re.search(
            r'\{\s*"check_details":\s*\[.*?\]\s*\}\s*',
            potential_json_block,
            re.DOTALL | re.IGNORECASE
        )

        if match:
            json_str = match.group(0).strip()

            # Final check: Ensure the extracted string is valid JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None

    return None

# Define function for full rag process
def run_compliance_graphrag(applicant_info: dict,
                            kb_data: dict,
                            embedding_fn,
                            llm_invoke_fn,
                            top_k: int = 2):

    # 1. Build graph
    G = build_graph_from_triples(kb_data["triples"])
    chunk_map = {c["id"]: c for c in kb_data["chunks"]}

    # 2. Hybrid retrieval
    retrieved = hybrid_retrieve_rules(applicant_info, kb_data, embedding_fn, top_k=top_k)

    # 3. Collect provenance chunk ids
    retrieved_chunk_ids_per_check = {}
    for check_name, retrieved_list in retrieved.items():
        retrieved_chunk_ids_per_check[check_name] = sorted([cid for cid, _ in retrieved_list])

    selected_chunk_ids = sorted({cid for sublist in retrieved_chunk_ids_per_check.values() for cid in sublist})

    #  4. Gather graph facts
    graph_facts_triples = collect_graph_provenance_facts_only(
        G, kb_data["triples"], selected_chunk_ids
    )

    graph_facts_context = []
    for h, r, t, prov in graph_facts_triples:
        graph_facts_context.append(f"[DIRECT] {h} --{r}--> {t} (from chunk {prov})")

    # 5. Deterministic thresholds & computed fields
    loan_type = applicant_info.get("PurposeoftheLoan", "Personal").capitalize()
    thresholds = get_thresholds_from_graph(G, loan_type)

    age = int(applicant_info.get("Age", 0))
    loan_term = int(applicant_info.get("LoanTerm", 0))
    age_at_maturity = age + loan_term
    applicant_info = dict(applicant_info)
    applicant_info["age_at_maturity"] = age_at_maturity

    dti_pass = float(applicant_info.get("DTI", 0)) <= thresholds["MAX_DTI"]
    min_age_pass = age >= thresholds["MIN_AGE"]
    max_age_pass = age_at_maturity <= thresholds["MAX_AGE_AT_MATURITY"]

    min_cs_required, max_dti_in_tier, cs_rule_node = get_min_credit_score_required_from_graph(G, applicant_info)
    applicant_cs = int(applicant_info.get("CreditScore", 0))
    applicant_dti = float(applicant_info.get("DTI", 0))

    # Apply the DTI check dependency: Credit Score is SKIPPED if DTI failed
    if not dti_pass:
         credit_score_pass = "SKIPPED"
    else:
        # Check if DTI falls into any defined tier (i.e., not the 999 fallback)
        if min_cs_required == 999:
             credit_score_pass = "FAIL" # Fails because DTI is too high for any tier
             # Use a generic fallback node for justification context
             cs_rule_node = f"DTI_TOO_HIGH_FOR_{loan_type.upper()}_RULES"
        else:
             credit_score_pass = "PASS" if applicant_cs >= min_cs_required else "FAIL"

    # Add deterministic CS data to thresholds for better LLM guidance
    thresholds["MIN_CREDIT_SCORE_REQUIRED"] = min_cs_required
    thresholds["CS_RULE_NODE"] = cs_rule_node
    thresholds["CS_DTI_TIER_MAX"] = max_dti_in_tier

    # 6. Build LLM context
    context_parts = ["Retrieved KB chunks for summarization:"]
    unique_retrieved_chunks = {(cid, text) for chk in retrieved for cid, text in retrieved[chk]}
    for cid, text in unique_retrieved_chunks:
        metadata = chunk_map.get(cid, {}).get("metadata", {})
        section = metadata.get("section", "N/A")
        header = metadata.get("header", "N/A")
        context_parts.append(
            f"\n--- [CHUNK ID: {cid}] [SECTION: {section}] [HEADER: {header}] ---\nCONTENT:\n{text}\n"
        )

    context_parts.append("\nGraph Facts (provenance filtered):")
    for f in graph_facts_context:
        context_parts.append(f"- {f}")
    context_text = "\n".join(context_parts)

    # 7. Build LLM prompt
    subjective_prompt = f"""
You are a strict financial compliance agent. Using the CONTEXT, THRESHOLDS, and FEATURES below,
evaluate the applicant for ALL checks and produce a JSON object with their results and detailed justifications.

Applicant info:
{json.dumps(applicant_info, indent=2)}

Deterministic results (guidance):
The results for the following checks have been **deterministically calculated in advance** and they should be used to for the results field of the corresponding check.
Do not contradict or override the results for these checks. Provide only justification to support the outcomes of these checks.
1. Debt-to-Income (DTI) (Ensure DTI is below maximum allowed for applicant's loan type). DTI_PASS: {dti_pass}
2. Credit Score (Ensure minimum credit score required is fulfilled). CREDIT_SCORE_PASS: {credit_score_pass}. MIN_CREDIT_SCORE_REQUIRED: {min_cs_required}
3. Income/Employment Verification (The check result should PASS if the applicant's employment is verified)
4. Minimum Age (Ensure applicant age meets minimum age required). MIN_AGE_PASS: {min_age_pass}
5. Age at Loan Maturity (Check if the applicant's age at loan maturity exceeds retirement age). MAX_AGE_AT_MATURITY_PASS: {max_age_pass}


CONTEXT (top-k retrieved chunks & graph facts):
{context_text}

Rules:
1. Identify the relevant CONTEXT to be used for each check and provide the justification for each check result, based on applicant's information. You MUST NOT use Personal loan rules in CONTEXT for Auto, Medical, Education and Travel loan types.
2. For DTI check, only use the CONTEXT for applicant's loan type: {applicant_info.get('PurposeoftheLoan')}. Do not use any rules for credit score when justifying DTI check result.
3. For the rule_summaries, you must synthesize a one- to two-sentence summary of only the most relevant rule used for justification and cite the chunk id.
4. You should cite only the relevant rules and EXACT graph rules in CONTEXT used for justification for the respective check.

Instructions for performing Credit Score Check:
1. Skip credit score check if DTI check result is FAIL and leave the rule summaries for Credit Score Check blank.
2. If DTI check passes, determine which DTI tier the applicant falls into based on CONTEXT for applicant's loan type: {applicant_info.get('PurposeoftheLoan')} loan and applicant DTI of {applicant_info.get('DTI')}%.
3. Identify the minimum required credit score for that tier and evaluate if the applicant's credit score passes the requirement.
4. Include this reasoning in the justification. Cite the relevant rules and EXACT graph facts from CONTEXT for {applicant_info.get('PurposeoftheLoan')} loan type for Credit Score check in rule summaries.

Other Instructions:
You MUST Output ONLY a single JSON object with the following format and NOTHING else should be included. DO NOT INCLUDE ANY INTRODUCTORY TEXT, EXPLANATIONS, OR CODE BLOCKS (```json, ```, ```python, etc.).
BEGIN YOUR RESPONSE IMMEDIATELY WITH THE OPENING CURLY BRACE:

{{
  "check_details": [
    {{
      "check":"DTI Check",
      "result":"PASS|FAIL",
      "justification":"...",
      "rule_summaries":[{{"chunk_id": int, "summary": "str"}}],
      "graph_facts_used":["str"]
    }},
    {{
      "check":"Credit Score Check",
      "result":"PASS|FAIL|SKIPPED",
      "justification":"...",
      "rule_summaries":[{{"chunk_id": int, "summary": "str"}}],
      "graph_facts_used":["str"]
    }},
    {{
      "check":"Income/Employment Check",
      "result":"PASS|FAIL",
      "justification":"...",
      "rule_summaries":[{{"chunk_id": int, "summary": "str"}}],
      "graph_facts_used":["str"]
    }},
    {{
      "check":"Min Age Check",
      "result":"PASS|FAIL",
      "justification":"...",
      "rule_summaries":[{{"chunk_id": int, "summary": "str"}}],
      "graph_facts_used":["str"]
    }},
    {{
      "check":"Max Age at Maturity Check",
      "result":"PASS|FAIL",
      "justification":"...",
      "rule_summaries":[{{"chunk_id": int, "summary": "str"}}],
      "graph_facts_used":["str"]
    }}
  ]
}}
"""

    # 8. Invoke LLM & parse safely
    raw_subjective = llm_invoke_fn(subjective_prompt)
    json_str = extract_first_json_object(raw_subjective) # Your initial extraction function

    raw_output_for_debug = None
    subj_report = None
    check_names = ["DTI Check","Credit Score Check","Income/Employment Check","Min Age Check","Max Age at Maturity Check"]

    # --- Initial Parsing Attempt ---
    if json_str:
        try:
            subj_report = json.loads(json_str)
        except Exception:
            # Parsing Failed (Extraction successful but content invalid)
            raw_output_for_debug = raw_subjective
            # subj_report remains None, moving to recovery/error assignment
    else:
        # No JSON Found (Extraction failed)
        raw_output_for_debug = raw_subjective
        # subj_report remains None, moving to recovery/error assignment

    # Recovery Attempt
    if subj_report is None and raw_output_for_debug is not None:
        try:
            # Attempt to extract the clean, final JSON object from the raw output
            recovered_data = extract_last_json_from_raw(raw_output_for_debug)

            if recovered_data:
                subj_report = recovered_data
                error_msg = None # Clear error message on successful recovery
                print("Successfully recovered clean JSON from debug output.")
            else:
                # Recovery failed: Assign final error report
                error_msg = f"Recovery failed. Initial extractor failed, and targeted recovery failed. Raw start: {raw_subjective[:200].replace('\n',' ')}..."
                subj_report = {"check_details":[
                    {"check": chk, "result":"PARSE_ERROR_RECOVER", "justification": error_msg,
                     "rule_summaries":[], "graph_facts_used":[]}
                    for chk in check_names
                ]}
        except Exception:
            # Critical error during recovery itself
            error_msg = f"Critical error during recovery attempt. Raw start: {raw_subjective[:200].replace('\n',' ')}..."
            subj_report = {"check_details":[
                {"check": chk, "result":"PARSE_ERROR_CRITICAL", "justification": error_msg,
                 "rule_summaries":[], "graph_facts_used":[]}
                for chk in check_names
            ]}

    # Final Error Assignment (Only if initial/recovery failed)
    # This block handles the case where the initial extraction failed and recovery was not possible/necessary.
    if subj_report is None:
        error_msg = f"No valid JSON object found. Raw start: {raw_subjective[:200].replace('\n',' ')}..."
        subj_report = {"check_details":[
            {"check": chk, "result":"PARSE_ERROR", "justification": error_msg,
             "rule_summaries":[], "graph_facts_used":[]}
            for chk in check_names
        ]}

    # 9. Ensure all checks present & fill missing keys
    check_map = {d["check"]: d for d in subj_report.get("check_details", [])}
    final_details = []
    default_error = lambda chk: {"check": chk, "result":"ERROR",
                                 "justification":"LLM missing output or parsing failed completely.",
                                 "rule_summaries": [], "graph_facts_used": []}

    for chk in check_names:
        if chk in check_map:
            detail = check_map[chk]
            detail["rule_summaries"] = detail.get("rule_summaries", [])
            detail["graph_facts_used"] = detail.get("graph_facts_used", [])
            final_details.append(detail)
        else:
            final_details.append(default_error(chk))

    overall = "PASS" if all(d["result"] in ["PASS", "SKIPPED"] for d in final_details) else "FAIL"

    # 10. Construct final debug dictionary
    debug_output = {
        "thresholds": thresholds,
        "retrieved_chunk_ids_per_check": retrieved_chunk_ids_per_check
    }

    if raw_output_for_debug is not None:
        debug_output["raw_llm_output"] = raw_output_for_debug

    return {
        "Overall_compliance_status": overall,
        "check_details": final_details,
        "debug": debug_output
    }

def llm_invoke_fn(prompt: str, model, tokenizer, max_new_tokens: int = 4000) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated[len(prompt):].strip()

# ---------------------Agent 3 Main Pipeline---------------
def setup_agent_3(policy_path: str = "Internal Policy.docx"):
    """
    Performs all heavy initialization ONCE and returns the necessary objects.
    """
    # 1. Load internal policy
    internal_policy_text = load_policy_from_docx(policy_path)

    # 2. Build knowledge base
    G, embedding_model, kb_data = build_and_index_knowledge_base(internal_policy_text)

    # 3. Load model 
    llm_model, llm_tokenizer = load_llama3_instruct_model()

    # 4. Define the partial invocation function for the LLM
    llm_invoke = lambda prompt: llm_invoke_fn(prompt, llm_model, llm_tokenizer)

    return kb_data, embedding_fn, llm_invoke


def agent_3(applicant_info_json: str, kb_data, embedding_fn, llm_invoke):
    """Wrapper to run the core RAG logic using pre-loaded resources."""
    import json
    applicant_info = json.loads(applicant_info_json)

    final_report = run_compliance_graphrag(
        applicant_info,
        kb_data,
        embedding_fn,
        llm_invoke,
        top_k=2
    )
    return json.dumps(final_report)


