import json
import pickle
import re
import numpy as np
import time
import hashlib
from typing import List, Tuple, Dict

import anthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIG ================= #
THRESHOLD_HIGH = 0.85       # strict for common symptoms
THRESHOLD_MID  = 0.75       # default
THRESHOLD_LOW  = 0.60       # lenient for rare/complex symptoms
VERBOSE        = True
MAX_RETRIES    = 3
USE_GPU        = False       # set True if CUDA available
# ========================================== #


# ───────────────────────────────────────────
# 1. Load SID.p
# ───────────────────────────────────────────
with open("SID.p", "rb") as f:
    sid = pickle.load(f)

sid_small    = {k.lower(): v for k, v in sid.items()}
known_symptoms = list(sid_small.keys())

# Frequency buckets for adaptive threshold
# Common symptoms → stricter match, rare → lenient
COMMON_SYMPTOMS = {
    "chest pain", "fatigue", "fever", "headache", "nausea",
    "vomiting", "shortness of breath", "dizziness", "cough",
    "abdominal pain", "back pain", "joint pain", "diarrhea",
    "constipation", "sweating", "chills", "loss of appetite"
}


# ───────────────────────────────────────────
# 2. Load embedding model
# ───────────────────────────────────────────
print("Loading embedding model...")
device = "cuda" if USE_GPU else "cpu"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

print("Computing SID embeddings...")
sid_embeddings = embedding_model.encode(known_symptoms)
print(f"SID embeddings shape: {sid_embeddings.shape}")


# ───────────────────────────────────────────
# 3. Load conversations + self reports
# ───────────────────────────────────────────
with open("conversations.json", "r") as f:
    conversations = json.load(f)

# FIX: Also load self reports
with open("self_reports.json", "r") as f:
    self_reports = json.load(f)

# Build lookup: dialog_id → self_report text
self_report_lookup: Dict[str, str] = {
    sr["dialog_id"]: sr.get("text", "")
    for sr in self_reports
}


# ───────────────────────────────────────────
# 4. Claude client
# ───────────────────────────────────────────
client = anthropic.Anthropic(api_key="YOUR_API_KEY_HERE")


# ───────────────────────────────────────────
# 5. Utilities
# ───────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def safe_json_parse(text: str) -> List[str]:
    text = text.strip()
    # Strip markdown code fences if LLM adds them
    text = re.sub(r'```json|```', '', text).strip()
    text = re.sub(r',\s*]', ']', text)

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        # Handle case where LLM returns {"symptoms": [...]}
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return v
        return []
    except:
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                cleaned = re.sub(r',\s*]', ']', match.group())
                return json.loads(cleaned)
            except:
                return []
        return []


# ───────────────────────────────────────────
# 6. LLM Extraction — DIALOG
# ───────────────────────────────────────────
llm_cache = {}

def extract_symptoms_from_dialog_llm(conversation_text: str) -> List[str]:
    key = "dialog_" + hashlib.md5(conversation_text.encode()).hexdigest()
    if key in llm_cache:
        return llm_cache[key]

    # FIX: Stronger prompt with negation, severity, normalization
    prompt = f"""You are a medical symptom extractor analyzing a doctor-patient conversation.

Extract ALL symptoms the patient has confirmed.

STRICT RULES:
1. If doctor asks about a symptom and patient says YES (or confirms) → INCLUDE it
2. If patient explicitly DENIES a symptom → DO NOT include it
3. Normalize to standard medical terms:
   - "tired" → "fatigue"
   - "throwing up" → "vomiting"
   - "tummy ache" → "abdominal pain"
   - "can't breathe well" → "shortness of breath"
4. Each symptom: 1-4 words, standardized medical term
5. Include severity ONLY if clearly stated (e.g., "severe chest pain")
6. Handle negation carefully:
   - "I don't have chest pain" → exclude
   - "No fever" → exclude
   - "Not really nauseous" → exclude
7. Output ONLY a valid JSON array, no explanation, no markdown

Example output: ["chest pain", "fatigue", "shortness of breath"]

Conversation:
{conversation_text}
"""

    for attempt in range(MAX_RETRIES):
        try:
            message = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = message.content[0].text.strip()
            symptoms = safe_json_parse(response_text)
            symptoms = [clean_text(s) for s in symptoms if s.strip()]
            llm_cache[key] = symptoms
            return symptoms

        except Exception as e:
            print(f"LLM dialog error (attempt {attempt+1}):", e)
            time.sleep(2)

    return []


# ───────────────────────────────────────────
# 7. LLM Extraction — SELF REPORT (NEW)
# ───────────────────────────────────────────
def extract_symptoms_from_self_report_llm(self_report_text: str) -> List[str]:
    """
    NEW: Dedicated extractor for self-report (patient's own description).
    Self-reports are free-form, so LLM handles them better than rules.
    """
    if not self_report_text.strip():
        return []

    key = "selfreport_" + hashlib.md5(self_report_text.encode()).hexdigest()
    if key in llm_cache:
        return llm_cache[key]

    prompt = f"""You are a medical symptom extractor analyzing a patient's self-reported complaint.

The patient wrote this in their own words. Extract ALL symptoms they are experiencing.

STRICT RULES:
1. Normalize to standard medical terms:
   - "I feel weak" → "weakness"
   - "my heart is racing" → "palpitations"
   - "I can't sleep" → "insomnia"
   - "feeling down/sad" → "depression"
2. Include duration/severity ONLY if clearly stated
3. Do NOT infer symptoms not mentioned
4. Handle negation:
   - "no pain" → exclude pain
   - "not feeling feverish" → exclude fever
5. Each symptom: 1-4 words, standardized medical term
6. Output ONLY a valid JSON array, no explanation, no markdown

Example output: ["fatigue", "chest pain", "insomnia"]

Patient self-report:
{self_report_text}
"""

    for attempt in range(MAX_RETRIES):
        try:
            message = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = message.content[0].text.strip()
            symptoms = safe_json_parse(response_text)
            symptoms = [clean_text(s) for s in symptoms if s.strip()]
            llm_cache[key] = symptoms
            return symptoms

        except Exception as e:
            print(f"LLM self-report error (attempt {attempt+1}):", e)
            time.sleep(2)

    return []


# ───────────────────────────────────────────
# 8. Rule-based YES/NO Extraction — IMPROVED
# ───────────────────────────────────────────
def extract_from_dialog_rules(conversation: List[dict]) -> List[str]:
    inferred = []

    for i in range(len(conversation) - 1):
        curr = conversation[i]
        nxt  = conversation[i + 1]

        doc_text = curr.get("text", "").lower()
        pat_text = nxt.get("text", "").lower()

        if curr.get("speaker", "").lower() == "doctor" and \
           nxt.get("speaker", "").lower() == "patient":

            is_yes = re.search(
                r'\b(yes|yeah|yep|yup|correct|right|i do|i have|indeed|absolutely|sure|definitely)\b',
                pat_text
            )
            # FIX: Stronger negation detection
            is_no = re.search(
                r'\b(no|not|never|nope|don\'t|do not|doesn\'t|does not|haven\'t|has not|neither|nor)\b',
                pat_text
            )

            if is_yes and not is_no:
                symptom = doc_text

                # FIX: Expanded filler removal
                symptom = re.sub(
                    r'(do you have|are you having|are you experiencing|any|do you feel|'
                    r'have you had|have you been experiencing|are you suffering from|'
                    r'did you have|do you suffer from|have you noticed|'
                    r'is there any|can you describe|tell me about)',
                    '',
                    symptom
                )

                symptom = re.sub(r'\?', '', symptom)
                symptom = re.sub(r'[^\w\s]', '', symptom)
                symptom = clean_text(symptom)

                # FIX: Filter out too-short or too-long extractions
                word_count = len(symptom.split())
                if symptom and 1 <= word_count <= 6:
                    inferred.append(symptom)

    return inferred


# ───────────────────────────────────────────
# 9. Multi-symptom splitting — IMPROVED
# ───────────────────────────────────────────

# FIX: Protect compound symptoms that should NOT be split
COMPOUND_SYMPTOMS = {
    "nausea and vomiting",
    "pain and discomfort",
    "shortness of breath",
    "loss of appetite",
    "loss of consciousness",
    "chest pain and tightness",
    "night sweats",
    "weight loss",
    "weight gain",
    "blurred vision",
    "runny nose",
    "sore throat",
    "muscle weakness",
    "joint pain",
    "back pain",
    "abdominal pain",
    "chest tightness"
}

def split_symptoms(symptoms: List[str]) -> List[str]:
    split_list = []

    for s in symptoms:
        # FIX: Don't split known compound symptoms
        if s in COMPOUND_SYMPTOMS:
            split_list.append(s)
            continue

        parts = re.split(r'\band\b|,|/|&', s)

        for p in parts:
            p = clean_text(p)
            if p and len(p.split()) >= 1:
                split_list.append(p)

    return list(set(split_list))


# ───────────────────────────────────────────
# 10. Adaptive threshold matching — IMPROVED
# ───────────────────────────────────────────
def get_threshold(symptom: str) -> float:
    """
    FIX: Adaptive threshold based on symptom type.
    Common symptoms → stricter (avoid false positives)
    Rare/complex symptoms → lenient (avoid false negatives)
    """
    if symptom in COMMON_SYMPTOMS:
        return THRESHOLD_HIGH
    elif len(symptom.split()) >= 3:
        # Multi-word symptoms are more specific → lenient
        return THRESHOLD_LOW
    else:
        return THRESHOLD_MID


def match_with_embeddings(
    extracted_symptoms: List[str]
) -> Tuple[List[dict], List[dict]]:

    if not extracted_symptoms:
        return [], []

    matched   = []
    unmatched = []

    extracted_symptoms   = list(set(extracted_symptoms))
    extracted_embeddings = embedding_model.encode(extracted_symptoms)

    similarity_matrix = cosine_similarity(
        extracted_embeddings,
        sid_embeddings
    )

    for i, symp in enumerate(extracted_symptoms):
        scores     = similarity_matrix[i]
        best_idx   = np.argmax(scores)
        best_score = scores[best_idx]
        best_match = known_symptoms[best_idx]

        # FIX: Adaptive threshold per symptom
        threshold = get_threshold(symp)

        if VERBOSE:
            print(f"'{symp}' → '{best_match}' ({best_score:.3f}) [threshold={threshold}]")

        if best_score >= threshold:
            matched.append({
                "extracted"  : symp,
                "matched_to" : best_match,
                "sid"        : sid_small[best_match],
                "similarity" : round(float(best_score), 4),
                "source"     : None   # filled later
            })
        else:
            unmatched.append({
                "extracted"  : symp,
                "best_attempt": best_match,
                "similarity" : round(float(best_score), 4),
                "reason"     : "below threshold"
            })

    return matched, unmatched


# ───────────────────────────────────────────
# 11. Format conversation
# ───────────────────────────────────────────
def format_conversation(conversation: List[dict]) -> str:
    return "\n".join(
        f"{turn.get('speaker', 'Unknown')}: {turn.get('text', '')}"
        for turn in conversation
    )


# ───────────────────────────────────────────
# 12. MAIN PIPELINE — IMPROVED
# ───────────────────────────────────────────
results = []

for dialog in conversations:

    dialog_id    = dialog.get("dialog_id")
    conversation = dialog.get("conversation", [])

    print(f"\n{'='*50}")
    print(f"Processing dialog {dialog_id}")
    print(f"{'='*50}")

    conversation_text = format_conversation(conversation)

    # FIX: Get self report for this dialog
    self_report_text = self_report_lookup.get(dialog_id, "")

    # ── DIALOG: STEP 1 — LLM extraction
    dialog_llm_symptoms = extract_symptoms_from_dialog_llm(conversation_text)

    # ── DIALOG: STEP 2 — Rule extraction
    dialog_rule_symptoms = extract_from_dialog_rules(conversation)

    # ── SELF REPORT: STEP 3 — LLM extraction (NEW)
    self_report_symptoms = extract_symptoms_from_self_report_llm(self_report_text)

    # ── STEP 4 — Combine all three sources
    combined_symptoms = dialog_llm_symptoms + dialog_rule_symptoms + self_report_symptoms

    print("Dialog LLM   :", dialog_llm_symptoms)
    print("Dialog Rules :", dialog_rule_symptoms)
    print("Self Report  :", self_report_symptoms)   # NEW
    print("Combined     :", combined_symptoms)

    # ── STEP 5 — Split multi-symptoms (compound-aware)
    processed_symptoms = split_symptoms(combined_symptoms)
    print("Final processed:", processed_symptoms)

    # ── STEP 6 — Adaptive threshold matching
    matched, unmatched = match_with_embeddings(processed_symptoms)

    # ── STEP 7 — Tag source of each matched symptom (NEW)
    all_dialog_symptoms = set(dialog_llm_symptoms + dialog_rule_symptoms)
    all_sr_symptoms     = set(self_report_symptoms)

    for m in matched:
        ext = m["extracted"]
        if ext in all_dialog_symptoms and ext in all_sr_symptoms:
            m["source"] = "both"
        elif ext in all_dialog_symptoms:
            m["source"] = "dialog"
        else:
            m["source"] = "self_report"

    # ── STEP 8 — Build model-ready inputs (NEW)
    # For SapBERT: join symptom names as space-separated text
    dialog_symptom_text      = " ".join(
        m["matched_to"] for m in matched if m["source"] in ("dialog", "both")
    )
    self_report_symptom_text = " ".join(
        m["matched_to"] for m in matched if m["source"] in ("self_report", "both")
    )

    results.append({
        "dialog_id"               : dialog_id,

        # Raw extractions
        "dialog_llm_symptoms"     : dialog_llm_symptoms,
        "dialog_rule_symptoms"    : dialog_rule_symptoms,
        "self_report_symptoms"    : self_report_symptoms,
        "processed_symptoms"      : processed_symptoms,

        # Matched results
        "matched_symptoms"        : matched,
        "unmatched_symptoms"      : unmatched,
        "final_symptom_list"      : list({m["matched_to"] for m in matched}),
        "sid_codes"               : list({m["sid"] for m in matched}),

        # NEW: Model-ready text inputs (feed directly to SapBERT tokenizer)
        "dialog_symptom_text"     : dialog_symptom_text,
        "self_report_symptom_text": self_report_symptom_text,
    })


# ───────────────────────────────────────────
# 13. Save output
# ───────────────────────────────────────────
with open("extracted_symptoms.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Done! Saved to extracted_symptoms.json")
print(f"Total dialogs processed: {len(results)}")
print(f"Cache hits saved: {len(llm_cache)} API calls cached")
