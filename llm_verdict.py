import os
import joblib
import json
import requests
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def load_model():
    """
    Load the trained Random Forest classifier and the expected feature column list.
    """
    BASE_DIR = os.path.dirname(__file__)
    model_path = os.path.join(BASE_DIR, "Models", "rf_classifier.pkl")
    features_path = os.path.join(BASE_DIR, "Models", "feature_cols.pkl")
    
    clf = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    return clf, feature_cols

def build_prompt(row, feature_cols, engine_id=None, recent_trend=None, rf_prob_bad=None):
    """
    Build the system and user prompt for the Groq API.
    row: pd.Series or dict of the current cycle's features.
    feature_cols: list of feature names.
    engine_id: the integer ID of the engine being analyzed.
    recent_trend: pd.DataFrame of the last 5 cycles (optional).
    rf_prob_bad: float, the Random Forest model's probability that this engine is 'Bad'.
                 Passed in so the LLM can calibrate its response against a validated model.
    """
    system_prompt = (
        f"You are a turbofan engine health expert acting as a second opinion for Engine {engine_id} ONLY. "
        "You analyze normalized sensor data (each value is scaled 0-1, "
        "where 0 = the minimum seen across all healthy engines in training, "
        "and 1 = the maximum seen across all degraded engines in training). "
        "Values near 0.5 are generally mid-range and not alarming on their own. "
        "You will also be given the Random Forest model's probability that the engine is 'Bad' — "
        "this is a validated, data-driven estimate and should strongly inform your verdict. "
        "Only override it if the sensor trend clearly shows accelerating degradation. "
        f"CRITICAL RULE: Your response must refer ONLY to Engine {engine_id}. "
        "DO NOT mention, reference, or compare any other engine number or ID in your response. "
        "Your output must be STRICTLY a JSON object with exactly these keys: "
        "'status' (either 'Good' or 'Bad'), 'confidence' (a float between 0.0 and 1.0), "
        f"and 'reason' (one short sentence about Engine {engine_id}'s specific sensors only). "
        "DO NOT add any text, markdown, or explanations outside the JSON object."
    )

    # Extract just the useful features in order
    current_values = {col: float(row[col]) for col in feature_cols}

    # RF context line
    if rf_prob_bad is not None:
        rf_label = "Bad" if rf_prob_bad >= 0.5 else "Good"
        rf_context = (
            f"Random Forest prediction for Engine {engine_id}: "
            f"Prob(Bad) = {rf_prob_bad:.3f} -> classified as '{rf_label}'.\n"
        )
    else:
        rf_context = ""

    user_content = (
        f"You are analyzing Engine {engine_id} ONLY.\n"
        f"{rf_context}"
        f"Current cycle sensor readings for Engine {engine_id} (normalized 0-1):\n"
        f"{json.dumps(current_values, indent=2)}\n"
    )

    if recent_trend is not None and not recent_trend.empty:
        trend_summary = {}
        for col in feature_cols:
            if col in recent_trend.columns:
                trend_summary[col] = [round(v, 4) for v in recent_trend[col].values]
        user_content += f"\nRecent trend for Engine {engine_id} (last {len(recent_trend)} cycles):\n{json.dumps(trend_summary, indent=2)}\n"

    user_content += (
        f"\nProvide your health verdict for Engine {engine_id} only. "
        f"Do NOT reference any other engine number. "
        f"If the RF model already shows a clear result and the sensor trends do not contradict it, "
        f"align your confidence accordingly."
    )

    return system_prompt, user_content


def call_llm(system_prompt, user_content, engine_id=None):
    """
    Call the Groq API chat completions endpoint.
    Returns a dictionary with 'status', 'confidence', 'reason'.
    engine_id: used to sanitize stray engine references from the LLM reason.
    """
    import re

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"status": "LLM_UNAVAILABLE", "confidence": 0.0, "reason": "GROQ_API_KEY not set"}
        
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.2
    }
    
    max_retries = 1
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            content = data['choices'][0]['message']['content'].strip()
            
            # Strip potential markdown code blocks
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
                
            result = json.loads(content.strip())
            
            # Validate required fields
            if 'status' not in result or 'confidence' not in result or 'reason' not in result:
                raise ValueError("Missing required fields in JSON")
                
            if result['status'] not in ['Good', 'Bad']:
                result['status'] = 'Bad'

            reason = str(result['reason'])

            # ── Sanitize stray engine ID references ───────────────────────
            # Replace any "Engine <number>" that is NOT the current engine_id
            if engine_id is not None:
                def replace_wrong_engine(match):
                    mentioned_id = int(match.group(1))
                    if mentioned_id != int(engine_id):
                        return f"Engine {engine_id}"
                    return match.group(0)
                reason = re.sub(r'[Ee]ngine\s+(\d+)', replace_wrong_engine, reason)

            return {
                "status": result['status'],
                "confidence": float(result['confidence']),
                "reason": reason
            }
            
        except Exception as e:
            if attempt == max_retries:
                return {
                    "status": "LLM_UNAVAILABLE",
                    "confidence": 0.0,
                    "reason": f"API call or parsing failed: {str(e)}"
                }
            time.sleep(1)  # Wait before retry

def combine_predictions(clf, row, feature_cols, llm_response):
    """
    Combine RF predictions with LLM predictions.
    """
    # 1. Get RF probability
    classes = list(clf.classes_)
    if 'Bad' in classes:
        bad_idx = classes.index('Bad')
    else:
        bad_idx = 0 
        
    if isinstance(row, pd.Series):
        X = pd.DataFrame([row[feature_cols]])
    else:
        X = row[feature_cols]
        
    rf_probs = clf.predict_proba(X)[0]
    rf_prob_bad = rf_probs[bad_idx]
    
    # 2. Get LLM probability of Bad
    if llm_response['status'] == "LLM_UNAVAILABLE":
        llm_prob_bad = rf_prob_bad 
        rf_weight = 1.0
        llm_weight = 0.0
    else:
        rf_weight = 0.6
        llm_weight = 0.4
        if llm_response['status'] == 'Bad':
            llm_prob_bad = llm_response['confidence']
        else:
            llm_prob_bad = 1.0 - llm_response['confidence']
            
    # 3. Combine
    combined_prob_bad = (rf_weight * rf_prob_bad) + (llm_weight * llm_prob_bad)
    
    # 4. Final Verdict
    if combined_prob_bad > 0.8:
        final_label = 'Critical'
        color = 'red'
    elif combined_prob_bad > 0.5:
        final_label = 'Warning'
        color = 'orange'
    else:
        final_label = 'Healthy'
        color = 'green'
        
    return {
        "rf_prob_bad": rf_prob_bad,
        "llm_prob_bad": llm_prob_bad,
        "combined_prob_bad": combined_prob_bad,
        "final_label": final_label,
        "color": color
    }