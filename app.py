import os
import time
import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from sklearn.preprocessing import MinMaxScaler
import llm_verdict

app = Flask(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "Dataset")
TRAIN_FILE = os.path.join(DATA_PATH, "train_FD001.txt")
TEST_FILE = os.path.join(DATA_PATH, "test_FD001.txt")
RUL_FILE = os.path.join(DATA_PATH, "RUL_FD001.txt")

# ── Global Cache for Data and Model ──────────────────────────────────────────
DATA_CACHE = {}
MODEL_CACHE = {}
LLM_CACHE = {}  # Keys: (engine_id, cycle)

def load_and_preprocess_data():
    if "test_df" in DATA_CACHE:
        return DATA_CACHE["train_df"], DATA_CACHE["test_df"], DATA_CACHE["rul_df"], DATA_CACHE["sensors"]

    columns = (
        ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
        + [f'sensor_{i}' for i in range(1, 22)]
    )
    train_df = pd.read_csv(TRAIN_FILE, sep=r'\s+', header=None, names=columns)
    test_df = pd.read_csv(TEST_FILE, sep=r'\s+', header=None, names=columns)
    rul_df = pd.read_csv(RUL_FILE, sep=r'\s+', header=None, names=['RUL_truth'])

    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    train_df = train_df.merge(max_cycles, on='unit_number', how='left')
    train_df['RUL'] = (train_df['max_cycle'] - train_df['time_in_cycles']).clip(upper=125)

    flat_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    train_df.drop(flat_sensors + ['max_cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'], axis=1, inplace=True)
    test_df.drop(flat_sensors + ['op_setting_1', 'op_setting_2', 'op_setting_3'], axis=1, inplace=True)

    remaining_sensors = [c for c in train_df.columns if 'sensor' in c]

    scaler = MinMaxScaler()
    train_df[remaining_sensors] = scaler.fit_transform(train_df[remaining_sensors])
    test_df[remaining_sensors] = scaler.transform(test_df[remaining_sensors])

    window_size = 5
    for col in remaining_sensors:
        train_df[f'{col}_roll_mean'] = (
            train_df.groupby('unit_number')[col].rolling(window_size, min_periods=1).mean().reset_index(0, drop=True)
        )
        train_df[f'{col}_roll_std'] = (
            train_df.groupby('unit_number')[col].rolling(window_size, min_periods=1).std().fillna(0).reset_index(0, drop=True)
        )
        test_df[f'{col}_roll_mean'] = (
            test_df.groupby('unit_number')[col].rolling(window_size, min_periods=1).mean().reset_index(0, drop=True)
        )
        test_df[f'{col}_roll_std'] = (
            test_df.groupby('unit_number')[col].rolling(window_size, min_periods=1).std().fillna(0).reset_index(0, drop=True)
        )

    DATA_CACHE["train_df"] = train_df
    DATA_CACHE["test_df"] = test_df
    DATA_CACHE["rul_df"] = rul_df
    DATA_CACHE["sensors"] = remaining_sensors
    return train_df, test_df, rul_df, remaining_sensors

def get_model():
    if "model" in MODEL_CACHE:
        return MODEL_CACHE["model"], MODEL_CACHE["feature_cols"]
    
    clf, feature_cols = llm_verdict.load_model()
    MODEL_CACHE["model"] = clf
    MODEL_CACHE["feature_cols"] = feature_cols
    return clf, feature_cols

# Warm up data and models
load_and_preprocess_data()
get_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/engines')
def get_engines():
    _, test_df, _, _ = load_and_preprocess_data()
    engines = sorted(test_df['unit_number'].unique().tolist())
    return jsonify({"engines": engines})

@app.route('/api/engine/<int:engine_id>')
def get_engine_data(engine_id):
    _, test_df, rul_df, sensors = load_and_preprocess_data()
    engine_data = test_df[test_df['unit_number'] == engine_id].copy()
    
    if engine_data.empty:
        return jsonify({"error": "Engine not found"}), 404
        
    true_rul = int(rul_df.iloc[engine_id - 1]['RUL_truth'])
    
    # Convert dataframe to list of dicts, converting nan/inf to None for JSON
    records = engine_data.replace([np.inf, -np.inf, np.nan], None).to_dict(orient='records')
    
    return jsonify({
        "engine_id": engine_id,
        "true_rul": true_rul,
        "sensors": sensors,
        "max_cycle": len(records),
        "data": records
    })

@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
    # Support both GET query parameters and POST json payload
    if request.method == 'POST':
        params = request.get_json() or {}
    else:
        params = request.args
        
    try:
        engine_id = int(params.get('engine_id'))
        cycle = int(params.get('cycle'))
    except (TypeError, ValueError):
        return jsonify({"error": "engine_id and cycle parameters are required and must be integers"}), 400
        
    run_llm = params.get('run_llm', 'false').lower() == 'true' if isinstance(params.get('run_llm'), str) else bool(params.get('run_llm'))

    _, test_df, _, _ = load_and_preprocess_data()
    model, feature_cols = get_model()
    
    engine_data = test_df[test_df['unit_number'] == engine_id]
    if engine_data.empty:
        return jsonify({"error": f"Engine {engine_id} not found"}), 404
        
    current_data = engine_data[engine_data['time_in_cycles'] <= cycle]
    if current_data.empty:
        return jsonify({"error": f"Cycle {cycle} not found for Engine {engine_id}"}), 404
        
    last_row = current_data.iloc[-1:]
    recent_trend = current_data.tail(5)
    
    llm_response = None
    
    if run_llm:
        cache_key = (engine_id, cycle)
        if cache_key in LLM_CACHE:
            llm_response = LLM_CACHE[cache_key]
        else:
            # First run prediction to calibrate LLM
            pre_combined = llm_verdict.combine_predictions(
                model, last_row, feature_cols,
                {"status": "LLM_UNAVAILABLE", "confidence": 0.0, "reason": ""}
            )
            rf_prob_bad_pre = pre_combined['rf_prob_bad']
            
            system_prompt, user_content = llm_verdict.build_prompt(
                last_row.iloc[0], feature_cols, engine_id=engine_id,
                recent_trend=recent_trend, rf_prob_bad=rf_prob_bad_pre
            )
            llm_response = llm_verdict.call_llm(system_prompt, user_content, engine_id=engine_id)
            LLM_CACHE[cache_key] = llm_response
    else:
        llm_response = {
            "status": "LLM_UNAVAILABLE",
            "confidence": 0.0,
            "reason": "LLM disabled during simulation"
        }
        
    combined_results = llm_verdict.combine_predictions(
        model, last_row, feature_cols, llm_response
    )
    
    return jsonify({
        "engine_id": engine_id,
        "cycle": cycle,
        "rf_prob_bad": float(combined_results['rf_prob_bad']),
        "llm_prob_bad": float(combined_results['llm_prob_bad']),
        "combined_prob_bad": float(combined_results['combined_prob_bad']),
        "final_label": combined_results['final_label'],
        "color": combined_results['color'],
        "llm_response": llm_response
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, host='0.0.0.0', port=port)
