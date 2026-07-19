import os
# Disable Gradio SSR — causes 503 during startup on HF Spaces
os.environ["GRADIO_SSR_MODE"] = "False"

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import llm_verdict
import gradio as gr
import spaces
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "Dataset")
TRAIN_FILE = os.path.join(DATA_PATH, "train_FD001.txt")
TEST_FILE  = os.path.join(DATA_PATH, "test_FD001.txt")
RUL_FILE   = os.path.join(DATA_PATH, "RUL_FD001.txt")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE   = os.path.join(BASE_DIR, "templates", "index.html")

# ── Global Cache ─────────────────────────────────────────────────────────────
DATA_CACHE  = {}
MODEL_CACHE = {}
LLM_CACHE   = {}

def load_and_preprocess_data():
    if "test_df" in DATA_CACHE:
        return DATA_CACHE["train_df"], DATA_CACHE["test_df"], DATA_CACHE["rul_df"], DATA_CACHE["sensors"]

    columns = (
        ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
        + [f'sensor_{i}' for i in range(1, 22)]
    )
    train_df = pd.read_csv(TRAIN_FILE, sep=r'\s+', header=None, names=columns)
    test_df  = pd.read_csv(TEST_FILE,  sep=r'\s+', header=None, names=columns)
    rul_df   = pd.read_csv(RUL_FILE,   sep=r'\s+', header=None, names=['RUL_truth'])

    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    train_df = train_df.merge(max_cycles, on='unit_number', how='left')
    train_df['RUL'] = (train_df['max_cycle'] - train_df['time_in_cycles']).clip(upper=125)

    flat_sensors = ['sensor_1','sensor_5','sensor_6','sensor_10','sensor_16','sensor_18','sensor_19']
    train_df.drop(flat_sensors + ['max_cycle','op_setting_1','op_setting_2','op_setting_3'], axis=1, inplace=True)
    test_df.drop( flat_sensors + ['op_setting_1','op_setting_2','op_setting_3'], axis=1, inplace=True)

    remaining_sensors = [c for c in train_df.columns if 'sensor' in c]

    scaler = MinMaxScaler()
    train_df[remaining_sensors] = scaler.fit_transform(train_df[remaining_sensors])
    test_df[remaining_sensors]  = scaler.transform(test_df[remaining_sensors])

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
    DATA_CACHE["test_df"]  = test_df
    DATA_CACHE["rul_df"]   = rul_df
    DATA_CACHE["sensors"]  = remaining_sensors
    return train_df, test_df, rul_df, remaining_sensors

def get_model():
    if "model" in MODEL_CACHE:
        return MODEL_CACHE["model"], MODEL_CACHE["feature_cols"]
    clf, feature_cols = llm_verdict.load_model()
    MODEL_CACHE["model"]       = clf
    MODEL_CACHE["feature_cols"] = feature_cols
    return clf, feature_cols

# Warm up on startup
load_and_preprocess_data()
get_model()

# ── Gradio demo (required by HF Spaces Gradio SDK) ───────────────────────────
@spaces.GPU
def dummy_gpu():
    return "ok"

with gr.Blocks(title="AeroShield") as demo:
    gr.HTML('<meta http-equiv="refresh" content="0; url=/dashboard">')
    _btn = gr.Button("Init", visible=False)
    _out = gr.Textbox(visible=False)
    _btn.click(dummy_gpu, outputs=_out)

demo.queue()

# ── Build combined dashboard HTML (inline CSS + JS so no /static mount needed) ─
def _build_dashboard_html():
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(STATIC_DIR, "style.css"), "r", encoding="utf-8") as f:
        css = f.read()
    with open(os.path.join(STATIC_DIR, "app.js"), "r", encoding="utf-8") as f:
        js = f.read()
    html = html.replace('<link rel="stylesheet" href="/static/style.css">', f'<style>{css}</style>')
    html = html.replace('<script src="/static/app.js"></script>', f'<script>{js}</script>')
    return html

DASHBOARD_HTML = _build_dashboard_html()

# ── Custom routes on demo.app ─────────────────────────────────────────────────
@demo.app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)

@demo.app.get("/api/engines")
async def api_engines():
    _, test_df, _, _ = load_and_preprocess_data()
    engines = sorted(test_df['unit_number'].unique().tolist())
    return JSONResponse({"engines": engines})

@demo.app.get("/api/engine/{engine_id}")
async def api_engine(engine_id: int):
    _, test_df, rul_df, sensors = load_and_preprocess_data()
    engine_data = test_df[test_df['unit_number'] == engine_id].copy()
    if engine_data.empty:
        return JSONResponse({"error": "Engine not found"}, status_code=404)
    true_rul = int(rul_df.iloc[engine_id - 1]['RUL_truth'])
    records  = engine_data.replace([np.inf, -np.inf, np.nan], None).to_dict(orient='records')
    return JSONResponse({
        "engine_id": engine_id,
        "true_rul":  true_rul,
        "sensors":   sensors,
        "max_cycle": len(records),
        "data":      records
    })

@demo.app.get("/api/predict")
@demo.app.post("/api/predict")
async def api_predict(request: Request):
    if request.method == "POST":
        params = await request.json()
    else:
        params = dict(request.query_params)

    try:
        engine_id = int(params.get('engine_id'))
        cycle     = int(params.get('cycle'))
    except (TypeError, ValueError):
        return JSONResponse({"error": "engine_id and cycle required"}, status_code=400)

    run_llm_raw = params.get('run_llm', 'false')
    run_llm = run_llm_raw is True or str(run_llm_raw).lower() == 'true'

    _, test_df, _, _ = load_and_preprocess_data()
    model, feature_cols = get_model()

    engine_data = test_df[test_df['unit_number'] == engine_id]
    if engine_data.empty:
        return JSONResponse({"error": f"Engine {engine_id} not found"}, status_code=404)

    current_data = engine_data[engine_data['time_in_cycles'] <= cycle]
    if current_data.empty:
        return JSONResponse({"error": f"Cycle {cycle} not found"}, status_code=404)

    last_row     = current_data.iloc[-1:]
    recent_trend = current_data.tail(5)

    if run_llm:
        cache_key = (engine_id, cycle)
        if cache_key in LLM_CACHE:
            llm_response = LLM_CACHE[cache_key]
        else:
            pre = llm_verdict.combine_predictions(
                model, last_row, feature_cols,
                {"status": "LLM_UNAVAILABLE", "confidence": 0.0, "reason": ""}
            )
            sys_p, usr_c = llm_verdict.build_prompt(
                last_row.iloc[0], feature_cols,
                engine_id=engine_id, recent_trend=recent_trend,
                rf_prob_bad=pre['rf_prob_bad']
            )
            llm_response = llm_verdict.call_llm(sys_p, usr_c, engine_id=engine_id)
            LLM_CACHE[cache_key] = llm_response
    else:
        llm_response = {"status": "LLM_UNAVAILABLE", "confidence": 0.0, "reason": "LLM disabled during simulation"}

    combined = llm_verdict.combine_predictions(model, last_row, feature_cols, llm_response)
    return JSONResponse({
        "engine_id":        engine_id,
        "cycle":            cycle,
        "rf_prob_bad":      float(combined['rf_prob_bad']),
        "llm_prob_bad":     float(combined['llm_prob_bad']),
        "combined_prob_bad": float(combined['combined_prob_bad']),
        "final_label":      combined['final_label'],
        "color":            combined['color'],
        "llm_response":     llm_response
    })

# ── Local dev ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)
