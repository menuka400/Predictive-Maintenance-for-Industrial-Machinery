import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---- CONFIG ----
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "Dataset")
MODEL_DIR = os.path.join(BASE_DIR, "Models")

index_names = ['unit_number', 'time_in_cycles']
setting_names = ['op_setting_1', 'op_setting_2', 'op_setting_3']
sensor_names = [f'sensor_{i}' for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

# ---- LOAD TRAIN ----
train_path = os.path.join(DATA_DIR, "train_FD001.txt")
train_df = pd.read_csv(train_path, sep=r'\s+', header=None)
train_df = train_df.iloc[:, :26]
train_df.columns = col_names

# ---- CALCULATE RUL ----
max_cycle = train_df.groupby('unit_number')['time_in_cycles'].transform('max')
train_df['RUL'] = max_cycle - train_df['time_in_cycles']

# ---- CREATE GOOD/BAD LABEL ----
THRESHOLD = 30
train_df['status'] = np.where(train_df['RUL'] > THRESHOLD, 'Good', 'Bad')

# ---- FEATURE SELECTION ----
flat_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
                 'sensor_16', 'sensor_18', 'sensor_19']
useful_sensors = [s for s in sensor_names if s not in flat_sensors]
model_df = train_df.drop(columns=flat_sensors + setting_names)

# ---- SPLIT BY ENGINE ----
unit_ids = model_df['unit_number'].unique()
train_units, val_units = train_test_split(unit_ids, test_size=0.2, random_state=42)

train_split = model_df[model_df['unit_number'].isin(train_units)]
val_split = model_df[model_df['unit_number'].isin(val_units)]

print("Train engines:", len(train_units), "| Train rows:", len(train_split))
print("Val engines:", len(val_units), "| Val rows:", len(val_split))

# ---- FEATURES / LABELS ----
feature_cols = useful_sensors
X_train = train_split[feature_cols]
y_train = train_split['status']

X_val = val_split[feature_cols]
y_val = val_split['status']

# ---- TRAIN MODEL ----
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# ---- EVALUATE ----
y_pred = clf.predict(X_val)

print("\n--- Classification Report ---")
print(classification_report(y_val, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_val, y_pred, labels=['Good', 'Bad']))

import joblib

# ---- SAVE MODEL ----
MODEL_DIR = os.path.join(os.path.dirname(__file__), "Models")
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "rf_classifier.pkl")
joblib.dump(clf, model_path)

# Also save the feature list — you'll need the EXACT same column order when predicting later
features_path = os.path.join(MODEL_DIR, "feature_cols.pkl")
joblib.dump(feature_cols, features_path)

print(f"Model saved to: {model_path}")
print(f"Feature list saved to: {features_path}")