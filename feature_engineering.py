import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "Dataset")

# Define columns according to NASA C-MAPSS dataset
index_names = ['unit_number', 'time_in_cycles']
setting_names = ['op_setting_1', 'op_setting_2', 'op_setting_3']
sensor_names = [f'sensor_{i}' for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

# ---- LOAD TRAIN ----
train_path = os.path.join(DATA_DIR, "train_FD001.txt")
train_df = pd.read_csv(train_path, sep=r'\s+', header=None)
train_df = train_df.iloc[:, :26]
train_df.columns = col_names

print("Train shape:", train_df.shape)

# ---- CALCULATE RUL ----
max_cycle = train_df.groupby('unit_number')['time_in_cycles'].transform('max')
train_df['RUL'] = max_cycle - train_df['time_in_cycles']

# ---- CREATE GOOD/BAD LABEL ----
THRESHOLD = 30  # cycles remaining — below this = "Bad"
train_df['status'] = np.where(train_df['RUL'] > THRESHOLD, 'Good', 'Bad')

print(train_df['status'].value_counts())
print(train_df[['unit_number', 'time_in_cycles', 'RUL', 'status']].head(10))

# Check variance of each sensor — flat ones contribute nothing
sensor_variance = train_df[sensor_names].std().sort_values()
print(sensor_variance)

print(train_df[setting_names].std())

# ---- FEATURE SELECTION ----
flat_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 
                 'sensor_16', 'sensor_18', 'sensor_19']

useful_sensors = [s for s in sensor_names if s not in flat_sensors]
print("Useful sensors:", useful_sensors)
print("Count:", len(useful_sensors))

# Drop flat sensors + op_settings (all confirmed constant for FD001)
model_df = train_df.drop(columns=flat_sensors + setting_names)

print(model_df.shape)
print(model_df.columns.tolist())