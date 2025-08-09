import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def calculate_nmae(y_true, y_pred):
    y_true, y_pred = np.array(y_true.fillna(0)), np.array(y_pred.fillna(0))
    denom = np.mean(np.abs(y_true))
    return mean_absolute_error(y_true, y_pred) / denom if denom != 0 else float('inf')

def evaluate_nmae(df):
    results = []
    for ds in df['Dataset'].unique():
        sub = df[df['Dataset'] == ds]
        if 'main_meter(kW)' in sub.columns and 'PredictedPowerConsumption' in sub.columns:
            results.append({
                'Dataset': ds,
                'NMAE': calculate_nmae(sub['main_meter(kW)'], sub['PredictedPowerConsumption'])
            })
    return pd.DataFrame(results)

