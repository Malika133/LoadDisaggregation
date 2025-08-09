import pandas as pd
import numpy as np
import logging

def ensure_utc(df, ts_col="timestamp"):
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    df.dropna(subset=[ts_col], inplace=True)
    if not pd.api.types.is_datetime64tz_dtype(df[ts_col]):
        df[ts_col] = df[ts_col].dt.tz_localize('UTC')
    else:
        df[ts_col] = df[ts_col].dt.tz_convert('UTC')
    return df

def detect_frequency_minutes(df, ts_col="timestamp"):
    deltas = df.sort_values(ts_col)[ts_col].diff().dropna()
    if len(deltas) == 0:
        return 60
    return max(1, int(deltas.mode().iloc[0].total_seconds() // 60))

def optimize_memory(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='unsigned')

