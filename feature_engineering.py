import pandas as pd
import numpy as np
from .data_utils import ensure_utc

def compute_fft_features(series, freq_min, n=5):
    y = series.fillna(0).values
    N = len(y)
    if N <= 1:
        return {f"fft_{i}": 0.0 for i in range(n)}
    fs = 60 / freq_min
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(N, d=1/fs)
    mags = np.abs(yf[xf >= 0])
    if len(mags) < n:
        mags = np.pad(mags, (0, n - len(mags)), mode='constant')
    return {f"fft_{i}": float(mags[i]) for i in range(n)}

def add_features(df, freq_min=60, use_fft=True):
    df = ensure_utc(df, "timestamp").sort_values(["Dataset", "timestamp"]).reset_index(drop=True)

    # Time features
    df['hour'] = df['timestamp'].dt.hour.astype('uint8')
    df['dayofweek'] = df['timestamp'].dt.dayofweek.astype('uint8')
    df['is_weekend'] = (df['dayofweek'] >= 5).astype('uint8')
    df['month'] = df['timestamp'].dt.month.astype('uint8')
    df['dayofyear'] = df['timestamp'].dt.dayofyear.astype('uint16')

    # Rolling & lag
    window = max(1, int(3*60/freq_min))
    df['rolling_3h'] = df.groupby('Dataset')['main_meter(kW)']\
                         .transform(lambda x: x.rolling(window, min_periods=1).mean().astype('float32'))
    df['lag_1'] = df.groupby('Dataset')['main_meter(kW)']\
                    .shift(1).fillna(0).astype('float32')

    # FFT if Augmented
    if use_fft:
        fft_frames = []
        for ds, grp in df.groupby('Dataset'):
            fft_feats = compute_fft_features(grp['main_meter(kW)'], freq_min)
            fft_df = pd.DataFrame([fft_feats] * len(grp), index=grp.index)
            fft_frames.append(fft_df)
        if fft_frames:
            df = pd.concat([df, pd.concat(fft_frames).sort_index()], axis=1)

    # Fill NaNs
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(0)

    # Target encoding
    if "HeatingOrCooling" in df.columns:
        df["HeatingOrCooling"] = df["HeatingOrCooling"].astype(str).fillna("missing").astype("category")

    # One-hot encode categorical (excluding target)
    cat_cols = df.select_dtypes(include=['category']).columns.tolist()
    if "HeatingOrCooling" in cat_cols: cat_cols.remove("HeatingOrCooling")
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

