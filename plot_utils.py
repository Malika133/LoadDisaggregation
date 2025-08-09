import os
import matplotlib.pyplot as plt
from scipy.signal import welch
import pandas as pd

def plot_psd(df, dataset_id, freq_min, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fs = 60 / freq_min
    plt.figure(figsize=(10, 5))
    for label in df['HeatingOrCooling'].unique():
        sig = df[df['HeatingOrCooling'] == label]['main_meter(kW)'].fillna(0).values
        if len(sig) > 1:
            f, pxx = welch(sig, fs=fs, nperseg=min(256, len(sig)))
            plt.semilogy(f, pxx, label=f"Label {label}")
    plt.legend(); plt.title(f"PSD - Dataset {dataset_id}")
    plt.savefig(f"{out_dir}/{dataset_id}_psd.png"); plt.close()

def plot_power_timeseries(df, dataset_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.plot(df['timestamp'], df['main_meter(kW)'], label='Power')
    plt.scatter(df['timestamp'], df['main_meter(kW)'], c=pd.Categorical(df['HeatingOrCooling']).codes, s=6)
    plt.legend(); plt.title(f"Power vs Time - Dataset {dataset_id}")
    plt.savefig(f"{out_dir}/{dataset_id}_timeseries.png"); plt.close()

