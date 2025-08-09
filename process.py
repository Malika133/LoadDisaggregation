import os, glob, gc, logging
import pandas as pd
from .data_utils import ensure_utc, detect_frequency_minutes
from .feature_engineering import add_features
from .evaluation import evaluate_nmae
from .plot_utils import plot_psd, plot_power_timeseries

def process_test_files(model_base, model_aug, input_folder, output_folder, consolidated_csv="nmae_consolidated.csv"):
    os.makedirs(output_folder, exist_ok=True)
    all_results = []
    for f in glob.glob(os.path.join(input_folder, '*')):
        try:
            logging.info(f"[INFO] Processing {f}")
            df_raw = pd.read_csv(f) if f.endswith('.csv') else pd.read_excel(f)

            if 'building_id' in df_raw.columns:
                df_raw.rename(columns={'building_id':'Dataset'}, inplace=True)

            freq_min = detect_frequency_minutes(ensure_utc(df_raw.copy(), 'timestamp'))

            # Baseline
            df_base = add_features(df_raw.copy(), freq_min, use_fft=model_base.use_fft)
            df_base = model_base.predict_labels(df_base)
            df_base = model_base.predict_power(df_base)

            # Augmented
            df_aug = add_features(df_raw.copy(), freq_min, use_fft=model_aug.use_fft)
            df_aug = model_aug.predict_labels(df_aug)
            df_aug = model_aug.predict_power(df_aug)

            df_aug.to_csv(os.path.join(output_folder, f"pred_{os.path.basename(f)}"), index=False)

            # Evaluate
            nmae_base = evaluate_nmae(df_base).rename(columns={"NMAE": "NMAE_Baseline"})
            nmae_aug = evaluate_nmae(df_aug).rename(columns={"NMAE": "NMAE_Augmented"})
            merged = pd.merge(nmae_base, nmae_aug, on="Dataset", how="outer")
            merged["Improvement"] = merged["NMAE_Augmented"] - merged["NMAE_Baseline"]
            merged["SourceFile"] = os.path.basename(f)
            all_results.append(merged)

            for ds in df_aug['Dataset'].unique():
                subset = df_aug[df_aug['Dataset'] == ds]
                plot_psd(subset, ds, freq_min, output_folder)
                plot_power_timeseries(subset, ds, output_folder)

            del df_raw, df_base, df_aug
            gc.collect()
        except Exception as e:
            logging.error(f"[ERROR] {f}: {e}", exc_info=True)
    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(os.path.join(output_folder, consolidated_csv), index=False)

