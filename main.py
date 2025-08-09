import logging
import pandas as pd
from .data_utils import ensure_utc, optimize_memory, detect_frequency_minutes
from .feature_engineering import add_features
from .models import HeatingCoolingModel
from .process import process_test_files

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    TRAIN_CSV = './data/train.csv'
    META_CSV = './data/building_metadata.csv'
    WEATHER_CSV = './data/weather_train.csv'
    TEST_FOLDER = './data/test/'
    OUTPUT_FOLDER = './output/'
    DATA_FRAC = 0.2

    train_df = pd.read_csv(TRAIN_CSV)
    meta_df = pd.read_csv(META_CSV)
    weather_df = pd.read_csv(WEATHER_CSV)

    # Rename
    train_df.rename(columns={'building_id': 'Dataset', 'meter': 'HeatingOrCooling', 'meter_reading': 'main_meter(kW)'}, inplace=True)
    meta_df.rename(columns={'building_id': 'Dataset', 'site_id': 'location', 'square_feet': 'Area'}, inplace=True)
    weather_df.rename(columns={'site_id': 'location'}, inplace=True)

    # Merge
    train_df = train_df.merge(meta_df[['Dataset','location','primary_use','Area','year_built','floor_count']], on='Dataset', how='left')
    weather_df = ensure_utc(weather_df, 'timestamp')
    train_df = ensure_utc(train_df, 'timestamp')

    train_df.dropna(subset=['location', 'timestamp'], inplace=True)
    weather_df.dropna(subset=['location', 'timestamp'], inplace=True)

    train_df.sort_values('timestamp', inplace=True)
    weather_df.sort_values('timestamp', inplace=True)

    train_df = pd.merge_asof(train_df, weather_df, on='timestamp', by='location', direction='nearest', tolerance=pd.Timedelta('1h'))
    for col in weather_df.columns.difference(['timestamp','location']):
        if col in train_df: train_df[col] = train_df[col].fillna(0)

    train_df = optimize_memory(train_df)
    if DATA_FRAC < 1.0:
        train_df = train_df.sample(frac=DATA_FRAC, random_state=42).reset_index(drop=True)

    freq_min = detect_frequency_minutes(train_df)

    # Train models
    model_base = HeatingCoolingModel(use_fft=False)
    model_aug = HeatingCoolingModel(use_fft=True)

    train_df_base = add_features(train_df.copy(), freq_min, use_fft=False)
    model_base.train_classifier(train_df_base)
    model_base.train_regressor(train_df_base)

    train_df_aug = add_features(train_df.copy(), freq_min, use_fft=True)
    model_aug.train_classifier(train_df_aug)
    model_aug.train_regressor(train_df_aug)

    # Process tests
    process_test_files(model_base, model_aug, TEST_FOLDER, OUTPUT_FOLDER)
    logging.info("[DONE] Pipeline completed.")

