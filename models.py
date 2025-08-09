import numpy as np
import xgboost as xgb

class HeatingCoolingModel:
    def __init__(self, use_fft=True):
        self.use_fft = use_fft
        self.feature_cols = []
        self.classifier = None
        self.regressor = None

    def _align_features(self, df):
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_cols]

    def train_classifier(self, df):
        exclude = ['HeatingOrCooling', 'timestamp', 'location', 'Dataset']
        self.feature_cols = [c for c in df.columns if c not in exclude]
        y = df['HeatingOrCooling'].cat.codes
        dtrain = xgb.DMatrix(df[self.feature_cols], label=y)
        params = dict(objective='multi:softmax', num_class=len(np.unique(y)),
                      eval_metric='mlogloss', tree_method='hist', max_depth=4,
                      subsample=0.6, colsample_bytree=0.6, min_child_weight=10, max_bin=64)
        self.classifier = xgb.train(params, dtrain, num_boost_round=50)

    def train_regressor(self, df):
        df['HeatingOrCooling_enc'] = df['HeatingOrCooling'].cat.codes
        features_reg = self.feature_cols + ['HeatingOrCooling_enc']
        for c in features_reg:
            if c not in df.columns:
                df[c] = 0
        dtrain = xgb.DMatrix(df[features_reg], label=df['main_meter(kW)'])
        params = dict(objective='reg:squarederror', eval_metric='mae',
                      tree_method='hist', max_depth=4, subsample=0.6,
                      colsample_bytree=0.6, min_child_weight=10, max_bin=64)
        self.regressor = xgb.train(params, dtrain, num_boost_round=50)

    def predict_labels(self, df):
        df['HeatingOrCooling'] = self.classifier.predict(xgb.DMatrix(self._align_features(df))).astype(int)
        return df

    def predict_power(self, df):
        df['HeatingOrCooling_enc'] = (df['HeatingOrCooling'] if np.issubdtype(df['HeatingOrCooling'].dtype, np.integer)
                                     else df['HeatingOrCooling'].astype(int))
        features = self.feature_cols + ['HeatingOrCooling_enc']
        for c in features:
            if c not in df.columns:
                df[c] = 0
        df['PredictedPowerConsumption'] = self.regressor.predict(xgb.DMatrix(df[features]))
        return df

