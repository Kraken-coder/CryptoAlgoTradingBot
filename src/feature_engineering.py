import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.feature_cols = [
            'log_average', 'vol_10', 'vol_20', 'vol_40', 'vol_ratio',
            'true_range', 'norm_range', 'atr_14', 'range_ratio',
            'vol_rel', 'vol_z', 'abs_r_x_vol', 'sum_r_6',
            'trend_regime_code', 'vol_regime_code'
        ]
        # These should ideally be loaded from the trained model metadata
        self.vol_20_q33 = 0.002 # Placeholder, should be updated with actual training data stats
        self.vol_20_q67 = 0.005 # Placeholder

    def set_quantiles(self, q33, q67):
        self.vol_20_q33 = q33
        self.vol_20_q67 = q67

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for the given DataFrame.
        Expects columns: ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        df = df.copy()
        
        # 1. Log Returns
        df['log_average'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Volatility Metrics
        df['vol_10'] = df['log_average'].rolling(window=10).std()
        df['vol_20'] = df['log_average'].rolling(window=20).std()
        df['vol_40'] = df['log_average'].rolling(window=40).std()
        df['vol_ratio'] = df['vol_10'] / df['vol_40']
        
        # 3. Range Metrics
        df['true_range'] = df['High'] - df['Low']
        df['norm_range'] = df['true_range'] / df['Close']
        df['atr_14'] = df['true_range'].rolling(window=14).mean()
        df['range_ratio'] = df['true_range'] / df['true_range'].rolling(window=20).mean()
        
        # 4. Volume Metrics
        vol_rolling_mean = df['Volume'].rolling(window=20).mean()
        vol_rolling_std = df['Volume'].rolling(window=20).std()
        
        df['vol_rel'] = df['Volume'] / vol_rolling_mean
        df['vol_z'] = (df['Volume'] - vol_rolling_mean) / vol_rolling_std
        df['abs_r_x_vol'] = df['log_average'].abs() * df['vol_rel']
        
        # 5. Regime Labels
        df['sum_r_6'] = df['log_average'].rolling(window=6).sum()
        
        # Trend Regime
        df['trend_regime'] = df['sum_r_6'].apply(lambda x: 'trend' if abs(x) > 0.01 else 'range')
        df['trend_regime_code'] = df['trend_regime'].astype('category').cat.codes
        # Note: In live inference, we need to ensure the coding is consistent. 
        # 'range' -> 0, 'trend' -> 1 (usually alphabetical)
        df['trend_regime_code'] = np.where(df['trend_regime'] == 'trend', 1, 0)

        # Volatility Regime
        # Use pre-calculated quantiles to avoid lookahead bias and ensure consistency
        def get_vol_regime(x):
            if pd.isna(x): return 'medium' # Default
            if x <= self.vol_20_q33: return 'low'
            if x >= self.vol_20_q67: return 'high'
            return 'medium'

        df['vol_regime'] = df['vol_20'].apply(get_vol_regime)
        
        # Manual encoding for consistency: high=0, low=1, medium=2 (alphabetical) -> check this!
        # Actually, let's use a fixed mapping
        # high, low, medium -> sorted: high, low, medium
        # But let's stick to a simple map
        vol_map = {'high': 0, 'low': 1, 'medium': 2}
        df['vol_regime_code'] = df['vol_regime'].map(vol_map)
        
        return df

    def get_feature_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the feature columns for the model."""
        return df[self.feature_cols]
