# Geon Kim 2024-11-21 

import pandas as pd
import numpy as np
from scipy import stats

class DataPreprocessor:
    """데이터 전처리기"""
    
    def __init__(self):
        self.scalers = {}
        
    def preprocess_timeseries(self, df):
        """시계열 데이터 전처리"""
        # 결측치 처리
        df = self._handle_missing_values(df)
        
        # 이상치 제거
        df = self._remove_outliers(df)
        
        # 정규화
        df = self._normalize_data(df)
        
        # 시계열 특성 추출
        df = self._extract_time_features(df)
        
        return df
    
    def _handle_missing_values(self, df):
        """결측치 처리"""
        # 선형 보간법으로 결측치 처리
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        return df
        
    def _remove_outliers(self, df):
        """이상치 제거"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            z_scores = stats.zscore(df[col])
            df = df[abs(z_scores) < 3]
            
        return df
    
    def _normalize_data(self, df):
        """데이터 정규화"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in self.scalers:
                self.scalers[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
            
            mean = self.scalers[col]['mean']
            std = self.scalers[col]['std']
            df[col] = (df[col] - mean) / std
            
        return df
        
    def _extract_time_features(self, df):
        """시계열 특성 추출"""
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df