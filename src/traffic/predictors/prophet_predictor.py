# Geon Kim 2024-11-21

from .base_predictor import BasePredictor
from prophet import Prophet
import pandas as pd

class ProphetPredictor(BasePredictor):
    """Facebook Prophet 기반 예측기"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model = Prophet(**kwargs)
        
    def train(self, data):
        """Prophet 모델 학습"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("데이터는 DataFrame 형식이어야 합니다.")
            
        if 'ds' not in data.columns or 'y' not in data.columns:
            raise ValueError("데이터는 'ds'와 'y' 컬럼을 포함해야 합니다.")
            
        self.model.fit(data)
        return self
        
    def predict(self, horizon):
        """예측 수행"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        future = self.model.make_future_dataframe(periods=horizon)
        forecast = self.model.predict(future)
        return forecast
        
    def update(self, new_data):
        """새로운 데이터로 모델 업데이트"""
        if self.model is None:
            self.train(new_data)
        else:
            self.model = Prophet(self.model.stan_init)
            combined_data = pd.concat([self.model.history, new_data])
            self.train(combined_data)