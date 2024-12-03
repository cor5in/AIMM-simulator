# Geon Kim 2024-11-21

from .base_predictor import BasePredictor
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np

class ArimaPredictor(BasePredictor):
    """ARIMA 기반 예측기"""
    
    def __init__(self, order=(1,1,1)):
        super().__init__()
        self.order = order
        
    def train(self, data):
        """ARIMA 모델 학습"""
        self.model = ARIMA(data, order=self.order)
        self.model = self.model.fit()
        return self
        
    def predict(self, horizon):
        """예측 수행"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        forecast = self.model.forecast(steps=horizon)
        return forecast
        
    def update(self, new_data):
        """모델 업데이트"""
        if self.model is None:
            self.train(new_data)
        else:
            self.model = self.model.append(new_data).fit()