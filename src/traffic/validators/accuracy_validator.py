# Geon Kim 2024-11-21

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class AccuracyValidator:
    """예측 정확도 검증기"""
    
    def __init__(self):
        self.metrics = {}
        
    def validate(self, y_true, y_pred):
        """예측 정확도 검증"""
        # RMSE 계산
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAE 계산
        mae = mean_absolute_error(y_true, y_pred)
        
        # R2 score 계산
        r2 = r2_score(y_true, y_pred)
        
        # MAPE 계산
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        self.metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        return self.metrics
        
    def get_metrics(self):
        """검증 지표 반환"""
        return self.metrics
        
    def print_report(self):
        """검증 결과 출력"""
        print("예측 정확도 검증 결과:")
        print(f"RMSE: {self.metrics['rmse']:.4f}")
        print(f"MAE: {self.metrics['mae']:.4f}")
        print(f"R2 Score: {self.metrics['r2']:.4f}")
        print(f"MAPE: {self.metrics['mape']:.2f}%")