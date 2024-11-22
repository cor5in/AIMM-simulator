# Geon Kim 2024-11-21

from abc import ABC, abstractmethod
import numpy as np

class BasePredictor(ABC):
    """기본 예측기 클래스"""
    
    def __init__(self):
        self.model = None
        
    @abstractmethod
    def train(self, data):
        """모델 학습"""
        pass
    
    @abstractmethod
    def predict(self, horizon):
        """예측 수행"""
        pass
    
    @abstractmethod
    def update(self, new_data):
        """모델 업데이트"""
        pass