# Geon Kim 2024-11-21

from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime

class BaseCollector(ABC):
    """기본 트래픽 수집기 클래스"""
    
    def __init__(self, sampling_interval=1.0):
        self.sampling_interval = sampling_interval
        self.last_collection_time = None
        self._buffer = []
        
    @abstractmethod
    def collect(self):
        """트래픽 데이터 수집 메서드"""
        pass
    
    def buffer_data(self, data):
        """데이터 임시 저장"""
        timestamp = datetime.now()
        self._buffer.append((timestamp, data))
        
    def clear_buffer(self):
        """버퍼 초기화"""
        self._buffer.clear()
        
    def get_buffer_data(self):
        """버퍼 데이터 반환"""
        return self._buffer