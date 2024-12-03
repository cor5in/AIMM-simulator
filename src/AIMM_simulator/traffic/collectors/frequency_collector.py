# Geon Kim 2024-11-21

from .base_collector import BaseCollector
import numpy as np

class FrequencyCollector(BaseCollector):
    """주파수별 트래픽 수집기"""
    
    def __init__(self, cell, sampling_interval=1.0):
        super().__init__(sampling_interval)
        self.cell = cell
        self.frequencies = cell.freq_config.keys()
        
    def collect(self):
        """주파수별 트래픽 부하 수집"""
        traffic_data = {}
        
        for freq in self.frequencies:
            if self.cell.freq_config[freq]['active']:
                # 주파수별 RB 사용률 계산
                rb_usage = self.cell.get_frequency_load(freq)
                # 주파수별 연결된 UE 수
                connected_ues = len(self.cell.get_frequency_users(freq))
                # 주파수별 처리량
                throughput = self.calculate_frequency_throughput(freq)
                
                traffic_data[freq] = {
                    'rb_usage': rb_usage,
                    'connected_ues': connected_ues,
                    'throughput': throughput
                }
        
        self.buffer_data(traffic_data)
        return traffic_data
    
    def calculate_frequency_throughput(self, freq):
        """주파수별 총 처리량 계산"""
        total_throughput = 0
        for ue_id in self.cell.get_frequency_users(freq):
            ue_throughput = self.cell.get_UE_throughput(ue_id)
            if ue_throughput is not None:
                total_throughput += ue_throughput
        return total_throughput