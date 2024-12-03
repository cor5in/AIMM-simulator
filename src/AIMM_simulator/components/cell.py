from traffic.predictors.prophet_predictor import ProphetPredictor
from traffic.storage.timeseries_db import TimeseriesDB
from ..utils.pathloss import UMa_pathloss
import numpy as np
import pandas as pd
import simpy
from collections import deque

class Cell:
    """5G 셀 타워를 표현하는 클래스"""
    i = 0  # 셀 인덱스 카운터
    
    def __init__(self, sim, interval=15*60, xyz=None, h_BS=20.0, 
                 MIMO_gain_dB=0.0, pattern=None, f_callback=None, 
                 f_callback_kwargs={}, verbosity=0):
        """
        Parameters
        ----------
        sim : Simulator
            시뮬레이터 인스턴스
        interval : float
            셀 업데이트 간격 (초)
        xyz : array-like
            셀의 3D 위치 [x, y, z]
        h_BS : float
            기지국 안테나 높이 (미터)
        MIMO_gain_dB : float
            MIMO 이득 (dB)
        pattern : array or function
            안테나 패턴
        """
        self.i = Cell.i
        Cell.i += 1
        self.sim = sim
        self.interval = interval
        
        # 트래픽 예측 관련 설정
        self.traffic_predictor = ProphetPredictor()
        self.traffic_db = TimeseriesDB()
        self.prediction_horizon = 4  # 1시간 (15분 * 4)
        
        # 주파수 설정
        self.freq_config = {
            800: {
                'bandwidth': 10.0,  # MHz
                'n_RBs': 50,       # Resource Blocks
                'active': False,
                'energy_consumed': 0.0
            },
            1800: {
                'bandwidth': 20.0,
                'n_RBs': 100,
                'active': False,
                'energy_consumed': 0.0
            },
            3600: {
                'bandwidth': 100.0,
                'n_RBs': 500,
                'active': False,
                'energy_consumed': 0.0
            }
        }
        
        # 트래픽 임계값 설정
        self.traffic_thresholds = {
            800: {'low': 0.2, 'high': 0.8},    # 20-80%
            1800: {'low': 0.3, 'high': 0.85},  # 30-85%
            3600: {'low': 0.4, 'high': 0.9}    # 40-90%
        }
        
        # 에너지 소비 설정
        self.energy_per_interval = 25  # Wh per 15min
        self.base_power_W = 130.0      # 기본 소비 전력
        self.frequency_power_W = 100.0  # 주파수당 추가 전력
        self.total_energy_consumed = 0.0
        
        # 리소스 관리
        self.rbs = simpy.Resource(self.sim.env, capacity=50)
        self.rb_masks = {freq: np.ones(config['n_RBs']) 
                        for freq, config in self.freq_config.items()}
        
        # 트래픽 이력
        self.traffic_history = {freq: [] for freq in self.freq_config}
        self.freq_users = {freq: set() for freq in self.freq_config}
        
        # 기타 설정
        self.pattern = pattern
        self.MIMO_gain_dB = MIMO_gain_dB
        self.f_callback = f_callback
        self.f_callback_kwargs = f_callback_kwargs
        self.verbosity = verbosity
        self.attached = set()
        
        # 위치 설정
        if xyz is not None:
            self.xyz = np.array(xyz)
        else:
            self.xyz = np.empty(3)
            self.xyz[:2] = 100.0 + 900.0 * self.sim.rng.random(2)
            self.xyz[2] = h_BS
            
        # 이웃 셀 관리
        self.neighbor_cells = []
        
        # 성능 모니터링
        self.reports = {
            freq: {
                'cqi': {},
                'rsrp': {},
                'throughput_Mbps': {}
            }
            for freq in self.freq_config.keys()
        }
        self.rsrp_history = {freq: {} for freq in self.freq_config.keys()}
        
    def predict_traffic(self, freq):
        """주파수별 트래픽 예측"""
        if len(self.traffic_history[freq]) < 4:  # 최소 1시간 데이터 필요
            return None
            
        data = pd.DataFrame({
            'ds': pd.date_range(end=pd.Timestamp.now(), 
                              periods=len(self.traffic_history[freq]), 
                              freq='15T'),
            'y': self.traffic_history[freq]
        })
        
        self.traffic_predictor.train(data)
        forecast = self.traffic_predictor.predict(self.prediction_horizon)
        return forecast['yhat'].values[-1]  # 마지막 예측값 반환
        
    def update_traffic_stats(self):
        """트래픽 통계 업데이트"""
        for freq in self.freq_config:
            if not self.freq_config[freq]['active']:
                continue
                
            current_load = self.get_frequency_load(freq)
            self.traffic_history[freq].append(current_load)
            
            # 최근 4시간만 유지
            if len(self.traffic_history[freq]) > 16:
                self.traffic_history[freq] = self.traffic_history[freq][-16:]
                
    def get_frequency_load(self, freq):
        """주파수별 현재 RB 사용률 계산"""
        if not self.freq_config[freq]['active']:
            return 0.0
            
        total_rbs = self.freq_config[freq]['n_RBs']
        used_rbs = sum(1 for ue in self.freq_users[freq] 
                      if self.get_UE_throughput(ue, freq) > 0)
        return used_rbs / total_rbs
        
    def manage_cell_state(self):
        """셀 상태 관리 (on/off 및 오프로딩)"""
        for freq in self.freq_config:
            if not self.freq_config[freq]['active']:
                continue
                
            current_load = self.get_frequency_load(freq)
            predicted_load = self.predict_traffic(freq)
            
            if predicted_load is None:
                continue
                
            # 트래픽이 낮을 것으로 예측되면 셀 off 고려
            if (current_load < self.traffic_thresholds[freq]['low'] and 
                predicted_load < self.traffic_thresholds[freq]['low']):
                if self.can_shift_traffic(freq):
                    self.deactivate_frequency(freq)
                    self.redistribute_traffic(freq)
                    
            # 트래픽이 높을 것으로 예측되면 셀 on 고려
            elif (current_load > self.traffic_thresholds[freq]['high'] or 
                  predicted_load > self.traffic_thresholds[freq]['high']):
                self.activate_frequency(freq)
                
    def can_shift_traffic(self, freq):
        """트래픽 이동 가능 여부 확인"""
        for ue in self.freq_users[freq]:
            # 다른 주파수로 이동 가능한지 확인
            alternative_freq = self.find_alternative_frequency(ue)
            if alternative_freq is None:
                return False
        return True
        
    def find_alternative_frequency(self, ue):
        """UE를 위한 대체 주파수 찾기"""
        best_freq = None
        best_rsrp = float('-inf')
        
        for freq in self.freq_config:
            if freq in self.active_freqs and self.get_frequency_load(freq) < 0.9:
                rsrp = self.get_rsrp(ue.i, freq)
                if rsrp > best_rsrp:
                    best_rsrp = rsrp
                    best_freq = freq
                    
        return best_freq
        
    def redistribute_traffic(self, freq):
        """트래픽 재분배"""
        users = list(self.freq_users[freq])
        for ue in users:
            alt_freq = self.find_alternative_frequency(ue)
            if alt_freq:
                self.freq_users[freq].remove(ue)
                self.freq_users[alt_freq].add(ue)
                
    def update_energy_consumption(self):
        """에너지 소비 업데이트"""
        # 기본 전력 소비
        interval_hours = self.interval / 3600  # 초를 시간으로 변환
        energy = self.base_power_W * interval_hours
        
        # 활성 주파수별 추가 전력 소비
        for freq in self.freq_config:
            if self.freq_config[freq]['active']:
                freq_energy = self.frequency_power_W * interval_hours
                self.freq_config[freq]['energy_consumed'] += freq_energy
                energy += freq_energy
                
        self.total_energy_consumed += energy
        
    def loop(self):
        """메인 루프"""
        while True:
            yield self.sim.wait(self.interval)
            
            # 트래픽 통계 업데이트
            self.update_traffic_stats()
            
            # 셀 상태 관리
            self.manage_cell_state()
            
            # 에너지 소비 업데이트
            self.update_energy_consumption()
            
            if self.f_callback is not None:
                self.f_callback(self, **self.f_callback_kwargs)
                
    def activate_frequency(self, freq):
        """주파수 대역 활성화"""
        if not self.freq_config[freq]['active']:
            self.freq_config[freq]['active'] = True
            self.active_freqs.add(freq)
            
    def deactivate_frequency(self, freq):
        """주파수 대역 비활성화"""
        if self.freq_config[freq]['active']:
            self.freq_config[freq]['active'] = False
            self.active_freqs.remove(freq)
            
    def get_energy_stats(self):
        """에너지 소비 통계"""
        return {
            'total': self.total_energy_consumed,
            'by_frequency': {
                freq: config['energy_consumed']
                for freq, config in self.freq_config.items()
            }
        }
        
    def get_rsrp(self, ue_i, freq=None):
        """주파수별 RSRP 반환"""
        if freq is None:
            return {f: self.reports[f]['rsrp'].get(ue_i, float('-inf')) 
                   for f in self.freq_config}
        return self.reports[freq]['rsrp'].get(ue_i, float('-inf'))
        
    def get_UE_throughput(self, ue_i, freq=None):
        """주파수별 UE 처리량 반환"""
        if freq is None:
            return {f: self.reports[f]['throughput_Mbps'].get(ue_i, 0.0) 
                   for f in self.freq_config}
        return self.reports[freq]['throughput_Mbps'].get(ue_i, 0.0)
        
    def get_UE_CQI(self, ue_i, freq=None):
        """주파수별 CQI 반환"""
        if freq is None:
            return {f: self.reports[f]['cqi'].get(ue_i, 0) 
                   for f in self.freq_config}
        return self.reports[freq]['cqi'].get(ue_i, 0)
        
    def set_pattern(self, pattern):
        """안테나 패턴 설정"""
        self.pattern = pattern
        
    def set_MIMO_gain(self, MIMO_gain_dB):
        """MIMO 이득 설정"""
        self.MIMO_gain_dB = MIMO_gain_dB
        
    def get_xyz(self):
        """셀 위치 반환"""
        return self.xyz
        
    def set_xyz(self, xyz):
        """셀 위치 설정"""
        self.xyz = np.array(xyz)
        
    def get_nattached(self):
        """연결된 UE 수 반환"""
        return len(self.attached)
        
    def get_neighbor_cells(self):
        """이웃 셀 목록 반환"""
        if not self.neighbor_cells:
            # 거리 기반으로 이웃 셀 찾기
            for cell in self.sim.cells:
                if cell.i != self.i:
                    dist = np.linalg.norm(self.xyz[:2] - cell.xyz[:2])
                    if dist < 1000:  # 1km 이내의 셀
                        self.neighbor_cells.append(cell)
        return self.neighbor_cells
        
    def __repr__(self):
        return f'Cell[{self.i}] at {self.xyz}'