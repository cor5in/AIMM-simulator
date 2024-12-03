import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from sys import stderr

class UE:
    """
    User Equipment (UE) 클래스
    이동성, 트래픽 생성, 네트워크 연결 관리
    
    Parameters
    ----------
    sim : Simulator
        시뮬레이터 인스턴스
    i : int
        UE 식별자
    xyz : array-like
        초기 위치 [x, y, z]
    movement_pattern : dict
        이동 패턴 설정
    traffic_pattern : dict
        트래픽 패턴 설정
    service_requirements : dict
        서비스 요구사항
    """
    
    def __init__(self, sim, i: int, xyz: np.ndarray,
                 movement_pattern: Optional[Dict] = None,
                 traffic_pattern: Optional[Dict] = None,
                 service_requirements: Optional[Dict] = None):
        self.sim = sim
        self.i = i
        self.xyz = np.array(xyz)
        
        # 이동성 설정
        self.movement_pattern = movement_pattern or {
            'type': 'random_walk',
            'velocity': 1.0,  # m/s
            'direction_change_interval': 30.0  # 초
        }
        
        # 트래픽 설정
        self.traffic_pattern = traffic_pattern or {
            'type': 'constant',
            'data_rate': 1.0  # Mbps
        }
        
        # QoS 요구사항
        self.service_requirements = service_requirements or {
            'min_throughput': 1.0,  # Mbps
            'max_latency': 100.0    # ms
        }
        
        # 상태 변수
        self.velocity = np.zeros(3)  # 현재 속도 벡터
        self.direction = np.zeros(3)  # 현재 이동 방향
        self.serving_cell = None     # 현재 서비스 중인 셀
        self.connected = False       # 연결 상태
        
        # 성능 메트릭
        self.metrics = {
            'throughput': deque(maxlen=100),  # 처리량 이력
            'latency': deque(maxlen=100),     # 지연 이력
            'rsrp': deque(maxlen=100),        # RSRP 이력
            'sinr': deque(maxlen=100)         # SINR 이력
        }
        
        # 이벤트 타이머
        self.last_direction_change = 0.0
        self.last_measurement = 0.0
        self.measurement_interval = 1.0  # 초
        
        # 트래픽 생성기 초기화
        self.initialize_traffic_generator()
        
    def initialize_traffic_generator(self):
        """트래픽 생성기 초기화"""
        if self.traffic_pattern['type'] == 'constant':
            self.generate_traffic = self._generate_constant_traffic
        elif self.traffic_pattern['type'] == 'poisson':
            self.generate_traffic = self._generate_poisson_traffic
        elif self.traffic_pattern['type'] == 'bursty':
            self.generate_traffic = self._generate_bursty_traffic
        else:
            raise ValueError(f"Unknown traffic pattern: {self.traffic_pattern['type']}")
            
    def _generate_constant_traffic(self) -> float:
        """일정한 트래픽 생성"""
        return self.traffic_pattern['data_rate']
        
    def _generate_poisson_traffic(self) -> float:
        """포아송 트래픽 생성"""
        mean_rate = self.traffic_pattern['mean_rate']
        return np.random.poisson(mean_rate)
        
    def _generate_bursty_traffic(self) -> float:
        """버스트 트래픽 생성"""
        if np.random.random() < self.traffic_pattern.get('burst_probability', 0.1):
            return self.traffic_pattern.get('burst_rate', 10.0)
        return self.traffic_pattern.get('base_rate', 0.1)
        
    def update_position(self):
        """위치 업데이트"""
        current_time = self.sim.env.now
        
        # 방향 변경 체크
        if (current_time - self.last_direction_change > 
            self.movement_pattern['direction_change_interval']):
            self.update_direction()
            self.last_direction_change = current_time
            
        # 속도 계산
        self.velocity = self.direction * self.movement_pattern['velocity']
        
        # 위치 업데이트
        self.xyz += self.velocity * self.sim.interval
        
        # 시뮬레이션 영역 내 제한
        self.xyz = np.clip(self.xyz, 
                          self.sim.area_bounds[0], 
                          self.sim.area_bounds[1])
                          
    def update_direction(self):
        """이동 방향 업데이트"""
        if self.movement_pattern['type'] == 'random_walk':
            # 무작위 방향 선택
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            self.direction = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            self.direction /= np.linalg.norm(self.direction)
            
    def measure_radio_conditions(self):
        """무선 환경 측정"""
        current_time = self.sim.env.now
        if current_time - self.last_measurement < self.measurement_interval:
            return
            
        self.last_measurement = current_time
        
        # 모든 셀에 대한 RSRP 측정
        best_rsrp = float('-inf')
        best_cell = None
        
        for cell in self.sim.cells:
            for freq in cell.freq_config:
                if not cell.freq_config[freq]['active']:
                    continue
                    
                rsrp = cell.get_rsrp(self.i, freq)
                if rsrp > best_rsrp:
                    best_rsrp = rsrp
                    best_cell = cell
                    
        # 측정 결과 저장
        self.metrics['rsrp'].append(best_rsrp)
        
        # 핸드오버 트리거
        if best_cell and best_cell != self.serving_cell:
            if best_rsrp > -95:  # -95 dBm 임계값
                self.trigger_handover(best_cell)
                
    def trigger_handover(self, target_cell):
        """핸드오버 트리거"""
        if self.serving_cell:
            self.serving_cell.detach(self)
        self.attach(target_cell)
        
    def attach(self, cell):
        """셀에 연결"""
        self.serving_cell = cell
        self.connected = True
        cell.attach(self)
        
    def detach(self):
        """셀에서 분리"""
        if self.serving_cell:
            self.serving_cell.detach(self)
        self.serving_cell = None
        self.connected = False
        
    def update_metrics(self):
        """성능 메트릭 업데이트"""
        if not self.connected:
            return
            
        # 처리량 측정
        current_throughput = self.serving_cell.get_UE_throughput(self.i)
        self.metrics['throughput'].append(current_throughput)
        
        # 지연 측정 (시뮬레이션 상의 가상 지연)
        current_latency = 20 + np.random.exponential(10)  # 기본 20ms + 랜덤 지연
        self.metrics['latency'].append(current_latency)
        
        # SINR 측정
        if self.serving_cell:
            current_sinr = self.calculate_sinr()
            self.metrics['sinr'].append(current_sinr)
            
    def calculate_sinr(self) -> float:
        """SINR 계산"""
        if not self.serving_cell:
            return float('-inf')
            
        # 서빙 셀로부터의 신호 전력
        serving_power = 10**(self.serving_cell.get_rsrp(self.i)/10)
        
        # 간섭 전력 계산
        interference = 0
        for cell in self.sim.cells:
            if cell != self.serving_cell:
                interference += 10**(cell.get_rsrp(self.i)/10)
                
        # 노이즈 파워 (-174 dBm/Hz + 대역폭)
        noise_power = 10**(-174/10) * self.serving_cell.bandwidth
        
        # SINR 계산
        sinr = serving_power / (interference + noise_power)
        return 10 * np.log10(sinr)
        
    def get_qos_satisfaction(self) -> Dict[str, bool]:
        """QoS 만족도 확인"""
        if not self.metrics['throughput'] or not self.metrics['latency']:
            return {'throughput': False, 'latency': False}
            
        avg_throughput = np.mean(self.metrics['throughput'])
        avg_latency = np.mean(self.metrics['latency'])
        
        return {
            'throughput': avg_throughput >= self.service_requirements['min_throughput'],
            'latency': avg_latency <= self.service_requirements['max_latency']
        }
        
    def get_serving_cell(self):
        """현재 서비스 중인 셀 반환"""
        return self.serving_cell
        
    def get_position(self) -> np.ndarray:
        """현재 위치 반환"""
        return self.xyz
        
    def get_velocity(self) -> np.ndarray:
        """현재 속도 반환"""
        return self.velocity
        
    def get_metrics(self) -> Dict:
        """성능 메트릭 반환"""
        return {
            'throughput': {
                'current': self.metrics['throughput'][-1] if self.metrics['throughput'] else 0,
                'average': np.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0
            },
            'latency': {
                'current': self.metrics['latency'][-1] if self.metrics['latency'] else 0,
                'average': np.mean(self.metrics['latency']) if self.metrics['latency'] else 0
            },
            'rsrp': {
                'current': self.metrics['rsrp'][-1] if self.metrics['rsrp'] else float('-inf'),
                'average': np.mean(self.metrics['rsrp']) if self.metrics['rsrp'] else float('-inf')
            },
            'sinr': {
                'current': self.metrics['sinr'][-1] if self.metrics['sinr'] else float('-inf'),
                'average': np.mean(self.metrics['sinr']) if self.metrics['sinr'] else float('-inf')
            }
        }
        
    def loop(self):
        """메인 루프"""
        while True:
            yield self.sim.wait(self.sim.interval)
            
            # 위치 업데이트
            self.update_position()
            
            # 무선 환경 측정
            self.measure_radio_conditions()
            
            # 성능 메트릭 업데이트
            self.update_metrics()
            
    def __repr__(self):
        return f'UE[{self.i}] at {self.xyz}'