import numpy as np
from sys import stderr
from collections import defaultdict

class RIC:
    """
    RAN Intelligent Controller (RIC) 클래스
    네트워크 최적화와 자동화를 담당
    
    Parameters
    ----------
    sim : Simulator
        시뮬레이터 인스턴스
    interval : float
        최적화 체크 간격 (초)
    verbosity : int
        디버깅 출력 레벨 (0=없음)
    """
    
    def __init__(self, sim, interval=10.0, verbosity=0):
        self.sim = sim
        self.interval = interval
        self.verbosity = verbosity
        
        # 성능 메트릭 저장
        self.metrics = {
            'network_load': [],
            'energy_efficiency': [],
            'user_satisfaction': [],
            'cell_states': defaultdict(list)
        }
        
        # 최적화 임계값
        self.thresholds = {
            'load_balance': 0.2,    # 셀 간 부하 차이 임계값
            'energy_saving': 0.3,   # 에너지 절약 모드 진입 임계값
            'qos_minimum': 0.7      # 최소 QoS 만족도
        }
        
    def optimize_network(self):
        """네트워크 최적화 수행"""
        self.update_metrics()
        
        # 부하 분산 최적화
        self.balance_network_load()
        
        # 에너지 효율성 최적화
        self.optimize_energy_efficiency()
        
        # QoS 최적화
        self.optimize_qos()
        
    def update_metrics(self):
        """성능 메트릭 업데이트"""
        # 네트워크 부하 계산
        total_load = 0
        active_cells = 0
        for cell in self.sim.cells:
            cell_load = max(cell.get_frequency_load(freq) 
                          for freq in cell.freq_config 
                          if cell.freq_config[freq]['active'])
            total_load += cell_load
            active_cells += 1
            self.metrics['cell_states'][cell.i].append({
                'load': cell_load,
                'active_freqs': [freq for freq in cell.freq_config 
                               if cell.freq_config[freq]['active']],
                'energy': cell.get_energy_stats()['total']
            })
            
        if active_cells > 0:
            avg_network_load = total_load / active_cells
            self.metrics['network_load'].append(avg_network_load)
            
        # 에너지 효율성 계산
        total_energy = sum(cell.get_energy_stats()['total'] 
                          for cell in self.sim.cells)
        total_throughput = sum(cell.get_UE_throughput(ue.i)
                             for cell in self.sim.cells
                             for ue in cell.attached)
        energy_efficiency = total_throughput / total_energy if total_energy > 0 else 0
        self.metrics['energy_efficiency'].append(energy_efficiency)
        
        # 사용자 만족도 계산
        satisfied_users = sum(1 for cell in self.sim.cells
                            for ue in cell.attached
                            if self.check_user_satisfaction(ue))
        total_users = sum(len(cell.attached) for cell in self.sim.cells)
        satisfaction_rate = satisfied_users / total_users if total_users > 0 else 1.0
        self.metrics['user_satisfaction'].append(satisfaction_rate)
        
    def balance_network_load(self):
        """셀 간 부하 분산"""
        cells_load = [(cell, self.get_cell_load(cell)) 
                     for cell in self.sim.cells]
        avg_load = np.mean([load for _, load in cells_load])
        
        for cell, load in cells_load:
            if abs(load - avg_load) > self.thresholds['load_balance']:
                if load > avg_load:
                    # 과부하 셀의 트래픽 분산
                    self.redistribute_cell_load(cell)
                elif load < avg_load:
                    # 저부하 셀의 커버리지 확장
                    self.expand_cell_coverage(cell)
                    
    def optimize_energy_efficiency(self):
        """에너지 효율성 최적화"""
        for cell in self.sim.cells:
            cell_load = self.get_cell_load(cell)
            if cell_load < self.thresholds['energy_saving']:
                # 저부하 주파수 비활성화
                self.optimize_cell_frequencies(cell)
                
    def optimize_qos(self):
        """QoS 최적화"""
        for cell in self.sim.cells:
            unsatisfied_users = [ue for ue in cell.attached 
                               if not self.check_user_satisfaction(ue)]
            if unsatisfied_users:
                self.improve_user_qos(cell, unsatisfied_users)
                
    def get_cell_load(self, cell):
        """셀의 전체 부하 계산"""
        return max(cell.get_frequency_load(freq) 
                  for freq in cell.freq_config 
                  if cell.freq_config[freq]['active'])
                  
    def check_user_satisfaction(self, ue):
        """사용자 QoS 만족도 확인"""
        serving_cell = ue.get_serving_cell()
        if serving_cell is None:
            return False
            
        throughput = serving_cell.get_UE_throughput(ue.i)
        rsrp = serving_cell.get_rsrp(ue.i)
        
        return (throughput >= 1.0 and  # 최소 1 Mbps
                rsrp >= -110)          # 최소 RSRP
                
    def redistribute_cell_load(self, cell):
        """과부하 셀의 부하 분산"""
        neighbor_cells = cell.get_neighbor_cells()
        if not neighbor_cells:
            return
            
        # 부하가 낮은 이웃 셀 찾기
        available_neighbors = [
            n for n in neighbor_cells
            if self.get_cell_load(n) < 0.7  # 70% 미만 부하
        ]
        
        if not available_neighbors:
            return
            
        # 과부하 셀의 UE들을 이웃 셀로 재분배
        for ue in list(cell.attached):
            best_neighbor = max(available_neighbors,
                              key=lambda n: n.get_rsrp(ue.i))
            if best_neighbor.get_rsrp(ue.i) > -105:  # 최소 RSRP 체크
                ue.attach(best_neighbor)
                
    def expand_cell_coverage(self, cell):
        """저부하 셀의 커버리지 확장"""
        # 주파수별 전력 증가
        for freq in cell.freq_config:
            if cell.freq_config[freq]['active']:
                # 전력 10% 증가 (최대 3dB까지)
                current_power = cell.freq_config[freq].get('power', 0)
                cell.freq_config[freq]['power'] = min(current_power + 0.5, 3.0)
                
    def optimize_cell_frequencies(self, cell):
        """셀의 주파수 사용 최적화"""
        active_freqs = [freq for freq in cell.freq_config 
                       if cell.freq_config[freq]['active']]
        
        for freq in active_freqs:
            freq_load = cell.get_frequency_load(freq)
            if freq_load < self.thresholds['energy_saving']:
                # 다른 주파수로 트래픽 이동이 가능한 경우
                if cell.can_shift_traffic(freq):
                    cell.deactivate_frequency(freq)
                    
    def improve_user_qos(self, cell, unsatisfied_users):
        """사용자 QoS 개선"""
        for ue in unsatisfied_users:
            current_freq = None
            for freq in cell.freq_config:
                if ue in cell.freq_users[freq]:
                    current_freq = freq
                    break
                    
            if current_freq is None:
                continue
                
            # 더 좋은 주파수 찾기
            best_freq = cell.find_alternative_frequency(ue)
            if best_freq and best_freq != current_freq:
                cell.freq_users[current_freq].remove(ue)
                cell.freq_users[best_freq].add(ue)
                
    def loop(self):
        """메인 루프"""
        while True:
            yield self.sim.wait(self.interval)
            self.optimize_network()
            
    def get_optimization_stats(self):
        """최적화 통계 반환"""
        return {
            'network_load': {
                'current': self.metrics['network_load'][-1],
                'average': np.mean(self.metrics['network_load'])
            },
            'energy_efficiency': {
                'current': self.metrics['energy_efficiency'][-1],
                'average': np.mean(self.metrics['energy_efficiency'])
            },
            'user_satisfaction': {
                'current': self.metrics['user_satisfaction'][-1],
                'average': np.mean(self.metrics['user_satisfaction'])
            },
            'cell_states': {
                cell_id: {
                    'load_history': [state['load'] for state in states],
                    'energy_consumption': states[-1]['energy']
                }
                for cell_id, states in self.metrics['cell_states'].items()
            }
        }
        
    def finalize(self):
        """시뮬레이션 종료 시 정리 작업"""
        if self.verbosity > 0:
            print('\nRIC Optimization Statistics:', file=stderr)
            stats = self.get_optimization_stats()
            print(f'Average network load: {stats["network_load"]["average"]:.2f}',
                  file=stderr)
            print(f'Average energy efficiency: {stats["energy_efficiency"]["average"]:.2f}',
                  file=stderr)
            print(f'Average user satisfaction: {stats["user_satisfaction"]["average"]:.2f}',
                  file=stderr)