from collections import deque
import numpy as np
from sys import stderr

class MME:
    """
    Mobility Management Entity (MME) 클래스
    UE의 핸드오버와 이동성 관리를 담당
    
    Parameters
    ----------
    sim : Simulator
        시뮬레이터 인스턴스
    interval : float
        핸드오버 체크 간격 (초)
    strategy : str
        핸드오버 전략 ('strongest_cell_simple_pathloss_model' 또는 'best_rsrp_cell')
    anti_pingpong : float
        핑퐁 방지를 위한 시간 간격 (초)
    verbosity : int
        디버깅 출력 레벨 (0=없음)
    """
    
    def __init__(self, sim, interval=10.0, 
                 strategy='strongest_cell_simple_pathloss_model',
                 anti_pingpong=30.0, verbosity=0):
        self.sim = sim
        self.interval = interval
        self.strategy = strategy
        self.anti_pingpong = anti_pingpong
        self.verbosity = verbosity
        
        # 핸드오버 이력 관리
        self.handover_history = {}  # {ue_id: [(시간, 이전셀, 새셀), ...]}
        
        print(f'MME: using handover strategy {self.strategy}.', file=stderr)
        
    def do_handovers(self):
        """핸드오버 수행"""
        for ue in self.sim.UEs:
            current_cell = ue.get_serving_cell()
            if current_cell is None:
                continue
                
            # 핸드오버 대상 셀 찾기
            if self.strategy == 'strongest_cell_simple_pathloss_model':
                target_cell_i = self.sim.get_strongest_cell_simple_pathloss_model(ue.xyz)
            elif self.strategy == 'best_rsrp_cell':
                target_cell_i = self.sim.get_best_rsrp_cell(ue.i)
            else:
                raise ValueError(f'Unknown handover strategy: {self.strategy}')
                
            if target_cell_i is None:
                continue
                
            target_cell = self.sim.cells[target_cell_i]
            
            # 현재 셀과 다른 경우에만 핸드오버 고려
            if target_cell.i != current_cell.i:
                # 핑퐁 체크
                if self.check_pingpong(ue.i, current_cell.i, target_cell.i):
                    continue
                    
                # QoS 체크
                if not self.check_target_cell_quality(target_cell, ue):
                    continue
                    
                # 핸드오버 수행
                self.execute_handover(ue, current_cell, target_cell)
                
    def check_pingpong(self, ue_id, current_cell_i, target_cell_i):
        """핑퐁 현상 체크"""
        if self.anti_pingpong <= 0:
            return False
            
        if ue_id not in self.handover_history:
            return False
            
        history = self.handover_history[ue_id]
        if not history:
            return False
            
        last_time, from_cell, to_cell = history[-1]
        current_time = self.sim.env.now
        
        # 최근 핸드오버가 anti_pingpong 시간 내에 발생했고
        # 이전 핸드오버와 반대 방향인 경우
        if (current_time - last_time < self.anti_pingpong and
            ((from_cell == target_cell_i and to_cell == current_cell_i) or
             (from_cell == current_cell_i and to_cell == target_cell_i))):
            return True
            
        return False
        
    def check_target_cell_quality(self, target_cell, ue):
        """대상 셀의 QoS 체크"""
        # RSRP 체크
        target_rsrp = target_cell.get_rsrp(ue.i)
        if target_rsrp < -110:  # -110 dBm 임계값
            return False
            
        # 부하 체크
        for freq in target_cell.freq_config:
            if (target_cell.freq_config[freq]['active'] and 
                target_cell.get_frequency_load(freq) < 0.9):
                return True
                
        return False
        
    def execute_handover(self, ue, source_cell, target_cell):
        """핸드오버 실행"""
        if self.verbosity > 0:
            print(f'Handover UE[{ue.i}] from Cell[{source_cell.i}] to Cell[{target_cell.i}]',
                  file=stderr)
            
        # 이전 셀에서 분리
        ue.detach()
        
        # 새로운 셀에 연결
        ue.attach(target_cell)
        
        # 핸드오버 이력 기록
        if ue.i not in self.handover_history:
            self.handover_history[ue.i] = deque(maxlen=10)
        self.handover_history[ue.i].append(
            (self.sim.env.now, source_cell.i, target_cell.i)
        )
        
    def loop(self):
        """메인 루프"""
        while True:
            yield self.sim.wait(self.interval)
            self.do_handovers()
            
    def get_handover_stats(self):
        """핸드오버 통계 반환"""
        stats = {}
        for ue_id, history in self.handover_history.items():
            stats[ue_id] = {
                'total_handovers': len(history),
                'last_handover_time': history[-1][0] if history else None,
                'frequent_transitions': self.get_frequent_transitions(history)
            }
        return stats
        
    def get_frequent_transitions(self, history):
        """자주 발생하는 셀 전환 패턴 분석"""
        if len(history) < 2:
            return {}
            
        transitions = {}
        for i in range(len(history)-1):
            from_cell = history[i][1]
            to_cell = history[i][2]
            key = (from_cell, to_cell)
            transitions[key] = transitions.get(key, 0) + 1
            
        return {k: v for k, v in transitions.items() if v > 1}
        
    def finalize(self):
        """시뮬레이션 종료 시 정리 작업"""
        if self.verbosity > 0:
            print('\nMME Handover Statistics:', file=stderr)
            stats = self.get_handover_stats()
            for ue_id, ue_stats in stats.items():
                print(f'UE[{ue_id}]:', file=stderr)
                print(f'  Total handovers: {ue_stats["total_handovers"]}', file=stderr)
                print(f'  Frequent transitions: {ue_stats["frequent_transitions"]}', 
                      file=stderr)