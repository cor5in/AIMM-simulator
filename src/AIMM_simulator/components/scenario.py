import numpy as np
from pathlib import Path
import json
import yaml
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from sys import stderr

@dataclass
class UEConfig:
    """UE 설정 데이터 클래스"""
    initial_positions: np.ndarray  # (N, 3) 배열
    movement_patterns: Dict[str, Dict]  # 이동 패턴
    traffic_patterns: Dict[str, Dict]   # 트래픽 패턴
    service_requirements: Dict[str, float]  # QoS 요구사항

@dataclass
class CellConfig:
    """셀 설정 데이터 클래스"""
    positions: np.ndarray  # (M, 3) 배열
    frequencies: Dict[str, Dict]  # 주파수 설정
    coverage_patterns: Dict[str, Dict]  # 커버리지 패턴
    power_configs: Dict[str, float]  # 전력 설정

class Scenario:
    """
    시뮬레이션 시나리오 관리 클래스
    
    Parameters
    ----------
    name : str
        시나리오 이름
    duration : float
        시뮬레이션 지속 시간 (초)
    config_path : str, optional
        설정 파일 경로
    """
    
    def __init__(self, name: str, duration: float, config_path: Optional[str] = None):
        self.name = name
        self.duration = duration
        self.config_path = Path(config_path) if config_path else None
        
        # 기본 설정
        self.ue_config = None
        self.cell_config = None
        self.network_params = {}
        self.events = []
        
        if self.config_path and self.config_path.exists():
            self.load_config()
            
    def load_config(self):
        """설정 파일 로드"""
        if self.config_path.suffix == '.json':
            with open(self.config_path) as f:
                config = json.load(f)
        elif self.config_path.suffix in ['.yml', '.yaml']:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {self.config_path.suffix}")
            
        self._parse_config(config)
        
    def _parse_config(self, config: Dict):
        """설정 파싱"""
        # UE 설정 파싱
        if 'ue_config' in config:
            ue_cfg = config['ue_config']
            self.ue_config = UEConfig(
                initial_positions=np.array(ue_cfg['initial_positions']),
                movement_patterns=ue_cfg.get('movement_patterns', {}),
                traffic_patterns=ue_cfg.get('traffic_patterns', {}),
                service_requirements=ue_cfg.get('service_requirements', {})
            )
            
        # 셀 설정 파싱
        if 'cell_config' in config:
            cell_cfg = config['cell_config']
            self.cell_config = CellConfig(
                positions=np.array(cell_cfg['positions']),
                frequencies=cell_cfg.get('frequencies', {}),
                coverage_patterns=cell_cfg.get('coverage_patterns', {}),
                power_configs=cell_cfg.get('power_configs', {})
            )
            
        # 네트워크 파라미터 파싱
        self.network_params = config.get('network_params', {})
        
        # 이벤트 파싱
        self.events = config.get('events', [])
        
    def create_default_config(self):
        """기본 설정 생성"""
        self.ue_config = UEConfig(
            initial_positions=np.array([[0, 0, 1.5]]),  # 기본 UE 위치
            movement_patterns={
                'random_walk': {
                    'velocity': 1.0,  # m/s
                    'direction_change_interval': 30.0  # 초
                }
            },
            traffic_patterns={
                'constant': {
                    'data_rate': 1.0  # Mbps
                }
            },
            service_requirements={
                'min_throughput': 1.0,  # Mbps
                'max_latency': 100.0    # ms
            }
        )
        
        self.cell_config = CellConfig(
            positions=np.array([[0, 0, 30]]),  # 기본 셀 위치
            frequencies={
                '800MHz': {
                    'bandwidth': 10,  # MHz
                    'n_RBs': 50
                },
                '1800MHz': {
                    'bandwidth': 20,  # MHz
                    'n_RBs': 100
                }
            },
            coverage_patterns={
                'omnidirectional': {
                    'gain': 0.0,  # dB
                    'pattern': 'uniform'
                }
            },
            power_configs={
                'max_power': 43.0,  # dBm
                'min_power': 33.0   # dBm
            }
        )
        
        self.network_params = {
            'handover_margin': 3.0,  # dB
            'time_to_trigger': 1.0,  # 초
            'load_threshold': 0.8,   # 80%
            'energy_threshold': 0.3  # 30%
        }
        
    def add_event(self, time: float, event_type: str, params: Dict):
        """이벤트 추가"""
        self.events.append({
            'time': time,
            'type': event_type,
            'params': params
        })
        # 시간순 정렬
        self.events.sort(key=lambda x: x['time'])
        
    def save_config(self, path: Optional[str] = None):
        """설정 저장"""
        save_path = Path(path) if path else self.config_path
        if not save_path:
            raise ValueError("No path specified for saving config")
            
        config = {
            'name': self.name,
            'duration': self.duration,
            'ue_config': asdict(self.ue_config) if self.ue_config else {},
            'cell_config': asdict(self.cell_config) if self.cell_config else {},
            'network_params': self.network_params,
            'events': self.events
        }
        
        # NumPy 배열을 리스트로 변환
        if self.ue_config:
            config['ue_config']['initial_positions'] = \
                self.ue_config.initial_positions.tolist()
        if self.cell_config:
            config['cell_config']['positions'] = \
                self.cell_config.positions.tolist()
            
        if save_path.suffix == '.json':
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif save_path.suffix in ['.yml', '.yaml']:
            with open(save_path, 'w') as f:
                yaml.dump(config, f)
        else:
            raise ValueError(f"Unsupported config file format: {save_path.suffix}")
            
    def validate(self) -> bool:
        """시나리오 유효성 검증"""
        try:
            if self.ue_config is None:
                raise ValueError("UE configuration is missing")
            if self.cell_config is None:
                raise ValueError("Cell configuration is missing")
                
            # UE 위치 검증
            if len(self.ue_config.initial_positions.shape) != 2 or \
               self.ue_config.initial_positions.shape[1] != 3:
                raise ValueError("Invalid UE positions shape")
                
            # 셀 위치 검증
            if len(self.cell_config.positions.shape) != 2 or \
               self.cell_config.positions.shape[1] != 3:
                raise ValueError("Invalid cell positions shape")
                
            # 주파수 설정 검증
            for freq, config in self.cell_config.frequencies.items():
                if 'bandwidth' not in config or 'n_RBs' not in config:
                    raise ValueError(f"Invalid frequency config for {freq}")
                    
            # 이벤트 검증
            for event in self.events:
                if not all(k in event for k in ['time', 'type', 'params']):
                    raise ValueError("Invalid event format")
                if event['time'] < 0 or event['time'] > self.duration:
                    raise ValueError(f"Event time {event['time']} out of range")
                    
            return True
            
        except Exception as e:
            print(f"Scenario validation failed: {str(e)}", file=stderr)
            return False
            
    def get_summary(self) -> Dict:
        """시나리오 요약 정보"""
        return {
            'name': self.name,
            'duration': self.duration,
            'n_ues': len(self.ue_config.initial_positions) \
                if self.ue_config else 0,
            'n_cells': len(self.cell_config.positions) \
                if self.cell_config else 0,
            'frequencies': list(self.cell_config.frequencies.keys()) \
                if self.cell_config else [],
            'n_events': len(self.events)
        }