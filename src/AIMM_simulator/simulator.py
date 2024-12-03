"""
AIMM 시뮬레이터 메인 모듈
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import simpy
import matplotlib.pyplot as plt
from datetime import datetime

from .components.cell import Cell
from .components.mme import MME
from .components.ric import RIC
from .components.ue import UE
from .components.scenario import Scenario
from .utils.logger import SimulationLogger
from .utils.pathloss import PathLossFactory

class Simulator:
    """AIMM 시뮬레이터 메인 클래스
    
    주요 기능:
    - 시뮬레이션 환경 초기화
    - 컴포넌트 생성 및 관리
    - 시뮬레이션 실행
    - 결과 수집 및 시각화
    """
    
    def __init__(self, config_path: str):
        """
        Parameters
        ----------
        config_path : str
            설정 파일 경로
        """
        # 설정 로드
        self.config = self._load_config(config_path)
        
        # SimPy 환경 초기화
        self.env = simpy.Environment()
        
        # 로거 초기화
        self.logger = SimulationLogger(
            log_dir=self.config['logging']['directory'],
            log_level=self.config['logging']['level']
        )
        
        # 컴포넌트 초기화
        self.cells = {}
        self.ues = {}
        self.mme = None
        self.ric = None
        self.scenario = None
        
        # 메트릭 저장소
        self.metrics = {
            'network': {'load': [], 'energy': [], 'time': []},
            'cells': {},
            'ues': {}
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드
        
        Parameters
        ----------
        config_path : str
            설정 파일 경로
            
        Returns
        -------
        Dict
            설정 데이터
        """
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            elif config_path.endswith('.json'):
                return json.load(f)
            else:
                raise ValueError("Unsupported config file format")
    
    def setup(self) -> None:
        """시뮬레이션 환경 설정"""
        # MME 생성
        self.mme = MME(self.config['mme'], self.env, self.logger)
        
        # RIC 생성
        self.ric = RIC(self.config['ric'], self.env, self.logger)
        
        # 셀 생성
        for cell_config in self.config['cells']:
            cell = Cell(cell_config, self.env, self.logger)
            self.cells[cell.id] = cell
            self.ric.register_cell(cell)
            self.metrics['cells'][cell.id] = {
                'load': [], 'energy': [], 'active': [], 'time': []
            }
        
        # UE 생성
        for ue_config in self.config['ues']:
            ue = UE(ue_config, self.env, self.mme, self.logger)
            self.ues[ue.id] = ue
            self.mme.register_ue(ue)
            self.metrics['ues'][ue.id] = {
                'traffic': [], 'rsrp': [], 'throughput': [], 'time': []
            }
        
        # 시나리오 생성
        self.scenario = Scenario(self.config['scenario'], self.env, self.logger)
    
    def run(self) -> None:
        """시뮬레이션 실행"""
        # 환경 설정 검증
        if not self._validate_setup():
            raise ValueError("Invalid simulation setup")
        
        # 시작 시간 기록
        start_time = datetime.now()
        self.logger.info("Starting simulation")
        
        try:
            # 시나리오 시작
            self.scenario.start()
            
            # 컴포넌트 프로세스 시작
            for cell in self.cells.values():
                self.env.process(cell.run())
            for ue in self.ues.values():
                self.env.process(ue.run())
            self.env.process(self.mme.run())
            self.env.process(self.ric.run())
            
            # 메트릭 수집 프로세스 시작
            self.env.process(self._collect_metrics())
            
            # 시뮬레이션 실행
            self.env.run(until=self.scenario.duration)
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {str(e)}")
            raise
        
        finally:
            # 종료 시간 기록
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self.logger.info(f"Simulation completed in {duration:.2f} seconds")
    
    def _validate_setup(self) -> bool:
        """설정 검증
        
        Returns
        -------
        bool
            검증 통과 여부
        """
        try:
            # 컴포넌트 존재 확인
            if not all([self.mme, self.ric, self.scenario]):
                raise ValueError("Missing core components")
            
            # 셀 존재 확인
            if not self.cells:
                raise ValueError("No cells configured")
            
            # UE 존재 확인
            if not self.ues:
                raise ValueError("No UEs configured")
            
            # 시나리오 설정 검증
            if not self.scenario.validate_config():
                raise ValueError("Invalid scenario configuration")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Setup validation failed: {str(e)}")
            return False
    
    def _collect_metrics(self) -> None:
        """메트릭 수집 프로세스"""
        while True:
            # 네트워크 메트릭
            self.metrics['network']['load'].append(self.ric.network_load)
            self.metrics['network']['energy'].append(self.ric.network_energy)
            self.metrics['network']['time'].append(self.env.now)
            
            # 셀 메트릭
            for cell_id, cell in self.cells.items():
                metrics = self.metrics['cells'][cell_id]
                metrics['load'].append(cell.load)
                metrics['energy'].append(cell.energy_consumption)
                metrics['active'].append(cell.is_active)
                metrics['time'].append(self.env.now)
            
            # UE 메트릭
            for ue_id, ue in self.ues.items():
                metrics = self.metrics['ues'][ue_id]
                metrics['traffic'].append(ue.total_traffic)
                if ue.rsrp_history:
                    metrics['rsrp'].append(ue.rsrp_history[-1])
                if ue.throughput_history:
                    metrics['throughput'].append(ue.throughput_history[-1])
                metrics['time'].append(self.env.now)
            
            # 15분 대기
            yield self.env.timeout(900)
    
    def save_results(self, output_dir: str) -> None:
        """결과 저장
        
        Parameters
        ----------
        output_dir : str
            출력 디렉토리
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 네트워크 메트릭 저장
        network_df = pd.DataFrame(self.metrics['network'])
        network_df.to_csv(os.path.join(output_dir, 'network_metrics.csv'), 
                         index=False)
        
        # 셀 메트릭 저장
        for cell_id, metrics in self.metrics['cells'].items():
            cell_df = pd.DataFrame(metrics)
            cell_df.to_csv(os.path.join(output_dir, f'cell_{cell_id}_metrics.csv'),
                          index=False)
        
        # UE 메트릭 저장
        for ue_id, metrics in self.metrics['ues'].items():
            ue_df = pd.DataFrame(metrics)
            ue_df.to_csv(os.path.join(output_dir, f'ue_{ue_id}_metrics.csv'),
                        index=False)
    
    def plot_results(self, output_dir: str) -> None:
        """결과 시각화
        
        Parameters
        ----------
        output_dir : str
            출력 디렉토리
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 네트워크 메트릭 플롯
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['network']['time'], 
                self.metrics['network']['load'])
        plt.title('Network Load')
        plt.xlabel('Time (s)')
        plt.ylabel('Load')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['network']['time'], 
                self.metrics['network']['energy'])
        plt.title('Network Energy')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'network_metrics.png'))
        plt.close()
        
        # 셀 메트릭 플롯
        plt.figure(figsize=(15, 5))
        for cell_id, metrics in self.metrics['cells'].items():
            plt.subplot(1, 3, 1)
            plt.plot(metrics['time'], metrics['load'], 
                    label=f'Cell {cell_id}')
            plt.subplot(1, 3, 2)
            plt.plot(metrics['time'], metrics['energy'], 
                    label=f'Cell {cell_id}')
            plt.subplot(1, 3, 3)
            plt.plot(metrics['time'], metrics['active'], 
                    label=f'Cell {cell_id}')
        
        plt.subplot(1, 3, 1)
        plt.title('Cell Load')
        plt.xlabel('Time (s)')
        plt.ylabel('Load')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.title('Cell Energy')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.title('Cell Status')
        plt.xlabel('Time (s)')
        plt.ylabel('Active')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cell_metrics.png'))
        plt.close()
        
        # UE 메트릭 플롯
        plt.figure(figsize=(15, 5))
        for ue_id, metrics in self.metrics['ues'].items():
            plt.subplot(1, 3, 1)
            plt.plot(metrics['time'], metrics['traffic'], 
                    label=f'UE {ue_id}')
            plt.subplot(1, 3, 2)
            plt.plot(metrics['time'], metrics['rsrp'], 
                    label=f'UE {ue_id}')
            plt.subplot(1, 3, 3)
            plt.plot(metrics['time'], metrics['throughput'], 
                    label=f'UE {ue_id}')
        
        plt.subplot(1, 3, 1)
        plt.title('UE Traffic')
        plt.xlabel('Time (s)')
        plt.ylabel('Traffic (Mbps)')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.title('UE RSRP')
        plt.xlabel('Time (s)')
        plt.ylabel('RSRP (dBm)')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.title('UE Throughput')
        plt.xlabel('Time (s)')
        plt.ylabel('Throughput (Mbps)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ue_metrics.png'))
        plt.close()