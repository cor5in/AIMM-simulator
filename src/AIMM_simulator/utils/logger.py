import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
import yaml
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

class SimulationLogger:
    """시뮬레이션 로깅 클래스
    
    시뮬레이션 이벤트, 메트릭, 디버그 정보를 기록
    """
    
    def __init__(self, name: str,
                 log_dir: str = 'logs',
                 level: str = 'INFO',
                 console_output: bool = True,
                 file_output: bool = True,
                 rotation: str = 'size',
                 max_bytes: int = 10*1024*1024,  # 10MB
                 backup_count: int = 5):
        """
        Parameters
        ----------
        name : str
            로거 이름
        log_dir : str
            로그 파일 저장 디렉토리
        level : str
            로깅 레벨 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        console_output : bool
            콘솔 출력 여부
        file_output : bool
            파일 출력 여부
        rotation : str
            로그 파일 순환 방식 ('size' 또는 'time')
        max_bytes : int
            최대 로그 파일 크기 (bytes)
        backup_count : int
            보관할 백업 파일 수
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 로거 생성
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 포매터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 콘솔 핸들러
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 파일 핸들러
        if file_output:
            log_file = self.log_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
            
            if rotation == 'size':
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
            else:  # time-based rotation
                file_handler = TimedRotatingFileHandler(
                    log_file,
                    when='midnight',
                    interval=1,
                    backupCount=backup_count
                )
                
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        # 메트릭 저장소
        self.metrics: Dict[str, Any] = {}
        
    def debug(self, message: str):
        """디버그 메시지 기록"""
        self.logger.debug(message)
        
    def info(self, message: str):
        """정보 메시지 기록"""
        self.logger.info(message)
        
    def warning(self, message: str):
        """경고 메시지 기록"""
        self.logger.warning(message)
        
    def error(self, message: str):
        """에러 메시지 기록"""
        self.logger.error(message)
        
    def critical(self, message: str):
        """치명적 에러 메시지 기록"""
        self.logger.critical(message)
        
    def log_event(self, event_type: str, details: Dict[str, Any]):
        """이벤트 기록
        
        Parameters
        ----------
        event_type : str
            이벤트 유형
        details : dict
            이벤트 상세 정보
        """
        self.logger.info(f"Event: {event_type} - {json.dumps(details)}")
        
    def log_metric(self, name: str, value: Any, timestamp: Optional[float] = None):
        """메트릭 기록
        
        Parameters
        ----------
        name : str
            메트릭 이름
        value : any
            메트릭 값
        timestamp : float, optional
            타임스탬프 (기본값: 현재 시간)
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        if name not in self.metrics:
            self.metrics[name] = []
            
        self.metrics[name].append({
            'timestamp': timestamp,
            'value': value
        })
        
    def log_network_state(self, cells: list, ues: list):
        """네트워크 상태 기록
        
        Parameters
        ----------
        cells : list
            셀 객체 리스트
        ues : list
            UE 객체 리스트
        """
        state = {
            'timestamp': datetime.now().timestamp(),
            'cells': [{
                'id': cell.i,
                'position': cell.xyz.tolist(),
                'active_freqs': [f for f in cell.freq_config if cell.freq_config[f]['active']],
                'load': cell.get_cell_load(),
                'energy': cell.get_energy_stats()
            } for cell in cells],
            'ues': [{
                'id': ue.i,
                'position': ue.xyz.tolist(),
                'serving_cell': ue.serving_cell.i if ue.serving_cell else None,
                'metrics': ue.get_metrics()
            } for ue in ues]
        }
        
        self.logger.debug(f"Network State: {json.dumps(state)}")
        
    def save_metrics(self, format: str = 'json'):
        """메트릭 저장
        
        Parameters
        ----------
        format : str
            저장 형식 ('json' 또는 'yaml')
        """
        metrics_file = self.log_dir / f"metrics_{datetime.now():%Y%m%d_%H%M%S}.{format}"
        
        if format == 'json':
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        elif format == 'yaml':
            with open(metrics_file, 'w') as f:
                yaml.dump(self.metrics, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def plot_metrics(self, metric_names: Optional[list] = None):
        """메트릭 플롯 생성
        
        Parameters
        ----------
        metric_names : list, optional
            플롯할 메트릭 이름 리스트 (기본값: 모든 메트릭)
        """
        try:
            import matplotlib.pyplot as plt
            
            if metric_names is None:
                metric_names = list(self.metrics.keys())
                
            for name in metric_names:
                if name not in self.metrics:
                    continue
                    
                data = self.metrics[name]
                timestamps = [d['timestamp'] for d in data]
                values = [d['value'] for d in data]
                
                plt.figure(figsize=(10, 6))
                plt.plot(timestamps, values)
                plt.title(f'Metric: {name}')
                plt.xlabel('Timestamp')
                plt.ylabel('Value')
                plt.grid(True)
                
                plot_file = self.log_dir / f"metric_{name}_{datetime.now():%Y%m%d_%H%M%S}.png"
                plt.savefig(plot_file)
                plt.close()
                
        except ImportError:
            self.warning("matplotlib is required for plotting metrics")
            
    def clear_metrics(self):
        """메트릭 초기화"""
        self.metrics.clear()
        
    def get_logger(self) -> logging.Logger:
        """로거 인스턴스 반환"""
        return self.logger