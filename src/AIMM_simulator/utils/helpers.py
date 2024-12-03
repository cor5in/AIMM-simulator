import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import json
import yaml
from pathlib import Path
from datetime import datetime
import logging
from sys import stderr

def setup_logger(name: str, level: str = 'INFO',
                log_file: Optional[str] = None) -> logging.Logger:
    """로깅 설정
    
    Parameters
    ----------
    name : str
        로거 이름
    level : str
        로깅 레벨 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    log_file : str, optional
        로그 파일 경로
        
    Returns
    -------
    logging.Logger
        설정된 로거 인스턴스
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def calculate_distance(point1: np.ndarray,
                      point2: np.ndarray) -> float:
    """3D 유클리드 거리 계산
    
    Parameters
    ----------
    point1, point2 : np.ndarray
        3D 좌표 [x, y, z]
        
    Returns
    -------
    float
        두 점 사이의 거리
    """
    return np.linalg.norm(point1 - point2)

def calculate_angle(point1: np.ndarray,
                   point2: np.ndarray) -> Tuple[float, float]:
    """두 점 사이의 방위각과 고도각 계산
    
    Parameters
    ----------
    point1, point2 : np.ndarray
        3D 좌표 [x, y, z]
        
    Returns
    -------
    tuple
        (방위각(azimuth), 고도각(elevation)) in radians
    """
    diff = point2 - point1
    
    # 방위각 (azimuth) 계산
    azimuth = np.arctan2(diff[1], diff[0])
    
    # 고도각 (elevation) 계산
    r_xy = np.sqrt(diff[0]**2 + diff[1]**2)
    elevation = np.arctan2(diff[2], r_xy)
    
    return azimuth, elevation

def db_to_linear(db_value: float) -> float:
    """dB 값을 선형 값으로 변환
    
    Parameters
    ----------
    db_value : float
        데시벨 값
        
    Returns
    -------
    float
        선형 값
    """
    return 10**(db_value/10)

def linear_to_db(linear_value: float) -> float:
    """선형 값을 dB 값으로 변환
    
    Parameters
    ----------
    linear_value : float
        선형 값
        
    Returns
    -------
    float
        데시벨 값
    """
    return 10 * np.log10(linear_value)

def calculate_throughput(sinr: float,
                        bandwidth: float,
                        efficiency: float = 0.8) -> float:
    """SINR과 대역폭으로부터 처리량 계산
    
    Parameters
    ----------
    sinr : float
        Signal to Interference plus Noise Ratio (dB)
    bandwidth : float
        대역폭 (Hz)
    efficiency : float
        스펙트럼 효율 (0~1)
        
    Returns
    -------
    float
        처리량 (bps)
    """
    # Shannon capacity formula with efficiency factor
    capacity = bandwidth * efficiency * np.log2(1 + db_to_linear(sinr))
    return capacity

def save_results(results: Dict,
                filepath: str,
                format: str = 'json') -> None:
    """시뮬레이션 결과 저장
    
    Parameters
    ----------
    results : dict
        저장할 결과 데이터
    filepath : str
        저장 경로
    format : str
        파일 형식 ('json' 또는 'yaml')
    """
    # NumPy 배열을 리스트로 변환
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    results = convert_numpy(results)
    
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'json':
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
    elif format.lower() == 'yaml':
        with open(path, 'w') as f:
            yaml.dump(results, f)
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_results(filepath: str) -> Dict:
    """시뮬레이션 결과 로드
    
    Parameters
    ----------
    filepath : str
        로드할 파일 경로
        
    Returns
    -------
    dict
        결과 데이터
    """
    path = Path(filepath)
    
    if path.suffix == '.json':
        with open(path) as f:
            return json.load(f)
    elif path.suffix in ['.yml', '.yaml']:
        with open(path) as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def generate_timestamp() -> str:
    """현재 시간 기반 타임스탬프 생성
    
    Returns
    -------
    str
        YYYYMMDD_HHMMSS 형식의 타임스탬프
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def calculate_cell_coverage(cell_position: np.ndarray,
                          area_bounds: Tuple[np.ndarray, np.ndarray],
                          resolution: int = 100) -> np.ndarray:
    """셀의 커버리지 영역 계산
    
    Parameters
    ----------
    cell_position : np.ndarray
        셀의 3D 위치
    area_bounds : tuple
        시뮬레이션 영역의 (최소, 최대) 좌표
    resolution : int
        그리드 해상도
        
    Returns
    -------
    np.ndarray
        커버리지 맵 (2D boolean array)
    """
    x = np.linspace(area_bounds[0][0], area_bounds[1][0], resolution)
    y = np.linspace(area_bounds[0][1], area_bounds[1][1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # 지상 높이 (z=0)에서의 커버리지 계산
    points = np.stack([X, Y, np.zeros_like(X)], axis=-1)
    distances = np.linalg.norm(points - cell_position, axis=-1)
    
    # 거리에 따른 경로손실 계산 (간단한 모델 사용)
    path_loss = 20 * np.log10(distances) + 20 * np.log10(2.4e9) - 147.55
    
    # -120 dBm 임계값 이상인 영역을 커버리지로 간주
    coverage = path_loss <= 120
    return coverage

def generate_grid_positions(area_bounds: Tuple[np.ndarray, np.ndarray],
                          n_points: int) -> np.ndarray:
    """균일한 그리드 위치 생성
    
    Parameters
    ----------
    area_bounds : tuple
        시뮬레이션 영역의 (최소, 최대) 좌표
    n_points : int
        생성할 포인트 수
        
    Returns
    -------
    np.ndarray
        생성된 위치 좌표 배열
    """
    n_per_dim = int(np.ceil(np.sqrt(n_points)))
    x = np.linspace(area_bounds[0][0], area_bounds[1][0], n_per_dim)
    y = np.linspace(area_bounds[0][1], area_bounds[1][1], n_per_dim)
    
    X, Y = np.meshgrid(x, y)
    positions = np.stack([X.flatten(), Y.flatten(), 
                         np.zeros_like(X.flatten())], axis=-1)
    
    return positions[:n_points]

def calculate_network_metrics(cells: List,
                            ues: List) -> Dict:
    """네트워크 전체의 성능 메트릭 계산
    
    Parameters
    ----------
    cells : list
        셀 객체 리스트
    ues : list
        UE 객체 리스트
        
    Returns
    -------
    dict
        네트워크 성능 메트릭
    """
    metrics = {
        'network_load': 0,
        'total_throughput': 0,
        'average_latency': 0,
        'energy_consumption': 0,
        'user_satisfaction': 0
    }
    
    # 네트워크 부하 및 처리량
    active_cells = 0
    for cell in cells:
        cell_metrics = cell.get_metrics()
        metrics['network_load'] += cell_metrics['load']
        metrics['total_throughput'] += cell_metrics['throughput']
        metrics['energy_consumption'] += cell_metrics['energy']
        if cell_metrics['load'] > 0:
            active_cells += 1
            
    if active_cells > 0:
        metrics['network_load'] /= active_cells
        
    # 사용자 성능
    satisfied_users = 0
    total_latency = 0
    for ue in ues:
        ue_metrics = ue.get_metrics()
        total_latency += ue_metrics['latency']['average']
        if ue.get_qos_satisfaction()['throughput']:
            satisfied_users += 1
            
    if ues:
        metrics['average_latency'] = total_latency / len(ues)
        metrics['user_satisfaction'] = satisfied_users / len(ues)
        
    return metrics