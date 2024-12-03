mport numpy as np
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod

class PathLossModel(ABC):
    """경로손실 모델 기본 클래스"""
    
    @abstractmethod
    def calculate_path_loss(self, distance: float, frequency: float) -> float:
        """경로손실 계산
        
        Parameters
        ----------
        distance : float
            거리 (meters)
        frequency : float
            주파수 (Hz)
            
        Returns
        -------
        float
            경로손실 (dB)
        """
        pass

class FreeSpaceModel(PathLossModel):
    """자유 공간 경로손실 모델"""
    
    def calculate_path_loss(self, distance: float, frequency: float) -> float:
        """자유 공간 경로손실 계산
        
        PL = 20log10(d) + 20log10(f) - 147.55
        
        Parameters
        ----------
        distance : float
            거리 (meters)
        frequency : float
            주파수 (Hz)
            
        Returns
        -------
        float
            경로손실 (dB)
        """
        return 20 * np.log10(distance) + 20 * np.log10(frequency) - 147.55

class OkumuraHataModel(PathLossModel):
    """오쿠무라-하타 경로손실 모델
    
    도시 환경에서의 경로손실 예측에 사용
    """
    
    def __init__(self, environment: str = 'urban'):
        """
        Parameters
        ----------
        environment : str
            환경 유형 ('urban', 'suburban', 'open')
        """
        self.environment = environment
    
    def calculate_path_loss(self, distance: float, frequency: float,
                           h_base: float = 30.0, h_mobile: float = 1.5) -> float:
        """오쿠무라-하타 경로손실 계산
        
        Parameters
        ----------
        distance : float
            거리 (km)
        frequency : float
            주파수 (MHz)
        h_base : float
            기지국 안테나 높이 (m)
        h_mobile : float
            단말기 안테나 높이 (m)
            
        Returns
        -------
        float
            경로손실 (dB)
        """
        a_hm = (1.1 * np.log10(frequency) - 0.7) * h_mobile - \
               (1.56 * np.log10(frequency) - 0.8)
        
        pl = 69.55 + 26.16 * np.log10(frequency) - 13.82 * np.log10(h_base) - \
             a_hm + (44.9 - 6.55 * np.log10(h_base)) * np.log10(distance)
        
        if self.environment == 'suburban':
            pl -= 2 * (np.log10(frequency/28))**2 + 5.4
        elif self.environment == 'open':
            pl -= 4.78 * (np.log10(frequency))**2 + 18.33 * np.log10(frequency) + 40.94
        
        return pl

class Cost231Model(PathLossModel):
    """COST-231 Hata 경로손실 모델
    
    오쿠무라-하타 모델의 확장 버전 (1500-2000 MHz)
    """
    
    def __init__(self, environment: str = 'urban'):
        """
        Parameters
        ----------
        environment : str
            환경 유형 ('urban', 'suburban')
        """
        self.environment = environment
    
    def calculate_path_loss(self, distance: float, frequency: float,
                           h_base: float = 30.0, h_mobile: float = 1.5) -> float:
        """COST-231 경로손실 계산
        
        Parameters
        ----------
        distance : float
            거리 (km)
        frequency : float
            주파수 (MHz)
        h_base : float
            기지국 안테나 높이 (m)
        h_mobile : float
            단말기 안테나 높이 (m)
            
        Returns
        -------
        float
            경로손실 (dB)
        """
        a_hm = (1.1 * np.log10(frequency) - 0.7) * h_mobile - \
               (1.56 * np.log10(frequency) - 0.8)
        
        pl = 46.3 + 33.9 * np.log10(frequency) - 13.82 * np.log10(h_base) - \
             a_hm + (44.9 - 6.55 * np.log10(h_base)) * np.log10(distance)
        
        if self.environment == 'suburban':
            pl += 0  # suburban area
        else:
            pl += 3  # urban area
        
        return pl

class Indoor3GPPModel(PathLossModel):
    """3GPP Indoor 경로손실 모델"""
    
    def calculate_path_loss(self, distance: float, frequency: float,
                           n_walls: int = 0, n_floors: int = 0) -> float:
        """3GPP Indoor 경로손실 계산
        
        Parameters
        ----------
        distance : float
            거리 (m)
        frequency : float
            주파수 (Hz)
        n_walls : int
            통과하는 벽의 수
        n_floors : int
            통과하는 층의 수
            
        Returns
        -------
        float
            경로손실 (dB)
        """
        fc = frequency / 1e9  # Convert Hz to GHz
        pl = 37 + 30 * np.log10(distance) + 18.3 * np.power((n_floors + 2)/(n_floors + 1), n_floors - 1)
        pl += 5 * n_walls
        pl += 20 * np.log10(fc)
        return pl

class KoreaUMaModel(PathLossModel):
    """한국 도시 매크로셀(UMa) 경로손실 모델
    
    3GPP 36.873 표준 기반, 한국 도시 환경에 최적화
    
    적용 범위:
    - 거리: 10m ~ 5000m
    - UE 높이: 1.5m ~ 2.0m
    - BS 높이: 25m ~ 30m
    - 주파수: 2GHz ~ 6GHz
    """
    
    def __init__(self, h_bs: float = 25.0, h_ut: float = 1.5, is_los: bool = True):
        self.h_bs = h_bs
        self.h_ut = h_ut
        self.is_los = is_los
    
    def calculate_path_loss(self, distance: float, frequency: float) -> float:
        """경로손실 계산
        
        Parameters
        ----------
        distance : float
            거리 (m)
        frequency : float
            주파수 (Hz)
            
        Returns
        -------
        float
            경로손실 (dB)
        """
        fc = frequency / 1e9  # Convert Hz to GHz
        d_2d = distance
        
        if self.is_los:
            if d_2d <= 18:
                pl = 28.0 + 22 * np.log10(d_2d) + 20 * np.log10(fc)
            else:
                pl = 28.0 + 40 * np.log10(d_2d) + 20 * np.log10(fc)
                pl -= 9 * np.log10((d_2d * d_2d + (self.h_bs - self.h_ut) ** 2))
        else:
            pl = 13.54 + 39.08 * np.log10(d_2d) + 20 * np.log10(fc)
            pl -= 0.6 * (self.h_ut - 1.5)
        
        return pl

class KoreaUMiModel(PathLossModel):
    """한국 도시 마이크로셀(UMi) 경로손실 모델
    
    3GPP 38.901 표준 기반, 한국 도시 환경에 최적화
    
    적용 범위:
    - 거리: 10m ~ 2000m
    - UE 높이: 1.5m ~ 2.0m
    - BS 높이: 10m
    - 주파수: 2GHz ~ 6GHz
    """
    
    def __init__(self, h_bs: float = 10.0, h_ut: float = 1.5, is_los: bool = True):
        self.h_bs = h_bs
        self.h_ut = h_ut
        self.is_los = is_los
    
    def calculate_path_loss(self, distance: float, frequency: float) -> float:
        fc = frequency / 1e9  # Convert Hz to GHz
        d_2d = distance
        
        if self.is_los:
            pl = 32.4 + 21 * np.log10(d_2d) + 20 * np.log10(fc)
        else:
            pl = 35.3 + 22.4 + 21.3 * np.log10(d_2d) + 20 * np.log10(fc)
            pl -= 0.3 * (self.h_ut - 1.5)
        
        return pl

class KoreaInHModel(PathLossModel):
    """한국 실내 핫스팟(InH) 경로손실 모델
    
    3GPP 36.873 표준 기반, 한국 실내 환경에 최적화
    
    적용 범위:
    - 거리: 1m ~ 150m
    - UE 높이: 1m ~ 2.5m
    - BS 높이: 2m ~ 6m
    - 주파수: 2GHz ~ 6GHz
    """
    
    def __init__(self, h_bs: float = 3.0, h_ut: float = 1.5, is_los: bool = True):
        self.h_bs = h_bs
        self.h_ut = h_ut
        self.is_los = is_los
    
    def calculate_path_loss(self, distance: float, frequency: float) -> float:
        fc = frequency / 1e9  # Convert Hz to GHz
        d_2d = distance
        
        if self.is_los:
            pl = 32.4 + 17.3 * np.log10(d_2d) + 20 * np.log10(fc)
        else:
            pl = 38.3 + 17.30 + 24.9 * np.log10(d_2d) + 20 * np.log10(fc)
        
        return pl

def get_path_loss_model(model_name: str, **kwargs) -> PathLossModel:
    """경로손실 모델 팩토리 함수
    
    Parameters
    ----------
    model_name : str
        모델 이름 ('free_space', 'okumura_hata', 'cost231', 'indoor_3gpp',
                  'korea_uma', 'korea_umi', 'korea_inh')
    **kwargs : dict
        모델 파라미터
        
    Returns
    -------
    PathLossModel
        경로손실 모델 인스턴스
    """
    models = {
        'free_space': FreeSpaceModel,
        'okumura_hata': OkumuraHataModel,
        'cost231': Cost231Model,
        'indoor_3gpp': Indoor3GPPModel,
        'korea_uma': KoreaUMaModel,
        'korea_umi': KoreaUMiModel,
        'korea_inh': KoreaInHModel
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name](**kwargs)

def calculate_shadowing(correlation_distance: float = 50.0,
                       std_dev: float = 8.0,
                       grid_size: Tuple[int, int] = (100, 100),
                       area_size: Tuple[float, float] = (1000, 1000)) -> np.ndarray:
    """상관된 쉐도잉 맵 생성
    
    Parameters
    ----------
    correlation_distance : float
        상관 거리 (m)
    std_dev : float
        표준편차 (dB)
    grid_size : tuple
        그리드 크기
    area_size : tuple
        영역 크기 (m)
        
    Returns
    -------
    np.ndarray
        쉐도잉 맵
    """
    # 그리드 간격 계산
    dx = area_size[0] / grid_size[0]
    dy = area_size[1] / grid_size[1]
    
    # 상관 행렬 생성
    x = np.arange(grid_size[0])
    y = np.arange(grid_size[1])
    X, Y = np.meshgrid(x, y)
    
    # 거리 행렬 계산
    D = np.sqrt((X[:,:,np.newaxis,np.newaxis] - X) ** 2 * dx ** 2 + 
                (Y[:,:,np.newaxis,np.newaxis] - Y) ** 2 * dy ** 2)
    
    # 상관 행렬 계산
    R = np.exp(-D / correlation_distance)
    
    # Cholesky 분해
    L = np.linalg.cholesky(R.reshape(-1, -1))
    
    # 무상관 가우시안 난수 생성
    z = np.random.normal(0, std_dev, grid_size[0] * grid_size[1])
    
    # 상관된 쉐도잉 맵 생성
    shadow = np.dot(L, z).reshape(grid_size)
    
    return shadow