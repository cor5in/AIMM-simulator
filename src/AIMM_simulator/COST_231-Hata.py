from math import log10
from numpy.linalg import norm
import numpy as np

class COST231_Hata_pathloss:
    '''
    COST 231-Hata pathloss model for urban environments, suitable for frequencies of 800 MHz and 1800 MHz.

    적용 범위:
    - 거리: 1km ~ 20km (하지만 시뮬레이션에서는 더 작은 거리도 사용 가능)
    - UE 높이: 1m ~ 10m
    - BS 높이: 20m (사용자 지정 가능)
    - 주파수: 800MHz 및 1800MHz

    Parameters
    ----------
    fc_MHz : float
        Carrier frequency in MHz (800MHz 또는 1800MHz).
    h_UE : float
        User equipment antenna height in meters.
    h_BS : float
        Base station antenna height in meters.
    city_size : str
        'small_medium' 또는 'large' 중 선택 (대도시 여부).
    environment : str
        'urban', 'suburban', 'open' 중 선택.
    '''

    def __init__(s, fc_MHz=1800, h_UE=1.5, h_BS=20.0, city_size='large', environment='urban'):
        # 기본 파라미터 초기화
        s.fc = fc_MHz  # 주파수 (MHz)
        s.h_UE = h_UE  # UE(단말) 높이 (m)
        s.h_BS = h_BS  # 기지국 높이 (m)
        s.city_size = city_size  # 도시 크기 ('small_medium' 또는 'large')
        s.environment = environment  # 환경 유형 ('urban', 'suburban', 'open')

        # 도시 크기에 따른 Cm 값 설정
        if s.environment == 'urban':
            if s.city_size == 'small_medium':
                s.Cm = 0
            elif s.city_size == 'large':
                s.Cm = 3
            else:
                raise ValueError("city_size must be 'small_medium' or 'large'")
        elif s.environment == 'suburban':
            s.Cm = -2
        elif s.environment == 'open':
            s.Cm = -5
        else:
            raise ValueError("environment must be 'urban', 'suburban', or 'open'")

    def a_h_UE(s):
        '''
        단말기 안테나 보정 인자 a(h_UE) 계산 (fc >= 300 MHz 가정)
        '''
        h_UE = s.h_UE
        return 3.2 * (log10(11.75 * h_UE))**2 - 4.97

    def __call__(s, xyz_cell, xyz_UE):
        '''
        3차원 공간상의 두 지점 간 경로손실 계산

        Parameters
        ----------
        xyz_cell : array_like
            기지국의 3차원 좌표 [x, y, z]
        xyz_UE : array_like
            단말의 3차원 좌표 [x, y, z]

        Returns
        -------
        float
            경로손실 값 (dB)
        '''
        # 거리 계산 (km 단위)
        d_m = norm(np.array(xyz_cell) - np.array(xyz_UE))
        d_km = max(d_m / 1000.0, 0.01)  # 최소 거리 10m로 설정하여 로그 무한대 방지

        # 경로손실 계산
        fc = s.fc
        h_BS = s.h_BS
        h_UE = s.h_UE
        a_h_UE = s.a_h_UE()
        Cm = s.Cm

        pathloss = (46.3 + 33.9 * log10(fc) - 13.82 * log10(h_BS)
                    - a_h_UE + (44.9 - 6.55 * log10(h_BS)) * log10(d_km) + Cm)
        return pathloss

    def plot():
        '''
        경로손실 모델의 특성을 시각화하는 함수
        주파수별로 경로손실을 거리에 따라 플롯
        '''
        import matplotlib.pyplot as plt

        # 거리 범위 설정 (10m-5km)
        d_m = np.linspace(10, 5000, 500)
        distances = d_m / 1000.0  # km 단위로 변환

        # 주파수 설정 (800MHz와 1800MHz)
        frequencies = [800, 1800]

        plt.figure(figsize=(8, 6))

        for fc in frequencies:
            pl_model = COST231_Hata_pathloss(fc_MHz=fc, h_UE=1.5, h_BS=20.0, city_size='large', environment='urban')
            path_losses = [pl_model([0, 0, pl_model.h_BS], [di * 1000, 0, pl_model.h_UE]) for di in distances]
            plt.plot(distances, path_losses, label=f'{fc} MHz')

        plt.xlabel('Distance (km)')
        plt.ylabel('Pathloss (dB)')
        plt.title('COST 231-Hata Pathloss Model (Large City, h_BS=20m, h_UE=1.5m)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if __name__ == '__main__':
        plot()  # 자체 테스트 실행
