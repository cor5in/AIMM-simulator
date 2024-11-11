# Geon Kim 2024-11-11: 한국에 맞춘 UMa 경로손실 모델을 추가

from math import log10,hypot
from numpy.linalg import norm

class UMa_pathloss:
    '''
    Urban macrocell (UMa) dual-slope pathloss model, from 3GPP standard 36.873,
    Table 7.2-1.
    
    적용 범위:
    - 거리: 10m ~ 5000m
    - UE 높이: 1.5m ~ 2.0m
    - BS 높이: 25m ~ 30m
    - 주파수: 2GHz ~ 6GHz
    '''
    def __init__(s,fc_GHz=2.1,h_UT=1.5,h_BS=25.0,LOS=True,h=25.0,W=20.0):
        '''
        매크로셀 경로손실 모델 초기화
        
        Parameters
        ----------
        fc_GHz : float
            중심 주파수 (GHz). 기본값 2.1GHz (한국 LTE 대역)
        h_UT : float
            단말 높이 (m). 기본값 1.5m (보행자 높이)
        h_BS : float
            기지국 높이 (m). 기본값 25.0m (도심 매크로셀 높이)
        LOS : bool
            Line of Sight 여부
        h : float
            평균 건물 높이 (m). NLOS에서만 사용
        W : float
            도로 폭 (m). NLOS에서만 사용
        '''
        # 기본 파라미터 초기화
        s.fc=fc_GHz    # 캐리어 주파수 (GHz)
        s.log10fc=log10(s.fc)  # log10(fc) 미리 계산
        s.h_UT=h_UT    # 단말 높이
        s.h_BS=h_BS    # 기지국 높이
        s.LOS=LOS      # LOS 여부
        s.h=h          # 평균 건물 높이 (NLOS용)
        s.W=W          # 도로 폭 (NLOS용)
        s.c=3e8        # 빛의 속도 (m/s)
        s.h_E=1.0      # 유효 환경 높이 보정값 (m)

        # Break point distance 계산
        # 전파 특성이 변화하는 지점까지의 거리
        s.dBP=4.0*(s.h_BS-s.h_E)*(s.h_UT-s.h_E)*s.fc*1e9/s.c

        # Break point 거리에서의 보정 계수 계산
        s.a=9.0*log10(s.dBP**2+(s.h_BS-s.h_UT)**2)

        # LOS 모델 상수 계산
        s.const_close=28.0+20.0*s.log10fc  # d ≤ dBP 구간
        s.const_far=28.0+20.0*s.log10fc-s.a  # d > dBP 구간

        # NLOS 모델 상수 계산
        s.c1=-9.1904695449517596702522e-4  # =3.2*(log10(17.625))**2-4.97

    def __call__(s,xyz_cell,xyz_UE):
        '''
        3차원 공간상의 두 지점 간 경로손실 계산
        
        Parameters
        ----------
        xyz_cell : array
            기지국의 3차원 좌표 [x, y, z]
        xyz_UE : array
            단말의 3차원 좌표 [x, y, z]
            
        Returns
        -------
        float
            경로손실 값 (dB)
        '''
        # 3차원 거리 계산
        d3D_m=norm(xyz_cell-xyz_UE)
        
        # LOS 경로손실 계산
        if d3D_m<s.dBP:
            # 근거리 경로손실 (dual slope 모델의 첫 번째 기울기)
            PL3D_UMa_LOS=s.const_close+22.0*log10(d3D_m)
        else:
            # 원거리 경로손실 (dual slope 모델의 두 번째 기울기)
            PL3D_UMa_LOS=s.const_far+40.0*log10(d3D_m)
            
        if s.LOS:
            return PL3D_UMa_LOS
            
        # NLOS 경로손실 계산
        # 3GPP TR 36.873 Table 7.2-1의 공식 사용
        PL3D_UMa_NLOS = (
            161.04                                             # 기본 상수
            - 7.1*log10(s.W)                                  # 도로 폭 영향
            + 7.5*log10(s.h)                                  # 건물 높이 영향
            - (24.37-3.7*(s.h/s.h_BS)**2)*log10(s.h_BS)      # BS 높이 영향
            + (43.42-3.1*log10(s.h_BS))*(log10(d3D_m)-3.0)   # 거리 의존성
            + 20*log10(s.fc)                                  # 주파수 의존성
            - (s.c1)                                          # 보정 상수
            - 0.6*(s.h_UT-1.5)                               # 단말 높이 보정
        )
        
        # NLOS 경로손실은 LOS 경로손실보다 작을 수 없음
        return max(PL3D_UMa_NLOS,PL3D_UMa_LOS)

def plot():
    '''
    경로손실 모델의 특성을 시각화하는 함수
    LOS와 NLOS 케이스의 경로손실을 거리에 따라 플롯
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from fig_timestamp import fig_timestamp
    
    # 그래프 기본 설정
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot()
    ax.grid(color='gray',alpha=0.7,lw=0.5)
    
    # 거리 범위 설정 (10m-5000m)
    d=np.linspace(10,5000,100)
    
    # NLOS 경우 계산
    PL=UMa_pathloss(fc_GHz=2.1,h_UT=1.5,h_BS=25.0,LOS=False)
    NLOS=np.array([PL(0,di) for di in d])
    ax.plot(d,NLOS,lw=2,label='NLOS ($\sigma=6$)')
    ax.fill_between(d,NLOS-6.0,NLOS+6.0,alpha=0.2)
    
    # LOS 경우 계산
    PL=UMa_pathloss(fc_GHz=2.1,h_UT=1.5,h_BS=25.0,LOS=True)
    LOS=np.array([PL(0,di) for di in d])
    ax.plot(d,LOS,lw=2,label='LOS ($\sigma=4$)')
    ax.fill_between(d,LOS-4.0,LOS+4.0,alpha=0.2)
    
    # 그래프 레이블 설정
    ax.set_xlabel('distance (metres)')
    ax.set_ylabel('pathloss (dB)')
    ax.set_xlim(0,np.max(d))
    ax.set_ylim(40)
    ax.legend()
    ax.set_title('3GPP UMa pathloss models')
    
    # 그래프 저장
    fig.tight_layout()
    fig_timestamp(fig,rotation=0,fontsize=6,author='Keith Briggs')
    fnbase='img/UMa_pathloss_model_01'
    fig.savefig(f'{fnbase}.png')
    print(f'eog {fnbase}.png &')
    fig.savefig(f'{fnbase}.pdf')
    print(f'evince {fnbase}.pdf &')

if __name__=='__main__':
    plot() # 자체 테스트 실행