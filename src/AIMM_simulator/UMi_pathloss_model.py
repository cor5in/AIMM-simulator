# Geon Kim: 2024-11-11: 한국에 맞춘 UMi 경로손실 모델 추가

from math import log10,hypot
from numpy.linalg import norm

class UMi_streetcanyon_pathloss:
    '''
    Urban microcell (UMi) streetcanyon pathloss model, from 3GPP standard 36.873,
    Table 7.2-1.
    
    This code covers both LOS and NLOS cases for 3D-UMi streetcanyon model.
    적용 범위:
    - 거리: 10m ~ 500m
    - UE 높이: 1.5m ~ 2.0m
    - BS 높이: 8m ~ 10m
    - 주파수: 2GHz ~ 6GHz
    '''
    def __init__(s,fc_GHz=3.5,h_UT=1.5,h_BS=8.0,LOS=True):
        # 기본 파라미터 초기화
        s.fc=fc_GHz    # 캐리어 주파수 (GHz 단위), 한국 5G NR n78 대역용 3.5GHz
        s.log10fc=log10(s.fc)   # 자주 사용되는 log10(fc) 값 미리 계산
        s.h_UT=h_UT    # UE(단말) 높이 (m), 보행자 높이 1.5m
        s.h_BS=h_BS    # 기지국 높이 (m), 스몰셀용 8m (가로등/전신주 높이)
        s.LOS=LOS      # Line of Sight 여부
        s.c=3e8        # 빛의 속도 (m/s)
        s.h_E=1.0      # 유효 환경 높이 보정값 (m)

        # Break point distance 계산
        # Break point: 전파 특성이 변화하는 지점까지의 거리
        # 공식: 4*(h_BS-h_E)*(h_UT-h_E)*fc*10^9/c
        s.dBP=4.0*(s.h_BS-s.h_E)*(s.h_UT-s.h_E)*s.fc*1e9/s.c

        # UMi streetcanyon LOS 모델 상수 계산
        # d ≤ dBP 구간에서 사용되는 상수
        s.const_close=32.4+20.0*s.log10fc  # UMi 모델 기본 상수 32.4 
        s.pl_coefficient=21.0               # UMi 경로손실 계수 21.0

        # d > dBP 구간에서 사용되는 상수
        s.a=9.5*log10(s.dBP**2)            # UMi break point 보정 계수
        s.const_far=32.4+20.0*s.log10fc-s.a  # 원거리 경로손실 계산용 상수

    def __call__(s,xyz_cell,xyz_UE):
        '''
        3차원 공간상의 두 지점 간 경로손실 계산
        
        Parameters
        ----------
        xyz_cell : array
            기지국의 3차원 좌표 [x, y, z]
        xyz_UE : array or float
            단말의 3차원 좌표 [x, y, z]
            
        Returns
        -------
        float
            경로손실 값 (dB)
        '''
        # 3차원 거리 계산
        d3D_m=norm(xyz_cell-xyz_UE)
        
        # LOS 경우의 경로손실 계산
        if d3D_m<s.dBP:
            PL3D_UMi_LOS = s.const_close + s.pl_coefficient*log10(d3D_m)
        else:
            PL3D_UMi_LOS = s.const_far + 40.0*log10(d3D_m)
            
        if s.LOS:
            return PL3D_UMi_LOS
            
        # NLOS 경우의 경로손실 계산
        PL3D_UMi_NLOS = 35.3*log10(d3D_m) + 22.4 + 21.3*log10(s.fc) - 0.3*(s.h_UT-1.5)
        
        # NLOS 경로손실은 LOS 경로손실보다 작을 수 없음
        return max(PL3D_UMi_NLOS, PL3D_UMi_LOS)

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
    
    # 거리 범위 설정 (10m-500m)
    d=np.linspace(10,500,100)
    
    # NLOS 경우 계산
    PL=UMi_streetcanyon_pathloss(fc_GHz=3.5,h_UT=1.5,h_BS=8.0,LOS=False)
    NLOS=np.array([PL(0,di) for di in d])
    ax.plot(d,NLOS,lw=2,label='NLOS ($\sigma=4$)')
    ax.fill_between(d,NLOS-4.0,NLOS+4.0,alpha=0.2)
    
    # LOS 경우 계산
    PL=UMi_streetcanyon_pathloss(fc_GHz=3.5,h_UT=1.5,h_BS=8.0,LOS=True)
    LOS=np.array([PL(0,di) for di in d])
    ax.plot(d,LOS,lw=2,label='LOS ($\sigma=3$)')
    ax.fill_between(d,LOS-3.0,LOS+3.0,alpha=0.2)
    
    # 그래프 레이블 설정
    ax.set_xlabel('distance (metres)')
    ax.set_ylabel('pathloss (dB)')
    ax.set_xlim(0,np.max(d))
    ax.set_ylim(40)
    ax.legend()
    ax.set_title('3GPP UMi Street Canyon pathloss models')
    
    # 그래프 저장
    fig.tight_layout()
    fig_timestamp(fig,rotation=0,fontsize=6,author='Keith Briggs')
    fnbase='img/UMi_streetcanyon_pathloss_model_01'
    fig.savefig(f'{fnbase}.png')
    print(f'eog {fnbase}.png &')
    fig.savefig(f'{fnbase}.pdf')
    print(f'evince {fnbase}.pdf &')

if __name__=='__main__':
    plot() # 자체 테스트 실행
