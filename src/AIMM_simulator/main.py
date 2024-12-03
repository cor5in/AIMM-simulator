#!/usr/bin/env python3
"""
AIMM 시뮬레이터 메인 실행 스크립트
"""

from simulator import Simulator

def main():
    # 시뮬레이터 초기화
    simulator = Simulator('config.yaml')

    # 환경 설정
    simulator.setup()

    # 시뮬레이션 실행
    simulator.run()

    # 결과 저장 및 시각화
    simulator.save_results('results')
    simulator.plot_results('plots')

if __name__ == '__main__':
    main()