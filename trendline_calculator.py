"""
추세선 계산 모듈
추세선 각도 계산 기능 제공
"""
import math
import numpy as np
import pandas as pd
import traceback


class TrendlineCalculator:
    """추세선 계산 클래스"""
    
    def __init__(self, window=14, aspect_ratio=4):
        """
        초기화
        
        Parameters:
        - window (int): 추세선을 계산할 때 사용할 과거 봉(캔들)의 개수. 기본값 14.
        - aspect_ratio (float): 차트의 시각적 종횡비 (너비 / 높이). 기본값 4.
        """
        self.window = window
        self.aspect_ratio = aspect_ratio
    
    def calculate_trendline(self, df):
        """
        주어진 OHLCV 데이터프레임에 대해, 각 시점에서 과거 'window' 기간 동안의 저가(low)를 기반으로
        지지 추세선(support trendline)을 선형 회귀로 피팅하고,
        이 추세선이 수평선과 이루는 '시각적 각도'(degree)를 계산하여 'angle' 컬럼에 저장합니다.
        
        Parameters:
        - df (pd.DataFrame): 'open', 'high', 'low', 'close', 'volume' 컬럼을 포함한 시계열 데이터.
                             인덱스는 datetime 타입이어야 함.
        
        Returns:
        - pd.DataFrame: 원본 데이터에 다음 두 컬럼이 추가된 복사본:
            - 'trendline': 해당 시점에서 추정된 추세선의 마지막 값 (즉, 현재 봉 직전의 추세선 예측값)
            - 'angle': 수평선(0°)과 추세선 사이의 시각적 각도 (단위: 도, degree). 
                       양수 = 상승 추세, 음수 = 하락 추세.
        """
        try:
            # 원본 데이터프레임을 수정하지 않기 위해 복사본 생성
            df = df.copy()
            
            # 추세선 값과 각도를 저장할 새로운 컬럼 초기화 (NaN으로 채움)
            # 첫 'window'개의 행은 충분한 과거 데이터가 없어 계산 불가 → NaN 유지
            df['trendline'] = np.nan
            df['angle'] = np.nan
            
            # 인덱스 'window'부터 끝까지 반복:
            #   → i번째 행에서, [i-window, i) 구간(총 'window'개 봉)을 사용해 추세선 계산
            for i in range(self.window, len(df)):
                
                # 1. 윈도우 내 데이터 추출: 현재 시점 직전 'window'일치의 OHLCV 데이터
                window_data = df.iloc[i - self.window : i]
                
                # 2. 추세선 기반: 저가(low)를 사용 → 지지선(support line) 추정
                #    (매수 전략에서 지지선 상승을 신호로 삼기 때문)
                low_points = window_data['low'].values  # shape: (window,)
                
                # 3. x축 정의: 시간을 단순 정수 인덱스로 변환 (0, 1, 2, ..., window-1)
                #    → 실제 날짜 대신 상대적 위치 사용 (선형 회귀에선 절대 시간 필요 없음)
                x = np.array(range(self.window))  # shape: (window,)
                
                # 4. 선형 회귀: y = slope * x + intercept 를 low_points에 피팅
                #    np.polyfit(x, y, deg=1) → 1차 다항식(직선)의 계수 [slope, intercept] 반환
                slope, intercept = np.polyfit(x, low_points, 1)
                #    - slope: 가격 단위/봉 (예: +5000 = 매일 5,000원 상승 추세)
                #    - intercept: x=0일 때(윈도우 시작 시점)의 추정 저가
                
                # 5. 추세선의 '마지막 시점'(x = window - 1)에서의 예측값 계산
                #    → 이 값은 '현재 봉'(i) 시점에서의 추세선 위치로 해석됨
                trendline_value_at_end = slope * (self.window - 1) + intercept
                
                # 6. 계산된 추세선 값을 데이터프레임의 'trendline' 컬럼에 저장
                #    → 시각화나 추가 분석에 활용 가능
                df.iloc[i, df.columns.get_loc('trendline')] = trendline_value_at_end
                
                # 7. 시각적 각도 계산을 위한 y축 스케일(가격 범위) 결정
                price_range = window_data['low'].max() - window_data['low'].min()
                
                # 8. 가격 범위가 0인 경우(모든 저가 동일) → 0으로 나누는 오류 방지
                #    → 평균 저가의 1%를 가상의 가격 범위로 설정 (작은 값이지만 0 아님)
                if price_range == 0:
                    price_range = window_data['low'].mean() * 0.01
                
                # 9. '시각적 기울기'(scaled_slope) 계산:
                #    실제 차트에서는 x축(시간)과 y축(가격)의 픽셀 스케일이 다름.
                #    예: 1봉 = 50px (가로), 100만원 = 20px (세로) → 가로가 상대적으로 김.
                #    이를 모방하기 위해:
                #        scaled_slope = (가격 기울기) × (x축 길이 / (y축 길이 × 종횡비))
                #    여기서:
                #        - x축 길이 = window (봉 수)
                #        - y축 길이 = price_range (가격 범위)
                #        - aspect_ratio = 차트 너비/높이 비율 (예: 4 = 800px/200px)
                scaled_slope = slope * (self.window / (price_range * self.aspect_ratio))
                #    → 이 값은 '픽셀 공간에서의 기울기'를 근사함.
                
                # 10. 시각적 각도 계산:
                #     - math.atan(scaled_slope): 아크탄젠트 → 라디안 단위의 각도
                #     - math.degrees(...): 라디안 → 도(degree) 변환
                #     결과:
                #        angle > 0 → 상승 추세 (값 클수록 가파름)
                #        angle < 0 → 하락 추세
                #        angle ≈ 0 → 수평에 가까운 추세
                angle = math.degrees(math.atan(scaled_slope))
                
                # 11. 계산된 각도를 'angle' 컬럼에 저장
                df.iloc[i, df.columns.get_loc('angle')] = angle
            
            # 모든 시점에 대해 처리 완료 → 결과 데이터프레임 반환
            return df
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def set_window(self, window):
        """
        윈도우 크기 설정
        
        Parameters:
        - window (int): 추세선을 계산할 때 사용할 과거 봉(캔들)의 개수
        """
        self.window = window
    
    def set_aspect_ratio(self, aspect_ratio):
        """
        종횡비 설정
        
        Parameters:
        - aspect_ratio (float): 차트의 시각적 종횡비 (너비 / 높이)
        """
        self.aspect_ratio = aspect_ratio

