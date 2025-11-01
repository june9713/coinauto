"""
시각화 모듈
백테스트 결과 시각화 기능 제공
"""
import traceback
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 백엔드를 Agg로 설정하여 속도 향상
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from config import Config


class Visualizer:
    """백테스트 결과 시각화 클래스"""
    
    def __init__(self, font_path=None, dpi=None):
        """
        초기화
        
        Parameters:
        - font_path (str, optional): 한글 폰트 경로. None이면 Config 기본값 사용
        - dpi (int, optional): 그래프 해상도. None이면 Config 기본값 사용
        """
        self.font_path = font_path or Config.FONT_PATH
        self.dpi = dpi or Config.GRAPH_DPI
        self._setup_font()
    
    def _setup_font(self):
        """한글 폰트 설정"""
        try:
            if os.path.exists(self.font_path):
                # 폰트를 matplotlib에 등록
                font_manager.fontManager.addfont(self.font_path)
                # 폰트 속성 가져오기
                font_prop = font_manager.FontProperties(fname=self.font_path)
                font_name = font_prop.get_name()
                plt.rcParams['font.family'] = font_name
            else:
                # 폰트 파일이 없으면 기본 폰트 사용 시도
                print(f"경고: 폰트 파일을 찾을 수 없습니다: {self.font_path}")
                plt.rcParams['font.family'] = 'DejaVu Sans'
            # 유니코드 마이너스 기호 대신 ASCII 마이너스(-) 사용하여 경고 방지
            plt.rcParams['axes.unicode_minus'] = True
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def plot_backtest_results(self, df, result, save_path=None, 
                             buy_threshold=None, sell_threshold=None,
                             volume_window=None, ma_window=None, volume_multiplier=None):
        """
        백테스트 결과를 matplotlib으로 시각화
        
        Parameters:
        - df (pd.DataFrame): 추세선 계산이 완료된 OHLCV 데이터프레임
        - result (dict): 백테스트 결과 딕셔너리
        - save_path (str, optional): 그래프 저장 경로. None이면 Config 기본값 사용
        - buy_threshold (float, optional): 매수 기준선 각도. None이면 Config 기본값 사용
        - sell_threshold (float, optional): 매도 기준선 각도. None이면 Config 기본값 사용
        - volume_window (int, optional): 거래량 평균 계산용 윈도우. None이면 QQC 그래프 미생성
        - ma_window (int, optional): 이동평균 계산용 윈도우. None이면 QQC 그래프 미생성
        - volume_multiplier (float, optional): 거래량 배수. None이면 QQC 그래프 미생성
        """
        try:
            if save_path is None:
                save_path = Config.DEFAULT_GRAPH_SAVE_PATH
            
            if buy_threshold is None:
                buy_threshold = Config.DEFAULT_BUY_ANGLE_THRESHOLD
            
            if sell_threshold is None:
                sell_threshold = Config.DEFAULT_SELL_ANGLE_THRESHOLD
            
            trades = result['trades']
            
            # 각도 데이터가 있는지 확인
            has_angle_data = 'angle' in df.columns
            
            # QQC 전략 파라미터 확인
            has_qqc_params = volume_window is not None and ma_window is not None and volume_multiplier is not None
            
            # 서브플롯 개수 결정
            subplot_count = 3  # 기본: 가격, 보유 구간, 누적 수익률
            if has_angle_data:
                subplot_count += 1  # 각도 차트 추가
            if has_qqc_params:
                subplot_count += 1  # 거래량 차트만 추가
            
            # 서브플롯 인덱스 설정
            current_idx = 0
            price_ax_idx = current_idx
            current_idx += 1
            
            angle_ax_idx = None
            if has_angle_data:
                angle_ax_idx = current_idx
                current_idx += 1
            
            volume_ax_idx = None
            if has_qqc_params:
                volume_ax_idx = current_idx
                current_idx += 1
            
            holding_ax_idx = current_idx
            current_idx += 1
            profit_ax_idx = current_idx
            
            # 서브플롯 생성
            fig, axes = plt.subplots(subplot_count, 1, figsize=(16, 4 * subplot_count), sharex=True)
            if subplot_count == 1:
                axes = [axes]
            
            fig.suptitle('BTC 백테스트 결과', fontsize=16, fontweight='bold')
            
            dates = df.index
            close_prices = df['close'].values
            
            # 1. BTC 가격 차트 (종가, 이동평균, 음봉/양봉)
            if has_qqc_params:
                self._plot_price_chart_with_qqc(axes[price_ax_idx], df, trades, ma_window)
            else:
                self._plot_price_chart(axes[price_ax_idx], dates, close_prices, trades)
            
            # 2. 추세선 각도 차트 (각도 데이터가 있는 경우만)
            if has_angle_data and angle_ax_idx is not None:
                self._plot_angle_chart(axes[angle_ax_idx], dates, df['angle'].values, 
                                       buy_threshold, sell_threshold)
            
            # 3. 거래량 차트 (QQC 파라미터가 있는 경우)
            if has_qqc_params and volume_ax_idx is not None:
                self._plot_volume_chart(axes[volume_ax_idx], df, volume_window, volume_multiplier)
            
            # 4. 보유 구간 표시
            self._plot_holding_periods(axes[holding_ax_idx], dates, close_prices, trades)
            
            # 9. 누적 수익률 차트
            self._plot_cumulative_profit(axes[profit_ax_idx], trades)
            
            plt.tight_layout(pad=1.0)  # pad 값 명시하여 계산 속도 향상
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            print(f"\n그래프 저장 완료: {save_path}")
            plt.close('all')  # 모든 figure 닫기하여 메모리 정리
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _plot_price_chart(self, ax, dates, close_prices, trades):
        """
        가격 차트 플롯
        
        Parameters:
        - ax: matplotlib axes 객체
        - dates: 날짜 인덱스
        - close_prices: 종가 배열
        - trades: 거래 리스트
        """
        try:
            ax.plot(dates, close_prices, label='BTC 종가', color='black', linewidth=1.5)
            
            # 매수/매도 신호 표시
            buy_dates = []
            buy_prices = []
            sell_dates = []
            sell_prices = []
            
            for trade in trades:
                trade_date = trade['date']
                trade_price = trade['price']
                
                if trade['action'].startswith('BUY'):
                    buy_dates.append(trade_date)
                    buy_prices.append(trade_price)
                elif trade['action'].startswith('SELL'):
                    sell_dates.append(trade_date)
                    sell_prices.append(trade_price)
            
            if buy_dates:
                ax.scatter(buy_dates, buy_prices, color='red', marker='^', s=100, 
                          label='매수', zorder=5)
            if sell_dates:
                ax.scatter(sell_dates, sell_prices, color='blue', marker='v', s=100, 
                          label='매도', zorder=5)
            
            ax.set_ylabel('가격 (원)', fontsize=10)
            ax.set_title('BTC 가격 및 매매 신호', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _plot_price_chart_with_qqc(self, ax, df, trades, ma_window):
        """
        QQC 전략용 가격 차트 플롯 (종가, 이동평균, 음봉/양봉)
        
        Parameters:
        - ax: matplotlib axes 객체
        - df: OHLCV 데이터프레임
        - trades: 거래 리스트
        - ma_window: 이동평균 윈도우
        """
        try:
            dates = df.index
            close_prices = df['close'].values
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            # 이동평균 계산 (현재 캔들을 제외한 과거 ma_window개 사용)
            # pandas rolling을 사용하여 벡터화된 계산으로 속도 향상
            ma_values = df['close'].shift(1).rolling(window=ma_window, min_periods=1).mean().values
            # 첫 번째 값은 현재 값으로 설정 (과거 데이터가 없는 경우)
            if len(ma_values) > 0:
                ma_values[0] = close_prices[0]
            
            # 음봉/양봉 판단
            is_bullish = open_prices < close_prices  # 양봉
            
            # 양봉은 빨간색, 음봉은 파란색으로 표시 (벡터화하여 속도 향상)
            colors = np.where(is_bullish, 'red', 'blue')
            # 높이와 낮이 라인 (벡터화)
            for color in ['red', 'blue']:
                mask = (colors == color)
                if np.any(mask):
                    masked_dates = dates[mask]
                    masked_lows = low_prices[mask]
                    masked_highs = high_prices[mask]
                    ax.vlines(masked_dates, masked_lows, masked_highs, 
                             colors=color, linewidth=1, alpha=0.5)
            # 오픈-종가 라인 (벡터화)
            for color in ['red', 'blue']:
                mask = (colors == color)
                if np.any(mask):
                    masked_dates = dates[mask]
                    masked_opens = open_prices[mask]
                    masked_closes = close_prices[mask]
                    ax.vlines(masked_dates, masked_opens, masked_closes, 
                             colors=color, linewidth=3, alpha=0.7)
            
            # 종가 라인
            ax.plot(dates, close_prices, label='BTC 종가', color='black', linewidth=1.5, alpha=0.7)
            
            # 이동평균 라인
            ax.plot(dates, ma_values, label=f'이동평균 ({ma_window}개)', 
                   color='green', linewidth=2, linestyle='--', alpha=0.8)
            
            # 매수/매도 신호 표시
            buy_dates = []
            buy_prices = []
            sell_dates = []
            sell_prices = []
            
            for trade in trades:
                trade_date = trade['date']
                trade_price = trade['price']
                
                if trade['action'].startswith('BUY'):
                    buy_dates.append(trade_date)
                    buy_prices.append(trade_price)
                elif trade['action'].startswith('SELL'):
                    sell_dates.append(trade_date)
                    sell_prices.append(trade_price)
            
            if buy_dates:
                ax.scatter(buy_dates, buy_prices, color='red', marker='^', s=100, 
                          label='매수', zorder=5)
            if sell_dates:
                ax.scatter(sell_dates, sell_prices, color='blue', marker='v', s=100, 
                          label='매도', zorder=5)
            
            ax.set_ylabel('가격 (원)', fontsize=10)
            ax.set_title('BTC 가격 (종가, 이동평균, 음봉/양봉)', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _plot_volume_chart(self, ax, df, volume_window, volume_multiplier):
        """
        거래량 차트 플롯 (조건 B: 과거 n개 캔들의 거래량 평균값과 현재 거래량 비교)
        
        Parameters:
        - ax: matplotlib axes 객체
        - df: OHLCV 데이터프레임
        - volume_window: 거래량 평균 계산용 윈도우
        - volume_multiplier: 거래량 배수
        """
        try:
            dates = df.index
            volumes = df['volume'].values
            
            # 거래량 평균 계산 (현재 캔들을 제외한 과거 volume_window개 사용)
            # QQC 엔진과 동일한 로직: pandas rolling을 사용하여 벡터화된 계산으로 속도 향상
            volume_avg = df['volume'].shift(1).rolling(window=volume_window, min_periods=1).mean().values
            # 첫 번째 값은 0으로 설정 (과거 데이터가 없는 경우)
            if len(volume_avg) > 0:
                volume_avg[0] = 0.0
            
            # 거래량 임계값 (평균 * 배수)
            volume_threshold = volume_avg * volume_multiplier
            
            # 현재 거래량
            ax.plot(dates, volumes, label='현재 거래량', color='blue', linewidth=1.5)
            
            # 거래량 평균
            ax.plot(dates, volume_avg, label=f'거래량 평균 ({volume_window}개)', 
                   color='orange', linewidth=2, linestyle='--', alpha=0.8)
            
            # 거래량 임계값
            ax.plot(dates, volume_threshold, label=f'거래량 임계값 (평균 * {volume_multiplier})', 
                   color='red', linewidth=2, linestyle=':', alpha=0.8)
            
            # 조건 B 만족 구간 표시 (현재 거래량 >= 임계값)
            condition_b_satisfied = volumes >= volume_threshold
            if np.any(condition_b_satisfied):
                ax.fill_between(dates, 0, volumes, where=condition_b_satisfied, 
                              alpha=0.3, color='green', label='조건 B 만족')
            
            ax.set_ylabel('거래량 (BTC)', fontsize=10)
            ax.set_title('거래량 비교 (조건 B)', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')  # 거래량은 로그 스케일이 적합할 수 있음
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _plot_condition_b_chart(self, ax, df, volume_window, volume_multiplier):
        """
        조건 B 만족 구간 차트 (현재 거래량 >= 거래량 평균 * 배수)
        
        Parameters:
        - ax: matplotlib axes 객체
        - df: OHLCV 데이터프레임
        - volume_window: 거래량 평균 계산용 윈도우
        - volume_multiplier: 거래량 배수
        """
        try:
            dates = df.index
            close_prices = df['close'].values
            volumes = df['volume'].values
            
            # 거래량 평균 계산 (벡터화하여 속도 향상)
            volume_avg = df['volume'].shift(1).rolling(window=volume_window, min_periods=1).mean().values
            if len(volume_avg) > 0:
                volume_avg[0] = 0.0
            
            # 조건 B 확인
            condition_b = volumes >= (volume_avg * volume_multiplier)
            
            # 종가 그래프
            ax.plot(dates, close_prices, label='BTC 종가', color='gray', linewidth=1.5, alpha=0.5)
            
            # 조건 B 만족 구간 색칠
            if np.any(condition_b):
                in_range = False
                start_idx = None
                label_added = False
                
                for i, satisfied in enumerate(condition_b):
                    if satisfied and not in_range:
                        in_range = True
                        start_idx = i
                    elif not satisfied and in_range:
                        if start_idx is not None:
                            label = '조건 B 만족' if not label_added else None
                            ax.axvspan(dates[start_idx], dates[i-1], alpha=0.3, 
                                     color='green', label=label)
                            label_added = True
                        in_range = False
                        start_idx = None
                
                if in_range and start_idx is not None:
                    label = '조건 B 만족' if not label_added else None
                    ax.axvspan(dates[start_idx], dates[-1], alpha=0.3, 
                             color='green', label=label)
            
            # 조건 B 만족 시점 표시
            condition_b_dates = []
            condition_b_prices = []
            for i, satisfied in enumerate(condition_b):
                if satisfied:
                    condition_b_dates.append(dates[i])
                    condition_b_prices.append(close_prices[i])
            
            if condition_b_dates:
                ax.scatter(condition_b_dates, condition_b_prices, color='green', marker='o', 
                          s=50, alpha=0.6, label='조건 B 만족', zorder=4)
            
            ax.set_ylabel('가격 (원)', fontsize=10)
            ax.set_title('조건 B 만족 구간 (거래량 >= 평균 * 배수)', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _plot_condition_d_chart(self, ax, df, ma_window):
        """
        조건 D 만족 구간 차트 (현재 종가 > 이동평균)
        
        Parameters:
        - ax: matplotlib axes 객체
        - df: OHLCV 데이터프레임
        - ma_window: 이동평균 윈도우
        """
        try:
            dates = df.index
            close_prices = df['close'].values
            
            # 이동평균 계산 (벡터화하여 속도 향상)
            ma_values = df['close'].shift(1).rolling(window=ma_window, min_periods=1).mean().values
            if len(ma_values) > 0:
                ma_values[0] = close_prices[0]
            
            # 조건 D 확인
            condition_d = close_prices > ma_values
            
            # 종가 및 이동평균 그래프
            ax.plot(dates, close_prices, label='BTC 종가', color='black', linewidth=1.5)
            ax.plot(dates, ma_values, label=f'이동평균 ({ma_window}개)', 
                   color='orange', linewidth=2, linestyle='--', alpha=0.8)
            
            # 조건 D 만족 구간 색칠
            if np.any(condition_d):
                in_range = False
                start_idx = None
                label_added = False
                
                for i, satisfied in enumerate(condition_d):
                    if satisfied and not in_range:
                        in_range = True
                        start_idx = i
                    elif not satisfied and in_range:
                        if start_idx is not None:
                            label = '조건 D 만족' if not label_added else None
                            ax.axvspan(dates[start_idx], dates[i-1], alpha=0.3, 
                                     color='blue', label=label)
                            label_added = True
                        in_range = False
                        start_idx = None
                
                if in_range and start_idx is not None:
                    label = '조건 D 만족' if not label_added else None
                    ax.axvspan(dates[start_idx], dates[-1], alpha=0.3, 
                             color='blue', label=label)
            
            # 조건 D 만족 시점 표시
            condition_d_dates = []
            condition_d_prices = []
            for i, satisfied in enumerate(condition_d):
                if satisfied:
                    condition_d_dates.append(dates[i])
                    condition_d_prices.append(close_prices[i])
            
            if condition_d_dates:
                ax.scatter(condition_d_dates, condition_d_prices, color='blue', marker='o', 
                          s=50, alpha=0.6, label='조건 D 만족', zorder=4)
            
            ax.set_ylabel('가격 (원)', fontsize=10)
            ax.set_title('조건 D 만족 구간 (종가 > 이동평균)', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _plot_condition_e_chart(self, ax, df):
        """
        조건 E 만족 구간 차트 (양봉: 오픈가 < 종가)
        
        Parameters:
        - ax: matplotlib axes 객체
        - df: OHLCV 데이터프레임
        """
        try:
            dates = df.index
            close_prices = df['close'].values
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            # 조건 E 확인 (양봉)
            condition_e = open_prices < close_prices
            
            # 종가 그래프
            ax.plot(dates, close_prices, label='BTC 종가', color='gray', linewidth=1.5, alpha=0.5)
            
            # 음봉/양봉 표시 (벡터화하여 속도 향상)
            colors = np.where(condition_e, 'red', 'blue')
            # 높이와 낮이 라인 (벡터화)
            for color in ['red', 'blue']:
                mask = (colors == color)
                if np.any(mask):
                    masked_dates = dates[mask]
                    masked_lows = low_prices[mask]
                    masked_highs = high_prices[mask]
                    ax.vlines(masked_dates, masked_lows, masked_highs, 
                             colors=color, linewidth=1, alpha=0.5)
            # 오픈-종가 라인 (벡터화)
            for color in ['red', 'blue']:
                mask = (colors == color)
                if np.any(mask):
                    masked_dates = dates[mask]
                    masked_opens = open_prices[mask]
                    masked_closes = close_prices[mask]
                    ax.vlines(masked_dates, masked_opens, masked_closes, 
                             colors=color, linewidth=3, alpha=0.7)
            
            # 조건 E 만족 구간 색칠 (양봉 구간)
            if np.any(condition_e):
                in_range = False
                start_idx = None
                label_added = False
                
                for i, satisfied in enumerate(condition_e):
                    if satisfied and not in_range:
                        in_range = True
                        start_idx = i
                    elif not satisfied and in_range:
                        if start_idx is not None:
                            label = '조건 E 만족 (양봉)' if not label_added else None
                            ax.axvspan(dates[start_idx], dates[i-1], alpha=0.2, 
                                     color='red', label=label)
                            label_added = True
                        in_range = False
                        start_idx = None
                
                if in_range and start_idx is not None:
                    label = '조건 E 만족 (양봉)' if not label_added else None
                    ax.axvspan(dates[start_idx], dates[-1], alpha=0.2, 
                             color='red', label=label)
            
            # 조건 E 만족 시점 표시
            condition_e_dates = []
            condition_e_prices = []
            for i, satisfied in enumerate(condition_e):
                if satisfied:
                    condition_e_dates.append(dates[i])
                    condition_e_prices.append(close_prices[i])
            
            if condition_e_dates:
                ax.scatter(condition_e_dates, condition_e_prices, color='red', marker='o', 
                          s=50, alpha=0.6, label='조건 E 만족', zorder=4)
            
            ax.set_ylabel('가격 (원)', fontsize=10)
            ax.set_title('조건 E 만족 구간 (양봉: 오픈가 < 종가)', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _plot_bde_conditions_chart(self, ax, df, volume_window, ma_window, volume_multiplier):
        """
        B, D, E 조건 모두 일치하는 구간을 색깔로 표시하는 그래프
        
        Parameters:
        - ax: matplotlib axes 객체
        - df: OHLCV 데이터프레임
        - volume_window: 거래량 평균 계산용 윈도우
        - ma_window: 이동평균 계산용 윈도우
        - volume_multiplier: 거래량 배수
        """
        try:
            dates = df.index
            close_prices = df['close'].values
            open_prices = df['open'].values
            volumes = df['volume'].values
            
            # 조건 계산 (벡터화하여 속도 향상)
            # 조건 B: 현재 거래량 >= 거래량 평균 * volume_multiplier
            # 현재 캔들을 제외한 과거 volume_window개 사용
            volume_avg = df['volume'].shift(1).rolling(window=volume_window, min_periods=1).mean().values
            if len(volume_avg) > 0:
                volume_avg[0] = 0.0
            condition_b = volumes >= (volume_avg * volume_multiplier)
            
            # 조건 D: 현재 종가 > 이동평균
            # 현재 캔들을 제외한 과거 ma_window개 사용
            ma_values = df['close'].shift(1).rolling(window=ma_window, min_periods=1).mean().values
            if len(ma_values) > 0:
                ma_values[0] = close_prices[0]
            condition_d = close_prices > ma_values
            
            # 조건 E: 양봉 (오픈가 < 종가)
            condition_e = open_prices < close_prices
            
            # B, D, E 모두 만족하는 구간
            condition_all = condition_b & condition_d & condition_e
            
            # 종가 그래프
            ax.plot(dates, close_prices, label='BTC 종가', color='gray', linewidth=1.5, alpha=0.5)
            
            # B, D, E 모두 만족하는 구간 색칠
            if np.any(condition_all):
                # 연속된 구간 찾기
                in_range = False
                start_idx = None
                label_added = False
                
                for i, satisfied in enumerate(condition_all):
                    if satisfied and not in_range:
                        # 구간 시작
                        in_range = True
                        start_idx = i
                    elif not satisfied and in_range:
                        # 구간 종료
                        if start_idx is not None:
                            label = 'B, D, E 모두 만족' if not label_added else None
                            ax.axvspan(dates[start_idx], dates[i-1], alpha=0.3, 
                                     color='yellow', label=label)
                            label_added = True
                        in_range = False
                        start_idx = None
                
                # 마지막까지 만족하는 경우
                if in_range and start_idx is not None:
                    label = 'B, D, E 모두 만족' if not label_added else None
                    ax.axvspan(dates[start_idx], dates[-1], alpha=0.3, 
                             color='yellow', label=label)
            
            # 매수 신호 표시 (B, D, E 만족 시점)
            buy_signals_dates = []
            buy_signals_prices = []
            for i, (date, satisfied) in enumerate(zip(dates, condition_all)):
                if satisfied:
                    buy_signals_dates.append(date)
                    buy_signals_prices.append(close_prices[i])
            
            if buy_signals_dates:
                ax.scatter(buy_signals_dates, buy_signals_prices, color='green', marker='o', 
                          s=50, alpha=0.6, label='BDE 조건 만족', zorder=4)
            
            ax.set_ylabel('가격 (원)', fontsize=10)
            ax.set_title('B, D, E 조건 모두 일치하는 구간', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _plot_angle_chart(self, ax, dates, angles, buy_threshold, sell_threshold):
        """
        추세선 각도 차트 플롯
        
        Parameters:
        - ax: matplotlib axes 객체
        - dates: 날짜 인덱스
        - angles: 각도 배열
        - buy_threshold: 매수 기준선 각도
        - sell_threshold: 매도 기준선 각도
        """
        try:
            ax.plot(dates, angles, label='추세선 각도', color='green', linewidth=1.5)
            ax.axhline(y=buy_threshold, color='red', linestyle='--', alpha=0.5, 
                      label=f'매수 기준선 ({buy_threshold}°)')
            ax.axhline(y=sell_threshold, color='blue', linestyle='--', alpha=0.5, 
                      label=f'매도 기준선 ({sell_threshold}°)')
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.fill_between(dates, 0, angles, where=(angles >= 0), alpha=0.2, 
                           color='red', label='상승 추세')
            ax.fill_between(dates, 0, angles, where=(angles < 0), alpha=0.2, 
                           color='blue', label='하락 추세')
            
            ax.set_ylabel('각도 (°)', fontsize=10)
            ax.set_title('추세선 각도', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _plot_holding_periods(self, ax, dates, close_prices, trades):
        """
        보유 구간 차트 플롯
        
        Parameters:
        - ax: matplotlib axes 객체
        - dates: 날짜 인덱스
        - close_prices: 종가 배열
        - trades: 거래 리스트
        """
        try:
            ax.plot(dates, close_prices, color='gray', linewidth=0.5, alpha=0.5)
            
            # 보유 구간 찾기
            holding_periods = self._find_holding_periods(trades, dates)
            
            # 보유 구간 색칠
            for idx, period in enumerate(holding_periods):
                start_idx = period['start_idx']
                end_idx = period['end_idx']
                
                if start_idx is not None and end_idx is not None:
                    period_dates = dates[start_idx:end_idx+1]
                    period_prices = close_prices[start_idx:end_idx+1]
                    # 첫 번째 구간만 레이블 추가하여 중복 방지
                    label = '보유 구간' if idx == 0 else None
                    ax.fill_between(period_dates, 0, period_prices, alpha=0.3, 
                                   color='yellow', label=label)
            
            ax.set_ylabel('가격 (원)', fontsize=10)
            ax.set_title('보유 구간', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _find_holding_periods(self, trades, dates):
        """
        보유 구간 찾기
        
        Parameters:
        - trades: 거래 리스트
        - dates: 날짜 인덱스
        
        Returns:
        - list: 보유 구간 리스트
        """
        try:
            holding_periods = []
            current_hold = None
            
            for trade in trades:
                if trade['action'].startswith('BUY'):
                    if current_hold is None:
                        current_hold = {'start': trade['date']}
                elif trade['action'].startswith('SELL'):
                    if current_hold is not None:
                        current_hold['end'] = trade['date']
                        holding_periods.append(current_hold)
                        current_hold = None
            
            # 마지막 보유 중인 경우
            if current_hold is not None:
                current_hold['end'] = dates[-1]
                holding_periods.append(current_hold)
            
            # 인덱스 변환
            for period in holding_periods:
                period['start_idx'] = dates.get_loc(period['start']) if period['start'] in dates else None
                period['end_idx'] = dates.get_loc(period['end']) if period['end'] in dates else None
            
            return holding_periods
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _plot_cumulative_profit(self, ax, trades):
        """
        누적 수익률 차트 플롯
        
        Parameters:
        - ax: matplotlib axes 객체
        - trades: 거래 리스트
        """
        try:
            # 누적 수익률 계산
            cumulative_profit = 0.0
            cumulative_profits = []
            cumulative_dates = []
            
            for trade in trades:
                if 'profit' in trade:
                    cumulative_profit += trade['profit']
                    cumulative_profits.append(cumulative_profit)
                    cumulative_dates.append(trade['date'])
            
            if cumulative_dates:
                ax.plot(cumulative_dates, cumulative_profits, marker='o', 
                       color='purple', linewidth=2, markersize=4, label='누적 수익')
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                
                # 최고점과 최저점 표시
                if len(cumulative_profits) > 0:
                    max_idx = np.argmax(cumulative_profits)
                    min_idx = np.argmin(cumulative_profits)
                    ax.scatter([cumulative_dates[max_idx]], [cumulative_profits[max_idx]], 
                             color='green', marker='*', s=200, zorder=5, label='최고점')
                    ax.scatter([cumulative_dates[min_idx]], [cumulative_profits[min_idx]], 
                             color='red', marker='*', s=200, zorder=5, label='최저점')
            
            ax.set_ylabel('누적 수익 (원)', fontsize=10)
            ax.set_xlabel('날짜', fontsize=10)
            ax.set_title('누적 수익률', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
            # x축 날짜 포맷팅
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _filter_data_by_period(self, df, result, start_date=None, end_date=None):
        """
        기간별로 데이터와 거래 기록 필터링
        
        Parameters:
        - df (pd.DataFrame): 전체 데이터프레임
        - result (dict): 전체 백테스트 결과
        - start_date (pd.Timestamp, optional): 시작 날짜
        - end_date (pd.Timestamp, optional): 종료 날짜
        
        Returns:
        - tuple: (필터링된 df, 필터링된 result)
        """
        try:
            # 데이터프레임 필터링
            filtered_df = df.copy()
            if start_date is not None:
                filtered_df = filtered_df[filtered_df.index >= start_date]
            if end_date is not None:
                filtered_df = filtered_df[filtered_df.index <= end_date]
            
            # 거래 기록 필터링
            filtered_trades = []
            for trade in result['trades']:
                trade_date = pd.to_datetime(trade['date'])
                if start_date is not None and trade_date < start_date:
                    continue
                if end_date is not None and trade_date > end_date:
                    continue
                filtered_trades.append(trade)
            
            # 필터링된 결과 딕셔너리 생성
            filtered_result = result.copy()
            filtered_result['trades'] = filtered_trades
            filtered_result['total_trades'] = len(filtered_trades)
            filtered_result['buy_count'] = len([t for t in filtered_trades if t['action'].startswith('BUY')])
            filtered_result['sell_count'] = len([t for t in filtered_trades if t['action'].startswith('SELL')])
            
            return filtered_df, filtered_result
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def plot_backtest_results_by_periods(self, df, result, base_dir='./images',
                                         buy_threshold=None, sell_threshold=None,
                                         volume_window=None, ma_window=None, volume_multiplier=None):
        """
        여러 기간별 백테스트 결과 그래프 생성
        
        Parameters:
        - df (pd.DataFrame): 추세선 계산이 완료된 OHLCV 데이터프레임
        - result (dict): 백테스트 결과 딕셔너리
        - base_dir (str): 이미지 저장 기본 디렉토리
        - buy_threshold (float, optional): 매수 기준선 각도. None이면 Config 기본값 사용
        - sell_threshold (float, optional): 매도 기준선 각도. None이면 Config 기본값 사용
        - volume_window (int, optional): 거래량 평균 계산용 윈도우. None이면 QQC 그래프 미생성
        - ma_window (int, optional): 이동평균 계산용 윈도우. None이면 QQC 그래프 미생성
        - volume_multiplier (float, optional): 거래량 배수. None이면 QQC 그래프 미생성
        """
        try:
            # 디렉토리 생성
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
                print(f"이미지 디렉토리 생성: {base_dir}")
            
            if buy_threshold is None:
                buy_threshold = Config.DEFAULT_BUY_ANGLE_THRESHOLD
            
            if sell_threshold is None:
                sell_threshold = Config.DEFAULT_SELL_ANGLE_THRESHOLD
            
            # 마지막 날짜 기준
            last_date = df.index[-1]
            
            # 1. 과거 3일 데이터
            days_3_start = last_date - pd.Timedelta(days=3)
            days_3_df, days_3_result = self._filter_data_by_period(df, result, start_date=days_3_start, end_date=last_date)
            if len(days_3_df) > 0:
                save_path = os.path.join(base_dir, 'backtest_result_3days.png')
                self.plot_backtest_results(days_3_df, days_3_result, save_path=save_path,
                                         buy_threshold=buy_threshold, sell_threshold=sell_threshold,
                                         volume_window=volume_window, ma_window=ma_window, volume_multiplier=volume_multiplier)
            
            # 2. 과거 5일 데이터
            days_5_start = last_date - pd.Timedelta(days=5)
            days_5_df, days_5_result = self._filter_data_by_period(df, result, start_date=days_5_start, end_date=last_date)
            if len(days_5_df) > 0:
                save_path = os.path.join(base_dir, 'backtest_result_5days.png')
                self.plot_backtest_results(days_5_df, days_5_result, save_path=save_path,
                                         buy_threshold=buy_threshold, sell_threshold=sell_threshold,
                                         volume_window=volume_window, ma_window=ma_window, volume_multiplier=volume_multiplier)
            
            # 3. 오늘 데이터 (마지막 날짜 하루)
            today_start = last_date.replace(hour=0, minute=0, second=0, microsecond=0)
            today_df, today_result = self._filter_data_by_period(df, result, start_date=today_start, end_date=last_date)
            if len(today_df) > 0:
                save_path = os.path.join(base_dir, 'backtest_result_today.png')
                self.plot_backtest_results(today_df, today_result, save_path=save_path,
                                         buy_threshold=buy_threshold, sell_threshold=sell_threshold,
                                         volume_window=volume_window, ma_window=ma_window, volume_multiplier=volume_multiplier)
            
            # 4. 전체기간 데이터
            save_path = os.path.join(base_dir, 'backtest_result_all.png')
            self.plot_backtest_results(df, result, save_path=save_path,
                                     buy_threshold=buy_threshold, sell_threshold=sell_threshold,
                                     volume_window=volume_window, ma_window=ma_window, volume_multiplier=volume_multiplier)
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise

