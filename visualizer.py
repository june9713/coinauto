"""
시각화 모듈
백테스트 결과 시각화 기능 제공
"""
import traceback
import os
import warnings
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 백엔드를 Agg로 설정하여 속도 향상
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib import rc
from matplotlib.ticker import FuncFormatter, LogFormatter
from config import Config

# matplotlib 폰트 관련 경고 억제
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
# 폰트 glyph 관련 경고 억제
warnings.filterwarnings('ignore', message=".*does not have a glyph for.*")

# 전역 matplotlib 설정: mathtext 완전 비활성화, ASCII만 사용
rc('axes', unicode_minus=False)  # 유니코드 마이너스 기호 대신 ASCII '-' 사용
rc('text', usetex=False)  # LaTeX 완전 비활성화
rc('font', **{'family': 'sans-serif'})

# mathtext 파서를 완전히 비활성화 (수식 포맷터 사용 안 함)
matplotlib.rcParams['axes.formatter.use_mathtext'] = False  # 수식 포맷터 사용 안 함
matplotlib.rcParams['axes.unicode_minus'] = False  # 유니코드 마이너스 사용 안 함
matplotlib.rcParams['mathtext.default'] = 'regular'  # mathtext 비활성화
matplotlib.rcParams['text.usetex'] = False  # LaTeX 사용 안 함


def safe_number_formatter(x, p):
    """
    안전한 숫자 포맷터 (NaN, inf, None 처리)
    
    Parameters:
    - x: 포맷팅할 숫자
    - p: matplotlib 포지션 파라미터 (사용하지 않음)
    
    Returns:
    - str: 포맷팅된 문자열
    """
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return ''
        return f'{x:,.0f}'
    except (ValueError, TypeError, OverflowError):
        return ''


class Visualizer:
    """백테스트 결과 시각화 클래스"""
    
    # 클래스 변수: 폰트 등록 여부 추적 (메모리 누수 방지)
    _font_registered = set()
    
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
        """한글 폰트 설정 (중복 등록 방지)"""
        try:
            if os.path.exists(self.font_path):
                # 폰트 경로를 키로 사용하여 중복 등록 방지
                if self.font_path not in Visualizer._font_registered:
                    # 폰트를 matplotlib에 등록
                    font_manager.fontManager.addfont(self.font_path)
                    Visualizer._font_registered.add(self.font_path)

                # 폰트 속성 가져오기
                font_prop = font_manager.FontProperties(fname=self.font_path)
                font_name = font_prop.get_name()

                # 폰트 설정 - 여러 방식으로 확실하게 적용
                rc('font', family=font_name)
                matplotlib.rcParams['font.family'] = font_name
                matplotlib.rcParams['font.sans-serif'] = [font_name]

                print(f"한글 폰트 설정 완료: {font_name}")
            else:
                # 폰트 파일이 없으면 기본 폰트 사용 시도
                print(f"경고: 폰트 파일을 찾을 수 없습니다: {self.font_path}")
                rc('font', family='DejaVu Sans')
            # 유니코드 마이너스 기호 대신 ASCII 마이너스(-) 사용하여 경고 방지
            rc('axes', unicode_minus=False)
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
                        result['trades']는 현재 백테스트 결과 + 기존 백테스트 결과(이미지 출력 기간 내)를 병합한 거래 리스트
                        이를 통해 과거 백테스트에서 발생한 거래도 현재 이미지에 함께 표시됨
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
            
            # 데이터 검증
            if df is None or len(df) == 0:
                print(f"  경고: 데이터프레임이 비어있습니다. 그래프를 생성하지 않습니다.")
                return
            
            if 'close' not in df.columns:
                print(f"  경고: 'close' 컬럼이 없습니다. 그래프를 생성하지 않습니다.")
                return
            
            trades = result.get('trades', [])
            
            # 각도 데이터가 있는지 확인
            has_angle_data = 'angle' in df.columns
            
            # QQC 전략 파라미터 확인
            has_qqc_params = volume_window is not None and ma_window is not None and volume_multiplier is not None

            # 서브플롯 개수 결정
            subplot_count = 3  # 기본: 가격, 보유 구간, 누적 수익률
            if has_angle_data:
                subplot_count += 1  # 각도 차트 추가
            if has_qqc_params:
                subplot_count += 2  # 거래량 차트 + 조건 상태 차트 추가

            # 서브플롯 인덱스 설정
            current_idx = 0
            price_ax_idx = current_idx
            current_idx += 1

            angle_ax_idx = None
            if has_angle_data:
                angle_ax_idx = current_idx
                current_idx += 1

            volume_ax_idx = None
            conditions_ax_idx = None
            if has_qqc_params:
                volume_ax_idx = current_idx
                current_idx += 1
                conditions_ax_idx = current_idx
                current_idx += 1

            holding_ax_idx = current_idx
            current_idx += 1
            profit_ax_idx = current_idx
            
            # 서브플롯 생성
            fig, axes = plt.subplots(subplot_count, 1, figsize=(16, 4 * subplot_count), sharex=True)
            if subplot_count == 1:
                axes = [axes]
            
            #fig.suptitle('BTC 백테스트 결과', fontsize=16, fontweight='bold')
            
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

            # 3-1. 조건 상태 차트 (QQC 파라미터가 있는 경우)
            if has_qqc_params and conditions_ax_idx is not None:
                self._plot_conditions_states_chart(axes[conditions_ax_idx], df, volume_window, ma_window, volume_multiplier)

            # 4. 보유 구간 표시
            self._plot_holding_periods(axes[holding_ax_idx], dates, close_prices, trades)
            
            # 9. 누적 수익률 차트
            self._plot_cumulative_profit(axes[profit_ax_idx], trades)
            
            # 모든 axes에 대해 unicode_minus=False 강제 설정 및 x축 날짜/시간 포맷팅
            for ax in axes:
                ax.tick_params(axis='both', which='major', labelsize=10)
            
            # x축 날짜/시간 포맷팅 (모든 차트에 공통 적용, sharex=True로 인해 마지막 차트에만 설정해도 모든 차트에 적용됨)
            try:
                if len(dates) > 0:
                    # 마지막 차트에 날짜/시간 포맷 설정 (sharex=True로 인해 모든 차트에 적용)
                    axes[profit_ax_idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
                    axes[profit_ax_idx].xaxis.set_major_locator(mdates.AutoDateLocator())
                    plt.setp(axes[profit_ax_idx].xaxis.get_majorticklabels(), rotation=45, ha='right')
            except Exception as e:
                # 날짜 포맷팅 실패 시 기본 포맷 사용
                pass
            
            # tight_layout 안전하게 처리 (mathtext 파서 오류 방지)
            try:
                plt.tight_layout(pad=1.0)  # pad 값 명시하여 계산 속도 향상
            except Exception as e:
                # tight_layout 실패 시 figure를 먼저 draw하여 렌더러 초기화
                print(f"  경고: tight_layout 실패, figure draw 후 저장: {e}")
                fig.canvas.draw()  # 렌더러 초기화를 위해 명시적으로 draw 호출
            
            # 렌더러 초기화를 위해 figure를 한 번 그리기
            try:
                fig.canvas.draw()
            except Exception as draw_error:
                # draw 실패해도 저장 시도
                print(f"  경고: figure draw 실패, 저장 시도: {draw_error}")
            
            # 파일 저장
            try:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white', format='png')
                # 저장 완료 확인
                if os.path.exists(save_path):
                    file_size = os.path.getsize(save_path)
                    if file_size > 0:
                        print(f"  그래프 저장 완료: {save_path} (크기: {file_size:,} bytes)")
                    else:
                        print(f"  경고: 저장된 파일이 비어있습니다: {save_path}")
                else:
                    print(f"  오류: 파일 저장 실패: {save_path}")
            except Exception as save_error:
                err = traceback.format_exc()
                print(f"  오류: 그래프 저장 실패: {err}")
                raise
            
            # 저장 완료 후 figure 닫기 (메모리 누수 방지)
            plt.close(fig)
            del fig  # 참조 제거
            
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
            
            # 매수/매도 신호 표시 (매도 유형별 구분)
            buy_dates = []
            buy_prices = []
            sell_profit_dates = []  # 이익실현
            sell_profit_prices = []
            sell_loss_dates = []  # 손절
            sell_loss_prices = []
            sell_expiry_dates = []  # 기간 만료
            sell_expiry_prices = []
            
            # trades 데이터 확인
            if len(trades) == 0:
                print(f"  경고: 거래 데이터가 없습니다 (trades 개수: {len(trades)})")
            else:
                print(f"  거래 데이터 확인: {len(trades)}개 거래")
                print(f"  그래프 데이터 범위: {dates[0]} ~ {dates[-1]}")
            
            for trade in trades:
                trade_date = trade['date']
                trade_price = trade['price']
                action = trade.get('action', '')
                
                # 날짜 형식 확인 및 변환
                if not isinstance(trade_date, pd.Timestamp):
                    trade_date = pd.to_datetime(trade_date)
                
                # 그래프 범위 내에 있는지 확인
                if trade_date < dates[0] or trade_date > dates[-1]:
                    # 범위 밖 거래는 건너뛰기 (디버깅용으로만 로그 출력)
                    continue
                
                if action.startswith('BUY'):
                    buy_dates.append(trade_date)
                    buy_prices.append(trade_price)
                elif action.startswith('SELL'):
                    if '이익실현' in action:
                        sell_profit_dates.append(trade_date)
                        sell_profit_prices.append(trade_price)
                    elif '손절' in action:
                        sell_loss_dates.append(trade_date)
                        sell_loss_prices.append(trade_price)
                    elif '기간 만료' in action:
                        sell_expiry_dates.append(trade_date)
                        sell_expiry_prices.append(trade_price)
                    else:
                        # 일반 매도 (유형 불명)
                        sell_expiry_dates.append(trade_date)
                        sell_expiry_prices.append(trade_price)
            
            # 결과 확인 로그
            print(f"  매수: {len(buy_dates)}개, 이익실현: {len(sell_profit_dates)}개, 손절: {len(sell_loss_dates)}개, 기간 만료: {len(sell_expiry_dates)}개")
            
            if buy_dates:
                ax.scatter(buy_dates, buy_prices, color='red', marker='^', s=100, 
                          label='매수', zorder=5)
            if sell_profit_dates:
                ax.scatter(sell_profit_dates, sell_profit_prices, color='blue', marker='v', s=100, 
                          label='매도 (이익실현)', zorder=5)
            if sell_loss_dates:
                ax.scatter(sell_loss_dates, sell_loss_prices, color='orange', marker='v', s=100, 
                          label='매도 (손절)', zorder=5)
            if sell_expiry_dates:
                ax.scatter(sell_expiry_dates, sell_expiry_prices, color='purple', marker='v', s=100, 
                          label='매도 (기간 만료)', zorder=5)
            
            ax.set_ylabel('가격 (원)', fontsize=10)
            ax.set_title('BTC 가격 및 매매 신호', fontsize=12, fontweight='bold')
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(safe_number_formatter))
            
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
            
            # CSV에 저장된 이동평균 사용, 없으면 계산
            if 'ma_values' in df.columns:
                ma_values = df['ma_values'].values
                print(f"  [그래프] CSV에서 저장된 이동평균 사용")
            else:
                # 이동평균 계산 (현재 캔들을 제외한 과거 ma_window개 사용)
                # pandas rolling을 사용하여 벡터화된 계산으로 속도 향상
                ma_values = df['close'].shift(1).rolling(window=ma_window, min_periods=1).mean().values
                # 첫 번째 값은 현재 값으로 설정 (과거 데이터가 없는 경우)
                if len(ma_values) > 0:
                    ma_values[0] = close_prices[0]
                print(f"  [그래프] 이동평균 실시간 계산 (CSV에 저장되지 않음)")
            
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
            
            # 매수/매도 신호 표시 (매도 유형별 구분)
            buy_dates = []
            buy_prices = []
            sell_profit_dates = []  # 이익실현
            sell_profit_prices = []
            sell_loss_dates = []  # 손절
            sell_loss_prices = []
            sell_expiry_dates = []  # 기간 만료
            sell_expiry_prices = []
            
            # trades 데이터 확인
            if len(trades) == 0:
                print(f"  경고: 거래 데이터가 없습니다 (trades 개수: {len(trades)})")
            else:
                print(f"  거래 데이터 확인: {len(trades)}개 거래")
                print(f"  그래프 데이터 범위: {dates[0]} ~ {dates[-1]}")
            
            for trade in trades:
                trade_date = trade['date']
                trade_price = trade['price']
                action = trade.get('action', '')
                
                # 날짜 형식 확인 및 변환
                if not isinstance(trade_date, pd.Timestamp):
                    trade_date = pd.to_datetime(trade_date)
                
                # 그래프 범위 내에 있는지 확인
                if trade_date < dates[0] or trade_date > dates[-1]:
                    # 범위 밖 거래는 건너뛰기 (디버깅용으로만 로그 출력)
                    continue
                
                if action.startswith('BUY'):
                    buy_dates.append(trade_date)
                    buy_prices.append(trade_price)
                elif action.startswith('SELL'):
                    if '이익실현' in action:
                        sell_profit_dates.append(trade_date)
                        sell_profit_prices.append(trade_price)
                    elif '손절' in action:
                        sell_loss_dates.append(trade_date)
                        sell_loss_prices.append(trade_price)
                    elif '기간 만료' in action:
                        sell_expiry_dates.append(trade_date)
                        sell_expiry_prices.append(trade_price)
                    else:
                        # 일반 매도 (유형 불명)
                        sell_expiry_dates.append(trade_date)
                        sell_expiry_prices.append(trade_price)
            
            # 결과 확인 로그
            print(f"  매수: {len(buy_dates)}개, 이익실현: {len(sell_profit_dates)}개, 손절: {len(sell_loss_dates)}개, 기간 만료: {len(sell_expiry_dates)}개")
            
            if buy_dates:
                ax.scatter(buy_dates, buy_prices, color='red', marker='^', s=100, 
                          label='매수', zorder=5)
            if sell_profit_dates:
                ax.scatter(sell_profit_dates, sell_profit_prices, color='blue', marker='v', s=100, 
                          label='매도 (이익실현)', zorder=5)
            if sell_loss_dates:
                ax.scatter(sell_loss_dates, sell_loss_prices, color='orange', marker='v', s=100, 
                          label='매도 (손절)', zorder=5)
            if sell_expiry_dates:
                ax.scatter(sell_expiry_dates, sell_expiry_prices, color='purple', marker='v', s=100, 
                          label='매도 (기간 만료)', zorder=5)
            
            ax.set_ylabel('가격 (원)', fontsize=10)
            ax.set_title('BTC 가격 (종가, 이동평균, 음봉/양봉)', fontsize=12, fontweight='bold')
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(safe_number_formatter))
            
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

            # CSV에 저장된 거래량 평균 사용, 없으면 계산
            if 'volume_avg' in df.columns:
                volume_avg = df['volume_avg'].values
                print(f"  [그래프] CSV에서 저장된 거래량 평균 사용")
            else:
                # 거래량 평균 계산 (현재 캔들을 제외한 과거 volume_window개 사용)
                # QQC 엔진과 동일한 로직: pandas rolling을 사용하여 벡터화된 계산으로 속도 향상
                volume_avg = df['volume'].shift(1).rolling(window=volume_window, min_periods=1).mean().values
                # 첫 번째 값은 0으로 설정 (과거 데이터가 없는 경우)
                if len(volume_avg) > 0:
                    volume_avg[0] = 0.0
                print(f"  [그래프] 거래량 평균 실시간 계산 (CSV에 저장되지 않음)")
            
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
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3)
            # 로그 스케일 사용 시 수식 렌더링 문제 방지를 위해 커스텀 포맷터 사용
            ax.set_yscale('log')  # 거래량은 로그 스케일이 적합할 수 있음
            # 로그 스케일의 수식 렌더링 문제 방지: mathtext를 사용하지 않는 포맷터 설정
            def log_formatter(x, pos):
                """로그 스케일용 안전한 포맷터 (수식 없이)"""
                try:
                    if x <= 0:
                        return ''
                    # 과학적 표기법 대신 일반 숫자로 표시
                    if x >= 1e6:
                        return f'{x/1e6:.1f}M'
                    elif x >= 1e3:
                        return f'{x/1e3:.1f}K'
                    else:
                        return f'{x:.2f}'
                except:
                    return ''
            ax.yaxis.set_major_formatter(FuncFormatter(log_formatter))
            
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

            # CSV에 저장된 거래량 평균 사용, 없으면 계산
            if 'volume_avg' in df.columns:
                volume_avg = df['volume_avg'].values
            else:
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
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(safe_number_formatter))
            
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

            # CSV에 저장된 이동평균 사용, 없으면 계산
            if 'ma_values' in df.columns:
                ma_values = df['ma_values'].values
            else:
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
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(safe_number_formatter))
            
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
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(safe_number_formatter))
            
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
            # CSV에 저장된 거래량 평균 사용, 없으면 계산
            if 'volume_avg' in df.columns:
                volume_avg = df['volume_avg'].values
            else:
                # 거래량 평균 계산 (현재 캔들을 제외한 과거 volume_window개 사용)
                volume_avg = df['volume'].shift(1).rolling(window=volume_window, min_periods=1).mean().values
                if len(volume_avg) > 0:
                    volume_avg[0] = 0.0
            condition_b = volumes >= (volume_avg * volume_multiplier)

            # 조건 D: 현재 종가 > 이동평균
            # CSV에 저장된 이동평균 사용, 없으면 계산
            if 'ma_values' in df.columns:
                ma_values = df['ma_values'].values
            else:
                # 이동평균 계산 (현재 캔들을 제외한 과거 ma_window개 사용)
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
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(safe_number_formatter))

        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise

    def _plot_conditions_states_chart(self, ax, df, volume_window, ma_window, volume_multiplier):
        """
        조건 B, D, E 및 B&&D&&E의 True/False 상태를 시간대별로 표시하는 그래프

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
            if 'volume_avg' in df.columns:
                volume_avg = df['volume_avg'].values
            else:
                volume_avg = df['volume'].shift(1).rolling(window=volume_window, min_periods=1).mean().values
                if len(volume_avg) > 0:
                    volume_avg[0] = 0.0
            condition_b = volumes >= (volume_avg * volume_multiplier)

            # 조건 D: 현재 종가 > 이동평균
            if 'ma_values' in df.columns:
                ma_values = df['ma_values'].values
            else:
                ma_values = df['close'].shift(1).rolling(window=ma_window, min_periods=1).mean().values
                if len(ma_values) > 0:
                    ma_values[0] = close_prices[0]
            condition_d = close_prices > ma_values

            # 조건 E: 양봉 (오픈가 < 종가)
            condition_e = open_prices < close_prices

            # B&&D&&E (모든 조건 만족)
            condition_all = condition_b & condition_d & condition_e

            # True를 1, False를 0으로 변환
            b_values = condition_b.astype(int)
            d_values = condition_d.astype(int)
            e_values = condition_e.astype(int)
            all_values = condition_all.astype(int)

            # 각 조건을 서로 다른 높이에 표시 (계단식으로 배치하여 보기 쉽게)
            # 4개의 레이어로 구성: B(맨 아래), D(두번째), E(세번째), B&&D&&E(맨 위)
            ax.fill_between(dates, 0, b_values, where=b_values > 0,
                           alpha=0.5, color='green', label='조건 B', step='post')
            ax.fill_between(dates, 1, 1 + d_values, where=d_values > 0,
                           alpha=0.5, color='blue', label='조건 D', step='post')
            ax.fill_between(dates, 2, 2 + e_values, where=e_values > 0,
                           alpha=0.5, color='red', label='조건 E', step='post')
            ax.fill_between(dates, 3, 3 + all_values, where=all_values > 0,
                           alpha=0.7, color='purple', label='B&&D&&E', step='post')

            # y축 설정
            ax.set_ylim(-0.2, 4.5)
            ax.set_yticks([0.5, 1.5, 2.5, 3.5])
            ax.set_yticklabels(['조건 B', '조건 D', '조건 E', 'B&&D&&E'])

            # 그리드 추가 (각 조건 구분선)
            for y in [1, 2, 3]:
                ax.axhline(y=y, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

            ax.set_ylabel('조건 상태', fontsize=10)
            ax.set_title('조건 B, D, E 및 B&&D&&E 상태', fontsize=12, fontweight='bold')
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.2, axis='x')

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
            ax.legend(loc='lower left')
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
            ax.yaxis.set_major_formatter(plt.FuncFormatter(safe_number_formatter))
            
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
            # legend는 라벨이 있는 아티스트가 있을 때만 표시
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(safe_number_formatter))
            
            # x축 날짜/시간 포맷팅은 plot_backtest_results 함수에서 공통으로 설정됨
            
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
                                         volume_window=None, ma_window=None, volume_multiplier=None,
                                         interval=None):
        """
        여러 기간별 백테스트 결과 그래프 생성 (순차 처리)

        Parameters:
        - df (pd.DataFrame): 추세선 계산이 완료된 OHLCV 데이터프레임
        - result (dict): 백테스트 결과 딕셔너리
                        result['trades']는 현재 백테스트 결과 + 기존 백테스트 결과(이미지 출력 기간 내)를 병합한 거래 리스트
                        이를 통해 과거 백테스트에서 발생한 거래도 현재 이미지에 함께 표시됨
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
            
            # 현재 시간에서 interval만큼 뺀 시간까지 그래프에 표시
            now = pd.Timestamp.now()
            if interval is not None:
                # interval을 timedelta로 변환
                from datetime import timedelta
                if interval.endswith('m'):
                    minutes = int(interval[:-1])
                    interval_timedelta = timedelta(minutes=minutes)
                elif interval.endswith('h'):
                    hours = int(interval[:-1])
                    if hours == 24:
                        interval_timedelta = timedelta(days=1)
                    else:
                        interval_timedelta = timedelta(hours=hours)
                else:
                    interval_timedelta = timedelta(days=1)  # 기본값
                
                # 그래프에 표시할 마지막 시간 = 현재 시간 - interval
                graph_end_time = now - interval_timedelta
                
                # 데이터프레임에서 graph_end_time 이전의 데이터만 사용
                df = df[df.index < graph_end_time]
                
                if len(df) == 0:
                    print("  경고: 그래프에 표시할 데이터가 없습니다 (현재 시간 기준 필터링 후).")
                    return
                
                last_date = df.index[-1]
                print(f"  현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  그래프 마지막 시간 (현재 - {interval}): {graph_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  그래프에 표시될 마지막 캔들: {last_date.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                # interval 정보가 없으면 데이터의 마지막 날짜 사용
                last_date = df.index[-1]
            
            # 작업 리스트 생성
            tasks = []
            
            # 1. 과거 3일 데이터
            days_3_start = last_date - pd.Timedelta(days=3)
            days_3_df, days_3_result = self._filter_data_by_period(df, result, start_date=days_3_start, end_date=last_date)
            if len(days_3_df) > 0:
                save_path = os.path.join(base_dir, 'backtest_result_3days.jpg')
                tasks.append(('3일', days_3_df, days_3_result, save_path))
            
            # 2. 과거 5일 데이터
            days_5_start = last_date - pd.Timedelta(days=5)
            days_5_df, days_5_result = self._filter_data_by_period(df, result, start_date=days_5_start, end_date=last_date)
            if len(days_5_df) > 0:
                save_path = os.path.join(base_dir, 'backtest_result_5days.jpg')
                tasks.append(('5일', days_5_df, days_5_result, save_path))
            
            # 3. 오늘 데이터 (마지막 날짜 하루)
            today_start = last_date.replace(hour=0, minute=0, second=0, microsecond=0)
            today_df, today_result = self._filter_data_by_period(df, result, start_date=today_start, end_date=last_date)
            if len(today_df) > 0:
                save_path = os.path.join(base_dir, 'backtest_result_today.jpg')
                tasks.append(('오늘', today_df, today_result, save_path))
            
            # 순차적으로 그래프 생성
            for period_name, period_df, period_result, save_path in tasks:
                try:
                    print(f"  → {period_name} 그래프 생성 중...")
                    self.plot_backtest_results(
                        period_df, period_result, save_path=save_path,
                        buy_threshold=buy_threshold, sell_threshold=sell_threshold,
                        volume_window=volume_window, ma_window=ma_window, volume_multiplier=volume_multiplier
                    )
                    print(f"  ✓ {period_name} 그래프 생성 완료: {save_path}")
                except Exception as e:
                    err = traceback.format_exc()
                    print(f"  ✗ {period_name} 그래프 생성 실패: {err}")
            
            # 메모리 정리
            plt.close('all')
            gc.collect()
            
            print(f"\n모든 그래프 생성 완료 ({len(tasks)}개)")
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise

