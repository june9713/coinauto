import math
import numpy as np
import pandas as pd
import traceback
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from makeHistory import make_history

def calculate_trendline(df, window=14, aspect_ratio=4):
    """
    주어진 OHLCV 데이터프레임에 대해, 각 시점에서 과거 'window' 기간 동안의 저가(low)를 기반으로
    지지 추세선(support trendline)을 선형 회귀로 피팅하고,
    이 추세선이 수평선과 이루는 '시각적 각도'(degree)를 계산하여 'angle' 컬럼에 저장합니다.
    
    Parameters:
    - df (pd.DataFrame): 'open', 'high', 'low', 'close', 'volume' 컬럼을 포함한 시계열 데이터.
                         인덱스는 datetime 타입이어야 함.
    - window (int): 추세선을 계산할 때 사용할 과거 봉(캔들)의 개수. 기본값 14.
    - aspect_ratio (float): 차트의 시각적 종횡비 (너비 / 높이). 
                            실제 차트에서 x축(시간)과 y축(가격)의 픽셀 비율을 모방하여
                            각도를 시각적으로 정확하게 계산하기 위해 사용됨. 기본값 4.

    Returns:
    - pd.DataFrame: 원본 데이터에 다음 두 컬럼이 추가된 복사본:
        - 'trendline': 해당 시점에서 추정된 추세선의 마지막 값 (즉, 현재 봉 직전의 추세선 예측값)
        - 'angle': 수평선(0°)과 추세선 사이의 시각적 각도 (단위: 도, degree). 
                   양수 = 상승 추세, 음수 = 하락 추세.
    """
    
    # 원본 데이터프레임을 수정하지 않기 위해 복사본 생성
    df = df.copy()
    
    # 추세선 값과 각도를 저장할 새로운 컬럼 초기화 (NaN으로 채움)
    # 첫 'window'개의 행은 충분한 과거 데이터가 없어 계산 불가 → NaN 유지
    df['trendline'] = np.nan
    df['angle'] = np.nan
    
    # 인덱스 'window'부터 끝까지 반복:
    #   → i번째 행에서, [i-window, i) 구간(총 'window'개 봉)을 사용해 추세선 계산
    for i in range(window, len(df)):
        
        # 1. 윈도우 내 데이터 추출: 현재 시점 직전 'window'일치의 OHLCV 데이터
        window_data = df.iloc[i - window : i]
        
        # 2. 추세선 기반: 저가(low)를 사용 → 지지선(support line) 추정
        #    (매수 전략에서 지지선 상승을 신호로 삼기 때문)
        low_points = window_data['low'].values  # shape: (window,)
        
        # 3. x축 정의: 시간을 단순 정수 인덱스로 변환 (0, 1, 2, ..., window-1)
        #    → 실제 날짜 대신 상대적 위치 사용 (선형 회귀에선 절대 시간 필요 없음)
        x = np.array(range(window))  # shape: (window,)
        
        # 4. 선형 회귀: y = slope * x + intercept 를 low_points에 피팅
        #    np.polyfit(x, y, deg=1) → 1차 다항식(직선)의 계수 [slope, intercept] 반환
        slope, intercept = np.polyfit(x, low_points, 1)
        #    - slope: 가격 단위/봉 (예: +5000 = 매일 5,000원 상승 추세)
        #    - intercept: x=0일 때(윈도우 시작 시점)의 추정 저가
        
        # 5. 추세선의 '마지막 시점'(x = window - 1)에서의 예측값 계산
        #    → 이 값은 '현재 봉'(i) 시점에서의 추세선 위치로 해석됨
        trendline_value_at_end = slope * (window - 1) + intercept
        
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
        scaled_slope = slope * (window / (price_range * aspect_ratio))
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
        
        # 루프 종료 → 다음 시점으로 이동
    
    # 모든 시점에 대해 처리 완료 → 결과 데이터프레임 반환
    return df


def backtest_strategy(df, buy_angle_threshold=5.4, sell_angle_threshold=-16.7, 
                     stop_loss_percent=-5.0, min_sell_price=20000, 
                     price_slippage=1000, initial_capital=1000000):
    """
    백테스트 전략 실행
    
    Parameters:
    - df (pd.DataFrame): 추세선 계산이 완료된 OHLCV 데이터프레임
    - buy_angle_threshold (float): 매수 조건 (각도 >= 이 값). 기본값 5.4
    - sell_angle_threshold (float): 매도 조건 (각도 <= 이 값). 기본값 -16.7
    - stop_loss_percent (float): 손절 기준 (수익률 %). 기본값 -5.0
    - min_sell_price (float): 최소 매도 가격 (원). 기본값 20000
    - price_slippage (int): 거래 가격 슬리퍼지 (원). 기본값 1000
    - initial_capital (float): 초기 자본 (원). 기본값 1,000,000
    
    Returns:
    - dict: 백테스트 결과 딕셔너리
    """
    try:
        # 보유 상태 변수
        holding = False  # 코인 보유 여부
        buy_price = 0.0  # 매수 가격
        buy_date = None  # 매수 날짜
        coin_amount = 0.0  # 보유 코인 수량
        
        # 자산 추적
        cash = initial_capital  # 현금 보유액
        total_asset = initial_capital  # 총 자산 (현금 + 코인 평가액)
        
        # 거래 기록
        trades = []
        asset_history = []  # 자산 추이 기록
        
        # 각 날짜를 순회하면서 백테스트 (09:00에 전날 데이터로 판단)
        # 일봉 데이터는 각 행이 하루를 나타내므로, 
        # i번째 행은 해당 날짜의 종가를 의미하며, 
        # 09:00에 전날까지의 데이터로 판단한다는 것은 i-1번째 행까지의 데이터로 판단한다는 의미
        
        for i in range(len(df)):
            current_date = df.index[i]
            current_close = df.iloc[i]['close']
            current_angle = df.iloc[i]['angle']
            
            # 각도가 계산되지 않은 경우 (window 초기 기간) 건너뛰기
            if pd.isna(current_angle):
                asset_history.append({
                    'date': current_date,
                    'total_asset': total_asset,
                    'cash': cash,
                    'coin_value': 0.0,
                    'holding': holding
                })
                continue
            
            # 09:00에 전날까지의 데이터로 판단하므로, 전날 종가 사용
            # 전날이 없는 경우 건너뛰기
            if i == 0:
                asset_history.append({
                    'date': current_date,
                    'total_asset': total_asset,
                    'cash': cash,
                    'coin_value': 0.0,
                    'holding': holding
                })
                continue
            
            prev_close = df.iloc[i-1]['close']
            prev_angle = df.iloc[i-1]['angle']
            
            # 전날 각도도 계산되지 않은 경우 건너뛰기
            if pd.isna(prev_angle):
                asset_history.append({
                    'date': current_date,
                    'total_asset': total_asset,
                    'cash': cash,
                    'coin_value': 0.0,
                    'holding': holding
                })
                continue
            
            # 현재 보유 상태에서 수익률 계산
            if holding:
                current_return = ((prev_close - buy_price) / buy_price) * 100
                
                # 매도 조건 1: 손절 (수익률 -5% 이하)
                if current_return <= stop_loss_percent:
                    sell_price = prev_close - price_slippage  # 종가 - 슬리퍼지
                    sell_amount = coin_amount * sell_price  # 매도 금액
                    profit = sell_amount - (coin_amount * buy_price)
                    profit_percent = (profit / (coin_amount * buy_price)) * 100
                    
                    # 현금 업데이트
                    cash += sell_amount
                    
                    trades.append({
                        'date': current_date,
                        'action': 'SELL (손절)',
                        'price': sell_price,
                        'amount': coin_amount,
                        'total_value': sell_amount,
                        'buy_price': buy_price,
                        'buy_date': buy_date,
                        'profit': profit,
                        'profit_percent': profit_percent,
                        'angle': prev_angle,
                        'total_asset': cash
                    })
                    
                    holding = False
                    coin_amount = 0.0
                    buy_price = 0.0
                    buy_date = None
                    
                    total_asset = cash
                    asset_history.append({
                        'date': current_date,
                        'total_asset': total_asset,
                        'cash': cash,
                        'coin_value': 0.0,
                        'holding': holding
                    })
                    continue
                
                # 매도 조건 2: 추세선 각도 <= -16.7도
                if prev_angle <= sell_angle_threshold:
                    # 매도가격이 20000원 이하면 매도하지 않음
                    if prev_close > min_sell_price:
                        sell_price = prev_close - price_slippage  # 종가 - 슬리퍼지
                        sell_amount = coin_amount * sell_price  # 매도 금액
                        profit = sell_amount - (coin_amount * buy_price)
                        profit_percent = (profit / (coin_amount * buy_price)) * 100
                        
                        # 현금 업데이트
                        cash += sell_amount
                        
                        trades.append({
                            'date': current_date,
                            'action': 'SELL (각도 신호)',
                            'price': sell_price,
                            'amount': coin_amount,
                            'total_value': sell_amount,
                            'buy_price': buy_price,
                            'buy_date': buy_date,
                            'profit': profit,
                            'profit_percent': profit_percent,
                            'angle': prev_angle,
                            'total_asset': cash
                        })
                        
                        holding = False
                        coin_amount = 0.0
                        buy_price = 0.0
                        buy_date = None
                        
                        total_asset = cash
                        asset_history.append({
                            'date': current_date,
                            'total_asset': total_asset,
                            'cash': cash,
                            'coin_value': 0.0,
                            'holding': holding
                        })
                    else:
                        # 매도가격이 20000원 이하인 경우 존버
                        # 자산 평가 업데이트
                        coin_value = coin_amount * current_close
                        total_asset = cash + coin_value
                        asset_history.append({
                            'date': current_date,
                            'total_asset': total_asset,
                            'cash': cash,
                            'coin_value': coin_value,
                            'holding': holding
                        })
                        continue
                else:
                    # 보유 중이지만 매도 조건 미충족
                    coin_value = coin_amount * current_close
                    total_asset = cash + coin_value
                    asset_history.append({
                        'date': current_date,
                        'total_asset': total_asset,
                        'cash': cash,
                        'coin_value': coin_value,
                        'holding': holding
                    })
            else:
                # 매수 조건: 추세선 각도 >= 5.4도
                if prev_angle >= buy_angle_threshold:
                    # 전날 종가로 매수 (종가 + 슬리퍼지)
                    buy_price = prev_close + price_slippage
                    buy_date = current_date
                    
                    # 보유 현금으로 최대한 매수
                    coin_amount = cash / buy_price
                    buy_total = coin_amount * buy_price
                    
                    # 현금 업데이트
                    cash -= buy_total
                    holding = True
                    
                    trades.append({
                        'date': current_date,
                        'action': 'BUY',
                        'price': buy_price,
                        'amount': coin_amount,
                        'total_value': buy_total,
                        'angle': prev_angle,
                        'total_asset': cash + (coin_amount * current_close)
                    })
                    
                    # 자산 평가 업데이트
                    coin_value = coin_amount * current_close
                    total_asset = cash + coin_value
                    asset_history.append({
                        'date': current_date,
                        'total_asset': total_asset,
                        'cash': cash,
                        'coin_value': coin_value,
                        'holding': holding
                    })
                else:
                    # 매수 조건 미충족
                    total_asset = cash
                    asset_history.append({
                        'date': current_date,
                        'total_asset': total_asset,
                        'cash': cash,
                        'coin_value': 0.0,
                        'holding': holding
                    })
        
        # 마지막 보유 중인 경우 강제 매도
        if holding and len(df) > 0:
            last_close = df.iloc[-1]['close']
            sell_price = last_close - price_slippage
            sell_amount = coin_amount * sell_price
            profit = sell_amount - (coin_amount * buy_price)
            profit_percent = (profit / (coin_amount * buy_price)) * 100
            
            # 현금 업데이트
            cash += sell_amount
            
            trades.append({
                'date': df.index[-1],
                'action': 'SELL (마지막 강제 매도)',
                'price': sell_price,
                'amount': coin_amount,
                'total_value': sell_amount,
                'buy_price': buy_price,
                'buy_date': buy_date,
                'profit': profit,
                'profit_percent': profit_percent,
                'angle': df.iloc[-1]['angle'] if not pd.isna(df.iloc[-1]['angle']) else None,
                'total_asset': cash
            })
            
            total_asset = cash
        
        # 최종 수익률 계산
        final_return = ((total_asset - initial_capital) / initial_capital) * 100
        
        return {
            'trades': trades,
            'asset_history': asset_history,
            'total_trades': len(trades),
            'buy_count': len([t for t in trades if t['action'].startswith('BUY')]),
            'sell_count': len([t for t in trades if t['action'].startswith('SELL')]),
            'initial_capital': initial_capital,
            'final_asset': total_asset,
            'total_return': final_return
        }
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


def print_backtest_results(result):
    """
    백테스트 결과 출력
    """
    try:
        trades = result['trades']
        
        print("\n" + "="*80)
        print("백테스트 결과")
        print("="*80)
        print(f"초기 자본: {result['initial_capital']:,.0f}원")
        print(f"최종 자산: {result['final_asset']:,.0f}원")
        print(f"총 수익률: {result['total_return']:+.2f}%")
        print(f"\n총 거래 횟수: {result['total_trades']}회")
        print(f"매수 횟수: {result['buy_count']}회")
        print(f"매도 횟수: {result['sell_count']}회")
        print("\n거래 내역:")
        print("-"*80)
        
        total_profit = 0.0
        profitable_trades = 0
        losing_trades = 0
        
        for trade in trades:
            date_str = trade['date'].strftime('%Y-%m-%d') if isinstance(trade['date'], pd.Timestamp) else str(trade['date'])
            action = trade['action']
            price = trade['price']
            
            print(f"\n[{date_str}] {action}")
            print(f"  가격: {price:,.0f}원")
            
            if 'amount' in trade:
                print(f"  수량: {trade['amount']:.8f} BTC")
                print(f"  총액: {trade['total_value']:,.0f}원")
            
            if 'buy_price' in trade:
                buy_price = trade['buy_price']
                buy_date_str = trade['buy_date'].strftime('%Y-%m-%d') if isinstance(trade['buy_date'], pd.Timestamp) else str(trade['buy_date'])
                profit = trade['profit']
                profit_percent = trade['profit_percent']
                total_profit += profit
                
                print(f"  매수 가격: {buy_price:,.0f}원 (매수일: {buy_date_str})")
                print(f"  수익: {profit:,.0f}원 ({profit_percent:+.2f}%)")
                
                if profit > 0:
                    profitable_trades += 1
                else:
                    losing_trades += 1
            
            if 'total_asset' in trade:
                print(f"  거래 후 총 자산: {trade['total_asset']:,.0f}원")
            
            if 'angle' in trade and trade['angle'] is not None:
                print(f"  추세선 각도: {trade['angle']:.2f}°")
        
        # 거래 간격 계산
        trade_dates = []
        for trade in trades:
            if isinstance(trade['date'], pd.Timestamp):
                trade_dates.append(trade['date'])
            else:
                trade_dates.append(pd.to_datetime(trade['date']))
        
        trade_intervals = []  # 거래 간격 (일수)
        if len(trade_dates) > 1:
            # 거래 날짜를 정렬
            sorted_dates = sorted(trade_dates)
            for i in range(1, len(sorted_dates)):
                interval_days = (sorted_dates[i] - sorted_dates[i-1]).days
                trade_intervals.append(interval_days)
        
        # 평균 거래 간격 계산
        avg_interval = 0.0
        if len(trade_intervals) > 0:
            avg_interval = sum(trade_intervals) / len(trade_intervals)
        
        # 최소/최대 거래 간격
        min_interval = min(trade_intervals) if len(trade_intervals) > 0 else 0
        max_interval = max(trade_intervals) if len(trade_intervals) > 0 else 0
        
        print("\n" + "-"*80)
        print(f"총 수익: {total_profit:,.0f}원")
        print(f"수익 거래: {profitable_trades}회")
        print(f"손실 거래: {losing_trades}회")
        if profitable_trades + losing_trades > 0:
            win_rate = (profitable_trades / (profitable_trades + losing_trades)) * 100
            print(f"승률: {win_rate:.2f}%")
        
        print("\n거래 간격 분석:")
        if len(trade_intervals) > 0:
            print(f"  평균 거래 간격: {avg_interval:.1f}일")
            print(f"  최소 거래 간격: {min_interval}일")
            print(f"  최대 거래 간격: {max_interval}일")
            print(f"  거래 간격 개수: {len(trade_intervals)}개")
        else:
            print(f"  거래가 1회 이하여서 간격을 계산할 수 없습니다.")
        
        print("="*80)
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


def plot_backtest_results(df, result, save_path='backtest_result.png'):
    """
    백테스트 결과를 matplotlib으로 시각화
    
    Parameters:
    - df (pd.DataFrame): 추세선 계산이 완료된 OHLCV 데이터프레임
    - result (dict): backtest_strategy() 함수의 반환값
    - save_path (str): 그래프 저장 경로. 기본값 'backtest_result.png'
    """
    try:
        trades = result['trades']
        
        # 한글 폰트 설정 (Windows)
        plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        
        # 4개의 서브플롯 생성: 가격, 각도, 보유 구간, 누적 수익률
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        fig.suptitle('BTC 백테스트 결과', fontsize=16, fontweight='bold')
        
        dates = df.index
        close_prices = df['close'].values
        
        # 1. BTC 가격 차트 (종가)
        ax1 = axes[0]
        ax1.plot(dates, close_prices, label='BTC 종가', color='black', linewidth=1.5)
        
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
            ax1.scatter(buy_dates, buy_prices, color='red', marker='^', s=100, 
                       label='매수', zorder=5)
        if sell_dates:
            ax1.scatter(sell_dates, sell_prices, color='blue', marker='v', s=100, 
                       label='매도', zorder=5)
        
        ax1.set_ylabel('가격 (원)', fontsize=10)
        ax1.set_title('BTC 가격 및 매매 신호', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # 2. 추세선 각도 차트
        ax2 = axes[1]
        angles = df['angle'].values
        ax2.plot(dates, angles, label='추세선 각도', color='green', linewidth=1.5)
        ax2.axhline(y=5.4, color='red', linestyle='--', alpha=0.5, label='매수 기준선 (5.4°)')
        ax2.axhline(y=-16.7, color='blue', linestyle='--', alpha=0.5, label='매도 기준선 (-16.7°)')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.fill_between(dates, 0, angles, where=(angles >= 0), alpha=0.2, color='red', label='상승 추세')
        ax2.fill_between(dates, 0, angles, where=(angles < 0), alpha=0.2, color='blue', label='하락 추세')
        
        ax2.set_ylabel('각도 (°)', fontsize=10)
        ax2.set_title('추세선 각도', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. 보유 구간 표시
        ax3 = axes[2]
        ax3.plot(dates, close_prices, color='gray', linewidth=0.5, alpha=0.5)
        
        # 보유 구간 찾기
        holding_periods = []
        current_hold = None
        
        for trade in trades:
            if trade['action'].startswith('BUY'):
                if current_hold is None:
                    current_hold = {'start': trade['date'], 'buy_price': trade['price']}
            elif trade['action'].startswith('SELL'):
                if current_hold is not None:
                    current_hold['end'] = trade['date']
                    holding_periods.append(current_hold)
                    current_hold = None
        
        # 마지막 보유 중인 경우
        if current_hold is not None:
            current_hold['end'] = dates[-1]
            holding_periods.append(current_hold)
        
        # 보유 구간 색칠
        for idx, period in enumerate(holding_periods):
            start_idx = df.index.get_loc(period['start']) if period['start'] in df.index else None
            end_idx = df.index.get_loc(period['end']) if period['end'] in df.index else None
            
            if start_idx is not None and end_idx is not None:
                period_dates = dates[start_idx:end_idx+1]
                period_prices = close_prices[start_idx:end_idx+1]
                # 첫 번째 구간만 레이블 추가하여 중복 방지
                label = '보유 구간' if idx == 0 else None
                ax3.fill_between(period_dates, 0, period_prices, alpha=0.3, color='yellow', label=label)
        
        ax3.set_ylabel('가격 (원)', fontsize=10)
        ax3.set_title('보유 구간', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # 4. 누적 수익률 차트
        ax4 = axes[3]
        
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
            ax4.plot(cumulative_dates, cumulative_profits, marker='o', 
                    color='purple', linewidth=2, markersize=4, label='누적 수익')
            ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # 최고점과 최저점 표시
            if len(cumulative_profits) > 0:
                max_idx = np.argmax(cumulative_profits)
                min_idx = np.argmin(cumulative_profits)
                ax4.scatter([cumulative_dates[max_idx]], [cumulative_profits[max_idx]], 
                          color='green', marker='*', s=200, zorder=5, label='최고점')
                ax4.scatter([cumulative_dates[min_idx]], [cumulative_profits[min_idx]], 
                          color='red', marker='*', s=200, zorder=5, label='최저점')
        
        ax4.set_ylabel('누적 수익 (원)', fontsize=10)
        ax4.set_xlabel('날짜', fontsize=10)
        ax4.set_title('누적 수익률', fontsize=12, fontweight='bold')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # x축 날짜 포맷팅
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n그래프 저장 완료: {save_path}")
        plt.close()
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


def main(start_date='2014-01-01', end_date=None , buy_angle_threshold=5.4, 
         sell_angle_threshold=-16.7, stop_loss_percent=-5.0, min_sell_price=20000, 
         price_slippage=1000,
         aspect_ratio=3,
         window=13,
         initial_capital=1000000):
    """
    메인 함수
    
    Parameters:
    - start_date (str): 백테스트 시작 날짜 (YYYY-MM-DD 형식). 기본값 '2014-01-01'
    - end_date (str, optional): 백테스트 종료 날짜 (YYYY-MM-DD 형식). None이면 모든 데이터 사용
    """
    try:
        print("="*80)
        print("BTC 가격 데이터 수집 및 백테스트 시작")
        print("="*80)
        
        # 시작 날짜 파싱
        try:
            start_dt = pd.to_datetime(start_date)
        except Exception as e:
            print(f"오류: 시작 날짜 형식이 올바르지 않습니다: {start_date}")
            print(f"  올바른 형식: YYYY-MM-DD (예: 2014-01-01)")
            return
        
        # 종료 날짜 파싱
        end_dt = None
        if end_date is not None:
            try:
                end_dt = pd.to_datetime(end_date)
            except Exception as e:
                print(f"오류: 종료 날짜 형식이 올바르지 않습니다: {end_date}")
                print(f"  올바른 형식: YYYY-MM-DD (예: 2023-12-31)")
                return
        
        # 날짜 범위 검증
        if end_dt is not None and end_dt <= start_dt:
            print(f"오류: 종료 날짜({end_dt.strftime('%Y-%m-%d')})가 시작 날짜({start_dt.strftime('%Y-%m-%d')})보다 이전이거나 같습니다.")
            return
        
        print(f"백테스트 기간: {start_dt.strftime('%Y-%m-%d')}", end='')
        if end_dt is not None:
            print(f" ~ {end_dt.strftime('%Y-%m-%d')}")
        else:
            print(" ~ (마지막)")
        
        # 0. makeHistory.py를 이용하여 전체 기간의 BTC 가격 데이터 수집
        print("\n[단계 1] BTC 일봉 데이터 수집 중...")
        df = make_history(ticker='BTC', interval='24h', collect_all_periods=True)
        
        if df is None or len(df) == 0:
            print("오류: 데이터 수집 실패")
            return
        
        print(f"수집 완료: {len(df)}개 데이터")
        print(f"전체 기간: {df.index[0]} ~ {df.index[-1]}")
        
        # 날짜 범위로 데이터 필터링
        original_count = len(df)
        
        # 시작 날짜 필터링
        df = df[df.index >= start_dt]
        
        # 종료 날짜 필터링 (설정된 경우)
        if end_dt is not None:
            df = df[df.index <= end_dt]
        
        filtered_count = len(df)
        
        if len(df) == 0:
            if end_dt is not None:
                print(f"오류: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')} 기간의 데이터가 없습니다.")
            else:
                print(f"오류: 시작 날짜 {start_dt.strftime('%Y-%m-%d')} 이후의 데이터가 없습니다.")
            return
        
        print(f"필터링 후: {filtered_count}개 데이터 (제외: {original_count - filtered_count}개)")
        print(f"백테스트 기간: {df.index[0]} ~ {df.index[-1]}")
        
        # 추세선 계산
        print("\n[단계 2] 추세선 각도 계산 중...")
        df = calculate_trendline(df, window=window, aspect_ratio=aspect_ratio)
        
        # 백테스트 실행
        print("\n[단계 3] 백테스트 실행 중...")
        print("매수 조건: 추세선 각도 >= 5.4°")
        print("매도 조건: 추세선 각도 <= -16.7° (단, 매도가격 > 20,000원)")
        print("손절 조건: 수익률 <= -5%")
        print("거래 가격: 종가 ± 1000원")
        
        result = backtest_strategy(
            df, 
            buy_angle_threshold=buy_angle_threshold,
            sell_angle_threshold=sell_angle_threshold,
            stop_loss_percent=stop_loss_percent,
            min_sell_price=min_sell_price,
            price_slippage=price_slippage,
            initial_capital=initial_capital
        )
        
        # 결과 출력
        print_backtest_results(result)
        
        # 그래프 생성
        print("\n[단계 4] 백테스트 결과 그래프 생성 중...")
        plot_backtest_results(df, result, save_path='backtest_result.png')
        
        print("\n백테스트 완료!")
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


if __name__ == "__main__":
    # 시작 날짜 설정 (기본값: 2014-01-01)
    # 원하는 날짜로 변경 가능 (예: '2020-01-01', '2021-06-01')
    start_date = '2025-01-01'
    
    # 종료 날짜 설정 (None이면 마지막까지 사용)
    # 원하는 날짜로 변경 가능 (예: '2023-12-31', '2024-06-30')
    # None으로 설정하면 모든 데이터 사용
    end_date =   None#'2016-01-01'# 예: '2023-12-31'
    
    # 시작 잔고 설정 (기본값: 1,000,000원)
    initial_capital = 100_000  # 예: 10000000 (1천만원)
    
    main(start_date=start_date, end_date=end_date , 
    buy_angle_threshold=3.0, sell_angle_threshold=-4.0, 
    stop_loss_percent=-3.0, min_sell_price=20000, 
    price_slippage=1000 ,aspect_ratio=3,window=13,
    initial_capital=initial_capital)