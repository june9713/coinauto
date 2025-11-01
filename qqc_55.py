"""
QQC 55개 데이터 분석 모듈
과거 55개의 데이터만 추출하여 현재 거래 상태와 결정 이유를 상세하게 추적
"""
import traceback
import datetime
import pandas as pd
from config import Config
from data_manager import DataManager
from qqc_test import QQCTestEngine
from condition_manager import ConditionManager


class LoggingQQCTestEngine(QQCTestEngine):
    """로깅 기능이 추가된 QQC 테스트 엔진"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_enabled = True
    
    def _check_buy_conditions(self, candle, current_candle_index):
        """
        매수 조건 확인 (상세 로그 추가)
        55개 데이터만 사용하는 경우를 고려하여 수정
        """
        try:
            # 버퍼에 최소 volume_window개가 있어야 함 (55개 데이터만 사용하는 경우 고려)
            # 원래 로직: volume_window+1개 필요 (volume_window개 평균 + 현재 캔들)
            # 수정 로직: volume_window개만 있어도 마지막 캔들 조건 확인 가능
            #           - 버퍼에 volume_window개가 있으면, 과거 volume_window-1개로 평균 계산 가능
            min_buffer_size_for_check = self.volume_window
            if len(self._candle_buffer) < min_buffer_size_for_check:
                if self._log_enabled:
                    print(f"\n[매수 조건 확인] {candle['date']} - 버퍼 데이터 부족 (필요: 최소 {min_buffer_size_for_check}개, 현재: {len(self._candle_buffer)}개)")
                return None
            
            # 인덱스 체크: volume_window-1개 이상이어야 과거 데이터로 평균 계산 가능
            # (55개 데이터의 경우, 마지막 캔들 인덱스 54에서 과거 54개 사용 가능)
            min_index_for_check = self.volume_window - 1
            if current_candle_index < min_index_for_check:
                if self._log_enabled:
                    print(f"\n[매수 조건 확인] {candle['date']} - 인덱스 부족 (필요: 최소 {min_index_for_check} 이상, 현재: {current_candle_index})")
                return None
            
            # 거래량 A 계산: 현재 캔들을 제외한 과거 데이터 사용
            # 버퍼 크기가 정확히 volume_window개인 경우(55개 데이터), 과거 volume_window-1개 사용
            # 버퍼 크기가 volume_window+1개 이상인 경우, 과거 volume_window개 사용
            available_past_data = min(current_candle_index, self.volume_window)
            if available_past_data < 1:
                if self._log_enabled:
                    print(f"\n[매수 조건 확인] {candle['date']} - 과거 데이터 부족 (과거 데이터: {available_past_data}개)")
                return None
            
            volume_window_start = max(0, current_candle_index - available_past_data)
            volume_window_end = current_candle_index - 1
            volumes = [self._candle_buffer[i]['volume'] for i in range(volume_window_start, volume_window_end + 1)]
            volume_a = sum(volumes) / len(volumes) if len(volumes) > 0 else 0.0
            
            # 조건 B: 현재 캔들의 거래량이 거래량 A의 배수 이상
            condition_b = candle['volume'] >= (volume_a * self.volume_multiplier)
            volume_threshold = volume_a * self.volume_multiplier
            
            # 이동평균 C 계산: 현재 캔들을 제외한 과거 데이터 사용
            min_ma_index = self.ma_window - 1
            if current_candle_index < min_ma_index:
                if self._log_enabled:
                    print(f"\n[매수 조건 확인] {candle['date']} - 이동평균 계산 불가 (필요: 최소 {min_ma_index} 이상, 현재: {current_candle_index})")
                return None
            
            available_ma_data = min(current_candle_index, self.ma_window)
            ma_window_start = max(0, current_candle_index - available_ma_data)
            ma_window_end = current_candle_index - 1
            closes = [self._candle_buffer[i]['close'] for i in range(ma_window_start, ma_window_end + 1)]
            ma_c = sum(closes) / len(closes) if len(closes) > 0 else 0.0
            
            # 조건 D: 현재 캔들의 종가가 이동평균 C보다 큼
            condition_d = candle['close'] > ma_c
            
            # 조건 E: 현재 캔들이 양봉 (오픈가 < 종가)
            condition_e = candle['open'] < candle['close']
            
            # 상세 로그 출력
            if self._log_enabled:
                print(f"\n{'='*80}")
                print(f"[매수 조건 확인] {candle['date']}")
                print(f"{'='*80}")
                print(f"[조건 B: 거래량 조건]")
                print(f"  현재 거래량: {candle['volume']:.8f} BTC")
                print(f"  과거 {len(volumes)}개 평균 거래량 (A): {volume_a:.8f} BTC (사용된 데이터: 인덱스 {volume_window_start}~{volume_window_end})")
                print(f"  거래량 임계값 (A * {self.volume_multiplier}): {volume_threshold:.8f} BTC")
                print(f"  조건 만족 여부: {condition_b} ({'✓ 만족' if condition_b else '✗ 불만족'})")
                print(f"[조건 D: 이동평균 조건]")
                print(f"  현재 종가: {candle['close']:,.0f}원")
                print(f"  과거 {len(closes)}개 이동평균 (C): {ma_c:,.0f}원 (사용된 데이터: 인덱스 {ma_window_start}~{ma_window_end})")
                print(f"  조건 만족 여부: {condition_d} ({'✓ 만족' if condition_d else '✗ 불만족'})")
                print(f"[조건 E: 양봉 조건]")
                print(f"  오픈가: {candle['open']:,.0f}원")
                print(f"  종가: {candle['close']:,.0f}원")
                print(f"  양봉 여부: {condition_e} ({'✓ 양봉' if condition_e else '✗ 음봉'})")
                print(f"[종합 판단]")
                all_conditions = condition_b and condition_d and condition_e
                print(f"  모든 조건 만족: {all_conditions} ({'✓ 매수 대기' if all_conditions else '✗ 매수 조건 불만족'})")
                if all_conditions:
                    print(f"  → 다음 캔들의 오픈가 + {self.price_slippage:,.0f}원으로 매수 예정")
            
            # 디버깅 통계 업데이트
            self._condition_check_count += 1
            if condition_b:
                self._condition_b_satisfied += 1
            if condition_d:
                self._condition_d_satisfied += 1
            if condition_e:
                self._condition_e_satisfied += 1
            if condition_b and condition_d and condition_e:
                self._all_conditions_satisfied += 1
            
            # B, D, E 모두 만족하면 다음 캔들의 오픈가+1000원으로 매수
            if condition_b and condition_d and condition_e:
                self._pending_buy = True
                self._pending_buy_condition_candle_index = current_candle_index
                return None
            
            return None
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _execute_buy_at_open(self, candle, current_candle_index):
        """매수 실행 (상세 로그 추가)"""
        try:
            buy_price = candle['open'] + self.price_slippage
            available_cash = self._cash * self.buy_cash_ratio
            self._coin_amount = available_cash / buy_price
            buy_total = self._coin_amount * buy_price
            
            if self._log_enabled:
                print(f"\n{'='*80}")
                print(f"[매수 실행] {candle['date']}")
                print(f"{'='*80}")
                print(f"  매수 가격: {buy_price:,.0f}원 (오픈가 {candle['open']:,.0f}원 + 슬리퍼지 {self.price_slippage:,.0f}원)")
                print(f"  매수 수량: {self._coin_amount:.8f} BTC")
                print(f"  매수 금액: {buy_total:,.0f}원 (현금의 {self.buy_cash_ratio*100:.0f}%)")
                print(f"  매수 전 현금: {self._cash + buy_total:,.0f}원")
                print(f"  매수 후 현금: {self._cash - buy_total:,.0f}원")
            
            # 현금 업데이트
            self._cash -= buy_total
            self._holding = True
            self._buy_price = buy_price
            self._buy_date = candle['date']
            self._buy_candle_index = current_candle_index
            self._buy_condition_candle_index = self._pending_buy_condition_candle_index
            
            # 자산 평가 업데이트
            coin_value = self._coin_amount * candle['close']
            self._total_asset = self._cash + coin_value
            
            # 거래량 A와 이동평균 C 계산
            prev_candle_index = current_candle_index - 1
            if prev_candle_index >= self.volume_window:
                volume_a = sum([self._candle_buffer[i]['volume'] for i in range(prev_candle_index - self.volume_window, prev_candle_index)]) / self.volume_window
            else:
                volume_a = 0.0
            
            if prev_candle_index >= self.ma_window:
                ma_c = sum([self._candle_buffer[i]['close'] for i in range(prev_candle_index - self.ma_window, prev_candle_index)]) / self.ma_window
            else:
                ma_c = 0.0
            
            trade = {
                'date': candle['date'],
                'action': 'BUY',
                'price': self._buy_price,
                'amount': self._coin_amount,
                'total_value': buy_total,
                'volume_a': volume_a,
                'ma_c': ma_c,
                'total_asset': self._total_asset
            }
            
            asset_history = {
                'date': candle['date'],
                'total_asset': self._total_asset,
                'cash': self._cash,
                'coin_value': coin_value,
                'holding': self._holding
            }
            
            return {
                'trade': trade,
                'asset_history': asset_history
            }
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _check_sell_conditions(self, candle, current_candle_index):
        """매도 조건 확인 (상세 로그 추가)"""
        try:
            if not self._holding or self._buy_candle_index is None or self._buy_condition_candle_index is None:
                return None
            
            # 현재 수익률 계산
            current_return = ((candle['close'] - self._buy_price) / self._buy_price) * 100
            
            # 조건 확인한 캔들(n) 기준으로 보유 기간 계산
            candles_passed = current_candle_index - self._buy_condition_candle_index
            
            # 상세 로그 출력
            if self._log_enabled:
                print(f"\n{'='*80}")
                print(f"[매도 조건 확인] {candle['date']}")
                print(f"{'='*80}")
                print(f"  매수 가격: {self._buy_price:,.0f}원")
                print(f"  매수 날짜: {self._buy_date}")
                print(f"  현재 종가: {candle['close']:,.0f}원")
                print(f"  현재 수익률: {current_return:.2f}%")
                print(f"  보유 기간: {candles_passed}개 캔들 경과 (조건 확인 기준: n+{self.hold_period} 캔들)")
                print(f"[매도 조건 확인]")
                print(f"  조건 1 (이익실현): 수익률 >= {self.profit_target}% → {current_return >= self.profit_target}")
                print(f"  조건 2 (손절): 수익률 <= {self.stop_loss}% → {current_return <= self.stop_loss}")
                print(f"  조건 3 (기간 만료): {candles_passed} >= {self.hold_period} → {candles_passed >= self.hold_period}")
            
            # 조건 1: 이익실현
            if current_return >= self.profit_target:
                if self._log_enabled:
                    print(f"  → 매도 결정: 이익실현 (수익률 {current_return:.2f}% >= {self.profit_target}%)")
                return self._execute_sell(candle, current_candle_index, 'SELL (이익실현)')
            
            # 조건 2: 손절
            if current_return <= self.stop_loss:
                if self._log_enabled:
                    print(f"  → 매도 결정: 손절 (수익률 {current_return:.2f}% <= {self.stop_loss}%)")
                return self._execute_sell(candle, current_candle_index, 'SELL (손절)')
            
            # 조건 3: 기간 만료
            if candles_passed >= self.hold_period:
                sell_price = candle['open']
                if self._log_enabled:
                    print(f"  → 매도 결정: 보유 기간 만료 (n+{self.hold_period} 캔들 경과, 오픈가 {sell_price:,.0f}원에서 매도)")
                return self._execute_sell_with_price(candle, current_candle_index, 'SELL (기간 만료)', sell_price)
            
            if self._log_enabled:
                print(f"  → 매도 조건 불만족, 보유 유지")
            
            return None
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _execute_sell_with_price(self, candle, current_candle_index, action_label, sell_price):
        """매도 실행 (상세 로그 추가)"""
        try:
            sell_amount = self._coin_amount * sell_price
            profit = sell_amount - (self._coin_amount * self._buy_price)
            profit_percent = (profit / (self._coin_amount * self._buy_price)) * 100
            
            if self._log_enabled:
                print(f"\n{'='*80}")
                print(f"[매도 실행] {candle['date']}")
                print(f"{'='*80}")
                print(f"  매도 사유: {action_label}")
                print(f"  매도 가격: {sell_price:,.0f}원")
                print(f"  매도 수량: {self._coin_amount:.8f} BTC")
                print(f"  매도 금액: {sell_amount:,.0f}원")
                print(f"  매수 가격: {self._buy_price:,.0f}원")
                print(f"  손익: {profit:,.0f}원 ({profit_percent:+.2f}%)")
                print(f"  매도 전 현금: {self._cash:,.0f}원")
                print(f"  매도 후 현금: {self._cash + sell_amount:,.0f}원")
            
            # 현금 업데이트
            self._cash += sell_amount
            
            trade = {
                'date': candle['date'],
                'action': action_label,
                'price': sell_price,
                'amount': self._coin_amount,
                'total_value': sell_amount,
                'buy_price': self._buy_price,
                'buy_date': self._buy_date,
                'profit': profit,
                'profit_percent': profit_percent,
                'total_asset': self._cash
            }
            
            # 상태 초기화
            self._holding = False
            coin_amount_sold = self._coin_amount
            self._coin_amount = 0.0
            self._buy_price = 0.0
            self._buy_date = None
            self._buy_candle_index = None
            self._buy_condition_candle_index = None
            
            self._total_asset = self._cash
            
            asset_history = {
                'date': candle['date'],
                'total_asset': self._total_asset,
                'cash': self._cash,
                'coin_value': 0.0,
                'holding': self._holding
            }
            
            return {
                'trade': trade,
                'asset_history': asset_history
            }
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise


def analyze_last_55_data(ticker='BTC', interval='3m',
                         volume_window=None, ma_window=None, volume_multiplier=None,
                         buy_cash_ratio=None, hold_period=None, profit_target=None,
                         stop_loss=None, initial_capital=None, price_slippage=None):
    """
    과거 55개 데이터만 추출하여 현재 거래 상태 분석
    
    Parameters:
    - ticker (str): 암호화폐 티커. 기본값 'BTC'
    - interval (str): 캔들스틱 간격. 기본값 '3m'
    - volume_window (int, optional): 거래량 평균 계산용 윈도우. None이면 기본값 55 사용
    - ma_window (int, optional): 이동평균 계산용 윈도우. None이면 기본값 9 사용
    - volume_multiplier (float, optional): 거래량 배수. None이면 기본값 1.4 사용
    - buy_cash_ratio (float, optional): 매수시 사용할 현금 비율. None이면 기본값 0.9 사용
    - hold_period (int, optional): 매수 후 보유 기간. None이면 기본값 15 사용
    - profit_target (float, optional): 이익실현 목표 수익률. None이면 기본값 17.6 사용
    - stop_loss (float, optional): 손절 기준 수익률. None이면 기본값 -28.6 사용
    - initial_capital (float, optional): 초기 자본. None이면 Config 기본값 사용
    - price_slippage (int, optional): 거래 가격 슬리퍼지. None이면 Config 기본값 사용
    
    Returns:
    - str: 현재 거래 상태 ('buy', 'sell', 'hold', 'none')
    """
    try:
        print("="*80)
        print("QQC 55개 데이터 분석 시작")
        print("="*80)
        
        # QQC 전략 변수 설정
        volume_window_val = volume_window if volume_window is not None else 55
        ma_window_val = ma_window if ma_window is not None else 9
        volume_multiplier_val = volume_multiplier if volume_multiplier is not None else 1.4
        buy_cash_ratio_val = buy_cash_ratio if buy_cash_ratio is not None else 0.9
        hold_period_val = hold_period if hold_period is not None else 15
        profit_target_val = profit_target if profit_target is not None else 17.6
        stop_loss_val = stop_loss if stop_loss is not None else -28.6
        
        initial_cap_val = initial_capital or Config.DEFAULT_INITIAL_CAPITAL
        price_slip_val = price_slippage or Config.DEFAULT_PRICE_SLIPPAGE
        
        print(f"\n[설정]")
        print(f"  티커: {ticker}")
        print(f"  간격: {interval}")
        print(f"  거래량 윈도우: {volume_window_val}")
        print(f"  이동평균 윈도우: {ma_window_val}")
        print(f"  거래량 배수: {volume_multiplier_val}")
        print(f"  매수 현금 비율: {buy_cash_ratio_val*100:.0f}%")
        print(f"  보유 기간: {hold_period_val} 캔들")
        print(f"  이익실현 목표: {profit_target_val}%")
        print(f"  손절 기준: {stop_loss_val}%")
        print(f"  초기 자본: {initial_cap_val:,.0f}원")
        print(f"  가격 슬리퍼지: {price_slip_val:,.0f}원")
        
        # 조건 딕셔너리 생성
        current_condition = ConditionManager.get_qqc_condition_key(
            volume_window=volume_window_val,
            ma_window=ma_window_val,
            volume_multiplier=volume_multiplier_val,
            buy_cash_ratio=buy_cash_ratio_val,
            hold_period=hold_period_val,
            profit_target=profit_target_val,
            stop_loss=stop_loss_val,
            price_slippage=price_slip_val,
            initial_capital=initial_cap_val,
            ticker=ticker,
            interval=interval
        )
        
        # 조건 기반 DataManager 초기화
        data_manager = DataManager(ticker=ticker, condition_dict=current_condition)
        
        # 데이터 로드
        print("\n[데이터 수집 중...]")
        df = data_manager.update_history_data(ticker=ticker, interval=interval, start_date='2014-01-01')
        
        if df is None or len(df) == 0:
            print("오류: 데이터 수집 실패")
            return 'none'
        
        print(f"전체 데이터 수: {len(df)}개")
        print(f"전체 기간: {df.index[0]} ~ {df.index[-1]}")
        
        # 마지막 57개를 추출한 후, 마지막 1개를 제외하여 56개 사용
        # 인덱스 0-54: 과거 평균용 (55개)
        # 인덱스 55: 현재 캔들
        # 인덱스 56: 제외 (실제 데이터의 마지막 캔들)
        # 안전하게 57개 이상이 있는지 확인
        required_count = volume_window_val + 2  # 57개 필요
        if len(df) < required_count:
            print(f"\n오류: 데이터가 부족합니다 (필요: 최소 {required_count}개, 현재: {len(df)}개)")
            return 'none'
        
        # 마지막 57개를 추출한 후, 마지막 1개를 제외하여 56개 사용
        df_last_57 = df.tail(required_count)
        df_last_56 = df_last_57.iloc[:-1]  # 마지막 인덱스 제외
        
        print(f"\n분석 대상 데이터: 마지막 {required_count}개 추출 후 마지막 1개 제외 → {len(df_last_56)}개 사용")
        print(f"분석 기간: {df_last_56.index[0]} ~ {df_last_56.index[-1]}")
        print(f"  - 인덱스 0-54: 과거 평균용 (55개)")
        print(f"  - 인덱스 55: 현재 캔들")
        print(f"  - 실제 데이터의 마지막 캔들(인덱스 56)은 제외됨")
        
        # QQC 백테스트 실행 (로깅 엔진 사용)
        print(f"\n[단계 1] QQC 백테스트 실행 중 (상세 로그 활성화)...")
        
        qqc_engine = LoggingQQCTestEngine(
            initial_capital=initial_cap_val,
            price_slippage=price_slip_val,
            volume_window=volume_window_val,
            ma_window=ma_window_val,
            volume_multiplier=volume_multiplier_val,
            buy_cash_ratio=buy_cash_ratio_val,
            hold_period=hold_period_val,
            profit_target=profit_target_val,
            stop_loss=stop_loss_val
        )
        
        result = qqc_engine.run(df_last_56)
        
        # 최종 상태 출력
        print(f"\n{'='*80}")
        print(f"[최종 거래 상태 분석 결과]")
        print(f"{'='*80}")
        
        last_status = result.get('last_trade_status', 'unknown')
        print(f"\n현재 거래 상태: {last_status.upper()}")
        
        if last_status == 'hold':
            print(f"\n[상태 설명] 보유 중 (HOLD)")
            print(f"  - 마지막 거래가 매수였으며, 아직 보유 중입니다.")
            if qqc_engine._holding:
                print(f"  - 매수 가격: {qqc_engine._buy_price:,.0f}원")
                print(f"  - 매수 날짜: {qqc_engine._buy_date}")
                if len(df_last_56) > 0:
                    last_candle = df_last_56.iloc[-1]
                    current_price = last_candle['close']
                    current_return = ((current_price - qqc_engine._buy_price) / qqc_engine._buy_price) * 100
                    print(f"  - 현재 가격: {current_price:,.0f}원")
                    print(f"  - 현재 수익률: {current_return:.2f}%")
                    if qqc_engine._buy_condition_candle_index is not None:
                        candles_passed = len(qqc_engine._candle_buffer) - 1 - qqc_engine._buy_condition_candle_index
                        print(f"  - 보유 기간: {candles_passed}개 캔들 경과 (기간 만료까지 {hold_period_val - candles_passed}개 남음)")
        elif last_status == 'buy':
            print(f"\n[상태 설명] 최근 매수 (BUY)")
            print(f"  - 마지막 거래가 매수였습니다.")
            if len(result.get('trades', [])) > 0:
                last_trade = [t for t in result['trades'] if t['action'].startswith('BUY')][-1]
                print(f"  - 매수 가격: {last_trade['price']:,.0f}원")
                print(f"  - 매수 날짜: {last_trade['date']}")
                print(f"  - 매수 수량: {last_trade['amount']:.8f} BTC")
        elif last_status == 'sell':
            print(f"\n[상태 설명] 최근 매도 (SELL)")
            print(f"  - 마지막 거래가 매도였습니다.")
            if len(result.get('trades', [])) > 0:
                last_trade = [t for t in result['trades'] if t['action'].startswith('SELL')][-1]
                print(f"  - 매도 가격: {last_trade['price']:,.0f}원")
                print(f"  - 매도 날짜: {last_trade['date']}")
                print(f"  - 매도 사유: {last_trade['action']}")
                print(f"  - 손익: {last_trade.get('profit', 0):,.0f}원 ({last_trade.get('profit_percent', 0):+.2f}%)")
        elif last_status == 'none':
            print(f"\n[상태 설명] 거래 없음 (NONE)")
            print(f"  - 분석 기간 동안 거래가 발생하지 않았습니다.")
            print(f"  - 모든 매수 조건을 만족하지 않았습니다.")
        
        print(f"\n[백테스트 통계]")
        print(f"  총 거래 횟수: {result.get('total_trades', 0)}회")
        print(f"  매수 횟수: {result.get('buy_count', 0)}회")
        print(f"  매도 횟수: {result.get('sell_count', 0)}회")
        print(f"  초기 자본: {result.get('initial_capital', 0):,.0f}원")
        print(f"  최종 자산: {result.get('final_asset', 0):,.0f}원")
        print(f"  총 수익률: {result.get('total_return', 0):+.2f}%")
        
        print(f"\n{'='*80}")
        print(f"분석 완료!")
        print(f"{'='*80}")
        
        return last_status
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


if __name__ == "__main__":
    # qqc_main.py와 동일한 설정 사용
    ticker = 'BTC'
    interval = '3m'
    price_slippage = 1000
    initial_capital = 1_000_000
    
    # QQC 전략 변수 설정
    volume_window = 55  # 거래량 평균 계산용 윈도우
    ma_window = 9  # 이동평균 계산용 윈도우
    volume_multiplier = 0.8  # 거래량 배수
    buy_cash_ratio = 0.9  # 매수시 사용할 현금 비율
    hold_period = 15  # 매수 후 보유 기간
    profit_target = 17.6  # 이익실현 목표 수익률
    stop_loss = -28.6  # 손절 기준 수익률
    
    # 분석 실행
    status = analyze_last_55_data(
        ticker=ticker,
        interval=interval,
        volume_window=volume_window,
        ma_window=ma_window,
        volume_multiplier=volume_multiplier,
        buy_cash_ratio=buy_cash_ratio,
        hold_period=hold_period,
        profit_target=profit_target,
        stop_loss=stop_loss,
        initial_capital=initial_capital,
        price_slippage=price_slippage
    )
    
    print(f"\n반환된 상태: {status}")

