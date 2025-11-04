"""
QQC 백테스트 엔진 모듈
거래량 기반 매수/매도 전략 실행 및 관리
"""
import traceback
import pandas as pd
from config import Config


class QQCTestEngine:
    """QQC 백테스트 엔진 클래스"""
    
    def __init__(self, initial_capital=None, price_slippage=None,
                 volume_window=None, ma_window=None, volume_multiplier=None,
                 buy_cash_ratio=None, hold_period=None, profit_target=None,
                 stop_loss=None, initial_holding_state=None,
                 saved_buy_price=None, saved_buy_condition_date=None,
                 saved_buy_execution_date=None, interval=None):
        """
        초기화

        Parameters:
        - initial_capital (float): 초기 자본 (원). None이면 Config 기본값 사용
        - price_slippage (int): 거래 가격 슬리퍼지 (원). None이면 Config 기본값 사용
        - volume_window (int): 거래량 평균 계산용 윈도우. None이면 기본값 55 사용
        - ma_window (int): 이동평균 계산용 윈도우. None이면 기본값 9 사용
        - volume_multiplier (float): 거래량 배수. None이면 기본값 1.4 사용
        - buy_cash_ratio (float): 매수시 사용할 현금 비율 (0.0~1.0). None이면 기본값 0.9 사용
        - hold_period (int): 매수 후 보유 기간 (캔들 수). None이면 기본값 15 사용
        - profit_target (float): 이익실현 목표 수익률 (%). None이면 기본값 17.6 사용
        - stop_loss (float): 손절 기준 수익률 (%). None이면 기본값 -28.6 사용
        - initial_holding_state (bool): 초기 코인 보유 상태 (백테스트용). None이면 False 사용
        - saved_buy_price (float): 저장된 매수 가격. None이면 없음
        - saved_buy_condition_date (datetime): 저장된 매수 조건 확인 날짜. None이면 없음
        - saved_buy_execution_date (datetime): 저장된 매수 실행 날짜. None이면 없음
        - interval (str): 캔들스틱 간격 (예: '3m', '5m', '1h'). None이면 기본값 '3m' 사용
        """
        self.initial_capital = initial_capital or Config.DEFAULT_INITIAL_CAPITAL
        self.price_slippage = price_slippage or Config.DEFAULT_PRICE_SLIPPAGE

        # 백테스트 설정
        self.volume_window = volume_window if volume_window is not None else 55
        self.ma_window = ma_window if ma_window is not None else 9
        self.volume_multiplier = volume_multiplier if volume_multiplier is not None else 1.4
        self.buy_cash_ratio = buy_cash_ratio if buy_cash_ratio is not None else 0.9
        self.hold_period = hold_period if hold_period is not None else 15
        self.profit_target = profit_target if profit_target is not None else 17.6
        self.stop_loss = stop_loss if stop_loss is not None else -28.6

        # interval 설정 및 분 단위 변환
        self.interval = interval if interval is not None else '3m'
        self._interval_minutes = self._parse_interval_to_minutes(self.interval)

        # 초기 코인 보유 상태 (백테스트용, 파일에서 복원됨)
        self.initial_holding_state = initial_holding_state if initial_holding_state is not None else False

        # 저장된 매수 시점 정보 (15캔들 경과 후 매도를 위해 필요)
        self.saved_buy_price = saved_buy_price
        self.saved_buy_condition_date = saved_buy_condition_date
        self.saved_buy_execution_date = saved_buy_execution_date

        # 상태 변수
        self._cash = 0.0
        self._coin_amount = 0.0
        self._total_asset = 0.0
        self._holding = False
        self._buy_price = 0.0
        self._buy_date = None
        self._buy_candle_index = None  # 매수한 캔들의 인덱스 (n+1 캔들, 매수 실행 캔들)
        self._buy_condition_candle_index = None  # 조건 확인한 캔들의 인덱스 (n 캔들, 보유 기간 계산 기준)
        self._buy_condition_date = None  # 조건 확인한 캔들의 날짜 (n 캔들, 보유 기간 계산 기준 - 절대값)
        self._buy_execution_date = None  # 매수 실행한 캔들의 날짜 (n+1 캔들, 절대값)
        self._buy_condition_absolute_index = None  # 조건 확인한 캔들의 절대 인덱스 (n 캔들, df 내 절대 위치)

        # 캔들 버퍼 (실시간 처리용)
        # 최신 캔들을 포함하여 최대 volume_window+1개까지 저장 (volume_window개 평균 계산용 + 현재 캔들)
        self._candle_buffer = []

        # 디버깅 통계 (run 메서드에서 사용)
        self._condition_check_count = 0
        self._condition_b_satisfied = 0
        self._condition_d_satisfied = 0
        self._condition_e_satisfied = 0
        self._all_conditions_satisfied = 0

        # 전체 조건 B, D, E 통계 (보유 여부와 관계없이 모든 캔들에 대해 수집, 그래프와 동일)
        self._condition_b_all_check_count = 0
        self._condition_b_all_satisfied = 0
        self._condition_d_all_satisfied = 0
        self._condition_e_all_satisfied = 0
        self._condition_bde_all_satisfied = 0
        self._condition_d_all_satisfied = 0
        self._condition_e_all_satisfied = 0
        self._condition_bde_all_satisfied = 0

    def _parse_interval_to_minutes(self, interval):
        """
        interval 문자열을 분 단위로 변환

        Parameters:
        - interval (str): 간격 문자열 (예: '3m', '5m', '1h', '1d')

        Returns:
        - int: 분 단위 간격
        """
        try:
            if interval.endswith('m'):
                return int(interval[:-1])
            elif interval.endswith('h'):
                return int(interval[:-1]) * 60
            elif interval.endswith('d'):
                return int(interval[:-1]) * 60 * 24
            else:
                print(f"[경고] 알 수 없는 interval 형식: {interval}, 기본값 3분 사용")
                return 3
        except Exception as e:
            print(f"[경고] interval 파싱 실패: {interval}, 기본값 3분 사용")
            return 3

    def reset(self):
        """상태 초기화"""
        self._cash = self.initial_capital
        self._coin_amount = 0.0
        self._total_asset = self.initial_capital
        # 초기 보유 상태 적용 (파일에서 복원된 상태)
        self._holding = self.initial_holding_state
        self._buy_price = self.saved_buy_price if self.saved_buy_price is not None else 0.0

        # 초기 보유 상태일 때 코인 수량 계산
        if self._holding and self._buy_price > 0:
            # 초기 자본의 buy_cash_ratio만큼 사용하여 매수했다고 가정
            available_cash = self.initial_capital * self.buy_cash_ratio
            self._coin_amount = available_cash / self._buy_price
            self._cash = self.initial_capital - available_cash
            print(f"[초기 보유 상태 복원] 매수가: {self._buy_price:,.0f}원, 코인 수량: {self._coin_amount:.8f}, 남은 현금: {self._cash:,.0f}원")

        self._buy_date = None
        self._buy_candle_index = None
        self._buy_condition_candle_index = None
        self._buy_condition_date = self.saved_buy_condition_date
        self._buy_execution_date = self.saved_buy_execution_date
        self._buy_condition_absolute_index = None  # run 메서드에서 계산될 예정
        self._candle_buffer = []
        self._condition_check_count = 0
        self._condition_b_satisfied = 0
        self._condition_d_satisfied = 0
        self._condition_e_satisfied = 0
        self._all_conditions_satisfied = 0
        self._condition_b_all_check_count = 0
        self._condition_b_all_satisfied = 0
        self._condition_d_all_satisfied = 0
        self._condition_e_all_satisfied = 0
        self._condition_bde_all_satisfied = 0
    
    def add_candle(self, candle_data):
        """
        새로운 캔들 추가 및 처리
        
        Parameters:
        - candle_data (dict or pd.Series): 캔들 데이터
            - date (datetime or pd.Timestamp): 날짜
            - open (float): 시가
            - high (float): 고가
            - low (float): 저가
            - close (float): 종가
            - volume (float): 거래량
            
        Returns:
        - dict: 처리 결과
            - trade (dict or None): 거래 정보 (거래 발생시)
            - asset_history (dict): 자산 기록
        """
        try:
            # 캔들 데이터 정규화
            if isinstance(candle_data, pd.Series):
                date = candle_data.name if hasattr(candle_data, 'name') else candle_data.get('date')
                open_price = candle_data['open']
                high_price = candle_data['high']
                low_price = candle_data['low']
                close_price = candle_data['close']
                volume = candle_data['volume']
            else:
                date = candle_data['date']
                open_price = candle_data['open']
                high_price = candle_data['high']
                low_price = candle_data['low']
                close_price = candle_data['close']
                volume = candle_data['volume']
            
            # 캔들을 버퍼에 추가
            # 날짜를 초 단위까지 포함하도록 처리 (pd.to_datetime은 기본적으로 초 단위 포함)
            candle_date = pd.to_datetime(date)
            candle = {
                'date': candle_date,
                'open': float(open_price),
                'high': float(high_price),
                'low': float(low_price),
                'close': float(close_price),
                'volume': float(volume)
            }
            
            self._candle_buffer.append(candle)
            
            # 버퍼 크기 제한 (최신 volume_window+1개만 유지: volume_window개 평균 계산용 + 현재 캔들)
            buffer_limit = self.volume_window + 1
            if len(self._candle_buffer) > buffer_limit:
                self._candle_buffer.pop(0)
            
            # 현재 캔들 인덱스 (버퍼 내에서의 위치)
            current_candle_index = len(self._candle_buffer) - 1

            # 보유 여부와 관계없이 모든 캔들에 대해 조건 B, D, E 통계 수집 (디버깅용, 그래프와 동일)
            self._update_all_conditions_statistics(candle, current_candle_index)
            
            # 보유 중인 경우 매도 조건 확인
            if self._holding:
                result = self._check_sell_conditions(candle, current_candle_index, getattr(self, '_current_absolute_index', None))
                if result:
                    return result

                # 매도하지 않은 경우 현재 보유 상태 로깅 (디버깅용)
                if hasattr(self, '_buy_condition_date') and self._buy_condition_date is not None:
                    current_date = candle['date']
                    time_diff = current_date - self._buy_condition_date
                    if hasattr(self, '_interval_minutes') and self._interval_minutes > 0:
                        candles_passed = int(time_diff.total_seconds() / 60 / self._interval_minutes)
                        current_return = ((candle['close'] - self._buy_price) / self._buy_price) * 100
                        print(f"[보유 중] 경과: {candles_passed}/{self.hold_period}캔들, 수익률: {current_return:+.2f}%, 현재가: {candle['close']:,.0f}원")
                
                # 매도하지 않은 경우 자산 평가만 업데이트
                coin_value = self._coin_amount * close_price
                self._total_asset = self._cash + coin_value
                # 보유 중일 때는 매수 조건 확인하지 않음 (기존 로직 유지)
                return {
                    'trade': None,
                    'asset_history': {
                        'date': candle['date'],
                        'total_asset': self._total_asset,
                        'cash': self._cash,
                        'coin_value': coin_value,
                        'holding': self._holding
                    }
                }
            else:
                # 보유하지 않은 경우 매수 조건 확인
                result = self._check_buy_conditions(candle, current_candle_index)
                if result:
                    return result
                
                # 매수하지 않은 경우 자산 평가 업데이트
                self._total_asset = self._cash
                return {
                    'trade': None,
                    'asset_history': {
                        'date': candle['date'],
                        'total_asset': self._total_asset,
                        'cash': self._cash,
                        'coin_value': 0.0,
                        'holding': self._holding
                    }
                }
                
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _update_all_conditions_statistics(self, candle, current_candle_index):
        """
        조건 B, D, E 통계 업데이트 (보유 여부와 관계없이 모든 캔들에 대해 확인, 그래프와 동일한 로직)
        
        Parameters:
        - candle (dict): 현재 캔들 데이터
        - current_candle_index (int): 현재 캔들 인덱스
        """
        try:
            # 최소 데이터가 필요함
            if len(self._candle_buffer) < 2:
                return
            
            # 버퍼 내에서 조건을 확인할 수 있는 범위인지 체크
            if current_candle_index < 1:
                return
            
            # 조건 B: 거래량 평균 계산 (그래프와 동일한 로직)
            if current_candle_index < self.volume_window:
                # 초기 구간: 현재까지의 모든 데이터로 평균 계산
                if current_candle_index == 0:
                    volume_avg = 0.0
                else:
                    volumes = [self._candle_buffer[i]['volume'] for i in range(0, current_candle_index)]
                    volume_avg = sum(volumes) / len(volumes) if len(volumes) > 0 else 0.0
            else:
                # 정상 구간: volume_window개 데이터로 평균 계산
                volume_window_start = current_candle_index - self.volume_window
                volume_window_end = current_candle_index - 1
                volumes = [self._candle_buffer[i]['volume'] for i in range(volume_window_start, volume_window_end + 1)]
                volume_avg = sum(volumes) / len(volumes) if len(volumes) > 0 else 0.0
            
            # 조건 B 확인
            condition_b = False
            if volume_avg > 0:
                self._condition_b_all_check_count += 1
                condition_b = candle['volume'] >= (volume_avg * self.volume_multiplier)
                if condition_b:
                    self._condition_b_all_satisfied += 1
            
            # 조건 D: 이동평균 계산 (그래프와 동일한 로직)
            condition_d = False
            if current_candle_index < self.ma_window:
                # 초기 구간: 현재까지의 모든 데이터로 평균 계산
                if current_candle_index == 0:
                    ma_c = candle['close']
                else:
                    closes = [self._candle_buffer[i]['close'] for i in range(0, current_candle_index)]
                    ma_c = sum(closes) / len(closes) if len(closes) > 0 else candle['close']
                # 조건 D: 현재 종가 > 이동평균
                condition_d = candle['close'] > ma_c
            else:
                # 정상 구간: ma_window개 데이터로 평균 계산
                ma_window_start = current_candle_index - self.ma_window
                ma_window_end = current_candle_index - 1
                closes = [self._candle_buffer[i]['close'] for i in range(ma_window_start, ma_window_end + 1)]
                ma_c = sum(closes) / len(closes) if len(closes) > 0 else candle['close']
                # 조건 D: 현재 종가 > 이동평균
                condition_d = candle['close'] > ma_c
            
            if condition_d:
                self._condition_d_all_satisfied += 1
            
            # 조건 E: 양봉 (오픈가 < 종가)
            condition_e = candle['open'] < candle['close']
            if condition_e:
                self._condition_e_all_satisfied += 1
            
            # B, D, E 모두 만족
            if condition_b and condition_d and condition_e:
                self._condition_bde_all_satisfied += 1
                
        except Exception as e:
            # 통계 수집 실패는 무시 (디버깅용이므로)
            pass
    
    def _check_buy_conditions(self, candle, current_candle_index):
        """
        매수 조건 확인 및 즉시 매수 실행

        Parameters:
        - candle (dict): 현재 캔들 데이터
        - current_candle_index (int): 현재 캔들 인덱스

        Returns:
        - dict or None: 매수 시 거래 정보와 자산 기록, 매수하지 않으면 None
        """
        try:
            # 필수 조건: 코인을 보유하지 않고 있어야 함 (중복 매수 방지)
            if self._holding:
                return None

            # 최소 volume_window+1개 캔들이 필요 (volume_window개 평균 계산용 + 현재 캔들)
            min_buffer_size = self.volume_window + 1
            if len(self._candle_buffer) < min_buffer_size:
                return None

            # 현재 캔들로 조건 확인하고, 조건 만족 시 즉시 매수
            if current_candle_index < self.volume_window:
                return None
            
            # 거래량 A 계산: 현재 캔들을 제외한 최신 volume_window개 캔들의 거래량 평균
            volume_window_start = current_candle_index - self.volume_window
            volume_window_end = current_candle_index - 1
            volumes = [self._candle_buffer[i]['volume'] for i in range(volume_window_start, volume_window_end + 1)]
            volume_a = sum(volumes) / len(volumes) if len(volumes) > 0 else 0.0
            
            # 조건 B: 현재 캔들의 거래량이 거래량 A의 1.4배 이상
            condition_b = candle['volume'] >= (volume_a * self.volume_multiplier)
            
            # 이동평균 C 계산: 현재 캔들을 제외한 최신 ma_window개 캔들의 종가 이동평균
            if current_candle_index < self.ma_window:
                return None
            
            ma_window_start = current_candle_index - self.ma_window
            ma_window_end = current_candle_index - 1
            closes = [self._candle_buffer[i]['close'] for i in range(ma_window_start, ma_window_end + 1)]
            ma_c = sum(closes) / len(closes) if len(closes) > 0 else 0.0
            
            # 조건 D: 현재 캔들의 종가가 이동평균 C보다 큼
            condition_d = candle['close'] > ma_c
            
            # 조건 E: 현재 캔들이 양봉 (오픈가 < 종가)
            condition_e = candle['open'] < candle['close']
            
            # 디버깅: 조건 확인 통계 업데이트
            self._condition_check_count += 1
            if condition_b:
                self._condition_b_satisfied += 1
            if condition_d:
                self._condition_d_satisfied += 1
            if condition_e:
                self._condition_e_satisfied += 1
            if condition_b and condition_d and condition_e:
                self._all_conditions_satisfied += 1
            
            # B, D, E 모두 만족하면 현재 캔들의 종가+슬리피지로 즉시 매수
            if condition_b and condition_d and condition_e:
                # 즉시 매수 실행 (현재 캔들의 종가 + 슬리피지)
                buy_price = candle['close'] + self.price_slippage

                # 현금의 buy_cash_ratio만큼 매수
                available_cash = self._cash * self.buy_cash_ratio
                self._coin_amount = available_cash / buy_price
                buy_total = self._coin_amount * buy_price

                # 현금 업데이트
                self._cash -= buy_total
                self._holding = True
                self._buy_price = buy_price
                self._buy_date = candle['date']
                self._buy_candle_index = current_candle_index
                self._buy_condition_candle_index = current_candle_index
                self._buy_condition_date = candle['date']
                self._buy_execution_date = candle['date']
                self._buy_condition_absolute_index = getattr(self, '_current_absolute_index', None)

                # 자산 평가 업데이트
                coin_value = self._coin_amount * candle['close']
                self._total_asset = self._cash + coin_value

                trade = {
                    'date': candle['date'],
                    'action': 'BUY',
                    'price': buy_price,
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
            
            return None

        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise

    def _check_sell_conditions(self, candle, current_candle_index, current_absolute_index):
        """
        매도 조건 확인

        Parameters:
        - candle (dict): 현재 캔들 데이터
        - current_candle_index (int): 현재 캔들 인덱스 (버퍼 내 상대값)
        - current_absolute_index (int): 현재 캔들 절대 인덱스 (df 내 절대 위치)

        Returns:
        - dict or None: 매도 시 거래 정보와 자산 기록, 매도하지 않으면 None
        """
        try:
            if not self._holding:
                return None

            # 현재 수익률 계산
            current_return = ((candle['close'] - self._buy_price) / self._buy_price) * 100

            # 조건 확인한 캔들(n) 기준으로 보유 기간 계산
            candles_passed = None

            # 방법 1: 매수 조건 날짜로 계산 (모든 모드에서 정확하게 동작, 시분초 기반)
            if self._buy_condition_date is not None:
                # 현재 캔들 날짜와 매수 조건 확인 날짜의 시간 차이를 이용하여 경과 캔들 계산
                current_date = candle['date']
                buy_date = self._buy_condition_date

                # 시간 차이 계산 (초 단위)
                time_diff = current_date - buy_date

                # interval에 따라 경과 캔들 수 계산
                # 예: 3m interval이면 180초 = 1캔들
                if hasattr(self, '_interval_minutes') and self._interval_minutes > 0:
                    candles_passed = int(time_diff.total_seconds() / 60 / self._interval_minutes)
                else:
                    # interval 정보가 없으면 경고하고 기본값 사용
                    print(f"\n[경고] interval 정보가 없어 보유 기간을 정확히 계산할 수 없습니다.")
                    return None

            # 방법 2: 절대 인덱스로 계산 (백업용, run() 메서드 내에서만 유효)
            elif current_absolute_index is not None and self._buy_condition_absolute_index is not None:
                candles_passed = current_absolute_index - self._buy_condition_absolute_index

            else:
                # 매수 시점 정보가 없으면 매도 불가
                print(f"\n[경고] 매수 시점 정보가 없어 보유 기간을 계산할 수 없습니다.")
                return None

            # 조건 1: n+15 캔들이 되기 전에 수익률 17.6%면 매도
            if current_return >= self.profit_target:
                return self._execute_sell(candle, current_candle_index, 'SELL (이익실현)')

            # 조건 2: n+15 캔들이 되기 전에 수익률 -28.6%면 손절
            if current_return <= self.stop_loss:
                return self._execute_sell(candle, current_candle_index, 'SELL (손절)')

            # 조건 3: n+15 캔들이 되면 무조건 매도 (조건 확인한 캔들 n 기준으로 15개 캔들 경과)
            if candles_passed is not None and candles_passed >= self.hold_period:
                # n+15 캔들의 오픈가에서 매도
                sell_price = candle['open']
                print(f"\n[매도 조건 만족] 보유 기간 만료")
                print(f"  매수 조건 날짜: {self._buy_condition_date}")
                print(f"  현재 캔들 날짜: {candle['date']}")
                print(f"  경과 캔들 수: {candles_passed}개 (목표: {self.hold_period}개)")
                print(f"  매도 가격: {sell_price:,.0f}원 (오픈가)")
                return self._execute_sell_with_price(candle, current_candle_index, 'SELL (기간 만료)', sell_price)

            return None

        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _execute_sell(self, candle, current_candle_index, action_label):
        """
        매도 실행 (종가 기준)
        
        Parameters:
        - candle (dict): 현재 캔들 데이터
        - current_candle_index (int): 현재 캔들 인덱스
        - action_label (str): 행동 레이블
        
        Returns:
        - dict: 거래 정보와 자산 기록
        """
        sell_price = candle['close'] - self.price_slippage
        return self._execute_sell_with_price(candle, current_candle_index, action_label, sell_price)
    
    def _execute_sell_with_price(self, candle, current_candle_index, action_label, sell_price):
        """
        매도 실행 (지정 가격)
        
        Parameters:
        - candle (dict): 현재 캔들 데이터
        - current_candle_index (int): 현재 캔들 인덱스
        - action_label (str): 행동 레이블
        - sell_price (float): 매도 가격
        
        Returns:
        - dict: 거래 정보와 자산 기록
        """
        try:
            # 상태 검증
            if self._coin_amount == 0 or self._buy_price == 0:
                print(f"경고: 잘못된 매도 시도 - coin_amount: {self._coin_amount}, buy_price: {self._buy_price}")
                print(f"  holding: {self._holding}, 매도 시도 날짜: {candle['date']}")
                # 상태 초기화
                self._holding = False
                self._coin_amount = 0.0
                self._buy_price = 0.0
                self._buy_date = None
                self._buy_condition_date = None
                return None

            sell_amount = self._coin_amount * sell_price
            profit = sell_amount - (self._coin_amount * self._buy_price)
            profit_percent = (profit / (self._coin_amount * self._buy_price)) * 100
            
            # 현금 업데이트
            self._cash += sell_amount
            
            trade = {
                'date': candle['date'],
                'action': action_label,
                'price': sell_price,
                'amount': self._coin_amount,
                'total_value': sell_amount,
                'buy_price': self._buy_price,
                'buy_date': self._buy_condition_date if self._buy_condition_date is not None else self._buy_date,  # 저장된 매수 조건 날짜 사용
                'profit': profit,
                'profit_percent': profit_percent,
                'total_asset': self._cash
            }
            
            # 상태 초기화
            self._holding = False
            self._coin_amount = 0.0
            self._buy_price = 0.0
            self._buy_date = None
            self._buy_candle_index = None
            self._buy_condition_candle_index = None
            self._buy_condition_date = None
            self._buy_execution_date = None
            
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
    
    def run(self, df):
        """
        백테스트 전략 실행 (일괄 처리)

        Parameters:
        - df (pd.DataFrame): OHLCV 데이터프레임

        Returns:
        - dict: 백테스트 결과 딕셔너리
        """
        try:
            # 상태 초기화
            self.reset()

            # 거래 기록
            trades = []
            asset_history = []

            # 디버깅: 매수 조건 확인 통계 초기화
            self._condition_check_count = 0
            self._condition_b_satisfied = 0
            self._condition_d_satisfied = 0
            self._condition_e_satisfied = 0
            self._all_conditions_satisfied = 0
            self._condition_b_all_check_count = 0
            self._condition_b_all_satisfied = 0

            # 거래량 평균과 이동평균 계산 (CSV 저장용)
            # pandas rolling을 사용하여 벡터화된 계산
            volume_avg_series = df['volume'].shift(1).rolling(window=self.volume_window, min_periods=1).mean()
            ma_series = df['close'].shift(1).rolling(window=self.ma_window, min_periods=1).mean()

            # 첫 번째 값 설정
            if len(df) > 0:
                volume_avg_series.iloc[0] = 0.0
                ma_series.iloc[0] = df['close'].iloc[0]

            # 저장된 매수 조건 날짜가 있으면 절대 인덱스 계산
            if self._buy_condition_date is not None:
                # df 인덱스에서 매수 조건 날짜와 일치하는 인덱스 찾기
                matching_indices = df.index == self._buy_condition_date
                if matching_indices.any():
                    self._buy_condition_absolute_index = matching_indices.argmax()
                    print(f"\n[매수 시점 복원] 매수 조건 확인 날짜: {self._buy_condition_date}")
                    print(f"  복원된 절대 인덱스: {self._buy_condition_absolute_index}")
                    print(f"  현재 백테스트 데이터 범위: {df.index[0]} ~ {df.index[-1]}")
                else:
                    print(f"\n[경고] 저장된 매수 조건 날짜({self._buy_condition_date})가 현재 백테스트 데이터 범위에 없습니다.")
                    print(f"  현재 백테스트 데이터 범위: {df.index[0]} ~ {df.index[-1]}")
                    print(f"  매도 조건 확인을 위해 첫 번째 캔들을 매수 시점으로 간주합니다.")
                    self._buy_condition_absolute_index = 0

            # 각 캔들을 하나씩 처리
            for idx in range(len(df)):
                candle_row = df.iloc[idx]

                # 캔들 데이터 생성
                candle_data = {
                    'date': candle_row.name if hasattr(candle_row, 'name') else df.index[idx],
                    'open': candle_row['open'],
                    'high': candle_row['high'],
                    'low': candle_row['low'],
                    'close': candle_row['close'],
                    'volume': candle_row['volume']
                }

                # 절대 인덱스 저장 (보유 기간 계산용)
                self._current_absolute_index = idx

                # 캔들 추가 및 처리
                result = self.add_candle(candle_data)

                # 결과 기록
                if result['trade'] is not None:
                    trades.append(result['trade'])

                asset_history.append(result['asset_history'])

            # 최종 수익률 계산
            final_return = ((self._total_asset - self.initial_capital) / self.initial_capital) * 100
            
            # 디버깅 정보 출력
            print(f"\n[디버깅] 매수 조건 확인 통계:")
            print(f"  총 캔들 수: {len(df)}개")
            print(f"  매수 조건 확인 가능한 캔들: {self._condition_check_count}개 (최소 {self.volume_window + 1}개 필요, 보유하지 않은 경우만)")
            if self._condition_check_count > 0:
                print(f"  조건 B 만족 (거래량 >= 평균 * {self.volume_multiplier}): {self._condition_b_satisfied}회 (백테스트 조건 확인 가능 범위 내, 보유하지 않은 경우만)")
                print(f"  조건 D 만족 (종가 > 이동평균): {self._condition_d_satisfied}회")
                print(f"  조건 E 만족 (양봉): {self._condition_e_satisfied}회")
                print(f"  모든 조건 만족 (B & D & E): {self._all_conditions_satisfied}회")
            print(f"  조건 B 확인 가능한 캔들 (모든 상태): {self._condition_b_all_check_count}개")
            print(f"  조건 B 만족 (거래량 >= 평균 * {self.volume_multiplier}): {self._condition_b_all_satisfied}회 (모든 캔들 기준, 그래프와 동일)")
            print(f"  조건 D 만족 (종가 > 이동평균): {self._condition_d_all_satisfied}회 (모든 캔들 기준, 그래프와 동일)")
            print(f"  조건 E 만족 (양봉): {self._condition_e_all_satisfied}회 (모든 캔들 기준, 그래프와 동일)")
            print(f"  모든 조건 만족 (B & D & E): {self._condition_bde_all_satisfied}회 (모든 캔들 기준, 그래프와 동일)")
            print(f"  실제 매수 발생: {len(trades)}회")
            
            # 마지막 거래 상태
            if self._holding:
                # 마지막 캔들에서 매수가 실행된 경우 'buy', 그 외에는 'hold'
                if len(trades) > 0 and trades[-1]['action'].startswith('BUY') and len(df) > 0 and trades[-1]['date'] == df.index[-1]:
                    last_trade_status = 'buy'
                else:
                    last_trade_status = 'hold'
            elif len(trades) > 0:
                last_trade = trades[-1]
                if last_trade['action'].startswith('BUY'):
                    last_trade_status = 'buy'
                elif last_trade['action'].startswith('SELL'):
                    last_trade_status = 'sell'
                else:
                    last_trade_status = 'unknown'
            else:
                last_trade_status = 'none'
            
            return {
                'trades': trades,
                'asset_history': asset_history,
                'total_trades': len(trades),
                'buy_count': len([t for t in trades if t['action'].startswith('BUY')]),
                'sell_count': len([t for t in trades if t['action'].startswith('SELL')]),
                'initial_capital': self.initial_capital,
                'final_asset': self._total_asset,
                'total_return': final_return,
                'last_trade_status': last_trade_status,
                'volume_avg': volume_avg_series,  # 거래량 평균 (CSV 저장용)
                'ma_values': ma_series  # 이동평균 (CSV 저장용)
            }
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise

