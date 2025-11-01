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
                 stop_loss=None):
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
        
        # 상태 변수
        self._cash = 0.0
        self._coin_amount = 0.0
        self._total_asset = 0.0
        self._holding = False
        self._buy_price = 0.0
        self._buy_date = None
        self._buy_candle_index = None  # 매수한 캔들의 인덱스 (n 캔들)
        
        # 캔들 버퍼 (실시간 처리용)
        # 최신 캔들을 포함하여 최대 56개까지 저장 (55개 평균 계산 + 최신 1개)
        self._candle_buffer = []
        
        # 매수 대기 플래그 (이전 캔들에서 조건 만족 시 다음 캔들의 오픈가로 매수)
        self._pending_buy = False
        
    def reset(self):
        """상태 초기화"""
        self._cash = self.initial_capital
        self._coin_amount = 0.0
        self._total_asset = self.initial_capital
        self._holding = False
        self._buy_price = 0.0
        self._buy_date = None
        self._buy_candle_index = None
        self._candle_buffer = []
        self._pending_buy = False
    
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
            
            # 버퍼 크기 제한 (최신 volume_window+1개만 유지: volume_window개 평균 계산용 + 최신 1개)
            buffer_limit = self.volume_window + 1
            if len(self._candle_buffer) > buffer_limit:
                self._candle_buffer.pop(0)
            
            # 현재 캔들 인덱스 (버퍼 내에서의 위치)
            current_candle_index = len(self._candle_buffer) - 1
            
            # 매수 대기 중이면 현재 캔들의 오픈가+1000원으로 매수
            if self._pending_buy and not self._holding:
                result = self._execute_buy_at_open(candle, current_candle_index)
                self._pending_buy = False
                if result:
                    return result
            
            # 보유 중인 경우 매도 조건 확인
            if self._holding:
                result = self._check_sell_conditions(candle, current_candle_index)
                if result:
                    return result
                
                # 매도하지 않은 경우 자산 평가만 업데이트
                coin_value = self._coin_amount * close_price
                self._total_asset = self._cash + coin_value
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
    
    def _check_buy_conditions(self, candle, current_candle_index):
        """
        매수 조건 확인
        
        Parameters:
        - candle (dict): 현재 캔들 데이터
        - current_candle_index (int): 현재 캔들 인덱스
        
        Returns:
        - dict or None: 매수 시 거래 정보와 자산 기록, 매수하지 않으면 None
        """
        try:
            # 최소 volume_window+1개 캔들이 필요 (volume_window개 평균 계산 + 최신 1개)
            min_buffer_size = self.volume_window + 1
            if len(self._candle_buffer) < min_buffer_size:
                return None
            
            # 이전 캔들 (현재 캔들 이전)로 조건 확인
            # 조건 확인은 이전 캔들(i-1)로 하고, 매수는 현재 캔들(i)의 오픈가로 함
            if current_candle_index == 0:
                return None
            
            prev_candle_index = current_candle_index - 1
            prev_candle = self._candle_buffer[prev_candle_index]
            
            # 거래량 A 계산: 이전 캔들 제외한 그 이전 volume_window개 캔들의 거래량 평균
            if prev_candle_index < self.volume_window:
                return None
            
            volume_window_start = prev_candle_index - self.volume_window
            volume_window_end = prev_candle_index - 1
            volumes = [self._candle_buffer[i]['volume'] for i in range(volume_window_start, volume_window_end + 1)]
            volume_a = sum(volumes) / len(volumes) if len(volumes) > 0 else 0.0
            
            # 조건 B: 이전 캔들의 거래량이 거래량 A의 1.4배 이상
            condition_b = prev_candle['volume'] >= (volume_a * self.volume_multiplier)
            
            # 이동평균 C 계산: 이전 캔들 제외한 그 이전 ma_window개 캔들의 종가 이동평균
            if prev_candle_index < self.ma_window:
                return None
            
            ma_window_start = prev_candle_index - self.ma_window
            ma_window_end = prev_candle_index - 1
            closes = [self._candle_buffer[i]['close'] for i in range(ma_window_start, ma_window_end + 1)]
            ma_c = sum(closes) / len(closes) if len(closes) > 0 else 0.0
            
            # 조건 D: 이전 캔들의 종가가 이동평균 C보다 큼
            condition_d = prev_candle['close'] > ma_c
            
            # 조건 E: 이전 캔들이 양봉 (오픈가 < 종가)
            condition_e = prev_candle['open'] < prev_candle['close']
            
            # B, D, E 모두 만족하면 다음 캔들의 오픈가+1000원으로 매수
            if condition_b and condition_d and condition_e:
                # 다음 캔들 추가 시 매수하도록 플래그 설정
                self._pending_buy = True
                # 현재는 거래 없이 자산 평가만 업데이트
                return None
            
            return None
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _execute_buy_at_open(self, candle, current_candle_index):
        """
        매수 실행 (다음 캔들의 오픈가+1000원)
        
        Parameters:
        - candle (dict): 현재 캔들 데이터 (매수 실행할 캔들)
        - current_candle_index (int): 현재 캔들 인덱스
        
        Returns:
        - dict: 거래 정보와 자산 기록
        """
        try:
            # 다음 오픈가+1000원으로 매수
            buy_price = candle['open'] + self.price_slippage
            
            # 현금의 0.9만큼만 매수
            available_cash = self._cash * self.buy_cash_ratio
            self._coin_amount = available_cash / buy_price
            buy_total = self._coin_amount * buy_price
            
            # 현금 업데이트
            self._cash -= buy_total
            self._holding = True
            self._buy_price = buy_price
            self._buy_date = candle['date']
            self._buy_candle_index = current_candle_index
            
            # 자산 평가 업데이트
            coin_value = self._coin_amount * candle['close']
            self._total_asset = self._cash + coin_value
            
            # 거래량 A와 이동평균 C 계산 (이전 캔들 기준)
            # 매수 조건은 이전 캔들에서 확인했으므로, 이전 캔들의 인덱스 사용
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
        """
        매도 조건 확인
        
        Parameters:
        - candle (dict): 현재 캔들 데이터
        - current_candle_index (int): 현재 캔들 인덱스
        
        Returns:
        - dict or None: 매도 시 거래 정보와 자산 기록, 매도하지 않으면 None
        """
        try:
            if not self._holding or self._buy_candle_index is None:
                return None
            
            # 현재 수익률 계산
            current_return = ((candle['close'] - self._buy_price) / self._buy_price) * 100
            
            # 조건 1: n+15 캔들이 되기 전에 수익률 17.6%면 매도
            if current_return >= self.profit_target:
                return self._execute_sell(candle, current_candle_index, 'SELL (이익실현)')
            
            # 조건 2: n+15 캔들이 되기 전에 수익률 -28.6%면 손절
            if current_return <= self.stop_loss:
                return self._execute_sell(candle, current_candle_index, 'SELL (손절)')
            
            # 조건 3: n+15 캔들이 되면 무조건 매도
            candles_passed = current_candle_index - self._buy_candle_index
            if candles_passed >= self.hold_period:
                # n+15 캔들의 오픈가에서 매도
                sell_price = candle['open']
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
                'buy_date': self._buy_date,
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
                
                # 캔들 추가 및 처리
                result = self.add_candle(candle_data)
                
                # 결과 기록
                if result['trade'] is not None:
                    trades.append(result['trade'])
                
                asset_history.append(result['asset_history'])
            
            # 최종 수익률 계산
            final_return = ((self._total_asset - self.initial_capital) / self.initial_capital) * 100
            
            # 마지막 거래 상태
            if self._holding:
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
                'last_trade_status': last_trade_status
            }
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise

