"""
백테스트 엔진 모듈
백테스트 전략 실행 및 관리
"""
import traceback
import pandas as pd
from config import Config


class BacktestEngine:
    """백테스트 엔진 클래스"""
    
    def __init__(self, buy_angle_threshold=None, sell_angle_threshold=None,
                 stop_loss_percent=None, min_sell_price=None,
                 price_slippage=None, initial_capital=None):
        """
        초기화
        
        Parameters:
        - buy_angle_threshold (float): 매수 조건 (각도 >= 이 값). None이면 Config 기본값 사용
        - sell_angle_threshold (float): 매도 조건 (각도 <= 이 값). None이면 Config 기본값 사용
        - stop_loss_percent (float): 손절 기준 (수익률 %). None이면 Config 기본값 사용
        - min_sell_price (float): 최소 매도 가격 (원). None이면 Config 기본값 사용
        - price_slippage (int): 거래 가격 슬리퍼지 (원). None이면 Config 기본값 사용
        - initial_capital (float): 초기 자본 (원). None이면 Config 기본값 사용
        """
        self.buy_angle_threshold = buy_angle_threshold or Config.DEFAULT_BUY_ANGLE_THRESHOLD
        self.sell_angle_threshold = sell_angle_threshold or Config.DEFAULT_SELL_ANGLE_THRESHOLD
        self.stop_loss_percent = stop_loss_percent or Config.DEFAULT_STOP_LOSS_PERCENT
        self.min_sell_price = min_sell_price or Config.DEFAULT_MIN_SELL_PRICE
        self.price_slippage = price_slippage or Config.DEFAULT_PRICE_SLIPPAGE
        self.initial_capital = initial_capital or Config.DEFAULT_INITIAL_CAPITAL
        
        # 보유 상태 변수 (백테스트 실행 중 상태 관리)
        self._holding = False
        self._buy_price = 0.0
        self._buy_date = None
        self._coin_amount = 0.0
        self._cash = 0.0
        self._total_asset = 0.0
    
    def run(self, df):
        """
        백테스트 전략 실행
        
        Parameters:
        - df (pd.DataFrame): 추세선 계산이 완료된 OHLCV 데이터프레임
        
        Returns:
        - dict: 백테스트 결과 딕셔너리
        """
        try:
            # 상태 초기화
            self._holding = False
            self._buy_price = 0.0
            self._buy_date = None
            self._coin_amount = 0.0
            self._cash = self.initial_capital
            self._total_asset = self.initial_capital
            
            # 거래 기록
            trades = []
            asset_history = []
            
            # 각 날짜를 순회하면서 백테스트 (09:00에 전날 데이터로 판단)
            for i in range(len(df)):
                current_date = df.index[i]
                current_close = df.iloc[i]['close']
                current_angle = df.iloc[i]['angle']
                
                # 각도가 계산되지 않은 경우 (window 초기 기간) 건너뛰기
                if pd.isna(current_angle):
                    asset_history.append({
                        'date': current_date,
                        'total_asset': self._total_asset,
                        'cash': self._cash,
                        'coin_value': 0.0,
                        'holding': self._holding
                    })
                    continue
                
                # 09:00에 전날까지의 데이터로 판단하므로, 전날 종가 사용
                # 전날이 없는 경우 건너뛰기
                if i == 0:
                    asset_history.append({
                        'date': current_date,
                        'total_asset': self._total_asset,
                        'cash': self._cash,
                        'coin_value': 0.0,
                        'holding': self._holding
                    })
                    continue
                
                prev_close = df.iloc[i-1]['close']
                prev_angle = df.iloc[i-1]['angle']
                
                # 전날 각도도 계산되지 않은 경우 건너뛰기
                if pd.isna(prev_angle):
                    asset_history.append({
                        'date': current_date,
                        'total_asset': self._total_asset,
                        'cash': self._cash,
                        'coin_value': 0.0,
                        'holding': self._holding
                    })
                    continue
                
                # 현재 보유 상태에서 처리
                if self._holding:
                    # 보유 중인 경우 매도 조건 확인
                    result = self._check_sell_conditions(
                        current_date, prev_close, prev_angle, current_close
                    )
                    if result:
                        trades.append(result['trade'])
                        asset_history.append(result['asset_history'])
                        continue
                    
                    # 매도하지 않은 경우 자산 평가만 업데이트
                    coin_value = self._coin_amount * current_close
                    self._total_asset = self._cash + coin_value
                    asset_history.append({
                        'date': current_date,
                        'total_asset': self._total_asset,
                        'cash': self._cash,
                        'coin_value': coin_value,
                        'holding': self._holding
                    })
                else:
                    # 보유하지 않은 경우 매수 조건 확인
                    result = self._check_buy_conditions(
                        current_date, prev_close, prev_angle, current_close
                    )
                    if result:
                        trades.append(result['trade'])
                        asset_history.append(result['asset_history'])
                    else:
                        # 매수하지 않은 경우 자산 평가 업데이트
                        self._total_asset = self._cash
                        asset_history.append({
                            'date': current_date,
                            'total_asset': self._total_asset,
                            'cash': self._cash,
                            'coin_value': 0.0,
                            'holding': self._holding
                        })
            
            # 연속 시뮬레이션이므로 마지막 강제 매도 불필요
            # 보유 중인 경우 다음 백테스트에서 상태를 이어받을 수 있음
            # 최종 자산은 이미 보유 중이면 현금 + 코인 가치로 계산되어 있음
            
            # 최종 수익률 계산
            final_return = ((self._total_asset - self.initial_capital) / self.initial_capital) * 100
            
            # 마지막 거래 상태: hold(보유 중), buy(마지막 거래가 매수), sell(마지막 거래가 매도)
            if self._holding:
                # 보유 중이면 'hold'
                last_trade_status = 'hold'
            elif len(trades) > 0:
                # 보유 중이 아니면 마지막 거래 확인
                last_trade = trades[-1]
                if last_trade['action'].startswith('BUY'):
                    last_trade_status = 'buy'
                elif last_trade['action'].startswith('SELL'):
                    last_trade_status = 'sell'
                else:
                    last_trade_status = 'unknown'
            else:
                # 거래가 없으면
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
    
    def _check_sell_conditions(self, current_date, prev_close, prev_angle, current_close):
        """
        매도 조건 확인
        
        Parameters:
        - current_date: 현재 날짜
        - prev_close: 전날 종가
        - prev_angle: 전날 각도
        - current_close: 현재 종가
        
        Returns:
        - dict or None: 매도 시 거래 정보와 자산 기록, 매도하지 않으면 None
        """
        try:
            current_return = ((prev_close - self._buy_price) / self._buy_price) * 100
            
            # 매도 조건 1: 손절 (수익률 threshold 이하)
            if current_return <= self.stop_loss_percent:
                return self._execute_sell(
                    current_date, prev_close, prev_angle,
                    'SELL (손절)', current_close
                )
            
            # 매도 조건 2: 추세선 각도 <= threshold
            if prev_angle <= self.sell_angle_threshold:
                # 매도가격이 최소 가격 이하면 매도하지 않음
                if prev_close > self.min_sell_price:
                    return self._execute_sell(
                        current_date, prev_close, prev_angle,
                        'SELL (각도 신호)', current_close
                    )
                else:
                    # 매도가격이 최소 가격 이하인 경우 존버
                    coin_value = self._coin_amount * current_close
                    self._total_asset = self._cash + coin_value
                    return {
                        'trade': None,
                        'asset_history': {
                            'date': current_date,
                            'total_asset': self._total_asset,
                            'cash': self._cash,
                            'coin_value': coin_value,
                            'holding': self._holding
                        }
                    }
            
            return None
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _check_buy_conditions(self, current_date, prev_close, prev_angle, current_close):
        """
        매수 조건 확인
        
        Parameters:
        - current_date: 현재 날짜
        - prev_close: 전날 종가
        - prev_angle: 전날 각도
        - current_close: 현재 종가
        
        Returns:
        - dict or None: 매수 시 거래 정보와 자산 기록, 매수하지 않으면 None
        """
        try:
            # 매수 조건: 추세선 각도 >= threshold
            if prev_angle >= self.buy_angle_threshold:
                return self._execute_buy(
                    current_date, prev_close, prev_angle, current_close
                )
            
            return None
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _execute_buy(self, current_date, prev_close, prev_angle, current_close):
        """
        매수 실행
        
        Parameters:
        - current_date: 현재 날짜
        - prev_close: 전날 종가
        - prev_angle: 전날 각도
        - current_close: 현재 종가
        
        Returns:
        - dict: 거래 정보와 자산 기록
        """
        try:
            # 전날 종가로 매수 (종가 + 슬리퍼지)
            self._buy_price = prev_close + self.price_slippage
            self._buy_date = current_date
            
            # 보유 현금으로 최대한 매수
            self._coin_amount = self._cash / self._buy_price
            buy_total = self._coin_amount * self._buy_price
            
            # 현금 업데이트
            self._cash -= buy_total
            self._holding = True
            
            # 자산 평가 업데이트
            coin_value = self._coin_amount * current_close
            self._total_asset = self._cash + coin_value
            
            trade = {
                'date': current_date,
                'action': 'BUY',
                'price': self._buy_price,
                'amount': self._coin_amount,
                'total_value': buy_total,
                'angle': prev_angle,
                'total_asset': self._total_asset
            }
            
            asset_history = {
                'date': current_date,
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
    
    def _execute_sell(self, current_date, prev_close, prev_angle, action_label, current_close):
        """
        매도 실행
        
        Parameters:
        - current_date: 현재 날짜
        - prev_close: 전날 종가
        - prev_angle: 전날 각도
        - action_label: 행동 레이블
        - current_close: 현재 종가
        
        Returns:
        - dict: 거래 정보와 자산 기록
        """
        try:
            sell_price = prev_close - self.price_slippage
            sell_amount = self._coin_amount * sell_price
            profit = sell_amount - (self._coin_amount * self._buy_price)
            profit_percent = (profit / (self._coin_amount * self._buy_price)) * 100
            
            # 현금 업데이트
            self._cash += sell_amount
            
            trade = {
                'date': current_date,
                'action': action_label,
                'price': sell_price,
                'amount': self._coin_amount,
                'total_value': sell_amount,
                'buy_price': self._buy_price,
                'buy_date': self._buy_date,
                'profit': profit,
                'profit_percent': profit_percent,
                'angle': prev_angle,
                'total_asset': self._cash
            }
            
            # 상태 초기화
            self._holding = False
            coin_amount_old = self._coin_amount
            buy_price_old = self._buy_price
            buy_date_old = self._buy_date
            
            self._coin_amount = 0.0
            self._buy_price = 0.0
            self._buy_date = None
            
            self._total_asset = self._cash
            
            asset_history = {
                'date': current_date,
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
    
    def set_parameters(self, buy_angle_threshold=None, sell_angle_threshold=None,
                       stop_loss_percent=None, min_sell_price=None,
                       price_slippage=None, initial_capital=None):
        """
        백테스트 파라미터 설정
        
        Parameters:
        - buy_angle_threshold (float, optional): 매수 조건 각도
        - sell_angle_threshold (float, optional): 매도 조건 각도
        - stop_loss_percent (float, optional): 손절 기준
        - min_sell_price (float, optional): 최소 매도 가격
        - price_slippage (int, optional): 거래 가격 슬리퍼지
        - initial_capital (float, optional): 초기 자본
        """
        if buy_angle_threshold is not None:
            self.buy_angle_threshold = buy_angle_threshold
        if sell_angle_threshold is not None:
            self.sell_angle_threshold = sell_angle_threshold
        if stop_loss_percent is not None:
            self.stop_loss_percent = stop_loss_percent
        if min_sell_price is not None:
            self.min_sell_price = min_sell_price
        if price_slippage is not None:
            self.price_slippage = price_slippage
        if initial_capital is not None:
            self.initial_capital = initial_capital

