"""
실시간 모니터링을 위한 공유 상태 객체
FastAPI 서버와 백그라운드 프로세스 간 데이터 공유
"""
import traceback
import threading
from datetime import datetime
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np


class SharedState:
    """스레드 안전한 공유 상태 객체"""
    
    def __init__(self):
        """초기화"""
        self._lock = threading.Lock()
        self._reset()
    
    def _reset(self):
        """상태 초기화 (내부용)"""
        self.last_backtest_result: Optional[Dict[str, Any]] = None
        self.last_backtest_trades: List[Dict[str, Any]] = []
        self.last_backtest_time: Optional[datetime] = None
        self.current_status: Optional[str] = None  # 'buy', 'sell', 'hold', 'none', 'wait'
        self.trade_state: Optional[Dict[str, Any]] = None
        self.image_paths: Dict[str, str] = {
            'today': None,
            '3days': None,
            '5days': None
        }
        self.is_running: bool = False
        self.last_error: Optional[str] = None
        self.error_logs: List[Dict[str, Any]] = []  # 에러 로그 저장
        self.current_balance: Optional[Dict[str, Any]] = None  # 현재 잔고 (KRW, BTC)
        self.total_assets_history: List[Dict[str, Any]] = []  # 총 자산 변동 이력
        self.initial_capital: Optional[float] = None  # 초기 자산
    
    def update_backtest_result(self, result: Dict[str, Any], trades: List[Dict[str, Any]]):
        """
        백테스트 결과 업데이트 (중복 거래 제거)

        Parameters:
        - result: 백테스트 결과 딕셔너리
        - trades: 거래 리스트
        """
        try:
            with self._lock:
                self.last_backtest_result = result

                # 중복 제거: (date, action) 조합이 같은 거래는 한 번만 저장
                seen = set()
                unique_trades = []
                for trade in trades:
                    trade_key = (
                        str(trade.get('date')),
                        trade.get('action')
                    )
                    if trade_key not in seen:
                        unique_trades.append(trade)
                        seen.add(trade_key)

                # trades를 JSON 직렬화 가능한 형태로 변환
                self.last_backtest_trades = self._serialize_trades(unique_trades)
                self.last_backtest_time = datetime.now()
                if 'last_trade_status' in result:
                    self.current_status = result['last_trade_status']
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def update_trade_state(self, state: Dict[str, Any]):
        """
        거래 상태 업데이트
        
        Parameters:
        - state: 거래 상태 딕셔너리
        """
        try:
            with self._lock:
                # 날짜 필드를 문자열로 변환
                state_copy = state.copy()
                if 'last_update' in state_copy:
                    if isinstance(state_copy['last_update'], pd.Timestamp):
                        state_copy['last_update'] = state_copy['last_update'].isoformat()
                    elif isinstance(state_copy['last_update'], datetime):
                        state_copy['last_update'] = state_copy['last_update'].isoformat()
                self.trade_state = state_copy
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def update_image_paths(self, today: Optional[str] = None, 
                          days_3: Optional[str] = None, 
                          days_5: Optional[str] = None):
        """
        이미지 경로 업데이트
        
        Parameters:
        - today: 오늘 그래프 이미지 경로
        - days_3: 3일 그래프 이미지 경로
        - days_5: 5일 그래프 이미지 경로
        """
        try:
            with self._lock:
                if today is not None:
                    self.image_paths['today'] = today
                if days_3 is not None:
                    self.image_paths['3days'] = days_3
                if days_5 is not None:
                    self.image_paths['5days'] = days_5
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def set_running(self, is_running: bool):
        """
        실행 상태 설정
        
        Parameters:
        - is_running: 실행 여부
        """
        with self._lock:
            self.is_running = is_running
    
    def set_error(self, error: Optional[str]):
        """
        오류 설정

        Parameters:
        - error: 오류 메시지
        """
        with self._lock:
            self.last_error = error
            if error:
                # 에러 로그에 추가 (최대 100개까지만 보관)
                self.error_logs.append({
                    'timestamp': datetime.now().isoformat(),
                    'error': error
                })
                # 로그가 100개를 초과하면 오래된 것부터 삭제
                if len(self.error_logs) > 100:
                    self.error_logs = self.error_logs[-100:]
    
    def get_status(self) -> Dict[str, Any]:
        """
        현재 상태 조회 (스레드 안전)
        
        Returns:
        - dict: 현재 상태 딕셔너리
        """
        try:
            with self._lock:
                status_data = {
                    'is_running': self.is_running,
                    'last_backtest_time': self.last_backtest_time.isoformat() if self.last_backtest_time else None,
                    'current_status': self.current_status,
                    'trade_state': self.trade_state,
                    'last_error': self.last_error,
                    'image_paths': self.image_paths.copy()
                }
                
                # 백테스트 결과 요약
                if self.last_backtest_result:
                    result = self.last_backtest_result
                    status_data['backtest_summary'] = {
                        'initial_capital': result.get('initial_capital', 0),
                        'final_asset': result.get('final_asset', 0),
                        'total_return': result.get('total_return', 0),
                        'total_trades': result.get('total_trades', 0),
                        'buy_count': result.get('buy_count', 0),
                        'sell_count': result.get('sell_count', 0)
                    }
                else:
                    status_data['backtest_summary'] = None
                
                return status_data
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """
        거래 기록 조회 (스레드 안전)
        
        Returns:
        - list: 거래 기록 리스트
        """
        with self._lock:
            return self.last_backtest_trades.copy()
    
    def get_backtest_result(self) -> Optional[Dict[str, Any]]:
        """
        백테스트 결과 조회 (스레드 안전)

        Returns:
        - dict: 백테스트 결과 딕셔너리 (직렬화된 형태)
        """
        try:
            with self._lock:
                if self.last_backtest_result is None:
                    return None

                # 결과를 JSON 직렬화 가능한 형태로 변환
                result = self._serialize_dict(self.last_backtest_result)

                # trades는 이미 직렬화되어 있음
                result['trades'] = self.last_backtest_trades.copy()

                return result
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise

    def get_error_logs(self) -> List[Dict[str, Any]]:
        """
        에러 로그 조회 (스레드 안전)

        Returns:
        - list: 에러 로그 리스트 (최신 순)
        """
        with self._lock:
            # 최신 로그가 먼저 오도록 역순으로 반환
            return list(reversed(self.error_logs.copy()))

    def update_balance(self, krw: float, btc: float, btc_price: float):
        """
        현재 잔고 업데이트

        Parameters:
        - krw (float): KRW 잔고
        - btc (float): BTC 보유량
        - btc_price (float): 현재 BTC 가격
        """
        try:
            with self._lock:
                total_assets = krw + (btc * btc_price)
                self.current_balance = {
                    'krw': krw,
                    'btc': btc,
                    'btc_price': btc_price,
                    'total_assets': total_assets,
                    'timestamp': datetime.now().isoformat()
                }

                # 총 자산 변동 이력에 추가 (최대 1000개까지만 보관)
                self.total_assets_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'total_assets': total_assets,
                    'krw': krw,
                    'btc': btc,
                    'btc_price': btc_price
                })

                # 히스토리가 1000개를 초과하면 오래된 것부터 삭제
                if len(self.total_assets_history) > 1000:
                    self.total_assets_history = self.total_assets_history[-1000:]
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise

    def get_balance(self) -> Optional[Dict[str, Any]]:
        """
        현재 잔고 조회 (스레드 안전)

        Returns:
        - dict: 현재 잔고 정보
        """
        with self._lock:
            if self.current_balance is None:
                return None
            balance = self.current_balance.copy()
            # Add initial_capital to the balance response
            balance['initial_capital'] = self.initial_capital
            return balance

    def get_total_assets_history(self) -> List[Dict[str, Any]]:
        """
        총 자산 변동 이력 조회 (스레드 안전)

        Returns:
        - list: 총 자산 변동 이력 리스트
        """
        with self._lock:
            return self.total_assets_history.copy()

    def set_initial_capital(self, initial_capital: float):
        """
        초기 자산 설정

        Parameters:
        - initial_capital (float): 초기 자산
        """
        with self._lock:
            self.initial_capital = initial_capital

    def get_initial_capital(self) -> Optional[float]:
        """
        초기 자산 조회 (스레드 안전)

        Returns:
        - float: 초기 자산
        """
        with self._lock:
            return self.initial_capital
    
    def _serialize_value(self, value: Any) -> Any:
        """
        단일 값을 JSON 직렬화 가능한 형태로 변환

        Parameters:
        - value: 변환할 값

        Returns:
        - Any: 직렬화된 값
        """
        # None 값 먼저 체크
        if value is None:
            return None
        # Timestamp와 datetime을 문자열로 변환
        elif isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        # numpy 스칼라 타입을 파이썬 기본 타입으로 변환 (배열 체크보다 먼저 처리)
        elif isinstance(value, (np.generic, np.number)):
            try:
                if pd.isna(value):
                    return None
                # numpy 스칼라를 파이썬 기본 타입으로 변환
                return value.item() if hasattr(value, 'item') else float(value)
            except (ValueError, TypeError):
                return None
        # numpy array와 pandas Series를 리스트로 변환 (스칼라가 아닌 배열만)
        elif hasattr(value, '__array__') and not isinstance(value, (np.generic, np.number)):
            try:
                # iterable인지 확인 (배열인 경우)
                iter(value)
                return [self._serialize_value(item) for item in value]
            except TypeError:
                # iterable이 아닌 경우 스칼라로 처리
                try:
                    if pd.isna(value):
                        return None
                    return value.item() if hasattr(value, 'item') else float(value)
                except (ValueError, TypeError):
                    return None
        elif isinstance(value, (pd.Series, pd.Index)):
            return [self._serialize_value(item) for item in value]
        # 딕셔너리 재귀 처리
        elif isinstance(value, dict):
            return self._serialize_dict(value)
        # 리스트 재귀 처리
        elif isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        # 스칼라 NaN 값을 None으로 변환 (배열이 아닌 경우만)
        elif isinstance(value, (int, float, str, bool)):
            try:
                if pd.isna(value):
                    return None
            except (ValueError, TypeError):
                pass
            return value
        # 그 외는 그대로 반환
        else:
            return value

    def _serialize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        딕셔너리를 JSON 직렬화 가능한 형태로 변환 (재귀적)

        Parameters:
        - data: 변환할 딕셔너리

        Returns:
        - dict: 직렬화된 딕셔너리
        """
        result = {}
        for key, value in data.items():
            result[key] = self._serialize_value(value)
        return result

    def _serialize_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        거래 리스트를 JSON 직렬화 가능한 형태로 변환

        Parameters:
        - trades: 거래 리스트

        Returns:
        - list: 직렬화된 거래 리스트
        """
        try:
            return [self._serialize_dict(trade) for trade in trades]
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise


# 전역 공유 상태 객체
shared_state = SharedState()

