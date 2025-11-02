"""
거래 상태 저장/로드 모듈
초기화 여부 및 실제 자산 상태 관리
"""
import traceback
import os
import json
import pandas as pd
from datetime import datetime


class TradeStateManager:
    """거래 상태 관리 클래스"""
    
    STATE_FILE = './trade_state.json'
    
    @classmethod
    def ensure_data_directory(cls):
        """데이터 디렉토리 생성"""
        try:
            dir_path = os.path.dirname(cls.STATE_FILE)
            # 루트 디렉토리('.')인 경우 디렉토리 생성 불필요
            if dir_path and dir_path != '.' and not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    @classmethod
    def load_state(cls):
        """
        저장된 상태 로드
        
        Returns:
        - dict: 상태 딕셔너리. None이면 저장된 상태 없음
        """
        try:
            cls.ensure_data_directory()
            
            if not os.path.exists(cls.STATE_FILE):
                return None
            
            with open(cls.STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # 날짜 필드를 datetime으로 변환
            if 'last_update' in state:
                state['last_update'] = pd.to_datetime(state['last_update'])
            
            return state
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    @classmethod
    def save_state(cls, state):
        """
        상태 저장
        
        Parameters:
        - state (dict): 저장할 상태 딕셔너리
        """
        try:
            cls.ensure_data_directory()
            
            # 복사본 생성 (날짜를 문자열로 변환)
            state_to_save = state.copy()
            if 'last_update' in state_to_save:
                if isinstance(state_to_save['last_update'], pd.Timestamp):
                    state_to_save['last_update'] = state_to_save['last_update'].isoformat()
                elif isinstance(state_to_save['last_update'], datetime):
                    state_to_save['last_update'] = state_to_save['last_update'].isoformat()
            
            with open(cls.STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state_to_save, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    @classmethod
    def clear_state(cls):
        """상태 초기화 (파일 삭제)"""
        try:
            if os.path.exists(cls.STATE_FILE):
                os.remove(cls.STATE_FILE)
                print("거래 상태 초기화 완료")
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    @classmethod
    def create_initial_state(cls, initial_capital, actual_cash, actual_coin_amount, ticker):
        """
        초기 상태 생성
        
        Parameters:
        - initial_capital (float): 초기 자본 (백테스트용)
        - actual_cash (float): 실제 계좌 현금 잔고
        - actual_coin_amount (float): 실제 계좌 코인 보유량
        - ticker (str): 암호화폐 티커
        
        Returns:
        - dict: 초기 상태 딕셔너리
        """
        try:
            state = {
                'initialized': True,
                'initial_capital': initial_capital,
                'actual_cash': actual_cash,
                'actual_coin_amount': actual_coin_amount,
                'ticker': ticker,
                'last_backtest_status': 'none',  # 'buy', 'sell', 'hold', 'none', 'wait'
                'last_update': pd.Timestamp.now(),
                'sync_needed': False  # 실제 자산과 백테스트 자산 동기화 필요 여부
            }
            return state
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise

