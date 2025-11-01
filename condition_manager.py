"""
조건 관리 모듈
백테스트 조건 저장/로드 및 파일명 생성
"""
import traceback
import json
import os
import hashlib


class ConditionManager:
    """백테스트 조건 관리 클래스"""
    
    # 순환 참조 방지를 위해 DATA_DIR을 직접 정의 (Config와 동일한 값 사용)
    DATA_DIR = './datas'
    CONDITION_FILE = os.path.join(DATA_DIR, 'backtest_conditions.json')
    
    @staticmethod
    def get_condition_key(buy_angle_threshold=None, sell_angle_threshold=None,
                          stop_loss_percent=None, min_sell_price=None,
                          price_slippage=None, window=None, aspect_ratio=None,
                          initial_capital=None, ticker='BTC', interval='24h'):
        """
        조건값들을 딕셔너리로 반환
        
        Parameters:
        - buy_angle_threshold (float, optional): 매수 조건 각도
        - sell_angle_threshold (float, optional): 매도 조건 각도
        - stop_loss_percent (float, optional): 손절 기준
        - min_sell_price (float, optional): 최소 매도 가격
        - price_slippage (float, optional): 거래 가격 슬리퍼지
        - window (int, optional): 추세선 계산 윈도우 크기
        - aspect_ratio (float, optional): 차트 종횡비
        - initial_capital (float, optional): 초기 자본
        - ticker (str): 암호화폐 티커
        - interval (str): 캔들스틱 간격
        
        Returns:
        - dict: 조건 딕셔너리
        """
        return {
            'buy_angle_threshold': buy_angle_threshold,
            'sell_angle_threshold': sell_angle_threshold,
            'stop_loss_percent': stop_loss_percent,
            'min_sell_price': min_sell_price,
            'price_slippage': price_slippage,
            'window': window,
            'aspect_ratio': aspect_ratio,
            'initial_capital': initial_capital,
            'ticker': ticker,
            'interval': interval
        }
    
    @staticmethod
    def get_qqc_condition_key(volume_window=None, ma_window=None, volume_multiplier=None,
                             buy_cash_ratio=None, hold_period=None, profit_target=None,
                             stop_loss=None, price_slippage=None, initial_capital=None,
                             ticker='BTC', interval='24h'):
        """
        QQC 전략 조건값들을 딕셔너리로 반환
        
        Parameters:
        - volume_window (int, optional): 거래량 평균 계산용 윈도우
        - ma_window (int, optional): 이동평균 계산용 윈도우
        - volume_multiplier (float, optional): 거래량 배수
        - buy_cash_ratio (float, optional): 매수시 사용할 현금 비율
        - hold_period (int, optional): 매수 후 보유 기간 (캔들 수)
        - profit_target (float, optional): 이익실현 목표 수익률 (%)
        - stop_loss (float, optional): 손절 기준 수익률 (%)
        - price_slippage (float, optional): 거래 가격 슬리퍼지
        - initial_capital (float, optional): 초기 자본
        - ticker (str): 암호화폐 티커
        - interval (str): 캔들스틱 간격
        
        Returns:
        - dict: 조건 딕셔너리
        """
        return {
            'volume_window': volume_window,
            'ma_window': ma_window,
            'volume_multiplier': volume_multiplier,
            'buy_cash_ratio': buy_cash_ratio,
            'hold_period': hold_period,
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'price_slippage': price_slippage,
            'initial_capital': initial_capital,
            'ticker': ticker,
            'interval': interval
        }
    
    @staticmethod
    def get_condition_string(condition_dict):
        """
        조건 딕셔너리를 파일명에 사용할 수 있는 문자열로 변환
        
        Parameters:
        - condition_dict (dict): 조건 딕셔너리
        
        Returns:
        - str: 조건 문자열 (예: "buy3.0_sell-4.0_sl-3.0_min20000_slip1000_w13_ar3_cap1_24h")
                또는 QQC 전략: "qqc_vw55_mw9_vm1.4_bcr0.9_hp15_pt17.6_sl-28.6_slip1000_cap1000000_btc_3m")
        """
        try:
            parts = []
            
            # 기존 전략 변수들
            if condition_dict.get('buy_angle_threshold') is not None:
                parts.append(f"buy{condition_dict['buy_angle_threshold']:.1f}")
            if condition_dict.get('sell_angle_threshold') is not None:
                parts.append(f"sell{condition_dict['sell_angle_threshold']:.1f}")
            if condition_dict.get('stop_loss_percent') is not None:
                parts.append(f"sl{condition_dict['stop_loss_percent']:.1f}")
            if condition_dict.get('min_sell_price') is not None:
                parts.append(f"min{condition_dict['min_sell_price']:.0f}")
            
            # QQC 전략 변수들
            if condition_dict.get('volume_window') is not None:
                parts.append(f"vw{condition_dict['volume_window']}")
            if condition_dict.get('ma_window') is not None:
                parts.append(f"mw{condition_dict['ma_window']}")
            if condition_dict.get('volume_multiplier') is not None:
                parts.append(f"vm{condition_dict['volume_multiplier']:.1f}")
            if condition_dict.get('buy_cash_ratio') is not None:
                parts.append(f"bcr{condition_dict['buy_cash_ratio']:.1f}")
            if condition_dict.get('hold_period') is not None:
                parts.append(f"hp{condition_dict['hold_period']}")
            if condition_dict.get('profit_target') is not None:
                parts.append(f"pt{condition_dict['profit_target']:.1f}")
            if condition_dict.get('stop_loss') is not None:  # QQC 전략의 stop_loss
                parts.append(f"sl{condition_dict['stop_loss']:.1f}")
            
            # 공통 변수들
            if condition_dict.get('price_slippage') is not None:
                parts.append(f"slip{condition_dict['price_slippage']:.0f}")
            if condition_dict.get('window') is not None:
                parts.append(f"w{condition_dict['window']}")
            if condition_dict.get('aspect_ratio') is not None:
                parts.append(f"ar{condition_dict['aspect_ratio']:.1f}")
            if condition_dict.get('initial_capital') is not None:
                parts.append(f"cap{condition_dict['initial_capital']:.0f}")
            
            ticker = condition_dict.get('ticker', 'BTC')
            interval = condition_dict.get('interval', '24h')
            parts.append(ticker.lower())
            parts.append(interval)
            
            return "_".join(parts)
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    @staticmethod
    def get_file_hash(condition_dict):
        """
        조건 딕셔너리를 해시값으로 변환 (파일명에 사용)
        
        Parameters:
        - condition_dict (dict): 조건 딕셔너리
        
        Returns:
        - str: 해시값 (16진수 문자열, 8자리)
        """
        try:
            # 딕셔너리를 정렬된 JSON 문자열로 변환
            json_str = json.dumps(condition_dict, sort_keys=True)
            # MD5 해시 생성 (8자리만 사용)
            hash_obj = hashlib.md5(json_str.encode())
            return hash_obj.hexdigest()[:8]
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    @classmethod
    def save_condition(cls, condition_dict, condition_file=None):
        """
        현재 조건을 파일에 저장
        
        Parameters:
        - condition_dict (dict): 조건 딕셔너리
        - condition_file (str, optional): 저장할 파일 경로. None이면 기본값 사용
        
        Returns:
        - bool: 성공 여부
        """
        try:
            if condition_file is None:
                condition_file = cls.CONDITION_FILE
            
            # 디렉토리 생성
            os.makedirs(os.path.dirname(condition_file), exist_ok=True)
            
            # 조건 저장
            with open(condition_file, 'w', encoding='utf-8') as f:
                json.dump(condition_dict, f, indent=2, ensure_ascii=False)
            
            print(f"조건 저장 완료: {condition_file}")
            return True
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    @classmethod
    def load_condition(cls, condition_file=None):
        """
        저장된 조건을 파일에서 로드
        
        Parameters:
        - condition_file (str, optional): 로드할 파일 경로. None이면 기본값 사용
        
        Returns:
        - dict or None: 조건 딕셔너리. 파일이 없으면 None 반환
        """
        try:
            if condition_file is None:
                condition_file = cls.CONDITION_FILE
            
            if not os.path.exists(condition_file):
                return None
            
            with open(condition_file, 'r', encoding='utf-8') as f:
                condition_dict = json.load(f)
            
            print(f"조건 로드 완료: {condition_file}")
            return condition_dict
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    @staticmethod
    def compare_conditions(condition1, condition2):
        """
        두 조건을 비교하여 같은지 확인
        
        Parameters:
        - condition1 (dict): 첫 번째 조건
        - condition2 (dict): 두 번째 조건
        
        Returns:
        - bool: 같은 조건이면 True, 다르면 False
        """
        try:
            # 비교할 키 목록 (기존 전략 + QQC 전략)
            keys_to_compare = [
                'buy_angle_threshold', 'sell_angle_threshold', 'stop_loss_percent',
                'min_sell_price', 'price_slippage', 'window', 'aspect_ratio',
                'initial_capital', 'ticker', 'interval',
                # QQC 전략 변수들
                'volume_window', 'ma_window', 'volume_multiplier', 'buy_cash_ratio',
                'hold_period', 'profit_target', 'stop_loss'
            ]
            
            for key in keys_to_compare:
                val1 = condition1.get(key)
                val2 = condition2.get(key)
                
                # None 비교 처리
                if val1 is None and val2 is None:
                    continue
                if val1 is None or val2 is None:
                    return False
                
                # 숫자 비교 (부동소수점 오차 고려)
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if abs(val1 - val2) > 1e-6:
                        return False
                elif val1 != val2:
                    return False
            
            return True
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise

