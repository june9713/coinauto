"""
실제 거래 실행 모듈
Bithumb API v2를 사용한 매수/매도 실행 (직접 API 호출 방식)
"""
import traceback
import os
import time
import uuid
import hashlib
import jwt
from urllib.parse import urlencode
import requests
from decimal import Decimal
from dotenv import load_dotenv


# .env 파일에서 환경 변수 로드
load_dotenv()


class Trader:
    """실제 거래 실행 클래스"""
    
    def __init__(self, api_key=None, api_secret=None):
        """
        초기화
        
        Parameters:
        - api_key (str, optional): API 키
        - api_secret (str, optional): API 시크릿 키
        """
        self.api_key = api_key or os.getenv('CONKEY')
        self.api_secret = api_secret or os.getenv('SECKEY')
        self.api_url = 'https://api.bithumb.com'
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API 키가 설정되지 않았습니다. .env 파일에 CONKEY와 SECKEY를 설정하세요.")
        
        print(f"api_key: {self.api_key[:10] if self.api_key else None}...")
        print(f"secret_key: {self.api_secret[:10] if self.api_secret else None}...")
    
    def _generate_jwt_token(self, query_params=None):
        """
        JWT 토큰 생성
        
        Parameters:
        - query_params (dict, optional): 쿼리 파라미터
        
        Returns:
        - str: Authorization 토큰
        """
        try:
            # 쿼리 파라미터가 있으면 인코딩
            if query_params:
                query = urlencode(query_params).encode()
                query_hash = hashlib.sha512(query).hexdigest()
            else:
                query_hash = ''
            
            # JWT payload 생성
            payload = {
                'access_key': self.api_key,
                'nonce': str(uuid.uuid4()),
                'timestamp': round(time.time() * 1000),
                'query_hash': query_hash,
                'query_hash_alg': 'SHA512',
            }
            
            # JWT 토큰 생성
            jwt_token = jwt.encode(payload, self.api_secret, algorithm='HS256')
            authorization_token = f'Bearer {jwt_token}'
            
            return authorization_token
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _get_order_chance(self, market='KRW-BTC'):
        """
        주문 가능 정보 조회
        
        Parameters:
        - market (str): 마켓 ID (예: 'KRW-BTC')
        
        Returns:
        - dict: 주문 가능 정보
        """
        try:
            param = {'market': market}
            authorization_token = self._generate_jwt_token(param)
            
            headers = {
                'Authorization': authorization_token
            }
            
            response = requests.get(
                f'{self.api_url}/v1/orders/chance',
                params=param,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise ValueError(f"주문 가능 정보 조회 실패: HTTP {response.status_code}, {response.text}")
                
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def get_balance(self, ticker='BTC'):
        """
        계좌 잔고 조회

        Parameters:
        - ticker (str): 암호화폐 티커. 기본값 'BTC'

        Returns:
        - dict: {
            'cash': float,  # KRW 전체 잔고 (주문 중 포함)
            'coin': float,  # 코인 전체 보유량 (주문 중 포함)
            'cash_available': float,  # KRW 주문 가능 금액
            'coin_available': float  # 코인 주문 가능 수량
          }
        """
        try:
            market = f'KRW-{ticker}'

            # 주문 가능 정보 조회 (잔고 정보 포함)
            chance_info = self._get_order_chance(market)

            print(f"\n[잔고 조회 디버깅] ticker={ticker}")

            # bid_account: 매수 시 사용하는 화폐 (KRW) 계좌
            # ask_account: 매도 시 사용하는 화폐 (BTC) 계좌
            coin_amount = 0.0
            coin_available = 0.0
            cash = 0.0
            cash_available = 0.0

            if 'bid_account' in chance_info:
                bid_account = chance_info['bid_account']
                if bid_account.get('currency') == 'KRW':
                    balance = float(bid_account.get('balance', 0))
                    locked = float(bid_account.get('locked', 0))  # 주문 중인 금액
                    cash = balance  # 전체 잔고 (주문 중 포함)
                    cash_available = balance - locked  # 실제 주문 가능한 금액
                    print(f"  KRW 전체 잔고: {balance:,.0f}원")
                    print(f"  KRW 주문 중: {locked:,.0f}원")
                    print(f"  KRW 주문 가능: {cash_available:,.0f}원")

            if 'ask_account' in chance_info:
                ask_account = chance_info['ask_account']
                if ask_account.get('currency') == ticker.upper():
                    balance = float(ask_account.get('balance', 0))
                    locked = float(ask_account.get('locked', 0))  # 주문 중인 수량
                    coin_amount = balance  # 전체 잔고 (주문 중 포함)
                    coin_available = balance - locked  # 실제 주문 가능한 수량
                    print(f"  {ticker} 전체 잔고: {balance:.8f}")
                    print(f"  {ticker} 주문 중: {locked:.8f}")
                    print(f"  {ticker} 주문 가능: {coin_available:.8f}")

            return {
                'cash': cash,
                'coin': coin_amount,
                'cash_available': cash_available,
                'coin_available': coin_available
            }
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def buy_market_order(self, ticker='BTC', cash_amount=None, cash_ratio=None):
        """
        시장가 매수 주문 실행
        
        Parameters:
        - ticker (str): 암호화폐 티커. 기본값 'BTC'
        - cash_amount (float, optional): 사용할 현금 금액 (원). None이면 전체 사용
        - cash_ratio (float, optional): 사용할 현금 비율 (0.0~1.0). None이면 전체 사용
        
        Returns:
        - dict: 주문 결과 {'success': bool, 'message': str, 'order_id': str, 'amount': float, 'price': float}
        """
        try:
            market = f'KRW-{ticker}'
            
            # 현재 잔고 조회 (주문 가능한 금액 포함)
            balance_info = self.get_balance(ticker)
            available_cash = balance_info['cash_available']  # 주문 가능한 KRW 금액
            
            # 사용할 현금 계산
            if cash_amount is not None:
                use_cash = min(cash_amount, available_cash)
            elif cash_ratio is not None:
                use_cash = available_cash * cash_ratio
            else:
                use_cash = available_cash
            
            if use_cash <= 0:
                return {
                    'success': False,
                    'message': '사용 가능한 현금이 없습니다.',
                    'order_id': None,
                    'amount': 0.0,
                    'price': 0.0
                }
            
            # 주문 파라미터 설정
            # 시장가 매수 주문: side='bid', ord_type='price' (금액으로 매수)
            order_params = {
                'market': market,
                'side': 'bid',  # 매수
                'ord_type': 'price',  # 금액으로 매수 (시장가)
                'price': str(int(use_cash))  # 사용할 금액
            }
            
            # JWT 토큰 생성
            authorization_token = self._generate_jwt_token(order_params)
            
            headers = {
                'Authorization': authorization_token
            }
            
            print(f"  주문 파라미터: {order_params}")
            
            # 주문 실행
            response = requests.post(
                f'{self.api_url}/v1/orders',
                json=order_params,
                headers=headers,
                timeout=10
            )
            
            print(f"  API 응답 상태 코드: {response.status_code}")
            print(f"  API 응답 내용: {response.text}")
            
            if response.status_code == 200 or response.status_code == 201:
                result = response.json()
                
                # 주문 결과 파싱
                order_id = result.get('uuid', result.get('id', ''))
                executed_volume = float(result.get('executed_volume', 0))
                executed_price = float(result.get('price', 0))
                
                return {
                    'success': True,
                    'message': '매수 주문 성공',
                    'order_id': str(order_id),
                    'amount': executed_volume,
                    'price': executed_price
                }
            else:
                error_msg = response.text
                try:
                    error_json = response.json()
                    error_msg = error_json.get('error', {}).get('message', error_msg)
                except:
                    pass
                
                return {
                    'success': False,
                    'message': f'매수 주문 실패: HTTP {response.status_code}, {error_msg}',
                    'order_id': None,
                    'amount': 0.0,
                    'price': 0.0
                }
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            return {
                'success': False,
                'message': f'매수 주문 실패: {str(e)}',
                'order_id': None,
                'amount': 0.0,
                'price': 0.0
            }
    
    def sell_market_order(self, ticker='BTC', coin_amount=None):
        """
        시장가 매도 주문 실행
        
        Parameters:
        - ticker (str): 암호화폐 티커. 기본값 'BTC'
        - coin_amount (float, optional): 매도할 코인 수량. None이면 전체 매도
        
        Returns:
        - dict: 주문 결과 {'success': bool, 'message': str, 'order_id': str, 'amount': float, 'price': float}
        """
        try:
            market = f'KRW-{ticker}'

            # 현재 잔고 조회 (주문 가능한 수량 포함)
            balance = self.get_balance(ticker)
            available_coin = balance['coin_available']  # 주문 가능한 코인 수량
            
            # 매도할 코인 수량 계산
            if coin_amount is not None:
                sell_amount = min(coin_amount, available_coin)
            else:
                sell_amount = available_coin
            
            if sell_amount <= 0:
                return {
                    'success': False,
                    'message': '매도할 코인이 없습니다.',
                    'order_id': None,
                    'amount': 0.0,
                    'price': 0.0
                }
            
            # 주문 파라미터 설정
            # 시장가 매도 주문: side='ask', ord_type='market' (수량으로 매도)
            order_params = {
                'market': market,
                'side': 'ask',  # 매도
                'ord_type': 'market',  # 시장가 매도
                'volume': str(Decimal(str(sell_amount)))  # 매도할 수량
            }
            
            # JWT 토큰 생성
            authorization_token = self._generate_jwt_token(order_params)
            
            headers = {
                'Authorization': authorization_token
            }
            
            print(f"  주문 파라미터: {order_params}")
            
            # 주문 실행
            response = requests.post(
                f'{self.api_url}/v1/orders',
                json=order_params,
                headers=headers,
                timeout=10
            )
            
            print(f"  API 응답 상태 코드: {response.status_code}")
            print(f"  API 응답 내용: {response.text}")
            
            if response.status_code == 200 or response.status_code == 201:
                result = response.json()
                
                # 주문 결과 파싱
                order_id = result.get('uuid', result.get('id', ''))
                executed_volume = float(result.get('executed_volume', 0))
                executed_price = float(result.get('price', 0))
                
                return {
                    'success': True,
                    'message': '매도 주문 성공',
                    'order_id': str(order_id),
                    'amount': executed_volume,
                    'price': executed_price
                }
            else:
                error_msg = response.text
                try:
                    error_json = response.json()
                    error_msg = error_json.get('error', {}).get('message', error_msg)
                except:
                    pass
                
                return {
                    'success': False,
                    'message': f'매도 주문 실패: HTTP {response.status_code}, {error_msg}',
                    'order_id': None,
                    'amount': 0.0,
                    'price': 0.0
                }
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            return {
                'success': False,
                'message': f'매도 주문 실패: {str(e)}',
                'order_id': None,
                'amount': 0.0,
                'price': 0.0
            }
