"""
실제 거래 실행 모듈
Bithumb API v2를 사용한 매수/매도 실행
"""
import traceback
import os
import time
from decimal import Decimal
from pybithumb2 import BithumbClient, MarketID, Currency, TradeSide, OrderType
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
        self.api_key = api_key
        self.api_secret = api_secret
        self._bithumb = None
    
    def _get_bithumb_client(self):
        """
        Bithumb 클라이언트 반환 (싱글톤 패턴)
        
        Returns:
        - BithumbClient: Bithumb 클라이언트 객체
        """
        if self._bithumb is None:
            # .env 파일에서 API 키 로드
            api_key = self.api_key or os.getenv('CONKEY')
            secret_key = self.api_secret or os.getenv('SECKEY')
            print(f"api_key: {api_key[:10] if api_key else None}...")
            print(f"secret_key: {secret_key[:10] if secret_key else None}...")
            if not api_key or not secret_key:
                raise ValueError("API 키가 설정되지 않았습니다. .env 파일에 CONKEY와 SECKEY를 설정하세요.")
            
            self._bithumb = BithumbClient(api_key=api_key, secret_key=secret_key)
        
        return self._bithumb
    
    def get_balance(self, ticker='BTC'):
        """
        계좌 잔고 조회
        
        Parameters:
        - ticker (str): 암호화폐 티커. 기본값 'BTC'
        
        Returns:
        - dict: {'cash': float, 'coin': float} - 현금(KRW) 잔고, 코인 보유량
        """
        try:
            bithumb = self._get_bithumb_client()
            
            # 디버깅: 전체 잔고 조회 (모든 코인 잔고 확인)
            print(f"\n[잔고 조회 디버깅] ticker={ticker}")
            
            # pybithumb2는 get_accounts()로 모든 계좌 잔고를 한 번에 조회
            accounts = bithumb.get_accounts()
            print(f"  전체 계좌 조회 결과 타입: {type(accounts)}, 개수: {len(accounts) if hasattr(accounts, '__len__') else 'N/A'}")
            
            # Account 객체들에서 코인과 KRW 잔고 추출
            coin_amount = 0.0
            cash = 0.0
            
            for account in accounts:
                currency_code = account.currency.code if hasattr(account.currency, 'code') else str(account.currency)
                balance = float(account.balance)
                locked = float(account.locked) if hasattr(account, 'locked') else 0.0
                
                print(f"  계좌: {currency_code}, 잔고={balance}, 잠김={locked}")
                
                if currency_code.upper() == ticker.upper():
                    coin_amount = balance
                    print(f"  코인 잔고 ({ticker}): {coin_amount}")
                elif currency_code.upper() == 'KRW':
                    cash = balance
                    print(f"  KRW 잔고: {cash}")
            
            print(f"  최종 결과: 현금={cash:,.0f}원, 코인={coin_amount:.8f} {ticker}")
            
            return {
                'cash': cash,
                'coin': coin_amount
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
            bithumb = self._get_bithumb_client()
            
            # 현재 잔고 조회
            balance = self.get_balance(ticker)
            available_cash = balance['cash']
            
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
            
            # MarketID 생성 (KRW-BTC 형식)
            market = MarketID.from_string(f"KRW-{ticker}")
            
            # 현재가 조회 (주문 가격 결정용)
            try:
                # pybithumb2는 get_snapshots()로 현재가 조회
                snapshots = bithumb.get_snapshots(markets=[market])
                if snapshots and len(snapshots) > 0:
                    current_price = float(snapshots[0].trade_price)
                    print(f"  현재가 조회: {current_price:,.0f}원")
                else:
                    current_price = 0
            except Exception as e:
                print(f"  경고: 현재가 조회 실패: {str(e)}")
                current_price = 0
            
            # 시장가 매수 주문 (pybithumb2 방식)
            # pybithumb2는 submit_order 메서드 사용
            try:
                # 주문할 수량 계산 (시장가이므로 가격은 자동 결정)
                if current_price > 0:
                    estimated_units = Decimal(str(use_cash / current_price))
                else:
                    # 현재가를 알 수 없으면 작은 수량으로 시작
                    estimated_units = Decimal("0.001")
                
                # 시장가 주문 시도
                order_result = bithumb.submit_order(
                    market=market,
                    side=TradeSide.BID,  # 매수: BID
                    volume=estimated_units,
                    price=Decimal("0"),  # 시장가 주문은 price를 0으로 설정
                    ord_type=OrderType.MARKET  # 시장가 주문
                )
                
                print(f"  주문 결과 타입: {type(order_result)}, 값: {order_result}")
                
            except Exception as e:
                # 시장가 주문이 안되면 지정가 주문으로 대체
                if current_price > 0:
                    buy_price = Decimal(str(int(current_price * 1.01)))  # 현재가의 1.01배로 지정가 주문
                    estimated_units = Decimal(str(use_cash / float(buy_price)))
                    order_result = bithumb.submit_order(
                        market=market,
                        side=TradeSide.BID,
                        volume=estimated_units,
                        price=buy_price,
                        ord_type=OrderType.LIMIT  # 지정가 주문
                    )
                else:
                    raise ValueError(f"현재가 조회 실패 및 주문 실패: {str(e)}")
            
            if order_result is None:
                return {
                    'success': False,
                    'message': '주문 실패: API 응답이 None입니다.',
                    'order_id': None,
                    'amount': 0.0,
                    'price': 0.0
                }
            
            # 주문 결과 파싱 (pybithumb2는 Order 모델 반환)
            if hasattr(order_result, 'uuid'):
                order_id = str(order_result.uuid)
                amount = float(order_result.volume)
                price = float(order_result.price)
                return {
                    'success': True,
                    'message': '매수 주문 성공',
                    'order_id': order_id,
                    'amount': amount,
                    'price': price
                }
            elif isinstance(order_result, dict):
                return {
                    'success': True,
                    'message': order_result.get('message', '매수 주문 성공'),
                    'order_id': str(order_result.get('uuid', order_result.get('order_id', ''))),
                    'amount': float(order_result.get('volume', 0)),
                    'price': float(order_result.get('price', 0))
                }
            else:
                return {
                    'success': True,
                    'message': '매수 주문 성공',
                    'order_id': str(order_result),
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
            bithumb = self._get_bithumb_client()
            
            # 현재 잔고 조회
            balance = self.get_balance(ticker)
            available_coin = balance['coin']
            
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
            
            # MarketID 생성 (KRW-BTC 형식)
            market = MarketID.from_string(f"KRW-{ticker}")
            
            # 현재가 조회 (주문 가격 결정용)
            try:
                # pybithumb2는 get_snapshots()로 현재가 조회
                snapshots = bithumb.get_snapshots(markets=[market])
                if snapshots and len(snapshots) > 0:
                    current_price = float(snapshots[0].trade_price)
                    print(f"  현재가 조회: {current_price:,.0f}원")
                else:
                    current_price = 0
            except Exception as e:
                print(f"  경고: 현재가 조회 실패: {str(e)}")
                current_price = 0
            
            # 시장가 매도 주문 (pybithumb2 방식)
            try:
                # 시장가 매도 주문
                order_result = bithumb.submit_order(
                    market=market,
                    side=TradeSide.ASK,  # 매도: ASK
                    volume=Decimal(str(sell_amount)),
                    price=Decimal("0"),  # 시장가 주문은 price를 0으로 설정
                    ord_type=OrderType.MARKET  # 시장가 주문
                )
                
                print(f"  주문 결과 타입: {type(order_result)}, 값: {order_result}")
                
            except Exception as e:
                # 시장가 주문이 안되면 지정가 주문으로 대체
                if current_price > 0:
                    sell_price = Decimal(str(int(current_price * 0.99)))  # 현재가의 0.99배로 지정가 주문
                    order_result = bithumb.submit_order(
                        market=market,
                        side=TradeSide.ASK,
                        volume=Decimal(str(sell_amount)),
                        price=sell_price,
                        ord_type=OrderType.LIMIT  # 지정가 주문
                    )
                else:
                    raise ValueError(f"현재가 조회 실패 및 주문 실패: {str(e)}")
            
            if order_result is None:
                return {
                    'success': False,
                    'message': '주문 실패: API 응답이 None입니다.',
                    'order_id': None,
                    'amount': 0.0,
                    'price': 0.0
                }
            
            # 주문 결과 파싱 (pybithumb2는 Order 모델 반환)
            if hasattr(order_result, 'uuid'):
                order_id = str(order_result.uuid)
                amount = float(order_result.volume)
                price = float(order_result.price)
                return {
                    'success': True,
                    'message': '매도 주문 성공',
                    'order_id': order_id,
                    'amount': amount,
                    'price': price
                }
            elif isinstance(order_result, dict):
                return {
                    'success': True,
                    'message': order_result.get('message', '매도 주문 성공'),
                    'order_id': str(order_result.get('uuid', order_result.get('order_id', ''))),
                    'amount': float(order_result.get('volume', sell_amount)),
                    'price': float(order_result.get('price', 0))
                }
            else:
                return {
                    'success': True,
                    'message': '매도 주문 성공',
                    'order_id': str(order_result),
                    'amount': sell_amount,
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

