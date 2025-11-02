"""
BTC 잔고 전액을 KRW로 매도 테스트 스크립트
"""
import traceback
from trader import Trader


def main():
    """
    현재 BTC 잔고 전액을 KRW로 매도
    """
    try:
        print("="*80)
        print("BTC 잔고 전액 KRW 매도 테스트")
        print("="*80)
        
        # Trader 객체 생성
        trader = Trader()
        
        # 현재 잔고 조회
        print("\n[1단계] 현재 잔고 조회 중...")
        balance = trader.get_balance(ticker='BTC')
        krw_balance = balance['cash']
        btc_balance = balance['coin']
        
        print(f"\n현재 계좌 상태:")
        print(f"  KRW 잔고: {krw_balance:,.0f}원")
        print(f"  BTC 보유량: {btc_balance:.8f} BTC")
        
        # BTC 잔고 확인
        if btc_balance <= 0:
            print("\n오류: 매도할 BTC 잔고가 없습니다.")
            return
        
        # 전액 매도 실행
        print(f"\n[2단계] BTC 잔고 전액({btc_balance:.8f} BTC)을 KRW로 매도 실행...")
        order_result = trader.sell_market_order(
            ticker='BTC',
            coin_amount=None  # None이면 전체 매도
        )
        
        # 결과 출력
        print("\n[3단계] 주문 결과:")
        if order_result['success']:
            print(f"  ✓ 매도 성공!")
            print(f"    주문 ID: {order_result['order_id']}")
            print(f"    매도 수량: {order_result['amount']:.8f} BTC")
            print(f"    매도 가격: {order_result['price']:,.0f}원")
        else:
            print(f"  ✗ 매도 실패: {order_result['message']}")
        
        # 매도 후 잔고 조회
        print("\n[4단계] 매도 후 잔고 조회 중...")
        balance_after = trader.get_balance(ticker='BTC')
        krw_balance_after = balance_after['cash']
        btc_balance_after = balance_after['coin']
        
        print(f"\n매도 후 계좌 상태:")
        print(f"  KRW 잔고: {krw_balance_after:,.0f}원")
        print(f"  BTC 보유량: {btc_balance_after:.8f} BTC")
        
        print("\n" + "="*80)
        print("BTC 잔고 전액 KRW 매도 테스트 완료")
        print("="*80)
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


if __name__ == "__main__":
    main()

