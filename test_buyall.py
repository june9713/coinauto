"""
KRW 잔고의 90%로 BTC 매수 테스트 스크립트
"""
import traceback
from trader import Trader


def main():
    """
    현재 KRW 잔고의 90%로 BTC를 구매
    """
    try:
        print("="*80)
        print("KRW 잔고 90% BTC 매수 테스트")
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
        
        # KRW 잔고 확인
        if krw_balance <= 0:
            print("\n오류: 사용 가능한 KRW 잔고가 없습니다.")
            return
        
        # 90% 매수 실행
        buy_cash = krw_balance * 0.9
        print(f"\n[2단계] KRW 잔고의 90%({buy_cash:,.0f}원)으로 BTC 매수 실행...")
        order_result = trader.buy_market_order(
            ticker='BTC',
            cash_ratio=0.9  # 90% 사용
        )
        
        # 결과 출력
        print("\n[3단계] 주문 결과:")
        if order_result['success']:
            print(f"  ✓ 매수 성공!")
            print(f"    주문 ID: {order_result['order_id']}")
            print(f"    매수 수량: {order_result['amount']:.8f} BTC")
            print(f"    매수 가격: {order_result['price']:,.0f}원")
        else:
            print(f"  ✗ 매수 실패: {order_result['message']}")
        
        # 매수 후 잔고 조회
        print("\n[4단계] 매수 후 잔고 조회 중...")
        balance_after = trader.get_balance(ticker='BTC')
        krw_balance_after = balance_after['cash']
        btc_balance_after = balance_after['coin']
        
        print(f"\n매수 후 계좌 상태:")
        print(f"  KRW 잔고: {krw_balance_after:,.0f}원")
        print(f"  BTC 보유량: {btc_balance_after:.8f} BTC")
        
        print("\n" + "="*80)
        print("KRW 잔고 90% BTC 매수 테스트 완료")
        print("="*80)
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


if __name__ == "__main__":
    main()

