"""
QQC 백테스트 메인 실행 모듈
qqc_test 모듈을 호출하여 백테스트 실행
"""
import traceback
import pandas as pd
from config import Config
from data_manager import DataManager
from qqc_test import QQCTestEngine
from result_reporter import ResultReporter
from visualizer import Visualizer
from condition_manager import ConditionManager


def main(start_date='2014-01-01', end_date=None, initial_capital=None,
         price_slippage=None, ticker='BTC', interval='24h',
         volume_window=None, ma_window=None, volume_multiplier=None,
         buy_cash_ratio=None, hold_period=None, profit_target=None,
         stop_loss=None):
    """
    QQC 백테스트 메인 함수
    
    Parameters:
    - start_date (str): 백테스트 시작 날짜 (YYYY-MM-DD 형식). 기본값 '2014-01-01'
    - end_date (str, optional): 백테스트 종료 날짜 (YYYY-MM-DD 형식). None이면 모든 데이터 사용
    - initial_capital (float, optional): 초기 자본 (원). None이면 Config 기본값 사용
    - price_slippage (int, optional): 거래 가격 슬리퍼지 (원). None이면 Config 기본값 사용
    - ticker (str): 암호화폐 티커. 기본값 'BTC'
    - interval (str): 캔들스틱 간격. 기본값 '24h'
    - volume_window (int, optional): 거래량 평균 계산용 윈도우. None이면 기본값 55 사용
    - ma_window (int, optional): 이동평균 계산용 윈도우. None이면 기본값 9 사용
    - volume_multiplier (float, optional): 거래량 배수. None이면 기본값 1.4 사용
    - buy_cash_ratio (float, optional): 매수시 사용할 현금 비율 (0.0~1.0). None이면 기본값 0.9 사용
    - hold_period (int, optional): 매수 후 보유 기간 (캔들 수). None이면 기본값 15 사용
    - profit_target (float, optional): 이익실현 목표 수익률 (%). None이면 기본값 17.6 사용
    - stop_loss (float, optional): 손절 기준 수익률 (%). None이면 기본값 -28.6 사용
    """
    try:
        print("="*80)
        print("QQC 백테스트 시작")
        print("="*80)
        
        # 시작 날짜 파싱
        start_dt = None
        if start_date is None:
            print("\n시작 날짜가 지정되지 않음. 데이터에서 첫 번째 날짜를 확인합니다.")
        else:
            try:
                start_dt = pd.to_datetime(start_date)
            except Exception as e:
                print(f"오류: 시작 날짜 형식이 올바르지 않습니다: {start_date}")
                print(f"  올바른 형식: YYYY-MM-DD (예: 2014-01-01)")
                return
        
        # 종료 날짜 파싱
        end_dt = None
        if end_date is not None:
            try:
                end_dt = pd.to_datetime(end_date)
            except Exception as e:
                print(f"오류: 종료 날짜 형식이 올바르지 않습니다: {end_date}")
                print(f"  올바른 형식: YYYY-MM-DD (예: 2023-12-31)")
                return
        
        # 백테스트 조건 값 설정
        initial_cap_val = initial_capital or Config.DEFAULT_INITIAL_CAPITAL
        price_slip_val = price_slippage or Config.DEFAULT_PRICE_SLIPPAGE
        
        # QQC 전략 변수 설정
        volume_window_val = volume_window if volume_window is not None else 55
        ma_window_val = ma_window if ma_window is not None else 9
        volume_multiplier_val = volume_multiplier if volume_multiplier is not None else 1.4
        buy_cash_ratio_val = buy_cash_ratio if buy_cash_ratio is not None else 0.9
        hold_period_val = hold_period if hold_period is not None else 15
        profit_target_val = profit_target if profit_target is not None else 17.6
        stop_loss_val = stop_loss if stop_loss is not None else -28.6
        
        # 조건 딕셔너리 생성
        current_condition = ConditionManager.get_qqc_condition_key(
            volume_window=volume_window_val,
            ma_window=ma_window_val,
            volume_multiplier=volume_multiplier_val,
            buy_cash_ratio=buy_cash_ratio_val,
            hold_period=hold_period_val,
            profit_target=profit_target_val,
            stop_loss=stop_loss_val,
            price_slippage=price_slip_val,
            initial_capital=initial_cap_val,
            ticker=ticker,
            interval=interval
        )
        
        # 조건 확인 및 저장
        print("\n[단계 0] 백테스트 조건 확인 중...")
        previous_condition = ConditionManager.load_condition()
        
        if previous_condition is not None:
            if ConditionManager.compare_conditions(current_condition, previous_condition):
                print("  조건 변경 없음. 기존 조건 파일 사용")
            else:
                print("  조건 변경 감지!")
                print(f"  이전 조건: {ConditionManager.get_condition_string(previous_condition)}")
                print(f"  현재 조건: {ConditionManager.get_condition_string(current_condition)}")
                print("  조건별 다른 파일을 사용합니다.")
        else:
            print("  저장된 조건 없음. 현재 조건 저장 중...")
        
        # 현재 조건 저장
        ConditionManager.save_condition(current_condition)
        
        # 조건 기반 DataManager 초기화
        data_manager = DataManager(ticker=ticker, condition_dict=current_condition)
        
        # start_date가 None인 경우 데이터에서 첫 번째 날짜 확인
        if start_dt is None:
            print("\n데이터에서 첫 번째 날짜 확인 중...")
            df = data_manager.update_history_data(ticker=ticker, interval=interval, start_date='2014-01-01')
            if df is None or len(df) == 0:
                print("오류: 데이터 수집 실패")
                return
            start_dt = df.index[0]
            print(f"데이터의 첫 번째 날짜를 시작 날짜로 사용: {start_dt.strftime('%Y-%m-%d')}")
        
        # 날짜 범위 검증
        if end_dt is not None and end_dt <= start_dt:
            print(f"오류: 종료 날짜({end_dt.strftime('%Y-%m-%d')})가 시작 날짜({start_dt.strftime('%Y-%m-%d')})보다 이전이거나 같습니다.")
            return
        
        print(f"백테스트 기간: {start_dt.strftime('%Y-%m-%d')}", end='')
        if end_dt is not None:
            print(f" ~ {end_dt.strftime('%Y-%m-%d')}")
        else:
            print(" ~ (마지막)")
        
        # 데이터 업데이트
        print("\n[단계 1] 데이터 수집 중...")
        df = data_manager.update_history_data(ticker=ticker, interval=interval, start_date='2014-01-01')
        
        if df is None or len(df) == 0:
            print("오류: 데이터 수집 실패")
            return
        
        print(f"수집 완료: {len(df)}개 데이터")
        print(f"전체 기간: {df.index[0]} ~ {df.index[-1]}")
        
        # 필터링 (시작 날짜 및 종료 날짜)
        original_count = len(df)
        df = df[df.index >= start_dt]
        if end_dt is not None:
            df = df[df.index <= end_dt]
        
        filtered_count = len(df)
        
        if len(df) == 0:
            print(f"백테스트 대상 데이터가 없습니다.")
            return
        
        print(f"백테스트 대상: {filtered_count}개 데이터 (제외: {original_count - filtered_count}개)")
        print(f"백테스트 기간: {df.index[0]} ~ {df.index[-1]}")
        
        # QQC 백테스트 실행
        print("\n[단계 2] QQC 백테스트 실행 중...")
        print(f"초기 자본: {initial_cap_val:,.0f}원")
        print(f"거래 가격 슬리퍼지: {price_slip_val:,.0f}원")
        print("매수 조건:")
        print(f"  - 조건 B: 현재 거래량 >= 거래량 A ({volume_window_val}개 평균) * {volume_multiplier_val}")
        print(f"  - 조건 D: 현재 종가 > 이동평균 C ({ma_window_val}개 평균)")
        print("  - 조건 E: 양봉 (오픈가 < 종가)")
        print(f"  - 매수 가격: 다음 오픈가 + {price_slip_val:,.0f}원")
        print(f"  - 매수 비율: 현금의 {buy_cash_ratio_val*100:.0f}%")
        print("매도 조건:")
        print(f"  - 수익률 >= {profit_target_val}%: 이익실현")
        print(f"  - 수익률 <= {stop_loss_val}%: 손절")
        print(f"  - 매수 후 {hold_period_val} 캔들 경과: 무조건 매도 (n+{hold_period_val} 캔들의 오픈가)")
        
        qqc_engine = QQCTestEngine(
            initial_capital=initial_cap_val,
            price_slippage=price_slip_val,
            volume_window=volume_window_val,
            ma_window=ma_window_val,
            volume_multiplier=volume_multiplier_val,
            buy_cash_ratio=buy_cash_ratio_val,
            hold_period=hold_period_val,
            profit_target=profit_target_val,
            stop_loss=stop_loss_val
        )
        
        result = qqc_engine.run(df)
        
        # 결과 출력
        print("\n[단계 3] 백테스트 결과 출력 중...")
        reporter = ResultReporter()
        reporter.print_backtest_results(result)
        
        # 그래프 생성
        print("\n[단계 4] 백테스트 결과 그래프 생성 중...")
        visualizer = Visualizer()
        visualizer.plot_backtest_results(
            df, result,
            buy_threshold=0,  # QQC 전략은 각도 기준이 아니므로 0으로 설정
            sell_threshold=0
        )
        
        # 백테스트 결과 저장
        print("\n[단계 5] 백테스트 결과 저장 중...")
        data_manager.save_qqc_backtest_result(
            df=df,
            result=result,
            test_start_date=start_dt,
            test_end_date=end_dt,
            volume_window=volume_window_val,
            ma_window=ma_window_val,
            volume_multiplier=volume_multiplier_val,
            buy_cash_ratio=buy_cash_ratio_val,
            hold_period=hold_period_val,
            profit_target=profit_target_val,
            stop_loss=stop_loss_val,
            price_slippage=price_slip_val,
            initial_capital=initial_cap_val,
            ticker=ticker,
            interval=interval
        )
        
        print("\nQQC 백테스트 완료!")
        
        # 마지막 거래 상태 반환
        return result.get('last_trade_status', 'unknown')
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


if __name__ == "__main__":
    # 시작 날짜 설정 (기본값: 2014-01-01)
    start_date = None  # '2014-01-01'
    
    # 종료 날짜 설정 (None이면 마지막까지 사용)
    end_date = None  # 예: '2023-12-31'
    
    # 시작 잔고 설정 (기본값: 1,000,000원)
    initial_capital = 1_000_000  # 예: 10000000 (1천만원)
    
    # QQC 전략 변수 설정
    volume_window = 55  # 거래량 평균 계산용 윈도우
    ma_window = 9  # 이동평균 계산용 윈도우
    volume_multiplier = 1.4  # 거래량 배수
    buy_cash_ratio = 0.9  # 매수시 사용할 현금 비율 (0.9 = 90%)
    hold_period = 15  # 매수 후 보유 기간 (캔들 수)
    profit_target = 17.6  # 이익실현 목표 수익률 (%)
    stop_loss = -28.6  # 손절 기준 수익률 (%)
    
    testresult = main(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        price_slippage=1000,
        ticker='BTC',
        interval='3m',
        volume_window=volume_window,
        ma_window=ma_window,
        volume_multiplier=volume_multiplier,
        buy_cash_ratio=buy_cash_ratio,
        hold_period=hold_period,
        profit_target=profit_target,
        stop_loss=stop_loss
    )
    print(f"last_trade_status: {testresult}")

