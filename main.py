"""
메인 실행 모듈
모든 클래스를 조합하여 백테스트 실행
"""
import traceback
import pandas as pd
import time
from config import Config
from data_manager import DataManager
from trendline_calculator import TrendlineCalculator
from backtest_engine import BacktestEngine
from result_reporter import ResultReporter
from visualizer import Visualizer
from condition_manager import ConditionManager


def main(start_date='2014-01-01', end_date=None, buy_angle_threshold=None, 
         sell_angle_threshold=None, stop_loss_percent=None, min_sell_price=None, 
         price_slippage=None, aspect_ratio=None, window=None, initial_capital=None,
         ticker='BTC', interval='24h'):
    """
    메인 함수
    
    Parameters:
    - start_date (str): 백테스트 시작 날짜 (YYYY-MM-DD 형식). 기본값 '2014-01-01'
    - end_date (str, optional): 백테스트 종료 날짜 (YYYY-MM-DD 형식). None이면 모든 데이터 사용
    - buy_angle_threshold (float, optional): 매수 조건 각도. None이면 Config 기본값 사용
    - sell_angle_threshold (float, optional): 매도 조건 각도. None이면 Config 기본값 사용
    - stop_loss_percent (float, optional): 손절 기준 (수익률 %). None이면 Config 기본값 사용
    - min_sell_price (float, optional): 최소 매도 가격 (원). None이면 Config 기본값 사용
    - price_slippage (int, optional): 거래 가격 슬리퍼지 (원). None이면 Config 기본값 사용
    - aspect_ratio (float, optional): 차트 종횡비. None이면 Config 기본값 사용
    - window (int, optional): 추세선 계산 윈도우 크기. None이면 Config 기본값 사용
    - initial_capital (float, optional): 초기 자본 (원). None이면 Config 기본값 사용
    - ticker (str): 암호화폐 티커. 기본값 'BTC'
    - interval (str): 캔들스틱 간격. 기본값 '24h'
    """
    try:
        print("="*80)
        print("BTC 가격 데이터 수집 및 백테스트 시작")
        print("="*80)
        
        # 시작 날짜 파싱
        start_dt = None
        if start_date is None:
            # start_date가 None이면 나중에 조건 확인 후 데이터를 확인하여 첫 번째 날짜 사용
            print("\n시작 날짜가 지정되지 않음. 조건 확인 후 데이터에서 첫 번째 날짜를 확인합니다.")
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
        buy_threshold = buy_angle_threshold or Config.DEFAULT_BUY_ANGLE_THRESHOLD
        sell_threshold = sell_angle_threshold or Config.DEFAULT_SELL_ANGLE_THRESHOLD
        stop_loss_val = stop_loss_percent or Config.DEFAULT_STOP_LOSS_PERCENT
        min_sell_val = min_sell_price or Config.DEFAULT_MIN_SELL_PRICE
        price_slip_val = price_slippage or Config.DEFAULT_PRICE_SLIPPAGE
        initial_cap_val = initial_capital or Config.DEFAULT_INITIAL_CAPITAL
        window_val = window or Config.DEFAULT_WINDOW
        aspect_ratio_val = aspect_ratio or Config.DEFAULT_ASPECT_RATIO
        
        # 조건 딕셔너리 생성
        current_condition = ConditionManager.get_condition_key(
            buy_angle_threshold=buy_threshold,
            sell_angle_threshold=sell_threshold,
            stop_loss_percent=stop_loss_val,
            min_sell_price=min_sell_val,
            price_slippage=price_slip_val,
            window=window_val,
            aspect_ratio=aspect_ratio_val,
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
            # 데이터 업데이트 (기존 데이터 로드 또는 수집)
            temp_df = data_manager.update_history_data(ticker=ticker, interval=interval, start_date='2014-01-01')
            if temp_df is None or len(temp_df) == 0:
                print("오류: 데이터 수집 실패")
                return
            start_dt = temp_df.index[0]
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
        
        # 기존 백테스트 기록 확인 (같은 파라미터로 실행된 마지막 기록)
        print("\n[단계 0-1] 기존 백테스트 기록 확인 중...")
        last_backtest_info = data_manager.get_last_backtest_info(
            ticker=ticker,
            buy_angle_threshold=buy_threshold,
            sell_angle_threshold=sell_threshold,
            stop_loss_percent=stop_loss_val,
            min_sell_price=min_sell_val,
            price_slippage=price_slip_val,
            initial_capital=initial_cap_val,
            window=window_val,
            aspect_ratio=aspect_ratio_val,
            interval=interval,
            condition_dict=current_condition
        )
        
        # 실제 백테스트 시작 날짜 결정
        actual_start_dt = start_dt
        use_previous_asset = False
        previous_final_asset = None
        
        if last_backtest_info is not None:
            last_trade_date = last_backtest_info['last_trade_date']
            previous_final_asset = last_backtest_info['last_final_asset']
            
            print(f"  기존 기록 발견:")
            print(f"    마지막 실행: {last_backtest_info['last_execution_time']}")
            print(f"    마지막 거래: {last_trade_date.strftime('%Y-%m-%d')}")
            if previous_final_asset is not None:
                print(f"    최종 자산: {previous_final_asset:,.0f}원")
            
            # 마지막 거래 날짜 이후부터 시작
            actual_start_dt = last_trade_date + pd.Timedelta(days=1)
            use_previous_asset = True
            
            # 요청한 시작 날짜보다 마지막 거래 날짜가 이후인 경우
            if actual_start_dt > start_dt:
                print(f"  누락 기간 백테스트: {actual_start_dt.strftime('%Y-%m-%d')} 이후부터")
            else:
                # 이미 모든 데이터가 처리된 경우
                if end_dt is None or last_trade_date >= end_dt:
                    print(f"  모든 데이터가 이미 백테스트되었습니다.")
                    print(f"  마지막 거래: {last_trade_date.strftime('%Y-%m-%d')}")
                    if end_dt is not None:
                        print(f"  요청 종료 날짜: {end_dt.strftime('%Y-%m-%d')}")
                    
                    # 기존 백테스트 결과에서 마지막 상태 조회
                    df_results = data_manager.load_backtest_results(ticker=ticker, condition_dict=current_condition)
                    if df_results is not None and len(df_results) > 0:
                        # 해당 조건의 최신 결과 찾기
                        filtered_results = df_results[
                            (df_results['buy_angle_threshold'] == buy_threshold) &
                            (df_results['sell_angle_threshold'] == sell_threshold) &
                            (df_results['stop_loss_percent'] == stop_loss_val) &
                            (df_results['min_sell_price'] == min_sell_val) &
                            (df_results['price_slippage'] == price_slip_val) &
                            (df_results['window'] == window_val) &
                            (df_results['aspect_ratio'] == aspect_ratio_val) &
                            (df_results['interval'] == interval)
                        ]
                        
                        if len(filtered_results) > 0:
                            # 마지막 거래 찾기
                            last_trade_row = filtered_results[
                                filtered_results['trade_date'].notna() &
                                (filtered_results['action'].str.startswith('BUY') | filtered_results['action'].str.startswith('SELL'))
                            ]
                            
                            if len(last_trade_row) > 0:
                                last_trade_action = last_trade_row.iloc[-1]['action']
                                if last_trade_action.startswith('BUY'):
                                    # 마지막이 매수면 보유 중
                                    return 'hold'
                                elif last_trade_action.startswith('SELL'):
                                    # 마지막이 매도면 매도 상태
                                    return 'sell'
                    
                    # 상태 조회 실패 시 기본값
                    return 'unknown'
        else:
            print(f"  기존 기록 없음. 처음부터 백테스트합니다.")
        
        # 데이터 업데이트 (기록 데이터 업데이트)
        print("\n[단계 1] BTC 일봉 데이터 기록 업데이트 중...")
        # start_date가 None인 경우 이미 데이터를 업데이트했으므로 다시 업데이트하지 않음
        if start_date is None:
            # 이미 temp_df로 데이터를 가져왔으므로 그대로 사용
            df = data_manager.load_history_from_file()
            if df is None or len(df) == 0:
                print("오류: 데이터 로드 실패")
                return
        else:
            df = data_manager.update_history_data(ticker=ticker, interval=interval, start_date='2014-01-01')
        
        if df is None or len(df) == 0:
            print("오류: 데이터 수집 실패")
            return
        
        print(f"수집 완료: {len(df)}개 데이터")
        print(f"전체 기간: {df.index[0]} ~ {df.index[-1]}")
        
        # 추세선 계산을 위해 window만큼 이전 데이터도 필요하므로
        # actual_start_dt 이전 window일만큼 포함
        window_size = window or Config.DEFAULT_WINDOW
        trendline_start_dt = actual_start_dt - pd.Timedelta(days=window_size * 2)  # 여유있게 2배
        
        # 실제 백테스트 시작 날짜로 필터링 (추세선 계산용 이전 데이터 포함)
        original_count = len(df)
        df = df[df.index >= trendline_start_dt]
        
        print(f"  추세선 계산용 데이터 포함 (window={window_size}): {trendline_start_dt.strftime('%Y-%m-%d')}부터")
        
        # 종료 날짜 필터링 (설정된 경우)
        if end_dt is not None:
            df = df[df.index <= end_dt]
        
        filtered_count = len(df)
        
        if len(df) == 0:
            print(f"누락된 백테스트 데이터가 없습니다.")
            print(f"  요청 시작 날짜: {start_dt.strftime('%Y-%m-%d')}")
            print(f"  실제 시작 날짜: {actual_start_dt.strftime('%Y-%m-%d')}")
            if end_dt is not None:
                print(f"  요청 종료 날짜: {end_dt.strftime('%Y-%m-%d')}")
            return
        
        print(f"백테스트 대상: {filtered_count}개 데이터 (제외: {original_count - filtered_count}개)")
        print(f"백테스트 기간: {df.index[0]} ~ {df.index[-1]}")
        
        if use_previous_asset and previous_final_asset is not None:
            print(f"  이전 최종 자산을 초기 자본으로 사용: {previous_final_asset:,.0f}원")
        
        # 추세선 계산
        print("\n[단계 2] 추세선 각도 계산 중...")
        trendline_calc = TrendlineCalculator(
            window=window or Config.DEFAULT_WINDOW,
            aspect_ratio=aspect_ratio or Config.DEFAULT_ASPECT_RATIO
        )
        df = trendline_calc.calculate_trendline(df)
        
        # 백테스트는 actual_start_dt 이후 데이터만 사용
        # 추세선 계산을 위해 이전 데이터를 포함했지만, 백테스트는 실제 시작 날짜부터만
        original_df_for_backtest = df.copy()
        df_for_backtest = df[df.index >= actual_start_dt]
        
        if len(df_for_backtest) == 0:
            print(f"누락된 백테스트 데이터가 없습니다.")
            return
        
        print(f"  백테스트 실행 데이터: {len(df_for_backtest)}개 ({df_for_backtest.index[0]} ~ {df_for_backtest.index[-1]})")
        
        # 백테스트 실행
        print("\n[단계 3] 백테스트 실행 중...")
        buy_threshold = buy_angle_threshold or Config.DEFAULT_BUY_ANGLE_THRESHOLD
        sell_threshold = sell_angle_threshold or Config.DEFAULT_SELL_ANGLE_THRESHOLD
        print(f"매수 조건: 추세선 각도 >= {buy_threshold}°")
        print(f"매도 조건: 추세선 각도 <= {sell_threshold}° (단, 매도가격 > {min_sell_price or Config.DEFAULT_MIN_SELL_PRICE:,.0f}원)")
        print(f"손절 조건: 수익률 <= {stop_loss_percent or Config.DEFAULT_STOP_LOSS_PERCENT}%")
        print(f"거래 가격: 종가 ± {price_slippage or Config.DEFAULT_PRICE_SLIPPAGE}원")
        
        # 이전 자산을 초기 자본으로 사용하는 경우
        actual_initial_capital = initial_capital
        if use_previous_asset and previous_final_asset is not None:
            actual_initial_capital = previous_final_asset
        
        backtest_engine = BacktestEngine(
            buy_angle_threshold=buy_angle_threshold,
            sell_angle_threshold=sell_angle_threshold,
            stop_loss_percent=stop_loss_percent,
            min_sell_price=min_sell_price,
            price_slippage=price_slippage,
            initial_capital=actual_initial_capital
        )
        result = backtest_engine.run(df_for_backtest)
        
        # 결과의 수익률을 원래 초기 자본 기준으로 재계산
        if use_previous_asset and previous_final_asset is not None and initial_capital is not None:
            original_initial_capital = initial_capital or Config.DEFAULT_INITIAL_CAPITAL
            # 전체 기간에 대한 수익률 계산
            original_total_return = ((result['final_asset'] - original_initial_capital) / original_initial_capital) * 100
            result['original_initial_capital'] = original_initial_capital
            result['original_total_return'] = original_total_return
        
        # 결과 출력
        print("\n[단계 4] 백테스트 결과 출력 중...")
        reporter = ResultReporter()
        reporter.print_backtest_results(result)
        
        # 그래프 생성 (전체 데이터 포함한 df 사용)
        print("\n[단계 5] 백테스트 결과 그래프 생성 중...")
        visualizer = Visualizer()
        visualizer.plot_backtest_results(
            original_df_for_backtest, result, 
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold
        )
        
        # 기간별 그래프 생성
        print("\n[단계 5-1] 기간별 백테스트 결과 그래프 생성 중...")
        visualizer.plot_backtest_results_by_periods(
            original_df_for_backtest, result,
            base_dir='./images',
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold
        )
        
        # 백테스트 결과 저장
        print("\n[단계 6] 백테스트 결과 저장 중...")
        # 저장 시에는 원래 요청한 시작 날짜와 실제 사용한 초기 자본을 저장
        data_manager.save_backtest_result(
            df=df_for_backtest,
            result=result,
            test_start_date=start_dt,  # 원래 요청한 시작 날짜 유지
            test_end_date=end_dt,
            buy_angle_threshold=buy_threshold,
            sell_angle_threshold=sell_threshold,
            stop_loss_percent=stop_loss_percent or Config.DEFAULT_STOP_LOSS_PERCENT,
            min_sell_price=min_sell_price or Config.DEFAULT_MIN_SELL_PRICE,
            price_slippage=price_slippage or Config.DEFAULT_PRICE_SLIPPAGE,
            initial_capital=initial_capital or Config.DEFAULT_INITIAL_CAPITAL,  # 원래 초기 자본 유지
            window=window or Config.DEFAULT_WINDOW,
            aspect_ratio=aspect_ratio or Config.DEFAULT_ASPECT_RATIO,
            ticker=ticker,
            interval=interval
        )
        
        print("\n백테스트 완료!")
        
        # 마지막 거래 상태 반환
        return result.get('last_trade_status', 'unknown')
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


if __name__ == "__main__":

    # 시작 날짜 설정 (기본값: 2014-01-01)
    # 원하는 날짜로 변경 가능 (예: '2020-01-01', '2021-06-01')
    start_date = None
    
    # 종료 날짜 설정 (None이면 마지막까지 사용)
    # 원하는 날짜로 변경 가능 (예: '2023-12-31', '2024-06-30')
    # None으로 설정하면 모든 데이터 사용
    end_date = None  # 예: '2016-01-01'
    
    # 시작 잔고 설정 (기본값: 1,000,000원)
    initial_capital = 1_000_000  # 예: 10000000 (1천만원)
    while 1:
        testresult = main(
            start_date=start_date, 
            end_date=end_date, 
            buy_angle_threshold=3.0, 
            sell_angle_threshold=-4.0, 
            stop_loss_percent=-3.0, 
            min_sell_price=20000, 
            price_slippage=1000,
            aspect_ratio=3,
            window=13,
            initial_capital=initial_capital,
            ticker='BTC',
            interval='24h'
        )
        print(f"last_trade_status: {testresult}")
        for i in range(3600):
            time.sleep(1)
            print(f"다음 테스트까지 {(3600*1 - i)//3600}시간 {(3600*3 - i)%3600//60}분 {(3600*3 - i)%60}초 남음", end='\r')


    
