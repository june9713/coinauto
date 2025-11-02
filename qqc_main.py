"""
QQC 백테스트 메인 실행 모듈
qqc_test 모듈을 호출하여 백테스트 실행
"""
import os

import warnings
# Python 3.13 GIL 경고 방지
# pandas._libs.pandas_parser가 GIL 없이 안전하게 실행될 수 있다고 선언하지 않아 발생하는 경고를 필터링
#warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*global interpreter lock.*')

from ast import Or
import traceback
import time
import datetime
from datetime import timedelta
import pandas as pd
from config import Config
from data_manager import DataManager
from qqc_test import QQCTestEngine
from result_reporter import ResultReporter
from visualizer import Visualizer
from condition_manager import ConditionManager
from trade_state import TradeStateManager
from trader import Trader



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
        
        # start_date가 None인 경우 저장된 데이터에서 첫 번째 날짜 확인
        if start_dt is None:
            print("\n데이터에서 첫 번째 날짜 확인 중...")
            # 먼저 저장된 데이터 확인 (전체 다운로드 없이)
            existing_df = data_manager.load_history_from_file()
            existing_df_before_update = data_manager.load_history_from_file()
            if existing_df is not None and len(existing_df) > 0:
                # 저장된 데이터가 있으면 첫 번째 날짜를 시작 날짜로 사용
                start_dt = existing_df.index[0]
                print(f"저장된 데이터에서 첫 번째 날짜 확인: {start_dt.strftime('%Y-%m-%d')}")
            else:
                # 저장된 데이터가 없으면 전체 데이터 다운로드하여 첫 번째 날짜 확인
                print("저장된 데이터가 없습니다. 전체 데이터 다운로드를 시작합니다...")
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
        
        # 데이터 업데이트 전에 기존 데이터 존재 여부 확인 (중요: 순서가 중요함!)
        #print("\n[단계 1-0] 기존 데이터 존재 여부 확인 중...")
        #existing_df_before_update = data_manager.load_history_from_file()
        
        if existing_df_before_update is not None and len(existing_df_before_update) > 0:
            print(f"  ✓ 기존 데이터 발견: {len(existing_df_before_update)}개 데이터")
            print(f"  기존 데이터 기간: {existing_df_before_update.index[0]} ~ {existing_df_before_update.index[-1]}")
            print(f"  → 증분 업데이트 후 마지막 56개만 백테스트 실행 예정")
            is_full_backtest = False
        else:
            print(f"  ✗ 기존 데이터 없음 (None 또는 빈 데이터)")
            print(f"  → 전체 데이터 다운로드 후 전체 백테스트 실행 예정")
            is_full_backtest = True
        
        # 데이터 업데이트 (저장된 데이터가 있으면 마지막 날짜 이후부터만 조회)
        print("\n[단계 1] 데이터 수집 중...")
        # update_history_data가 저장된 데이터가 있으면 마지막 날짜 이후부터만 조회하고,
        # 저장된 데이터가 없으면 start_date부터 전체 다운로드합니다.
        df = data_manager.update_history_data(ticker=ticker, interval=interval, start_date='2014-01-01')
        
        if df is None or len(df) == 0:
            print("오류: 데이터 수집 실패")
            return
        
        print(f"수집 완료: {len(df)}개 데이터")
        print(f"전체 기간: {df.index[0]} ~ {df.index[-1]}")
        
        # 원본 데이터 보관 (기간별 그래프 생성을 위해)
        original_df_for_backtest = df.copy()
        
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
        
        # 백테스트 데이터 선택 로직
        # - 기존 데이터가 없었던 경우 (전체 다운로드): 전체 데이터로 백테스트
        # - 기존 데이터가 있었던 경우 (증분 업데이트): 마지막 56개만 사용
        print(f"\n[백테스트 데이터 선택]")
        print(f"  is_full_backtest 플래그: {is_full_backtest}")
        if is_full_backtest:
            # 전체 데이터로 백테스트 실행
            print(f"  → 전체 백테스트 모드: 기존 데이터가 없었으므로 전체 데이터로 백테스트 실행")
            print(f"  전체 다운로드된 데이터: {len(df)}개 데이터")
            print(f"  백테스트 기간: {df.index[0]} ~ {df.index[-1]}")
            df_for_backtest = df.copy()  # 복사본 사용
        else:
            # 마지막 56개만 사용 (과거 55개 + 현재 1개)
            # qqc_55.py와 동일한 로직: 마지막 57개 추출 후 마지막 1개 제외하여 56개 사용
            required_count = volume_window_val + 2  # 57개 필요
            if len(df) < required_count:
                print(f"\n경고: 데이터가 부족합니다 (필요: 최소 {required_count}개, 현재: {len(df)}개)")
                print("전체 데이터를 사용하여 백테스트를 실행합니다.")
                df_for_backtest = df
            else:
                # 마지막 57개를 추출한 후, 마지막 1개를 제외하여 56개 사용
                df_last_57 = df.tail(required_count)
                df_for_backtest = df_last_57.iloc[:-1]  # 마지막 인덱스 제외
                
                print(f"  → 부분 백테스트 모드: 기존 데이터가 있어서 마지막 56개만 백테스트 실행")
                print(f"  백테스트 실행 데이터: 마지막 {required_count}개 추출 후 마지막 1개 제외 → {len(df_for_backtest)}개 사용")
                print(f"  백테스트 실행 기간: {df_for_backtest.index[0]} ~ {df_for_backtest.index[-1]}")
                print(f"    - 인덱스 0-{volume_window_val-1}: 과거 평균용 ({volume_window_val}개)")
                print(f"    - 인덱스 {volume_window_val}: 현재 캔들")
                print(f"    - 실제 데이터의 마지막 캔들(인덱스 {required_count-1})은 제외됨")
        
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
        
        # 백테스트 실행
        print(f"\n[백테스트 실행] 실제 사용 데이터: {len(df_for_backtest)}개")
        print(f"  실행 기간: {df_for_backtest.index[0]} ~ {df_for_backtest.index[-1]}")
        result = qqc_engine.run(df_for_backtest)
        
        # 백테스트 결과에서 마지막 거래 상태 확인
        last_trade_status = result.get('last_trade_status', 'unknown')
        
        # 전체 백테스트인 경우 확인 로그
        if is_full_backtest:
            print(f"\n[전체 백테스트 확인]")
            print(f"  백테스트 실행 데이터: {len(df_for_backtest)}개")
            print(f"  총 거래 횟수: {result.get('total_trades', 0)}회")
            print(f"  초기 자본: {result.get('initial_capital', 0):,.0f}원")
            print(f"  최종 자산: {result.get('final_asset', 0):,.0f}원")
            print(f"  총 수익률: {result.get('total_return', 0):+.2f}%")
        
        # 실제 거래 실행 로직
        # 초기화 여부는 main 함수 호출 전에 확인하므로 여기서는 실행만
        # 실제 거래 실행은 main 함수 외부에서 처리 (상태 확인 후)
        
        # 결과 출력
        print("\n[단계 3] 백테스트 결과 출력 중...")
        reporter = ResultReporter()
        reporter.print_backtest_results(result)
        
        # 그래프 생성 (전체 데이터 포함한 df 사용)
        print("\n[단계 4] 백테스트 결과 그래프 생성 중...")
        visualizer = Visualizer()
        visualizer.plot_backtest_results(
            original_df_for_backtest, result,
            buy_threshold=0,  # QQC 전략은 각도 기준이 아니므로 0으로 설정
            sell_threshold=0,
            volume_window=volume_window_val,
            ma_window=ma_window_val,
            volume_multiplier=volume_multiplier_val
        )
        
        # 기간별 그래프 생성
        print("\n[단계 4-1] 기간별 백테스트 결과 그래프 생성 중...")
        visualizer.plot_backtest_results_by_periods(
            original_df_for_backtest, result,
            base_dir='./images',
            buy_threshold=0,  # QQC 전략은 각도 기준이 아니므로 0으로 설정
            sell_threshold=0,
            volume_window=volume_window_val,
            ma_window=ma_window_val,
            volume_multiplier=volume_multiplier_val
        )
        
        # 백테스트 결과 저장
        print("\n[단계 5] 백테스트 결과 저장 중...")
        if is_full_backtest:
            print(f"  전체 백테스트 결과를 폴더별/시간단위별로 저장합니다.")
            print(f"  저장 대상 데이터: 백테스트에 사용된 {len(df_for_backtest)}개 데이터")
        else:
            print(f"  증분 백테스트 결과를 저장합니다.")
            print(f"  저장 대상 데이터: 백테스트에 사용된 {len(df_for_backtest)}개 데이터")
        
        # 저장 시 백테스트에 사용된 데이터(df_for_backtest)를 전달
        # 전체 백테스트일 때는 전체 데이터, 증분 백테스트일 때는 마지막 56개
        data_manager.save_qqc_backtest_result(
            df=df_for_backtest,  # 백테스트에 사용된 데이터를 그대로 전달
            result=result,
            test_start_date=df_for_backtest.index[0] if len(df_for_backtest) > 0 else start_dt,
            test_end_date=df_for_backtest.index[-1] if len(df_for_backtest) > 0 else end_dt,
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
        
        # 마지막 거래 상태와 결과 반환 (실제 거래 실행을 위해)
        return {
            'last_trade_status': result.get('last_trade_status', 'unknown'),
            'result': result,
            'trades': result.get('trades', []),
            'final_asset': result.get('final_asset', initial_cap_val)
        }
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


def wait_start(interval):
    now = datetime.datetime.now()
    #3m 이면 datetime now 가 3,6,9,12,15,18,....57 분 0초 가 아닌! 10초까지 대기하도록 함
    #만야 5m 이면 5,10,15,20,25,30,35,40,45,50,55 분 0초 가 아닌! 10초까지 대기하도록 함
    #만야 10m 이면 10,20,30,40,50 분 0초 가 아닌! 10초까지 대기하도록 함
    #만야 30m 이면 30,40,50 분 0초 가 아닌! 10초까지 대기하도록 함
    #만야 1h 이면 1시간 0분 0초 가 아닌! 10초까지 대기하도록 함
    #만야 6h 이면 6시간 0분 0초 가 아닌! 10초까지 대기하도록 함
    #만야 12h 이면 12시간 0분 0초 가 아닌! 10초까지 대기하도록 함
    #만야 24h 이면 24시간 0분 0초 가 아닌! 10초까지 대기하도록 함
    if interval == '3m':
        force_wait = False
        if now.minute %3 == 0:
            force_wait = True

        while ( force_wait or now.minute % 3 != 0):
            time.sleep(1)
            if now.minute % 3 !=0:
                force_wait = False
            now = datetime.datetime.now()
            # 다음 3분 간격 시간 계산 (minute이 60 이상이 되지 않도록)
            next_minute = ((now.minute // 3) + 1) * 3
            if next_minute >= 60:
                next_start_time = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
            else:
                next_start_time = now.replace(minute=next_minute, second=0, microsecond=0)
            if next_start_time <= now:
                next_start_time = next_start_time + timedelta(minutes=3)
            left_seconds = int((next_start_time - now).total_seconds() + 10)
            print(f"대기 시간: {left_seconds//3600}시간 {left_seconds%3600//60}분 {left_seconds%60}초 남음", end='\r')
    elif interval == '5m':
        force_wait = False
        if now.minute %5 == 0:
            force_wait = True

        while ( force_wait or now.minute % 5 != 0):
            time.sleep(1)
            if now.minute % 5 !=0:
                force_wait = False
            now = datetime.datetime.now()
            # 다음 5분 간격 시간 계산 (minute이 60 이상이 되지 않도록)
            next_minute = ((now.minute // 5) + 1) * 5
            if next_minute >= 60:
                next_start_time = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
            else:
                next_start_time = now.replace(minute=next_minute, second=0, microsecond=0)
            if next_start_time <= now:
                next_start_time = next_start_time + timedelta(minutes=5)
            left_seconds = int((next_start_time - now).total_seconds() + 10)
            print(f"대기 시간: {left_seconds//3600}시간 {left_seconds%3600//60}분 {left_seconds%60}초 남음", end='\r')
    elif interval == '10m':
        force_wait = False
        if now.minute %10 == 0:
            force_wait = True

        while ( force_wait or now.minute % 10 != 0):
            time.sleep(1)
            if now.minute % 10 !=0:
                force_wait = False
            now = datetime.datetime.now()
            # 다음 10분 간격 시간 계산 (minute이 60 이상이 되지 않도록)
            next_minute = ((now.minute // 10) + 1) * 10
            if next_minute >= 60:
                next_start_time = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
            else:
                next_start_time = now.replace(minute=next_minute, second=0, microsecond=0)
            if next_start_time <= now:
                next_start_time = next_start_time + timedelta(minutes=10)
            left_seconds = int((next_start_time - now).total_seconds() + 10)
            print(f"대기 시간: {left_seconds//3600}시간 {left_seconds%3600//60}분 {left_seconds%60}초 남음", end='\r')
    elif interval == '30m':
        force_wait = False
        if now.minute %30 == 0:
            force_wait = True

        while ( force_wait or now.minute % 30 != 0):
            time.sleep(1)
            if now.minute % 30 !=0:
                force_wait = False
            now = datetime.datetime.now()
            # 다음 30분 간격 시간 계산 (minute이 60 이상이 되지 않도록)
            next_minute = ((now.minute // 30) + 1) * 30
            if next_minute >= 60:
                next_start_time = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
            else:
                next_start_time = now.replace(minute=next_minute, second=0, microsecond=0)
            if next_start_time <= now:
                next_start_time = next_start_time + timedelta(minutes=30)
            left_seconds = int((next_start_time - now).total_seconds() + 10)
            print(f"대기 시간: {left_seconds//3600}시간 {left_seconds%3600//60}분 {left_seconds%60}초 남음", end='\r')
    
    else:
        pass
    left_seconds = 10
    for i in range(10):
        time.sleep(1)
        left_seconds -= 1
        print(f"대기 시간: {left_seconds//3600}시간 {left_seconds%3600//60}분 {left_seconds%60}초 남음", end='\r')

def execute_real_trade(trader, ticker, backtest_status, prev_status, state, buy_cash_ratio):
    """
    실제 거래 실행
    
    Parameters:
    - trader: Trader 객체
    - ticker: 암호화폐 티커
    - backtest_status: 백테스트 결과 상태 ('buy', 'sell', 'hold', 'none', 'wait')
    - prev_status: 이전 상태
    - state: 현재 저장된 상태
    - buy_cash_ratio: 매수시 사용할 현금 비율
    
    Returns:
    - bool: 거래 실행 여부
    """
    try:
        # 초기화 직후 첫 백테스트에서 홀드/매도 상태면 실제 거래 안 함
        if state.get('last_backtest_status') == 'none' and backtest_status in ['hold', 'sell']:
            print(f"\n[실제 거래] 초기화 직후 홀드/매도 상태로 실제 거래 생략")
            print(f"  백테스트 상태: {backtest_status} (실제로는 코인 미보유 상태)")
            return False
        
        # 실제 자산 상태 확인
        balance = trader.get_balance(ticker)
        actual_cash = balance['cash']
        actual_coin = balance['coin']
        
        # 상태 매핑:
        # - 'buy', 'none', 'wait': 코인 미보유, 현금 보유 상태
        # - 'hold', 'sell': 코인 보유, 현금은 잔액만 보유 상태
        
        # 매수 조건: 백테스트에서 매수가 발생하고, 실제로 코인 미보유 상태인 경우
        if backtest_status == 'buy' and prev_status in ['none', 'wait', 'sell']:
            if actual_coin > 0:
                print(f"\n[실제 거래] 매수 생략: 이미 코인 보유 중 ({actual_coin:.8f} {ticker})")
                return False
            
            # 실제 매수 실행
            print(f"\n[실제 거래] 매수 실행")
            print(f"  백테스트 상태: {prev_status} -> {backtest_status}")
            print(f"  현재 현금: {actual_cash:,.0f}원")
            print(f"  사용할 현금 비율: {buy_cash_ratio*100:.0f}%")
            
            order_result = trader.buy_market_order(
                ticker=ticker,
                cash_ratio=buy_cash_ratio
            )
            
            if order_result['success']:
                print(f"  ✓ 매수 성공: 주문 ID {order_result['order_id']}")
                print(f"    매수 수량: {order_result['amount']:.8f} {ticker}")
                print(f"    매수 가격: {order_result['price']:,.0f}원")
            else:
                print(f"  ✗ 매수 실패: {order_result['message']}")
            
            return order_result['success']
        
        # 매도 조건: 백테스트에서 매도가 발생하고, 실제로 코인 보유 상태인 경우
        elif backtest_status == 'sell' and prev_status in ['hold', 'buy']:
            if actual_coin <= 0:
                print(f"\n[실제 거래] 매도 생략: 코인 미보유 상태")
                return False
            
            # 실제 매도 실행
            print(f"\n[실제 거래] 매도 실행")
            print(f"  백테스트 상태: {prev_status} -> {backtest_status}")
            print(f"  현재 코인 보유량: {actual_coin:.8f} {ticker}")
            
            order_result = trader.sell_market_order(
                ticker=ticker,
                coin_amount=actual_coin  # 전체 매도
            )
            
            if order_result['success']:
                print(f"  ✓ 매도 성공: 주문 ID {order_result['order_id']}")
                print(f"    매도 수량: {order_result['amount']:.8f} {ticker}")
                print(f"    매도 가격: {order_result['price']:,.0f}원")
            else:
                print(f"  ✗ 매도 실패: {order_result['message']}")
            
            return order_result['success']
        
        else:
            # 거래 불필요한 상태 변화
            return False
            
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        return False


if __name__ == "__main__":
    # 시작 날짜 설정 (기본값: 2014-01-01)
    start_date = None  # '2014-01-01'
    
    # 종료 날짜 설정 (None이면 마지막까지 사용)
    end_date = None  # 예: '2023-12-31'
    
    # QQC 전략 변수 설정
    volume_window = 55  # 거래량 평균 계산용 윈도우
    ma_window = 9  # 이동평균 계산용 윈도우
    volume_multiplier = 1.4  # 거래량 배수
    buy_cash_ratio = 0.9  # 매수시 사용할 현금 비율 (0.9 = 90%)
    hold_period = 15  # 매수 후 보유 기간 (캔들 수)
    profit_target = 17.6  # 이익실현 목표 수익률 (%)
    stop_loss = -28.6  # 손절 기준 수익률 (%)
    ticker = 'BTC'
    interval = '3m'
    price_slippage = 1000
    
    # 초기화 여부 확인
    print("="*80)
    print("QQC 백테스트 및 자동 거래 시스템")
    print("="*80)
    
    saved_state = TradeStateManager.load_state()
    
    if saved_state is None:
        print("\n저장된 거래 상태가 없습니다.")
        initialize = input("초기화하시겠습니까? (y/n): ").strip().lower()
        
        if initialize == 'y':
            print("\n초기화 모드: 실제 계좌 잔고를 조회하여 초기 자본으로 설정합니다.")
            try:
                trader = Trader()
                balance = trader.get_balance(ticker)
                actual_cash = balance['cash']
                actual_coin = balance['coin']
                
                print(f"\n현재 계좌 상태:")
                print(f"  현금 잔고: {actual_cash:,.0f}원")
                print(f"  코인 보유량: {actual_coin:.8f} {ticker}")
                
                # 초기 상태 저장
                initial_capital = actual_cash  # 실제 현금 잔고를 초기 자본으로 사용
                state = TradeStateManager.create_initial_state(
                    initial_capital=initial_capital,
                    actual_cash=actual_cash,
                    actual_coin_amount=actual_coin,
                    ticker=ticker
                )
                state['last_backtest_status'] = 'none'  # 초기 상태는 'none'
                TradeStateManager.save_state(state)
                
                print(f"\n초기화 완료:")
                print(f"  초기 자본: {initial_capital:,.0f}원")
                print(f"  실제 현금: {actual_cash:,.0f}원")
                print(f"  실제 코인: {actual_coin:.8f} {ticker}")
                
            except Exception as e:
                err = traceback.format_exc()
                print("err", err)
                print("\n오류: 초기화 실패. 기본값으로 진행합니다.")
                initial_capital = Config.DEFAULT_INITIAL_CAPITAL
                saved_state = None
        else:
            print("\n초기화하지 않습니다. 기본값으로 진행합니다.")
            initial_capital = Config.DEFAULT_INITIAL_CAPITAL
            saved_state = None
    else:
        print("\n이전 거래 상태를 불러왔습니다.")
        initialize = input("초기화하시겠습니까? (y/n): ").strip().lower()
        
        if initialize == 'y':
            print("\n초기화 모드: 실제 계좌 잔고를 조회하여 초기 자본으로 설정합니다.")
            try:
                trader = Trader()
                balance = trader.get_balance(ticker)
                actual_cash = balance['cash']
                actual_coin = balance['coin']
                
                print(f"\n현재 계좌 상태:")
                print(f"  현금 잔고: {actual_cash:,.0f}원")
                print(f"  코인 보유량: {actual_coin:.8f} {ticker}")
                
                # 초기 상태 저장
                initial_capital = actual_cash
                state = TradeStateManager.create_initial_state(
                    initial_capital=initial_capital,
                    actual_cash=actual_cash,
                    actual_coin_amount=actual_coin,
                    ticker=ticker
                )
                state['last_backtest_status'] = 'none'
                TradeStateManager.save_state(state)
                saved_state = state
                
                print(f"\n초기화 완료:")
                print(f"  초기 자본: {initial_capital:,.0f}원")
                
            except Exception as e:
                err = traceback.format_exc()
                print("err", err)
                print("\n오류: 초기화 실패. 이전 상태를 유지합니다.")
                initial_capital = saved_state.get('initial_capital', Config.DEFAULT_INITIAL_CAPITAL)
        else:
            print("\n이전 거래를 계속합니다.")
            initial_capital = saved_state.get('initial_capital', Config.DEFAULT_INITIAL_CAPITAL)
    
    # interval 의 갱신 주기의 갱신 시점에서 백테스트가 시작할 수 있도록 현재 시간 기준으로 갱신시간까지 대기
    # 모든 interval 시간의 대기 시간을 계산하여 대기함
    wait_start("3m")
    print("백테스트 시작")
    
    # Trader 객체 생성 (실제 거래용)
    trader = None
    try:
        trader = Trader()
    except Exception as e:
        print(f"경고: 실제 거래 기능 초기화 실패: {str(e)}")
        print("백테스트만 실행됩니다.")
    
    while 1:
        testresult = main(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            price_slippage=price_slippage,
            ticker=ticker,
            interval=interval,
            volume_window=volume_window,
            ma_window=ma_window,
            volume_multiplier=volume_multiplier,
            buy_cash_ratio=buy_cash_ratio,
            hold_period=hold_period,
            profit_target=profit_target,
            stop_loss=stop_loss
        )
        
        # 백테스트 결과 처리
        if isinstance(testresult, dict):
            backtest_status = testresult.get('last_trade_status', 'unknown')
            print(f"\n백테스트 결과 상태: {backtest_status}")
            
            # 상태 업데이트 및 실제 거래 실행
            if saved_state is not None and trader is not None:
                prev_status = saved_state.get('last_backtest_status', 'none')
                
                # 실제 거래 실행
                trade_executed = execute_real_trade(
                    trader=trader,
                    ticker=ticker,
                    backtest_status=backtest_status,
                    prev_status=prev_status,
                    state=saved_state,
                    buy_cash_ratio=buy_cash_ratio
                )
                
                # 상태 업데이트
                saved_state['last_backtest_status'] = backtest_status
                saved_state['last_update'] = pd.Timestamp.now()
                
                # 실제 잔고 재조회 및 업데이트
                try:
                    balance = trader.get_balance(ticker)
                    saved_state['actual_cash'] = balance['cash']
                    saved_state['actual_coin_amount'] = balance['coin']
                except Exception as e:
                    print(f"경고: 잔고 조회 실패: {str(e)}")
                
                TradeStateManager.save_state(saved_state)
            else:
                # 상태 저장되지 않았거나 trader가 없는 경우
                if saved_state is None:
                    print("\n경고: 거래 상태가 저장되지 않았습니다. 실제 거래를 실행하지 않습니다.")
                if trader is None:
                    print("\n경고: 실제 거래 기능이 사용 불가능합니다. 백테스트만 실행됩니다.")
        else:
            print(f"last_trade_status: {testresult}")
        
        wait_start("3m")

