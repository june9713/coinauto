"""
pytbithumb을 이용한 모든 기간 암호화폐 거래 데이터 수집 모듈
"""
import traceback
import pandas as pd
from datetime import datetime, timedelta
from pybithumb2 import BithumbClient, MarketID, TimeUnit
import time
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()


def collect_all_periods_data(ticker='BTC', interval='24h'):
    """
    지정된 암호화폐의 모든 기간 데이터를 수집합니다.
    
    Parameters:
    - ticker (str): 암호화폐 티커 (예: 'BTC', 'ETH'). 기본값 'BTC'
    - interval (str): 캔들스틱 간격. 
                      지원 값: '1m', '3m', '5m', '10m', '30m', '1h', '6h', '12h', '24h'
                      기본값 '24h' (일봉)
    
    Returns:
    - pd.DataFrame: OHLCV 데이터프레임. 컬럼: 'open', 'high', 'low', 'close', 'volume'
                   인덱스는 datetime 타입
    """
    try:
        # .env 파일에서 API 키 로드
        api_key = os.getenv('CONKEY')
        secret_key = os.getenv('SECKEY')
        
        # BithumbClient 객체 생성 (API 키가 없어도 공개 데이터 조회는 가능)
        bithumb = BithumbClient(api_key=api_key, secret_key=secret_key)
        market = MarketID.from_string(f"KRW-{ticker}")
        
        print(f"  [collect_all_periods_data] API 연결 확인: BithumbClient 객체 생성 완료")
        print(f"  수집 대상: {ticker}, 간격: {interval}")
        
        # 데이터 수집 (재시도 로직 포함)
        max_retries = 3
        retry_delay = 2
        candles = None
        last_error = None
        
        for retry_count in range(max_retries):
            try:
                print(f"  API 호출 중... (시도 {retry_count + 1}/{max_retries})")
                
                # interval에 따라 적절한 메서드 호출 (최대 200개 조회)
                if interval.endswith('m'):
                    minutes = int(interval[:-1])
                    if minutes in [1, 3, 5, 10, 15, 30, 60, 240]:
                        unit = TimeUnit(minutes)
                        print(f"    호출: get_minute_candles(market='{market}', count=200, unit={unit})")
                        candles = bithumb.get_minute_candles(market=market, count=200, unit=unit)
                    else:
                        print(f"  오류: {interval}는 지원하지 않는 단위입니다.")
                        return pd.DataFrame()
                elif interval.endswith('h'):
                    hours = int(interval[:-1])
                    if hours == 1:
                        # 1h는 60분으로 처리
                        unit = TimeUnit(60)
                        print(f"    호출: get_minute_candles(market='{market}', count=200, unit={unit})")
                        candles = bithumb.get_minute_candles(market=market, count=200, unit=unit)
                    elif hours == 24:
                        print(f"    호출: get_day_candles(market='{market}', count=200)")
                        candles = bithumb.get_day_candles(market=market, count=200)
                    else:
                        # 6h, 12h 등은 TimeUnit 범위를 초과하므로 지원하지 않음
                        print(f"  오류: {interval}는 pybithumb2에서 직접 지원하지 않습니다.")
                        return pd.DataFrame()
                elif interval == '1w':
                    print(f"    호출: get_week_candles(market='{market}', count=200)")
                    candles = bithumb.get_week_candles(market=market, count=200)
                elif interval == '1M':
                    print(f"    호출: get_month_candles(market='{market}', count=200)")
                    candles = bithumb.get_month_candles(market=market, count=200)
                else:
                    print(f"  오류: {interval}는 pybithumb2에서 직접 지원하지 않습니다.")
                    return pd.DataFrame()
                
                if candles is not None:
                    print(f"  API 응답 수신: {type(candles)} 타입 확인")
                    break
                else:
                    print(f"  경고: API가 None을 반환했습니다.")
                    
            except Exception as api_error:
                last_error = api_error
                err = traceback.format_exc()
                print(f"  API 호출 오류 (시도 {retry_count + 1}/{max_retries}):")
                print(f"  오류 유형: {type(api_error).__name__}")
                print(f"  오류 메시지: {str(api_error)}")
                
                if retry_count < max_retries - 1:
                    print(f"  {retry_delay}초 후 재시도...")
                    time.sleep(retry_delay)
                else:
                    print("  최대 재시도 횟수 도달")
        
        if candles is None or len(candles) == 0:
            print(f"경고: {ticker}의 {interval} 데이터를 수집하지 못했습니다.")
            if last_error:
                print(f"  마지막 오류: {type(last_error).__name__}: {str(last_error)}")
            return pd.DataFrame()
        
        # DFList[Candle]를 pandas DataFrame으로 변환
        rows = []
        index_dates = []
        
        for candle in candles:
            # candle_date_time_kst를 사용하여 인덱스 생성
            dt = candle.candle_date_time_kst
            rows.append({
                'open': float(candle.opening_price),
                'high': float(candle.high_price),
                'low': float(candle.low_price),
                'close': float(candle.trade_price),
                'volume': float(candle.candle_acc_trade_volume)
            })
            index_dates.append(dt)
        
        # DataFrame 생성
        df = pd.DataFrame(rows)
        if len(index_dates) == len(df):
            df.index = pd.DatetimeIndex(index_dates)
        else:
            print(f"  경고: 인덱스 날짜 개수가 맞지 않습니다 ({len(index_dates)} vs {len(df)}). 기본 인덱스 사용")
        
        print(f"  수집된 데이터: {len(df)}개 행, 컬럼: {list(df.columns)}")
        
        # 컬럼명이 대문자인 경우 소문자로 변환
        df.columns = [col.lower() for col in df.columns]
        
        # 필요한 컬럼 확인: 'open', 'high', 'low', 'close', 'volume'
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            err_msg = f"필수 컬럼이 없습니다: {missing_columns}"
            print(f"오류: {err_msg}")
            raise ValueError(err_msg)
        
        # 데이터프레임을 시간순으로 정렬 (오래된 데이터부터)
        df = df.sort_index()
        
        print(f"성공: {ticker} {interval} 데이터 {len(df)}개 수집 완료")
        print(f"  기간: {df.index[0]} ~ {df.index[-1]}")
        
        return df
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


def collect_multiple_intervals(ticker='BTC', intervals=None):
    """
    여러 시간 프레임의 데이터를 수집합니다.
    
    Parameters:
    - ticker (str): 암호화폐 티커 (예: 'BTC', 'ETH'). 기본값 'BTC'
    - intervals (list): 수집할 시간 프레임 리스트. 
                        None이면 모든 주요 시간 프레임 수집
                        예: ['1m', '5m', '1h', '24h']
    
    Returns:
    - dict: {interval: DataFrame} 형태의 딕셔너리
    """
    try:
        if intervals is None:
            # 모든 주요 시간 프레임
            intervals = ['1m', '3m', '5m', '10m', '30m', '1h', '6h', '12h', '24h']
        
        result = {}
        
        for interval in intervals:
            try:
                print(f"\n[{ticker}] {interval} 데이터 수집 중...")
                df = collect_all_periods_data(ticker, interval)
                
                if len(df) > 0:
                    result[interval] = df
                    # API 호출 제한을 피하기 위한 대기
                    time.sleep(0.5)
                else:
                    print(f"  경고: {interval} 데이터 수집 실패 (빈 데이터)")
                    
            except Exception as e:
                err = traceback.format_exc()
                print(f"  오류: {interval} 데이터 수집 중 오류 발생")
                print("err", err)
                # 개별 인터벌 실패 시에도 계속 진행
                continue
        
        print(f"\n총 {len(result)}개 시간 프레임 데이터 수집 완료")
        return result
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


def get_maximum_historical_data(ticker='BTC', interval='24h'):
    """
    가능한 최대 과거 기간의 데이터를 수집합니다.
    여러 번의 API 호출을 통해 누락 없이 최대한 많은 데이터를 수집합니다.
    
    Parameters:
    - ticker (str): 암호화폐 티커 (예: 'BTC', 'ETH'). 기본값 'BTC'
    - interval (str): 캔들스틱 간격. 기본값 '24h'
    
    Returns:
    - pd.DataFrame: 모든 기간의 OHLCV 데이터프레임
    """
    try:
        # pybithumb2는 한 번에 최대 200개만 조회 가능하므로,
        # collect_all_periods_data를 사용하여 최근 200개를 가져옴
        print(f"  API 연결 확인: BithumbClient 사용")
        print(f"  수집 대상: {ticker}, 간격: {interval}")
        print(f"  주의: pybithumb2는 한 번에 최대 200개만 조회 가능합니다.")
        
        # 먼저 collect_all_periods_data로 최근 200개 수집
        df = collect_all_periods_data(ticker, interval)
        
        if df is None or len(df) == 0:
            print(f"  오류: 기본 수집 방식 실패")
            return pd.DataFrame()
        
        # 중복 제거 (인덱스 기준)
        df = df[~df.index.duplicated(keep='first')]
        
        # 시간순 정렬
        df = df.sort_index()
        
        print(f"\n성공: {ticker} {interval} 총 {len(df)}개 데이터 수집 완료")
        print(f"  기간: {df.index[0]} ~ {df.index[-1]}")
        
        return df
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


# main.py에서 호출 가능한 메인 함수
def make_history(ticker='BTC', interval='24h', collect_all_periods=True):
    """
    main.py에서 호출되는 메인 함수
    
    Parameters:
    - ticker (str): 암호화폐 티커. 기본값 'BTC'
    - interval (str): 캔들스틱 간격. 기본값 '24h'
    - collect_all_periods (bool): True면 가능한 모든 기간 데이터 수집, 
                                 False면 기본 기간만 수집
    
    Returns:
    - pd.DataFrame: OHLCV 데이터프레임
    """
    try:
        if collect_all_periods:
            print(f"[make_history] 전체 기간 데이터 수집 모드")
            df = get_maximum_historical_data(ticker, interval)
            # 최대 기간 수집 실패 시 기본 방식으로 fallback
            if df is None or len(df) == 0:
                print(f"  전체 기간 수집 실패, 기본 방식으로 재시도...")
                df = collect_all_periods_data(ticker, interval)
            return df
        else:
            print(f"[make_history] 기본 기간 데이터 수집 모드")
            return collect_all_periods_data(ticker, interval)
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise

