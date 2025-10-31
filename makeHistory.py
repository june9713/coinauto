"""
pytbithumb을 이용한 모든 기간 암호화폐 거래 데이터 수집 모듈
"""
import traceback
import pandas as pd
from datetime import datetime, timedelta
from pybithumb import Bithumb
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
        conkey = os.getenv('CONKEY')
        seckey = os.getenv('SECKEY')
        
        # Bithumb 객체 생성 (API 키가 있으면 전달)
        if conkey and seckey:
            bithumb = Bithumb(conkey, seckey)
        else:
            # API 키가 없어도 공개 데이터 조회는 가능
            bithumb = Bithumb()
        
        print(f"  [collect_all_periods_data] API 연결 확인: Bithumb 객체 생성 완료")
        print(f"  수집 대상: {ticker}, 간격: {interval}")
        
        # 데이터 수집 (재시도 로직 포함)
        # 문서에 따르면: Bithumb.get_candlestick("BTC", chart_intervals="30m")
        max_retries = 3
        retry_delay = 2
        df = None
        last_error = None
        
        for retry_count in range(max_retries):
            try:
                print(f"  API 호출 중... (시도 {retry_count + 1}/{max_retries})")
                print(f"    호출: get_candlestick('{ticker}', chart_intervals='{interval}')")
                # 문서에 맞는 올바른 호출 방식: chart_intervals 키워드 인자 사용
                df = bithumb.get_candlestick(ticker, chart_intervals=interval)
                
                if df is not None:
                    print(f"  API 응답 수신: DataFrame 타입 확인")
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
        
        if df is None or len(df) == 0:
            print(f"경고: {ticker}의 {interval} 데이터를 수집하지 못했습니다.")
            if last_error:
                print(f"  마지막 오류: {type(last_error).__name__}: {str(last_error)}")
            return pd.DataFrame()
        
        # DataFrame 타입 확인
        if not isinstance(df, pd.DataFrame):
            print(f"  오류: API 응답이 DataFrame이 아닙니다. 타입: {type(df)}")
            return pd.DataFrame()
        
        print(f"  수집된 데이터: {len(df)}개 행, 컬럼: {list(df.columns)}")
        
        # 인덱스를 datetime으로 변환 (이미 datetime이 아닌 경우)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
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
        # .env 파일에서 API 키 로드
        conkey = os.getenv('CONKEY')
        seckey = os.getenv('SECKEY')
        
        # Bithumb 객체 생성 (API 키가 있으면 전달)
        if conkey and seckey:
            bithumb = Bithumb(conkey, seckey)
        else:
            # API 키가 없어도 공개 데이터 조회는 가능
            bithumb = Bithumb()
        
        print(f"  API 연결 확인: Bithumb 객체 생성 완료")
        print(f"  수집 대상: {ticker}, 간격: {interval}")
        
        # 먼저 최근 데이터부터 수집
        # 주의: pybithumb의 get_candlestick은 개수 지정 파라미터가 없음
        # 기본적으로 최근 데이터만 반환하므로, 한 번의 호출로 가능한 모든 데이터를 수집
        all_data = []
        collected_count = 0
        max_retries = 3  # 최대 재시도 횟수
        retry_delay = 2  # 재시도 대기 시간 (초)
        
        # 문서에 따르면 get_candlestick은 기본적으로 최근 데이터를 반환
        # 여러 번 호출해도 같은 데이터가 반환될 수 있으므로 한 번만 호출
        try:
            # API 호출 (재시도 로직 포함)
            df = None
            last_error = None
            
            for retry_count in range(max_retries):
                try:
                    print(f"  API 호출 중... (시도 {retry_count + 1}/{max_retries})")
                    print(f"    호출: get_candlestick('{ticker}', chart_intervals='{interval}')")
                    # 문서에 맞는 올바른 호출 방식: chart_intervals 키워드 인자 사용
                    df = bithumb.get_candlestick(ticker, chart_intervals=interval)
                    
                    if df is not None:
                        print(f"  API 응답 수신: DataFrame 타입 확인")
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
            
            if df is None or len(df) == 0:
                print(f"  오류: API 호출 실패 - 데이터가 없습니다.")
                if last_error:
                    print(f"  마지막 오류: {type(last_error).__name__}: {str(last_error)}")
                return pd.DataFrame()
            
            # DataFrame 타입 확인
            if not isinstance(df, pd.DataFrame):
                print(f"  오류: API 응답이 DataFrame이 아닙니다. 타입: {type(df)}")
                return pd.DataFrame()
            
            print(f"  수집된 데이터: {len(df)}개 행, 컬럼: {list(df.columns)}")
            
            # 인덱스를 datetime으로 변환
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # 컬럼명 소문자 변환
            df.columns = [col.lower() for col in df.columns]
            
            # 데이터 정렬
            df = df.sort_index()
            
            all_data.append(df)
            collected_count += len(df)
            
            print(f"  진행: {len(df)}개 데이터 수집 완료")
                        
        except Exception as e:
            err = traceback.format_exc()
            print(f"  데이터 수집 중 오류 발생:")
            print("err", err)
        
        if len(all_data) == 0:
            print(f"\n경고: {ticker}의 {interval} 데이터를 수집하지 못했습니다.")
            print(f"  fallback: 기본 수집 방식으로 재시도합니다...")
            # fallback: 기본 방식으로 재시도
            try:
                fallback_df = collect_all_periods_data(ticker, interval)
                if fallback_df is not None and len(fallback_df) > 0:
                    print(f"  성공: fallback 방식으로 {len(fallback_df)}개 데이터 수집 완료")
                    return fallback_df
            except Exception as fallback_error:
                err = traceback.format_exc()
                print(f"  fallback도 실패:")
                print("err", err)
            
            return pd.DataFrame()
        
        # 모든 데이터프레임을 결합 (현재는 하나의 데이터프레임만 있지만, 일관성을 위해)
        final_df = pd.concat(all_data, ignore_index=False)
        
        # 중복 제거 (인덱스 기준)
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        
        # 시간순 정렬
        final_df = final_df.sort_index()
        
        print(f"\n성공: {ticker} {interval} 총 {len(final_df)}개 데이터 수집 완료")
        print(f"  기간: {final_df.index[0]} ~ {final_df.index[-1]}")
        
        return final_df
        
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

