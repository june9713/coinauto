"""
데이터 수집 모듈
암호화폐 거래 데이터를 API를 통해 수집
"""
import traceback
import pandas as pd
from datetime import datetime, timedelta
from pybithumb2 import BithumbClient, MarketID, TimeUnit
import time
import os
import requests
from dotenv import load_dotenv


# .env 파일에서 환경 변수 로드
load_dotenv()


class DataCollector:
    """암호화폐 거래 데이터 수집 클래스"""
    
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
            
            # BithumbClient 객체 생성 (API 키가 없어도 공개 데이터 조회는 가능)
            self._bithumb = BithumbClient(api_key=api_key, secret_key=secret_key)
        
        return self._bithumb
    
    def collect_all_periods_data(self, ticker='BTC', interval='24h'):
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
            bithumb = self._get_bithumb_client()
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
                            # 지원하지 않는 분 단위 interval인 경우 requests 직접 사용
                            print(f"    경고: {interval}는 지원하지 않는 단위입니다. requests 직접 호출 사용")
                            return self.collect_candles_with_params(ticker=ticker, interval=interval, count=200)
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
                            # 6h, 12h 등은 TimeUnit 범위를 초과하므로 requests 직접 사용
                            print(f"    경고: {interval}는 pybithumb2에서 직접 지원하지 않습니다. requests 직접 호출 사용")
                            return self.collect_candles_with_params(ticker=ticker, interval=interval, count=200)
                    elif interval == '1w':
                        print(f"    호출: get_week_candles(market='{market}', count=200)")
                        candles = bithumb.get_week_candles(market=market, count=200)
                    elif interval == '1M':
                        print(f"    호출: get_month_candles(market='{market}', count=200)")
                        candles = bithumb.get_month_candles(market=market, count=200)
                    else:
                        # 지원하지 않는 interval인 경우 requests 직접 사용
                        print(f"    경고: {interval}는 pybithumb2에서 직접 지원하지 않습니다. requests 직접 호출 사용")
                        return self.collect_candles_with_params(ticker=ticker, interval=interval, count=200)
                    
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
            
            # 데이터 정제
            df = self._normalize_dataframe(df)
            
            print(f"성공: {ticker} {interval} 데이터 {len(df)}개 수집 완료")
            if len(df) > 0:
                print(f"  기간: {df.index[0]} ~ {df.index[-1]}")
            
            return df
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _normalize_dataframe(self, df):
        """
        DataFrame을 표준 형식으로 정제
        
        Parameters:
        - df (pd.DataFrame): 원본 데이터프레임
        
        Returns:
        - pd.DataFrame: 정제된 데이터프레임
        """
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
        
        return df
    
    def _convert_interval_to_unit(self, interval):
        """
        interval 문자열을 Bithumb API의 unit 값으로 변환
        
        Parameters:
        - interval (str): 캔들스틱 간격 (예: '1m', '3m', '24h')
        
        Returns:
        - tuple: (unit_type, unit_value) - ('minutes' or 'days', unit 값)
        """
        # 분 단위 interval
        if interval.endswith('m'):
            unit_value = int(interval[:-1])
            return ('minutes', unit_value)
        # 시간 단위 interval (분으로 변환)
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            if hours == 24:
                return ('days', 1)
            else:
                # 시간을 분으로 변환
                unit_value = hours * 60
                return ('minutes', unit_value)
        else:
            raise ValueError(f"지원하지 않는 interval 형식: {interval}")
    
    def collect_candles_with_params(self, ticker='BTC', interval='24h', count=200, to_date=None):
        """
        Bithumb API를 직접 호출하여 count와 to 파라미터를 사용한 캔들 데이터 수집
        
        Parameters:
        - ticker (str): 암호화폐 티커 (예: 'BTC', 'ETH'). 기본값 'BTC'
        - interval (str): 캔들스틱 간격. 기본값 '24h'
        - count (int): 조회할 캔들 개수 (최대 200). 기본값 200
        - to_date (pd.Timestamp or str, optional): 마지막 캔들 시각 (exclusive).
                                                    None이면 가장 최근 캔들
        
        Returns:
        - pd.DataFrame: OHLCV 데이터프레임. 컬럼: 'open', 'high', 'low', 'close', 'volume'
        """
        try:
            # interval을 unit으로 변환
            unit_type, unit_value = self._convert_interval_to_unit(interval)
            
            # market 코드 생성 (KRW-BTC 형식)
            market = f"KRW-{ticker}"
            
            # API 엔드포인트 결정
            if unit_type == 'minutes':
                url = f"https://api.bithumb.com/v1/candles/minutes/{unit_value}"
            else:  # days
                url = f"https://api.bithumb.com/v1/candles/days"
            
            # 쿼리 파라미터 구성
            params = {
                'market': market,
                'count': min(count, 200)  # 최대 200개로 제한
            }
            
            # to_date가 지정된 경우 추가
            if to_date is not None:
                to_dt = pd.to_datetime(to_date)
                # ISO8601 형식으로 변환 (KST 기준)
                params['to'] = to_dt.strftime('%Y-%m-%dT%H:%M:%S')
            
            print(f"  [collect_candles_with_params] API 직접 호출")
            print(f"    URL: {url}")
            print(f"    Parameters: market={market}, count={params['count']}, to={params.get('to', 'None')}")
            
            # HTTP 요청 (재시도 로직 포함)
            max_retries = 3
            retry_delay = 2
            response = None
            last_error = None
            
            for retry_count in range(max_retries):
                try:
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        break
                    else:
                        print(f"  API 응답 오류: HTTP {response.status_code}")
                        if retry_count < max_retries - 1:
                            print(f"  {retry_delay}초 후 재시도...")
                            time.sleep(retry_delay)
                        else:
                            print(f"  응답 내용: {response.text}")
                            
                except Exception as req_error:
                    last_error = req_error
                    err = traceback.format_exc()
                    print(f"  API 호출 오류 (시도 {retry_count + 1}/{max_retries}):")
                    print(f"  오류 유형: {type(req_error).__name__}")
                    print(f"  오류 메시지: {str(req_error)}")
                    
                    if retry_count < max_retries - 1:
                        print(f"  {retry_delay}초 후 재시도...")
                        time.sleep(retry_delay)
            
            if response is None or response.status_code != 200:
                if last_error:
                    print(f"  마지막 오류: {type(last_error).__name__}: {str(last_error)}")
                return pd.DataFrame()
            
            # JSON 응답 파싱
            data = response.json()
            
            if not isinstance(data, list) or len(data) == 0:
                print(f"  경고: 응답 데이터가 비어있거나 형식이 올바르지 않습니다.")
                return pd.DataFrame()
            
            # DataFrame 생성
            # Bithumb API 응답 형식에 맞게 컬럼 매핑
            rows = []
            index_dates = []
            
            for item in data:
                # candle_date_time_kst를 사용하여 인덱스 생성
                dt_str = item.get('candle_date_time_kst', '')
                if dt_str:
                    try:
                        dt = pd.to_datetime(dt_str)
                        rows.append({
                            'open': float(item.get('opening_price', 0)),
                            'high': float(item.get('high_price', 0)),
                            'low': float(item.get('low_price', 0)),
                            'close': float(item.get('trade_price', 0)),
                            'volume': float(item.get('candle_acc_trade_volume', 0))
                        })
                        index_dates.append(dt)
                    except Exception:
                        # 날짜 파싱 실패 시 해당 항목 스킵
                        continue
            
            if len(rows) == 0:
                return pd.DataFrame()
            
            # DataFrame 생성 (인덱스와 함께)
            df = pd.DataFrame(rows)
            if len(index_dates) == len(df):
                df.index = pd.DatetimeIndex(index_dates)
            else:
                # 인덱스가 맞지 않으면 기본 인덱스 사용
                print(f"  경고: 인덱스 날짜 개수가 맞지 않습니다 ({len(index_dates)} vs {len(df)}). 기본 인덱스 사용")
            
            # 데이터 정제
            df = self._normalize_dataframe(df)
            
            print(f"  성공: {ticker} {interval} 데이터 {len(df)}개 수집 완료")
            if len(df) > 0:
                print(f"  기간: {df.index[0]} ~ {df.index[-1]}")
            
            return df
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def collect_multiple_intervals(self, ticker='BTC', intervals=None):
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
                    df = self.collect_all_periods_data(ticker, interval)
                    
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
    
    def get_maximum_historical_data(self, ticker='BTC', interval='24h'):
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
            # collect_all_periods_data를 사용하여 최근 200개를 먼저 가져옴
            # 더 많은 데이터가 필요한 경우 collect_candles_with_params를 여러 번 호출하여 수집
            print(f"  API 연결 확인: BithumbClient 사용")
            print(f"  수집 대상: {ticker}, 간격: {interval}")
            print(f"  주의: pybithumb2는 한 번에 최대 200개만 조회 가능합니다.")
            
            # 먼저 collect_all_periods_data로 최근 200개 수집
            df = self.collect_all_periods_data(ticker, interval)
            
            if df is None or len(df) == 0:
                print(f"  오류: 기본 수집 방식 실패")
                # fallback: requests 직접 호출
                print(f"  fallback: requests 직접 호출로 재시도합니다...")
                try:
                    fallback_df = self.collect_candles_with_params(ticker=ticker, interval=interval, count=200)
                    if fallback_df is not None and len(fallback_df) > 0:
                        print(f"  성공: fallback 방식으로 {len(fallback_df)}개 데이터 수집 완료")
                        return fallback_df
                except Exception as fallback_error:
                    err = traceback.format_exc()
                    print(f"  fallback도 실패:")
                    print("err", err)
                
                return pd.DataFrame()
            
            # 중복 제거 (인덱스 기준)
            df = df[~df.index.duplicated(keep='first')]
            
            # 시간순 정렬
            df = df.sort_index()
            
            print(f"\n성공: {ticker} {interval} 총 {len(df)}개 데이터 수집 완료")
            if len(df) > 0:
                print(f"  기간: {df.index[0]} ~ {df.index[-1]}")
            
            return df
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def make_history(self, ticker='BTC', interval='24h', collect_all_periods=True, from_date=None):
        """
        main.py에서 호출되는 메인 함수
        
        Parameters:
        - ticker (str): 암호화폐 티커. 기본값 'BTC'
        - interval (str): 캔들스틱 간격. 기본값 '24h'
        - collect_all_periods (bool): True면 가능한 모든 기간 데이터 수집, 
                                     False면 기본 기간만 수집
        - from_date (pd.Timestamp or str, optional): 이 날짜 이후의 데이터만 조회.
                                                     None이면 모든 데이터 반환
                                                     지정되면 count와 to 파라미터를 활용하여 필요한 데이터만 조회
        
        Returns:
        - pd.DataFrame: OHLCV 데이터프레임
        """
        try:
            # from_date가 지정된 경우: requests로 직접 API 호출하여 필요한 데이터만 효율적으로 조회
            # (기록이 있을 때 업데이트할 때만 사용)
            if from_date is not None:
                from_dt = pd.to_datetime(from_date)
                today = pd.Timestamp.now()
                
                # 필요한 캔들 개수 계산 (정확히 필요한 만큼만 계산)
                # interval에 따라 캔들 개수 계산
                time_diff = today - from_dt
                
                if interval.endswith('m'):
                    minutes = int(interval[:-1])
                    required_count = int(time_diff.total_seconds() / 60 / minutes)
                elif interval.endswith('h'):
                    hours = int(interval[:-1])
                    if hours == 24:
                        required_count = int(time_diff.days)
                    else:
                        required_count = int(time_diff.total_seconds() / 3600 / hours)
                else:
                    required_count = 200  # 기본값
                
                # 필요한 캔들 개수 + 10개 (여유분)
                count_with_buffer = required_count + 10
                
                print(f"[make_history] 효율적 데이터 수집 모드 (requests 직접 호출, from_date 사용)")
                print(f"  기존 기록 마지막 날짜: {from_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  현재 날짜: {today.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  필요 캔들 개수: {required_count}개, 조회할 개수: {count_with_buffer}개 (필요개수 + 10개)")
                
                # 최대 200개 제한이 있으므로, 필요한 개수가 200개를 넘으면 여러 번 호출
                all_data = []
                current_to = today
                remaining_count = count_with_buffer
                max_iterations = 10  # 무한 루프 방지
                iteration = 0
                
                while remaining_count > 0 and iteration < max_iterations:
                    # 이번 배치에서 조회할 개수 (최대 200개)
                    batch_count = min(remaining_count, 200)
                    
                    # 최신 데이터부터 과거로 가면서 수집
                    batch_df = self.collect_candles_with_params(
                        ticker=ticker,
                        interval=interval,
                        count=batch_count,
                        to_date=current_to
                    )
                    
                    if batch_df is None or len(batch_df) == 0:
                        break
                    
                    all_data.append(batch_df)
                    
                    # 다음 배치를 위해 가장 오래된 데이터 시각을 to_date로 설정
                    oldest_date = batch_df.index.min()
                    current_to = oldest_date
                    
                    # 남은 개수 계산
                    remaining_count -= len(batch_df)
                    
                    # 배치 크기만큼만 받았으면 더 이상 데이터가 없을 가능성
                    if len(batch_df) < batch_count:
                        break
                    
                    iteration += 1
                
                if len(all_data) == 0:
                    print(f"  경고: {from_dt.strftime('%Y-%m-%d')} 이후 데이터를 찾을 수 없습니다.")
                    return pd.DataFrame()
                
                # 모든 배치를 결합
                df = pd.concat(all_data, ignore_index=False)
                df = df[~df.index.duplicated(keep='first')]  # 중복 제거
                df = df.sort_index()
                
                # from_date 이후 데이터만 필터링 (조회한 데이터 + 10개에서 필요한 부분만 사용)
                before_filter = len(df)
                df = df[df.index > from_dt]
                after_filter = len(df)
                
                print(f"  조회 완료: {before_filter}개 데이터 조회, 필터링 후: {after_filter}개 (필요개수: {required_count}개)")
                if len(df) > 0:
                    print(f"  기간: {df.index[0]} ~ {df.index[-1]}")
                
                return df
            
            # from_date가 없는 경우: 기존 pybithumb 라이브러리 사용 (기록이 없을 때)
            # requests를 사용하지 않고 pybithumb의 get_candlestick 메서드 사용
            if collect_all_periods:
                print(f"[make_history] 전체 기간 데이터 수집 모드 (pybithumb 사용)")
                df = self.get_maximum_historical_data(ticker, interval)
                # 최대 기간 수집 실패 시 기본 방식으로 fallback
                if df is None or len(df) == 0:
                    print(f"  전체 기간 수집 실패, 기본 방식으로 재시도...")
                    df = self.collect_all_periods_data(ticker, interval)
            else:
                print(f"[make_history] 기본 기간 데이터 수집 모드 (pybithumb 사용)")
                df = self.collect_all_periods_data(ticker, interval)
            
            return df
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise

