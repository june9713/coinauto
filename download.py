"""
과거 데이터 다운로드 스크립트
pybithumb을 이용하여 최대 기록의 과거 데이터를 조회하고 저장합니다.

사용법:
    python download.py 3m BTC

Args:
    interval: 캔들스틱 간격 (예: 3m, 5m, 1h, 24h)
    ticker: 암호화폐 티커 (예: BTC, ETH)
"""
import traceback
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from pybithumb2 import BithumbClient, MarketID, TimeUnit
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()


class SimpleCandle:
    """간단한 Candle 객체 (pybithumb2의 Candle과 호환)"""
    def __init__(self, item_data, market_str=''):
        self.market = item_data.get('market', market_str)
        self.candle_date_time_utc = item_data.get('candle_date_time_utc', '')
        self.candle_date_time_kst = pd.to_datetime(item_data.get('candle_date_time_kst', ''))
        self.opening_price = float(item_data.get('opening_price', 0))
        self.high_price = float(item_data.get('high_price', 0))
        self.low_price = float(item_data.get('low_price', 0))
        self.trade_price = float(item_data.get('trade_price', 0))
        self.candle_acc_trade_volume = float(item_data.get('candle_acc_trade_volume', 0))
        self.candle_acc_trade_price = float(item_data.get('candle_acc_trade_price', 0))


def _fetch_candle_batch(ticker, interval, market, to_date, api_key, secret_key, batch_id):
    """
    단일 배치의 캔들 데이터를 조회하는 함수 (스레드에서 실행)
    
    Parameters:
    - ticker (str): 암호화폐 티커
    - interval (str): 캔들스틱 간격
    - market (MarketID): 마켓 ID
    - to_date (pd.Timestamp or None): 조회할 마지막 시각 (None이면 최신)
    - api_key (str): API 키
    - secret_key (str): 시크릿 키
    - batch_id (int): 배치 ID
    
    Returns:
    - tuple: (batch_id, pd.DataFrame or None, error or None)
    """
    try:
        # BithumbClient 객체 생성 (각 스레드마다 독립적으로)
        bithumb = BithumbClient(api_key=api_key, secret_key=secret_key)
        
        candles = None
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            if minutes in [1, 3, 5, 10, 15, 30, 60, 240]:
                unit = TimeUnit(minutes)
                if to_date is None:
                    candles = bithumb.get_minute_candles(market=market, count=200, unit=unit)
                else:
                    candles = _get_minute_candles_with_to(market, minutes, to_date, count=200)
            else:
                return (batch_id, None, f"지원하지 않는 단위: {interval}")
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            if hours == 1:
                unit = TimeUnit(60)
                if to_date is None:
                    candles = bithumb.get_minute_candles(market=market, count=200, unit=unit)
                else:
                    candles = _get_minute_candles_with_to(market, 60, to_date, count=200)
            elif hours == 24:
                if to_date is None:
                    candles = bithumb.get_day_candles(market=market, count=200)
                else:
                    candles = _get_day_candles_with_to(market, to_date, count=200)
            else:
                return (batch_id, None, f"지원하지 않는 시간 단위: {interval}")
        else:
            return (batch_id, None, f"지원하지 않는 형식: {interval}")
        
        if candles is None or len(candles) == 0:
            return (batch_id, None, "데이터 없음")
        
        # DFList[Candle]를 pandas DataFrame으로 변환
        batch_rows = []
        batch_dates = []
        
        for candle in candles:
            dt = candle.candle_date_time_kst
            batch_rows.append({
                'open': float(candle.opening_price),
                'high': float(candle.high_price),
                'low': float(candle.low_price),
                'close': float(candle.trade_price),
                'volume': float(candle.candle_acc_trade_volume)
            })
            batch_dates.append(dt)
        
        if len(batch_rows) == 0:
            return (batch_id, None, "배치 데이터 비어있음")
        
        # DataFrame 생성
        batch_df = pd.DataFrame(batch_rows)
        if len(batch_dates) == len(batch_df):
            batch_df.index = pd.DatetimeIndex(batch_dates)
        
        # 시간순 정렬
        batch_df = batch_df.sort_index()
        
        return (batch_id, batch_df, None)
        
    except Exception as e:
        err = traceback.format_exc()
        return (batch_id, None, str(e))


def collect_maximum_historical_data(ticker='BTC', interval='3m', max_iterations=100, max_workers=30):
    """
    가능한 최대 과거 기간의 데이터를 수집합니다.
    스레드를 이용하여 여러 번의 API 호출을 병렬로 처리합니다.
    
    Parameters:
    - ticker (str): 암호화폐 티커 (예: 'BTC', 'ETH'). 기본값 'BTC'
    - interval (str): 캔들스틱 간격. 기본값 '3m'
    - max_iterations (int): 최대 API 호출 횟수. 기본값 100
    - max_workers (int): 최대 동시 스레드 수. 기본값 30
    
    Returns:
    - pd.DataFrame: 모든 기간의 OHLCV 데이터프레임
    """
    try:
        # .env 파일에서 API 키 로드
        api_key = os.getenv('CONKEY')
        secret_key = os.getenv('SECKEY')
        
        # BithumbClient 객체 생성
        bithumb = BithumbClient(api_key=api_key, secret_key=secret_key)
        market = MarketID.from_string(f"KRW-{ticker}")
        
        print(f"[데이터 수집 시작]")
        print(f"  티커: {ticker}")
        print(f"  간격: {interval}")
        print(f"  최대 API 호출 횟수: {max_iterations}")
        print(f"  동시 스레드 수: {max_workers}")
        
        # 첫 번째 호출: 최신 데이터부터 시작
        print(f"\n[첫 번째 호출] 최신 데이터 조회 중...")
        first_result = _fetch_candle_batch(ticker, interval, market, None, api_key, secret_key, 0)
        
        if first_result[1] is None:
            print(f"  오류: 첫 번째 데이터 수집 실패 - {first_result[2]}")
            return pd.DataFrame()
        
        first_df = first_result[1]
        print(f"  수집된 데이터: {len(first_df)}개")
        if len(first_df) > 0:
            print(f"  기간: {first_df.index[0]} ~ {first_df.index[-1]}")
        
        all_data = [first_df]
        current_to = first_df.index[0]  # 가장 오래된 데이터 시각
        
        # 첫 번째 배치가 200개 미만이면 더 이상 데이터가 없음
        if len(first_df) < 200:
            print(f"  마지막 배치입니다 (200개 미만).")
            df = pd.concat(all_data, ignore_index=False)
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
            print(f"\n[데이터 수집 완료]")
            print(f"  총 {len(df)}개 데이터 수집")
            if len(df) > 0:
                print(f"  기간: {df.index[0]} ~ {df.index[-1]}")
            return df
        
        # 이후 호출들을 병렬로 처리
        print(f"\n[병렬 데이터 수집 시작]")
        iteration = 1
        completed_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 작업 큐에 모든 배치 추가
            futures = {}
            while iteration < max_iterations:
                future = executor.submit(
                    _fetch_candle_batch,
                    ticker, interval, market, current_to, api_key, secret_key, iteration
                )
                futures[future] = iteration
                iteration += 1
                
                # 다음 배치를 위해 to_date 업데이트 (대략적으로 계산)
                # 실제로는 API 응답을 받아야 정확하지만, 병렬 처리이므로 추정값 사용
                if interval.endswith('m'):
                    minutes = int(interval[:-1])
                    current_to = current_to - pd.Timedelta(minutes=200 * minutes)
                elif interval.endswith('h'):
                    hours = int(interval[:-1])
                    if hours == 24:
                        current_to = current_to - pd.Timedelta(days=200)
                    else:
                        current_to = current_to - pd.Timedelta(hours=200 * hours)
                
                # 너무 과거로 가면 중단
                if current_to < pd.Timestamp('2014-01-01'):
                    break
            
            # 결과 수집
            batch_results = {}  # {batch_id: DataFrame}
            
            for future in as_completed(futures):
                batch_id = futures[future]
                try:
                    result = future.result()
                    batch_id_result, batch_df, error = result
                    
                    if batch_df is not None and len(batch_df) > 0:
                        batch_results[batch_id_result] = batch_df
                        completed_count += 1
                        oldest_date = batch_df.index[0]
                        print(f"  [배치 {batch_id_result}] 완료: {len(batch_df)}개 데이터 (기간: {oldest_date} ~ {batch_df.index[-1]})")
                        
                        # 배치 크기가 200개 미만이면 더 이상 데이터가 없을 가능성
                        if len(batch_df) < 200:
                            print(f"  [배치 {batch_id_result}] 마지막 배치로 판단 (200개 미만)")
                    else:
                        failed_count += 1
                        if error:
                            print(f"  [배치 {batch_id_result}] 실패: {error}")
                except Exception as e:
                    failed_count += 1
                    err = traceback.format_exc()
                    print(f"  [배치 {batch_id}] 예외 발생: {str(e)}")
        
        print(f"\n[병렬 수집 완료]")
        print(f"  성공: {completed_count}개 배치")
        print(f"  실패: {failed_count}개 배치")
        
        # 첫 번째 데이터와 병렬로 수집한 데이터 결합
        if len(batch_results) > 0:
            # batch_id 순서대로 정렬하여 추가 (선택사항, 인덱스로 최종 정렬하므로 필수는 아님)
            sorted_batch_ids = sorted(batch_results.keys())
            for batch_id in sorted_batch_ids:
                all_data.append(batch_results[batch_id])
        
        if len(all_data) == 0:
            print(f"\n오류: 데이터 수집 실패")
            return pd.DataFrame()
        
        # 모든 배치를 결합
        df = pd.concat(all_data, ignore_index=False)
        df = df[~df.index.duplicated(keep='first')]  # 중복 제거
        df = df.sort_index()
        
        print(f"\n[데이터 수집 완료]")
        print(f"  총 {len(df)}개 데이터 수집")
        if len(df) > 0:
            print(f"  기간: {df.index[0]} ~ {df.index[-1]}")
            print(f"  API 호출 횟수: {completed_count + 1}회 (첫 번째 + 병렬)")
        
        return df
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


def _get_minute_candles_with_to(market, minutes, to_date, count=200):
    """
    to 파라미터를 사용하여 분봉 데이터를 조회합니다.
    
    Parameters:
    - market (MarketID): 마켓 ID
    - minutes (int): 분 단위
    - to_date (pd.Timestamp): 마지막 캔들 시각 (exclusive)
    - count (int): 조회할 캔들 개수 (최대 200)
    
    Returns:
    - list: Candle 객체 리스트 (pybithumb2의 DFList[Candle] 형식)
    """
    import requests
    
    try:
        market_str = market.value if hasattr(market, 'value') else str(market)
        url = f"https://api.bithumb.com/v1/candles/minutes/{minutes}"
        
        params = {
            'market': market_str,
            'count': min(count, 200),
            'to': to_date.strftime('%Y-%m-%dT%H:%M:%S')
        }
        
        response = requests.get(url, params=params, timeout=10, verify=False)
        
        if response.status_code != 200:
            print(f"  API 응답 오류: HTTP {response.status_code}")
            return None
        
        data = response.json()
        
        if not isinstance(data, list) or len(data) == 0:
            return None
        
        # JSON 데이터를 Candle 객체로 변환
        candles = []
        for item in data:
            candle = SimpleCandle(item, market_str)
            candles.append(candle)
        
        return candles
        
    except Exception as e:
        err = traceback.format_exc()
        print(f"  오류: {err}")
        return None


def _get_day_candles_with_to(market, to_date, count=200):
    """
    to 파라미터를 사용하여 일봉 데이터를 조회합니다.
    
    Parameters:
    - market (MarketID): 마켓 ID
    - to_date (pd.Timestamp): 마지막 캔들 시각 (exclusive)
    - count (int): 조회할 캔들 개수 (최대 200)
    
    Returns:
    - list: Candle 객체 리스트 (pybithumb2의 DFList[Candle] 형식)
    """
    import requests
    
    try:
        market_str = market.value if hasattr(market, 'value') else str(market)
        url = f"https://api.bithumb.com/v1/candles/days"
        
        params = {
            'market': market_str,
            'count': min(count, 200),
            'to': to_date.strftime('%Y-%m-%dT%H:%M:%S')
        }
        
        response = requests.get(url, params=params, timeout=10, verify=False)
        
        if response.status_code != 200:
            print(f"  API 응답 오류: HTTP {response.status_code}")
            return None
        
        data = response.json()
        
        if not isinstance(data, list) or len(data) == 0:
            return None
        
        # JSON 데이터를 Candle 객체로 변환
        candles = []
        for item in data:
            candle = SimpleCandle(item, market_str)
            candles.append(candle)
        
        return candles
        
    except Exception as e:
        err = traceback.format_exc()
        print(f"  오류: {err}")
        return None


def add_moving_averages(df):
    """
    데이터프레임에 이동평균 컬럼을 추가합니다.
    
    Parameters:
    - df (pd.DataFrame): OHLCV 데이터프레임 (close 컬럼 필요)
    
    Returns:
    - pd.DataFrame: 이동평균 컬럼이 추가된 데이터프레임
    """
    try:
        if df is None or len(df) == 0:
            return df
        
        if 'close' not in df.columns:
            print("  경고: 'close' 컬럼이 없어 이동평균을 계산할 수 없습니다.")
            return df
        
        # 데이터프레임 복사
        df_copy = df.copy()
        
        # 시간순 정렬 (이동평균 계산을 위해 필수)
        df_copy = df_copy.sort_index()
        
        # 이동평균 계산 (ma5, ma7, ma10)
        ma_periods = [5, 7, 10]
        for period in ma_periods:
            col_name = f'ma{period}'
            df_copy[col_name] = df_copy['close'].rolling(window=period, min_periods=1).mean()
        
        return df_copy
        
    except Exception as e:
        err = traceback.format_exc()
        print(f"  이동평균 계산 오류: {err}")
        return df


def save_history_to_dumps(df, ticker='BTC', interval='3m', base_dir='./dumps'):
    """
    데이터프레임을 dumps 폴더에 날짜별/시간별로 저장합니다.
    
    Parameters:
    - df (pd.DataFrame): 저장할 OHLCV 데이터프레임
    - ticker (str): 암호화폐 티커. 기본값 'BTC'
    - interval (str): 캔들스틱 간격. 기본값 '3m'
    - base_dir (str): 기본 디렉토리. 기본값 './dumps'
    """
    try:
        if df is None or len(df) == 0:
            print("저장할 데이터가 없습니다.")
            return
        
        # 저장 경로: ./dumps/BTC/3m/2025-11-06/~~~
        ticker_upper = ticker.upper()
        data_dir = os.path.join(base_dir, ticker_upper, interval)
        
        # 디렉토리 생성
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"디렉토리 생성: {data_dir}")
        
        # 파일명 형식: history_{ticker}_{interval}_{YYYYMMDD_HH}.csv
        # 기존 datas 폴더의 파일명 형식과 유사하게
        base_filename = f"history_{ticker.lower()}_{interval}"
        
        # 데이터를 날짜별로 그룹화
        df_copy = df.copy()
        df_copy['_temp_date'] = df_copy.index.date
        df_copy['_temp_hour'] = df_copy.index.hour
        
        saved_files = 0
        for date, date_group in df_copy.groupby('_temp_date'):
            date_folder = pd.Timestamp(date).strftime('%Y-%m-%d')
            date_dir = os.path.join(data_dir, date_folder)
            
            # 날짜 폴더 생성
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)
            
            # 해당 날짜의 데이터를 시간별로 그룹화
            for hour, hour_group in date_group.groupby('_temp_hour'):
                # 시간별 데이터프레임 (임시 컬럼 제거)
                hour_df = hour_group.drop(['_temp_date', '_temp_hour'], axis=1)
                
                # 시간별 파일명 생성 (YYYYMMDD_HH 형식)
                date_str = pd.Timestamp(date).strftime('%Y%m%d')
                hour_str = f"{date_str}_{hour:02d}"
                filename = f"{base_filename}_{hour_str}.csv"
                save_path = os.path.join(date_dir, filename)
                
                # 기존 파일이 있으면 로드하여 덮어쓰기
                if os.path.exists(save_path):
                    try:
                        existing_df = pd.read_csv(save_path, index_col=0, parse_dates=True)
                        
                        # 인덱스를 datetime으로 변환
                        if not isinstance(existing_df.index, pd.DatetimeIndex):
                            existing_df.index = pd.to_datetime(existing_df.index)
                        
                        # 중복 제거 및 정렬
                        existing_df = existing_df[~existing_df.index.duplicated(keep='first')]
                        existing_df = existing_df.sort_index()
                        
                        # 덮어쓰기될 데이터 개수 계산 (덮어쓰기 전에)
                        overwritten_count = len(hour_df.index.intersection(existing_df.index))
                        original_count = len(existing_df)
                        
                        # 기존 데이터에서 새 데이터와 겹치는 인덱스 제거 (덮어쓰기)
                        existing_df = existing_df[~existing_df.index.isin(hour_df.index)]
                        
                        # 기존 데이터(겹치지 않는 부분)와 새 데이터 결합
                        combined_df = pd.concat([existing_df, hour_df])
                        combined_df = combined_df.sort_index()
                        
                        # 이동평균 추가
                        combined_df = add_moving_averages(combined_df)
                        
                        # 덮어쓰기된 데이터 저장
                        combined_df.to_csv(save_path, index=True)
                        print(f"  {filename}: 기존 {original_count}개 + 새 {len(hour_df)}개 (덮어쓰기: {overwritten_count}개) = {len(combined_df)}개")
                    except Exception as e:
                        # 기존 파일 로드 실패 시 새 데이터만 저장
                        print(f"  경고: 기존 파일 로드 실패 ({filename}): {e}")
                        # 이동평균 추가
                        hour_df = add_moving_averages(hour_df)
                        hour_df.to_csv(save_path, index=True)
                        print(f"  {filename}: 새 데이터 {len(hour_df)}개 저장")
                else:
                    # 기존 파일이 없으면 새로 저장
                    # 이동평균 추가
                    hour_df = add_moving_averages(hour_df)
                    hour_df.to_csv(save_path, index=True)
                    print(f"  {filename}: 새 데이터 {len(hour_df)}개 저장")
                
                saved_files += 1
        
        print(f"\n[파일 저장 완료]")
        print(f"  저장된 파일 수: {saved_files}개")
        print(f"  저장 경로: {data_dir}")
        print(f"  총 {len(df)}개 데이터")
        if len(df) > 0:
            print(f"  기간: {df.index[0]} ~ {df.index[-1]}")
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


def main():
    """
    메인 함수
    명령줄 인자를 파싱하여 데이터를 다운로드하고 저장합니다.
    """
    try:
        # 명령줄 인자 파싱
        if len(sys.argv) < 3:
            print("사용법: python download.py <interval> <ticker>")
            print("예시: python download.py 3m BTC")
            sys.exit(1)
        
        interval = sys.argv[1]
        ticker = sys.argv[2].upper()
        
        print("="*80)
        print(f"과거 데이터 다운로드 시작")
        print("="*80)
        print(f"티커: {ticker}")
        print(f"간격: {interval}")
        print("="*80)
        
        # 최대 기록 데이터 수집
        df = collect_maximum_historical_data(ticker=ticker, interval=interval, max_iterations=1000)
        
        if df is None or len(df) == 0:
            print("\n오류: 데이터 수집 실패")
            sys.exit(1)
        
        # dumps 폴더에 저장
        save_history_to_dumps(df, ticker=ticker, interval=interval, base_dir='./dumps')
        
        print("\n" + "="*80)
        print("다운로드 완료!")
        print("="*80)
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


if __name__ == '__main__':
    main()

