"""
데이터 관리 모듈
데이터 로드, 저장, 업데이트 관리
"""
import traceback
import pandas as pd
import os
from datetime import datetime, timedelta
from config import Config
from data_collector import DataCollector


class DataManager:
    """데이터 로드, 저장, 업데이트 관리 클래스"""
    
    def __init__(self, ticker='BTC', data_dir=None, data_file=None, condition_dict=None):
        """
        초기화
        
        Parameters:
        - ticker (str): 암호화폐 티커. 기본값 'BTC'
        - data_dir (str, optional): 데이터 디렉토리 경로. None이면 Config 사용
        - data_file (str, optional): 데이터 파일 경로. None이면 Config 사용
        - condition_dict (dict, optional): 조건 딕셔너리. None이면 기본 파일명 사용
        """
        self.ticker = ticker
        self.data_dir = data_dir or Config.DATA_DIR
        if data_file is None:
            self.data_file = Config.get_data_file_path(ticker=ticker, condition_dict=condition_dict)
        else:
            self.data_file = data_file
        self.condition_dict = condition_dict
        self.collector = DataCollector()
    
    def ensure_data_directory(self):
        """
        데이터 디렉토리가 존재하는지 확인하고, 없으면 생성합니다.
        
        Returns:
        - bool: 성공 여부
        """
        try:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
                print(f"데이터 폴더 생성: {self.data_dir}")
            return True
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def load_history_from_file(self):
        """
        저장된 데이터 파일에서 데이터를 로드합니다.
        단일 파일이 없으면 날짜별 폴더에서 시간별 분할 파일들을 찾아 로드합니다.
        현재 설정의 데이터가 없으면 ticker와 interval만 일치하는 다른 조건의 파일도 찾아서 로드합니다.
        
        Returns:
        - pd.DataFrame: 저장된 OHLCV 데이터프레임. 파일이 없으면 None 반환
        """
        try:
            # 먼저 단일 파일 확인
            if os.path.exists(self.data_file):
                print(f"기록 파일 로드 중: {self.data_file}")
                df = pd.read_csv(self.data_file, index_col=0, parse_dates=True)
                
                # 컬럼명 확인 및 정리
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in required_columns:
                    if col not in df.columns:
                        print(f"경고: 필수 컬럼 '{col}'이 파일에 없습니다.")
                
                # 인덱스를 datetime으로 변환
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                # 중복 제거 및 정렬
                df = df[~df.index.duplicated(keep='first')]
                df = df.sort_index()
                
                print(f"기록 파일 로드 완료: {len(df)}개 데이터")
                if len(df) > 0:
                    print(f"  기간: {df.index[0]} ~ {df.index[-1]}")
                
                return df
            
            # 단일 파일이 없으면 날짜별 폴더에서 시간별 분할 파일들 찾기
            print("날짜별 폴더에서 시간별 분할 파일을 검색합니다...")
            
            # 파일명에서 조건 부분 추출 (확장자 제외)
            base_filename = os.path.basename(self.data_file)
            name_without_ext = os.path.splitext(base_filename)[0]
            
            # ticker와 interval만 일치하는 파일을 찾기 위한 패턴 생성
            # 파일명 형식: history_..._{ticker}_{interval}.csv (예: history_vw55_mw9_..._btc_3m.csv)
            # condition_dict에서 ticker와 interval 추출
            ticker_interval_suffix = None
            if self.condition_dict and 'ticker' in self.condition_dict and 'interval' in self.condition_dict:
                ticker = self.condition_dict['ticker'].lower()
                interval = self.condition_dict['interval']
                # 파일명 끝에 _{ticker}_{interval} 형식으로 포함되어 있음
                ticker_interval_suffix = f"_{ticker}_{interval}"
            else:
                ticker_interval_suffix = None
            
            # 날짜별 폴더에서 해당 조건으로 시작하는 모든 시간별 파일 찾기
            all_dataframes = []
            
            if os.path.exists(self.data_dir):
                # 날짜별 폴더 목록 가져오기 (YYYY-MM-DD 형식)
                date_folders = []
                for item in os.listdir(self.data_dir):
                    item_path = os.path.join(self.data_dir, item)
                    if os.path.isdir(item_path) and len(item) == 10 and item.count('-') == 2:
                        try:
                            # 날짜 형식 검증
                            datetime.strptime(item, '%Y-%m-%d')
                            date_folders.append(item)
                        except ValueError:
                            continue
                
                # 날짜순으로 정렬
                date_folders.sort()
                
                # 각 날짜 폴더에서 시간별 파일 찾기
                for date_folder in date_folders:
                    date_dir = os.path.join(self.data_dir, date_folder)
                    if not os.path.isdir(date_dir):
                        continue
                    
                    # 해당 날짜 폴더의 모든 파일 확인
                    for filename in os.listdir(date_dir):
                        if not filename.endswith('.csv'):
                            continue
                        
                        # 현재 조건의 파일명으로 시작하는지 확인
                        matches_current = filename.startswith(name_without_ext)
                        
                        # 현재 조건이 아니면 ticker와 interval만 일치하는 파일인지 확인
                        matches_ticker_interval = False
                        if not matches_current and ticker_interval_suffix:
                            # 파일명이 history_로 시작하고 _{ticker}_{interval}로 끝나는지 확인
                            if filename.startswith('history_') and filename.endswith(f"{ticker_interval_suffix}.csv"):
                                matches_ticker_interval = True
                        
                        # 현재 조건 또는 ticker+interval 일치하는 파일이면 로드
                        if matches_current or matches_ticker_interval:
                            file_path = os.path.join(date_dir, filename)
                            try:
                                file_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                                if len(file_df) > 0:
                                    all_dataframes.append(file_df)
                            except Exception as e:
                                print(f"경고: 파일 로드 실패 ({file_path}): {e}")
                                continue
            
            if len(all_dataframes) == 0:
                print(f"기록 파일이 없습니다: {self.data_file}")
                if ticker_interval_suffix:
                    print(f"  검색 패턴: history_*{ticker_interval_suffix}.csv")
                return None
            
            # 모든 데이터프레임 병합
            combined_df = pd.concat(all_dataframes)
            
            # 컬럼명 확인 및 정리
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in combined_df.columns:
                    print(f"경고: 필수 컬럼 '{col}'이 데이터에 없습니다.")
            
            # 인덱스를 datetime으로 변환
            if not isinstance(combined_df.index, pd.DatetimeIndex):
                combined_df.index = pd.to_datetime(combined_df.index)
            
            # 중복 제거 및 정렬
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
            
            print(f"기록 파일 로드 완료 (시간별 분할 파일 병합): {len(combined_df)}개 데이터")
            if len(combined_df) > 0:
                print(f"  기간: {combined_df.index[0]} ~ {combined_df.index[-1]}")
                print(f"  로드된 파일 수: {len(all_dataframes)}개")
            
            return combined_df
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def save_history_to_file(self, df, is_realtime_update=False, interval=None):
        """
        데이터프레임을 파일에 저장합니다.
        
        Parameters:
        - df (pd.DataFrame): 저장할 OHLCV 데이터프레임
        - is_realtime_update (bool): 실시간 업데이트 여부. True면 날짜별 폴더에 시간별 파일로 저장
        - interval (str, optional): 캔들스틱 간격. None이면 condition_dict에서 가져오거나 기본값 사용
        """
        try:
            self.ensure_data_directory()
            
            # 현재 시간 기준으로 미래 캔들 필터링 (완료된 캔들만 저장)
            now = pd.Timestamp.now()
            original_count = len(df)
            
            # interval을 고려하여 현재 시간에서 interval만큼 빼서 비교
            # 예: 현재 시간이 23:03이고 interval이 3m이면, 23:03 캔들은 미래 캔들로 간주
            #     따라서 23:00 이하의 캔들만 저장
            # interval 파라미터가 없으면 condition_dict에서 가져오기
            if interval is None:
                if self.condition_dict and 'interval' in self.condition_dict:
                    interval = self.condition_dict['interval']
                else:
                    # condition_dict에도 없으면 기본값 사용 (하지만 이 경우는 드뭄)
                    interval = '24h'
            
            interval_timedelta = self._get_interval_timedelta(interval)
            cutoff_time = now - interval_timedelta
            
            # cutoff_time보다 이전의 캔들만 저장 (현재 시간과 같은 시간의 캔들 제외)
            future_candles_df = df[df.index >= cutoff_time]
            df = df[df.index < cutoff_time]
            filtered_count = len(df)
            
            if original_count > filtered_count:
                print(f"미래 캔들 필터링: {original_count}개 → {filtered_count}개 (제외: {original_count - filtered_count}개)")
                print(f"  현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  cutoff 시간 (현재 - {interval}): {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 버려진 미래 캔들의 시간 출력
                if len(future_candles_df) > 0:
                    print(f"  [버려진 미래 캔들 시간]")
                    for idx, candle_time in enumerate(future_candles_df.index):
                        print(f"    - {candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        if idx >= 9:  # 최대 10개만 출력 (너무 많으면 생략)
                            remaining = len(future_candles_df) - 10
                            if remaining > 0:
                                print(f"    ... 외 {remaining}개 더 있음")
                            break
                
                # 현재 캔들(완료된 캔들 중 가장 최신)의 시간 출력
                if len(df) > 0:
                    current_candle_time = df.index[-1]
                    print(f"  [현재 캔들 시간] {current_candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 중복 제거 및 정렬
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
            
            if is_realtime_update and len(df) > 0:
                # 실시간 업데이트: 날짜별 폴더에 시간별 파일로 저장
                date_folder = now.strftime('%Y-%m-%d')
                date_dir = os.path.join(self.data_dir, date_folder)
                
                # 날짜 폴더 생성
                if not os.path.exists(date_dir):
                    os.makedirs(date_dir)
                
                # 파일명에서 조건 문자열 추출
                base_filename = os.path.basename(self.data_file)
                # 확장자 제거
                name_without_ext = os.path.splitext(base_filename)[0]
                # 시간별 파일명 생성 (YYYYMMDD_HH 형식)
                hour_str = now.strftime('%Y%m%d_%H')
                filename = f"{name_without_ext}_{hour_str}.csv"
                save_path = os.path.join(date_dir, filename)
                
                # 기존 파일이 있으면 로드하여 병합
                if os.path.exists(save_path):
                    try:
                        existing_df = pd.read_csv(save_path, index_col=0, parse_dates=True)
                        
                        # 인덱스를 datetime으로 변환
                        if not isinstance(existing_df.index, pd.DatetimeIndex):
                            existing_df.index = pd.to_datetime(existing_df.index)
                        
                        # 중복 제거 및 정렬
                        existing_df = existing_df[~existing_df.index.duplicated(keep='first')]
                        existing_df = existing_df.sort_index()
                        
                        # 기존 데이터와 새 데이터 병합
                        combined_df = pd.concat([existing_df, df])
                        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]  # 중복 시 새로운 데이터 우선
                        combined_df = combined_df.sort_index()
                        
                        print(f"기존 파일 발견: {len(existing_df)}개 데이터")
                        print(f"새로운 데이터: {len(df)}개")
                        print(f"병합 후: {len(combined_df)}개 데이터")
                        
                        # 병합된 데이터 저장
                        combined_df.to_csv(save_path, index=True)
                        print(f"기록 파일 저장 완료 (실시간 업데이트, 기존 데이터 병합): {save_path}")
                        print(f"  총 {len(combined_df)}개 데이터")
                        if len(combined_df) > 0:
                            print(f"  기간: {combined_df.index[0]} ~ {combined_df.index[-1]}")
                    except Exception as e:
                        # 기존 파일 로드 실패 시 새 데이터만 저장
                        print(f"경고: 기존 파일 로드 실패 ({save_path}): {e}")
                        print(f"  새 데이터만 저장합니다.")
                        df.to_csv(save_path, index=True)
                        print(f"기록 파일 저장 완료 (실시간 업데이트): {save_path}")
                        print(f"  총 {len(df)}개 데이터")
                        if len(df) > 0:
                            print(f"  기간: {df.index[0]} ~ {df.index[-1]}")
                else:
                    # 기존 파일이 없으면 새로 저장
                    df.to_csv(save_path, index=True)
                    print(f"기록 파일 저장 완료 (실시간 업데이트, 새 파일 생성): {save_path}")
                    print(f"  총 {len(df)}개 데이터")
                    if len(df) > 0:
                        print(f"  기간: {df.index[0]} ~ {df.index[-1]}")
            else:
                # 초기 다운로드: 날짜별 폴더에 시간별 파일로 저장
                if len(df) > 0:
                    # 파일명에서 조건 문자열 추출
                    base_filename = os.path.basename(self.data_file)
                    name_without_ext = os.path.splitext(base_filename)[0]
                    
                    # 원본 데이터 보존을 위해 복사본 사용
                    df_copy = df.copy()
                    
                    # 데이터를 날짜별로 그룹화 (임시 컬럼 추가)
                    df_copy['_temp_date'] = df_copy.index.date
                    df_copy['_temp_hour'] = df_copy.index.hour
                    
                    saved_files = 0
                    for date, date_group in df_copy.groupby('_temp_date'):
                        date_folder = pd.Timestamp(date).strftime('%Y-%m-%d')
                        date_dir = os.path.join(self.data_dir, date_folder)
                        
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
                            filename = f"{name_without_ext}_{hour_str}.csv"
                            save_path = os.path.join(date_dir, filename)
                            
                            # 기존 파일이 있으면 로드하여 병합
                            if os.path.exists(save_path):
                                try:
                                    existing_df = pd.read_csv(save_path, index_col=0, parse_dates=True)
                                    
                                    # 인덱스를 datetime으로 변환
                                    if not isinstance(existing_df.index, pd.DatetimeIndex):
                                        existing_df.index = pd.to_datetime(existing_df.index)
                                    
                                    # 중복 제거 및 정렬
                                    existing_df = existing_df[~existing_df.index.duplicated(keep='first')]
                                    existing_df = existing_df.sort_index()
                                    
                                    # 기존 데이터와 새 데이터 병합
                                    combined_df = pd.concat([existing_df, hour_df])
                                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]  # 중복 시 새로운 데이터 우선
                                    combined_df = combined_df.sort_index()
                                    
                                    # 병합된 데이터 저장
                                    combined_df.to_csv(save_path, index=True)
                                    print(f"  {filename}: 기존 {len(existing_df)}개 + 새 {len(hour_df)}개 = {len(combined_df)}개 (병합)")
                                except Exception as e:
                                    # 기존 파일 로드 실패 시 새 데이터만 저장
                                    print(f"  경고: 기존 파일 로드 실패 ({filename}): {e}")
                                    hour_df.to_csv(save_path, index=True)
                                    print(f"  {filename}: 새 데이터 {len(hour_df)}개 저장")
                            else:
                                # 기존 파일이 없으면 새로 저장
                                hour_df.to_csv(save_path, index=True)
                                print(f"  {filename}: 새 데이터 {len(hour_df)}개 저장")
                            
                            saved_files += 1
                    
                    print(f"기록 파일 저장 완료 (초기 다운로드, 시간별 분할): {saved_files}개 파일 생성")
                    print(f"  총 {len(df)}개 데이터")
                    if len(df) > 0:
                        print(f"  기간: {df.index[0]} ~ {df.index[-1]}")
                else:
                    print("저장할 데이터가 없습니다.")
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def get_last_date_from_file(self):
        """
        저장된 기록 파일의 마지막 날짜를 반환합니다.
        
        Returns:
        - pd.Timestamp or None: 마지막 날짜. 파일이 없거나 비어있으면 None 반환
        """
        try:
            df = self.load_history_from_file()
            if df is None or len(df) == 0:
                return None
            
            last_date = df.index[-1]
            return last_date
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def _get_interval_timedelta(self, interval):
        """
        interval 문자열을 timedelta로 변환합니다.
        
        Parameters:
        - interval (str): 캔들스틱 간격 (예: '3m', '24h')
        
        Returns:
        - timedelta: interval에 해당하는 시간 간격
        """
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            return timedelta(minutes=minutes)
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            if hours == 24:
                return timedelta(days=1)
            else:
                return timedelta(hours=hours)
        else:
            raise ValueError(f"지원하지 않는 interval 형식: {interval}")
    
    def _check_missing_data_in_recent(self, df, interval, recent_count=3000):
        """
        최근 데이터 중 중간에 비어있는 데이터가 있는지 확인합니다.
        
        Parameters:
        - df (pd.DataFrame): 전체 데이터프레임
        - interval (str): 캔들스틱 간격
        - recent_count (int): 확인할 최근 데이터 개수. 기본값 3000
        
        Returns:
        - bool: 중간에 비어있는 데이터가 있으면 True, 없으면 False
        """
        try:
            if df is None or len(df) < 2:
                return False
            
            # 최근 recent_count개만 확인
            if len(df) > recent_count:
                recent_df = df.tail(recent_count).copy()
            else:
                recent_df = df.copy()
            
            if len(recent_df) < 2:
                return False
            
            # interval에 따른 예상 시간 간격
            interval_delta = self._get_interval_timedelta(interval)
            
            # 인덱스를 기준으로 시간 차이 계산
            time_diffs = recent_df.index.to_series().diff().dropna()
            
            # 예상 간격의 1.5배 이상 차이나는 경우 누락으로 간주
            # (예: 3분 간격인데 6분 이상 차이나면 누락)
            threshold = interval_delta * 1.5
            
            missing_count = (time_diffs > threshold).sum()
            
            if missing_count > 0:
                print(f"  최근 {len(recent_df)}개 데이터 중 {missing_count}개 구간에서 누락 감지")
                print(f"  첫 번째 누락 구간: {time_diffs[time_diffs > threshold].iloc[0]}")
                return True
            
            return False
            
        except Exception as e:
            err = traceback.format_exc()
            print(f"  경고: 최근 데이터 누락 확인 중 오류 발생: {err}")
            # 오류 발생 시 안전하게 False 반환 (누락 확인 실패)
            return False
    
    def update_history_data(self, ticker='BTC', interval='24h', start_date='2014-01-01'):
        """
        기록 파일의 마지막 날짜부터 현재까지의 데이터를 다운로드하여 파일에 누적 저장합니다.
        최근 3000개 데이터를 확인하여 중간에 비어있는 데이터가 있으면 전체를 다시 다운로드합니다.
        
        Parameters:
        - ticker (str): 암호화폐 티커. 기본값 'BTC'
        - interval (str): 캔들스틱 간격. 기본값 '24h'
        - start_date (str): 최초 시작 날짜. 기본값 '2014-01-01'
        
        Returns:
        - pd.DataFrame: 업데이트된 전체 데이터프레임
        """
        try:
            print("\n" + "="*80)
            print("기록 데이터 업데이트 검사")
            print("="*80)
            
            # 시작 날짜 파싱
            start_dt = pd.to_datetime(start_date)
            
            # 기존 데이터 로드
            existing_df = self.load_history_from_file()
            
            # 현재 시간 (시분초 포함)
            now = pd.Timestamp.now()
            
            # 기존 데이터가 있는 경우
            if existing_df is not None and len(existing_df) > 0:
                last_date = existing_df.index[-1]  # normalize() 제거 - 시분초 포함
                print(f"기존 기록 마지막 시간: {last_date.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 최근 3000개 데이터 확인
                print("\n최근 3000개 데이터 누락 확인 중...")
                has_missing = self._check_missing_data_in_recent(existing_df, interval, recent_count=3000)
                
                if has_missing:
                    # 최근 3000개 데이터 범위 계산
                    if len(existing_df) >= 3000:
                        recent_start_date = existing_df.index[-3000]
                    else:
                        recent_start_date = existing_df.index[0]
                    
                    print(f"\n최근 3000개 데이터 중 누락 감지!")
                    print(f"최근 3000개 범위: {recent_start_date.strftime('%Y-%m-%d %H:%M:%S')} ~ {last_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"최근 3000개 전체를 다시 다운로드하여 누락 부분을 채웁니다...")
                    
                    # 최근 3000개를 다시 다운로드
                    # from_date=None으로 전체 데이터를 받은 후 최근 3000개만 선택
                    # 또는 필요한 개수만 계산하여 받기
                    all_recent_df = self.collector.make_history(
                        ticker=ticker, 
                        interval=interval, 
                        collect_all_periods=True,
                        from_date=None  # 전체 최신 데이터 받기
                    )
                    
                    if all_recent_df is not None and len(all_recent_df) > 0:
                        # 최근 3000개만 선택
                        if len(all_recent_df) >= 3000:
                            recent_df = all_recent_df.tail(3000).copy()
                        else:
                            recent_df = all_recent_df.copy()
                        
                        # recent_start_date 이후의 데이터만 사용 (기존 데이터와 겹치는 부분)
                        recent_df = recent_df[recent_df.index >= recent_start_date]
                        
                        if len(recent_df) > 0:
                            print(f"최근 3000개 재다운로드 완료: {len(recent_df)}개 데이터")
                            print(f"  기간: {recent_df.index[0]} ~ {recent_df.index[-1]}")
                            
                            # 기존 데이터와 병합 (최근 부분은 재다운로드한 데이터로 교체)
                            # 최근 부분 제거 후 재다운로드한 데이터 추가
                            if len(existing_df) >= 3000:
                                existing_df_before_recent = existing_df.iloc[:-3000].copy()
                            else:
                                # 전체 데이터가 3000개 미만이면 recent_start_date 이전 데이터만 유지
                                existing_df_before_recent = existing_df[existing_df.index < recent_start_date].copy()
                            
                            # 재다운로드한 데이터와 기존 데이터를 병합
                            combined_df = pd.concat([existing_df_before_recent, recent_df])
                            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                            combined_df = combined_df.sort_index()
                            
                            # 재다운로드한 데이터를 파일에 저장 (날짜별/시간별로 나눠서 저장)
                            print(f"\n재다운로드한 데이터 파일 저장 중...")
                            self.save_history_to_file(recent_df, is_realtime_update=False, interval=interval)
                            
                            # 기존 데이터를 업데이트된 것으로 교체
                            existing_df = combined_df
                            last_date = existing_df.index[-1]
                            print(f"누락 부분 채우기 완료. 업데이트된 마지막 시간: {last_date.strftime('%Y-%m-%d %H:%M:%S')}")
                        else:
                            print("  경고: 재다운로드한 데이터가 범위에 맞지 않습니다.")
                    else:
                        print("  경고: 최근 3000개 재다운로드 실패. 기존 데이터 유지.")
                else:
                    print("  최근 3000개 데이터에 누락 없음")
                
                # 마지막 캔들 시간이 현재 시간보다 최신이거나 같으면 업데이트 불필요
                # (시분초까지 비교)
                if last_date >= now:
                    print("기록이 최신 상태입니다. 업데이트 불필요.")
                    return existing_df
                
                # 마지막 캔들 이후부터 데이터 수집
                print(f"\n누락된 시간부터 다운로드: {last_date.strftime('%Y-%m-%d %H:%M:%S')} ~ {now.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 최신 데이터 다운로드 (API 제약으로 전체를 받지만, from_date로 즉시 필터링하여 필요한 부분만 사용)
                print(f"\n최신 데이터 다운로드 중... (마지막 기록 이후만 사용)")
                new_df = self.collector.make_history(ticker=ticker, interval=interval, 
                                                     collect_all_periods=True, from_date=last_date)
                
                if new_df is None or len(new_df) == 0:
                    print("경고: 새로운 데이터를 다운로드할 수 없습니다.")
                    return existing_df
                
                if len(new_df) == 0:
                    print("추가할 새로운 데이터가 없습니다.")
                    return existing_df
                
                print(f"새로운 데이터 {len(new_df)}개 추가됨")
                print(f"  기간: {new_df.index[0]} ~ {new_df.index[-1]}")
                
                # 기존 데이터와 새로운 데이터 비교하여 값이 다른 경우에만 교체
                common_indices = existing_df.index.intersection(new_df.index)
                replaced_count = 0
                unchanged_count = 0
                updated_indices = []
                
                if len(common_indices) > 0:
                    print(f"  기존 데이터와 겹치는 캔들: {len(common_indices)}개 발견")
                    # 값이 같은 캔들은 새로운 데이터에서 제외 (기존 데이터 유지)
                    new_df_to_use = new_df.copy()
                    
                    for idx in common_indices:
                        existing_row = existing_df.loc[idx]
                        new_row = new_df.loc[idx]
                        
                        # 값 비교 (open, high, low, close, volume)
                        values_different = False
                        different_fields = []
                        
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            if col in existing_row and col in new_row:
                                # 부동소수점 비교를 위해 작은 차이는 무시
                                if abs(float(existing_row[col]) - float(new_row[col])) > 0.01:
                                    values_different = True
                                    different_fields.append(f"{col}: {existing_row[col]:.2f} → {new_row[col]:.2f}")
                        
                        if values_different:
                            replaced_count += 1
                            updated_indices.append(idx)
                            print(f"    [값 변경] {idx.strftime('%Y-%m-%d %H:%M:%S')}: {', '.join(different_fields)}")
                        else:
                            unchanged_count += 1
                            # 값이 같은 경우 새로운 데이터에서 제외 (기존 데이터 유지)
                            new_df_to_use = new_df_to_use.drop(idx)
                else:
                    # 겹치는 캔들이 없으면 모든 새로운 데이터 사용
                    new_df_to_use = new_df.copy()
                
                if replaced_count > 0:
                    print(f"  총 {replaced_count}개 캔들의 값이 업데이트되었습니다.")
                if unchanged_count > 0:
                    print(f"  {unchanged_count}개 캔들의 값은 동일하여 기존 데이터를 유지합니다.")
                
                # 기존 데이터와 병합 (값이 다른 경우만 새로운 데이터로 교체, 값이 같으면 기존 데이터 유지)
                combined_df = pd.concat([existing_df, new_df_to_use])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]  # 중복 시 마지막 데이터 우선
                combined_df = combined_df.sort_index()
                
                # 실시간 업데이트: 새로운 데이터만 별도 파일로 저장
                self.save_history_to_file(new_df, is_realtime_update=True, interval=interval)
                
                return combined_df
                
            else:
                # 기존 데이터가 없는 경우: 처음부터 전체 다운로드
                now = pd.Timestamp.now()
                print("기존 기록이 없습니다. 전체 데이터 다운로드를 시작합니다.")
                print(f"다운로드 기간: {start_date} ~ {now.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 전체 데이터 다운로드
                df = self.collector.make_history(ticker=ticker, interval=interval, collect_all_periods=True)
                
                if df is None or len(df) == 0:
                    print("오류: 데이터 다운로드 실패")
                    return None
                
                # 시작 날짜 이후만 필터링
                df = df[df.index >= start_dt]
                
                if len(df) == 0:
                    print(f"경고: {start_date} 이후의 데이터가 없습니다.")
                    return None
                
                print(f"전체 데이터 다운로드 완료: {len(df)}개 데이터")
                print(f"  기간: {df.index[0]} ~ {df.index[-1]}")
                
                # 파일에 저장
                self.save_history_to_file(df, is_realtime_update=False, interval=interval)
                
                return df
                
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def save_backtest_result(self, df, result, test_start_date, test_end_date,
                            buy_angle_threshold, sell_angle_threshold,
                            stop_loss_percent, min_sell_price, price_slippage,
                            initial_capital, window, aspect_ratio,
                            ticker='BTC', interval='24h'):
        """
        백테스트 결과를 파일에 저장합니다.
        각 거래를 별도 행으로 저장합니다.
        
        Parameters:
        - df (pd.DataFrame): 백테스트에 사용된 데이터프레임 (주가 정보 포함)
        - result (dict): 백테스트 결과 딕셔너리
        - test_start_date (str or pd.Timestamp): 백테스트 시작 날짜
        - test_end_date (str or pd.Timestamp or None): 백테스트 종료 날짜
        - buy_angle_threshold (float): 매수 조건 각도
        - sell_angle_threshold (float): 매도 조건 각도
        - stop_loss_percent (float): 손절 기준
        - min_sell_price (float): 최소 매도 가격
        - price_slippage (float): 거래 가격 슬리퍼지
        - initial_capital (float): 초기 자본
        - window (int): 추세선 계산 윈도우 크기
        - aspect_ratio (float): 차트 종횡비
        - ticker (str): 암호화폐 티커. 기본값 'BTC'
        - interval (str): 캔들스틱 간격. 기본값 '24h'
        """
        try:
            self.ensure_data_directory()
            
            # 백테스트 결과 파일 경로 (조건 기반)
            result_file = Config.get_backtest_result_file_path(ticker=ticker, condition_dict=self.condition_dict)
            
            # 실행 시점
            execution_time = pd.Timestamp.now()
            
            # 백테스트 시점의 주가 정보
            if len(df) > 0:
                price_at_start = df.iloc[0]['close']
                price_at_end = df.iloc[-1]['close']
            else:
                price_at_start = None
                price_at_end = None
            
            # 날짜 형식 변환
            if isinstance(test_start_date, str):
                test_start_date_str = pd.to_datetime(test_start_date).strftime('%Y-%m-%d')
            elif isinstance(test_start_date, pd.Timestamp):
                test_start_date_str = test_start_date.strftime('%Y-%m-%d')
            else:
                test_start_date_str = str(test_start_date)
            
            if test_end_date is None:
                test_end_date_str = 'None'
            elif isinstance(test_end_date, str):
                test_end_date_str = pd.to_datetime(test_end_date).strftime('%Y-%m-%d')
            elif isinstance(test_end_date, pd.Timestamp):
                test_end_date_str = test_end_date.strftime('%Y-%m-%d')
            else:
                test_end_date_str = str(test_end_date)
            
            # 각 거래를 별도 행으로 변환
            trades = result.get('trades', [])
            new_rows = []
            
            for trade in trades:
                # 거래 날짜 형식 변환 (초 단위까지 포함)
                if isinstance(trade['date'], pd.Timestamp):
                    trade_date_ts = trade['date']
                    trade_date_str = trade_date_ts.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    trade_date_ts = pd.to_datetime(trade['date'])
                    trade_date_str = trade_date_ts.strftime('%Y-%m-%d %H:%M:%S')
                
                # 매수 날짜 형식 변환 (매도 거래인 경우) (초 단위까지 포함)
                buy_date_str = None
                if 'buy_date' in trade and trade['buy_date'] is not None:
                    if isinstance(trade['buy_date'], pd.Timestamp):
                        buy_date_str = trade['buy_date'].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        buy_date = pd.to_datetime(trade['buy_date'])
                        buy_date_str = buy_date.strftime('%Y-%m-%d %H:%M:%S')
                
                # execution_time은 캔들의 시간(trade_date)을 사용
                execution_time_for_trade = trade_date_ts
                
                # 각 거래 행 데이터 생성
                row = {
                    'execution_time': execution_time_for_trade.strftime('%Y-%m-%d %H:%M:%S'),
                    'test_start_date': test_start_date_str,
                    'test_end_date': test_end_date_str,
                    'ticker': ticker,
                    'interval': interval,
                    'price_at_start': price_at_start,
                    'price_at_end': price_at_end,
                    'buy_angle_threshold': buy_angle_threshold,
                    'sell_angle_threshold': sell_angle_threshold,
                    'stop_loss_percent': stop_loss_percent,
                    'min_sell_price': min_sell_price,
                    'price_slippage': price_slippage,
                    'initial_capital': initial_capital,
                    'window': window,
                    'aspect_ratio': aspect_ratio,
                    'total_trades': result['total_trades'],
                    'buy_count': result['buy_count'],
                    'sell_count': result['sell_count'],
                    'final_asset': result['final_asset'],
                    'total_return': result['total_return'],
                    # 거래 상세 정보
                    'trade_date': trade_date_str,
                    'action': trade.get('action', ''),
                    'trade_price': trade.get('price', None),
                    'trade_amount': trade.get('amount', None),
                    'trade_total_value': trade.get('total_value', None),
                    'buy_price': trade.get('buy_price', None),
                    'buy_date': buy_date_str,
                    'profit': trade.get('profit', None),
                    'profit_percent': trade.get('profit_percent', None),
                    'angle': trade.get('angle', None),
                    'total_asset_after_trade': trade.get('total_asset', None)
                }
                new_rows.append(row)
            
            # 거래가 없는 경우 요약 정보만 저장
            if len(new_rows) == 0:
                row = {
                    'execution_time': execution_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'test_start_date': test_start_date_str,
                    'test_end_date': test_end_date_str,
                    'ticker': ticker,
                    'interval': interval,
                    'price_at_start': price_at_start,
                    'price_at_end': price_at_end,
                    'buy_angle_threshold': buy_angle_threshold,
                    'sell_angle_threshold': sell_angle_threshold,
                    'stop_loss_percent': stop_loss_percent,
                    'min_sell_price': min_sell_price,
                    'price_slippage': price_slippage,
                    'initial_capital': initial_capital,
                    'window': window,
                    'aspect_ratio': aspect_ratio,
                    'total_trades': result['total_trades'],
                    'buy_count': result['buy_count'],
                    'sell_count': result['sell_count'],
                    'final_asset': result['final_asset'],
                    'total_return': result['total_return'],
                    # 거래 상세 정보 (거래 없음)
                    'trade_date': None,
                    'action': None,
                    'trade_price': None,
                    'trade_amount': None,
                    'trade_total_value': None,
                    'buy_price': None,
                    'buy_date': None,
                    'profit': None,
                    'profit_percent': None,
                    'angle': None,
                    'total_asset_after_trade': None
                }
                new_rows.append(row)
            
            # 새 행들을 DataFrame으로 생성
            new_df = pd.DataFrame(new_rows)
            
            # 파일명에서 조건 문자열 추출
            base_filename = os.path.basename(result_file)
            name_without_ext = os.path.splitext(base_filename)[0]
            
            # 거래 기록이 있는 경우: 거래 날짜별, 시간별로 분할 저장
            if len(new_rows) > 0 and any(row.get('trade_date') is not None for row in new_rows):
                # trade_date가 있는 행만 필터링
                new_df_with_trades = new_df[new_df['trade_date'].notna()].copy()
                
                if len(new_df_with_trades) > 0:
                    # trade_date를 datetime으로 변환
                    new_df_with_trades['_temp_trade_date'] = pd.to_datetime(new_df_with_trades['trade_date'])
                    new_df_with_trades['_temp_date'] = new_df_with_trades['_temp_trade_date'].dt.date
                    new_df_with_trades['_temp_hour'] = new_df_with_trades['_temp_trade_date'].dt.hour
                    
                    saved_files = 0
                    # 거래 날짜별로 그룹화
                    for date, date_group in new_df_with_trades.groupby('_temp_date'):
                        date_folder = pd.Timestamp(date).strftime('%Y-%m-%d')
                        date_dir = os.path.join(self.data_dir, date_folder)
                        
                        # 날짜 폴더 생성
                        if not os.path.exists(date_dir):
                            os.makedirs(date_dir)
                        
                        # 해당 날짜의 데이터를 시간별로 그룹화
                        for hour, hour_group in date_group.groupby('_temp_hour'):
                            # 시간별 데이터프레임 (임시 컬럼 제거)
                            hour_df = hour_group.drop(['_temp_trade_date', '_temp_date', '_temp_hour'], axis=1)
                            
                            # 시간별 파일명 생성 (YYYYMMDD_HH 형식)
                            date_str = pd.Timestamp(date).strftime('%Y%m%d')
                            hour_str = f"{date_str}_{hour:02d}"
                            filename = f"{name_without_ext}_{hour_str}.csv"
                            save_path = os.path.join(date_dir, filename)
                            
                            # 기존 파일이 있으면 로드하고 병합, 없으면 새로 생성
                            if os.path.exists(save_path):
                                try:
                                    existing_df = pd.read_csv(save_path)
                                    if len(existing_df) > 0:
                                        # 기존 데이터와 병합 (중복 제거)
                                        combined_df = pd.concat([existing_df, hour_df], ignore_index=True, sort=False)
                                        # trade_date와 action을 기준으로 중복 제거 (execution_time은 이제 trade_date와 같음)
                                        if 'trade_date' in combined_df.columns and 'action' in combined_df.columns:
                                            combined_df = combined_df.drop_duplicates(subset=['trade_date', 'action'], keep='last')
                                        elif 'trade_date' in combined_df.columns:
                                            combined_df = combined_df.drop_duplicates(subset=['trade_date'], keep='last')
                                        if 'trade_date' in combined_df.columns:
                                            combined_df = combined_df.sort_values('trade_date')
                                        else:
                                            combined_df = combined_df.sort_values('execution_time')
                                        combined_df.to_csv(save_path, index=False)
                                    else:
                                        hour_df.to_csv(save_path, index=False)
                                except Exception as e:
                                    # 파일 읽기 실패 시 새로 저장
                                    hour_df.to_csv(save_path, index=False)
                            else:
                                hour_df.to_csv(save_path, index=False)
                            
                            saved_files += 1
                    
                    print(f"\n백테스트 결과 저장 완료 (날짜별/시간별 분할): {saved_files}개 파일 생성")
                    print(f"  총 {len(new_df_with_trades)}개 거래 기록")
                else:
                    # 거래 기록이 없지만 실행 정보가 있는 경우: 실행 시간 기준으로 저장
                    date_folder = execution_time.strftime('%Y-%m-%d')
                    date_dir = os.path.join(self.data_dir, date_folder)
                    if not os.path.exists(date_dir):
                        os.makedirs(date_dir)
                    
                    hour_str = execution_time.strftime('%Y%m%d_%H')
                    filename = f"{name_without_ext}_{hour_str}.csv"
                    save_path = os.path.join(date_dir, filename)
                    new_df.to_csv(save_path, index=False)
                    print(f"\n백테스트 결과 저장 완료: {save_path}")
                    print(f"  이번 실행: {len(new_rows)}개 기록 (거래 없음)")
            else:
                # 거래 기록이 없는 경우: 실행 시간 기준으로 저장
                date_folder = execution_time.strftime('%Y-%m-%d')
                date_dir = os.path.join(self.data_dir, date_folder)
                if not os.path.exists(date_dir):
                    os.makedirs(date_dir)
                
                hour_str = execution_time.strftime('%Y%m%d_%H')
                filename = f"{name_without_ext}_{hour_str}.csv"
                save_path = os.path.join(date_dir, filename)
                new_df.to_csv(save_path, index=False)
                print(f"\n백테스트 결과 저장 완료: {save_path}")
                print(f"  이번 실행: {len(new_rows)}개 기록 (거래 없음)")
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def save_qqc_backtest_result(self, df, result, test_start_date, test_end_date,
                                volume_window, ma_window, volume_multiplier,
                                buy_cash_ratio, hold_period, profit_target, stop_loss,
                                price_slippage, initial_capital,
                                ticker='BTC', interval='24h',
                                current_krw=None, current_btc=None):
        """
        QQC 백테스트 결과를 파일에 저장합니다.
        각 거래를 별도 행으로 저장합니다.
        
        Parameters:
        - df (pd.DataFrame): 백테스트에 사용된 데이터프레임 (주가 정보 포함)
        - result (dict): 백테스트 결과 딕셔너리
        - test_start_date (str or pd.Timestamp): 백테스트 시작 날짜
        - test_end_date (str or pd.Timestamp or None): 백테스트 종료 날짜
        - volume_window (int): 거래량 평균 계산용 윈도우
        - ma_window (int): 이동평균 계산용 윈도우
        - volume_multiplier (float): 거래량 배수
        - buy_cash_ratio (float): 매수시 사용할 현금 비율
        - hold_period (int): 매수 후 보유 기간 (캔들 수)
        - profit_target (float): 이익실현 목표 수익률 (%)
        - stop_loss (float): 손절 기준 수익률 (%)
        - price_slippage (float): 거래 가격 슬리퍼지
        - initial_capital (float): 초기 자본
        - ticker (str): 암호화폐 티커. 기본값 'BTC'
        - interval (str): 캔들스틱 간격. 기본값 '24h'
        """
        try:
            self.ensure_data_directory()
            
            # 백테스트 결과 파일 경로 (조건 기반)
            result_file = Config.get_backtest_result_file_path(ticker=ticker, condition_dict=self.condition_dict)
            
            # 실행 시점
            execution_time = pd.Timestamp.now()
            
            # 백테스트 시점의 주가 정보
            if len(df) > 0:
                price_at_start = df.iloc[0]['close']
                price_at_end = df.iloc[-1]['close']
            else:
                price_at_start = None
                price_at_end = None
            
            # 날짜 형식 변환
            if isinstance(test_start_date, str):
                test_start_date_str = pd.to_datetime(test_start_date).strftime('%Y-%m-%d')
            elif isinstance(test_start_date, pd.Timestamp):
                test_start_date_str = test_start_date.strftime('%Y-%m-%d')
            else:
                test_start_date_str = str(test_start_date)
            
            if test_end_date is None:
                test_end_date_str = 'None'
            elif isinstance(test_end_date, str):
                test_end_date_str = pd.to_datetime(test_end_date).strftime('%Y-%m-%d')
            elif isinstance(test_end_date, pd.Timestamp):
                test_end_date_str = test_end_date.strftime('%Y-%m-%d')
            else:
                test_end_date_str = str(test_end_date)
            
            # 각 거래를 별도 행으로 변환
            trades = result.get('trades', [])
            new_rows = []
            
            for trade in trades:
                # 거래 날짜 형식 변환 (초 단위까지 포함)
                if isinstance(trade['date'], pd.Timestamp):
                    trade_date_ts = trade['date']
                    trade_date_str = trade_date_ts.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    trade_date_ts = pd.to_datetime(trade['date'])
                    trade_date_str = trade_date_ts.strftime('%Y-%m-%d %H:%M:%S')
                
                # 매수 날짜 형식 변환 (매도 거래인 경우) (초 단위까지 포함)
                buy_date_str = None
                if 'buy_date' in trade and trade['buy_date'] is not None:
                    if isinstance(trade['buy_date'], pd.Timestamp):
                        buy_date_str = trade['buy_date'].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        buy_date = pd.to_datetime(trade['buy_date'])
                        buy_date_str = buy_date.strftime('%Y-%m-%d %H:%M:%S')
                
                # execution_time은 실제 백테스트 실행 시간을 사용 (각 실행별로 구분)
                # 이렇게 하면 같은 거래라도 실행 시점별로 구분되어 누적 저장됨
                execution_time_for_trade = execution_time
                
                # 각 거래 행 데이터 생성
                row = {
                    'execution_time': execution_time_for_trade.strftime('%Y-%m-%d %H:%M:%S'),
                    'test_start_date': test_start_date_str,
                    'test_end_date': test_end_date_str,
                    'ticker': ticker,
                    'interval': interval,
                    'price_at_start': price_at_start,
                    'price_at_end': price_at_end,
                    'volume_window': volume_window,
                    'ma_window': ma_window,
                    'volume_multiplier': volume_multiplier,
                    'buy_cash_ratio': buy_cash_ratio,
                    'hold_period': hold_period,
                    'profit_target': profit_target,
                    'stop_loss': stop_loss,
                    'price_slippage': price_slippage,
                    'initial_capital': initial_capital,
                    'total_trades': result['total_trades'],
                    'buy_count': result['buy_count'],
                    'sell_count': result['sell_count'],
                    'final_asset': result['final_asset'],
                    'total_return': result['total_return'],
                    # 거래 상세 정보
                    'trade_date': trade_date_str,
                    'action': trade.get('action', ''),
                    'trade_price': trade.get('price', None),
                    'trade_amount': trade.get('amount', None),
                    'trade_total_value': trade.get('total_value', None),
                    'buy_price': trade.get('buy_price', None),
                    'buy_date': buy_date_str,
                    'profit': trade.get('profit', None),
                    'profit_percent': trade.get('profit_percent', None),
                    'volume_a': trade.get('volume_a', None),
                    'ma_c': trade.get('ma_c', None),
                    'total_asset_after_trade': trade.get('total_asset', None),
                    # 실제 잔고 정보
                    'current_krw_balance': current_krw,
                    'current_btc_balance': current_btc,
                    'total_balance': (current_krw + (current_btc * trade.get('price', 0))) if (current_krw is not None and current_btc is not None) else None
                }
                new_rows.append(row)
            
            # 거래가 없는 경우 요약 정보만 저장
            if len(new_rows) == 0:
                row = {
                    'execution_time': execution_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'test_start_date': test_start_date_str,
                    'test_end_date': test_end_date_str,
                    'ticker': ticker,
                    'interval': interval,
                    'price_at_start': price_at_start,
                    'price_at_end': price_at_end,
                    'volume_window': volume_window,
                    'ma_window': ma_window,
                    'volume_multiplier': volume_multiplier,
                    'buy_cash_ratio': buy_cash_ratio,
                    'hold_period': hold_period,
                    'profit_target': profit_target,
                    'stop_loss': stop_loss,
                    'price_slippage': price_slippage,
                    'initial_capital': initial_capital,
                    'total_trades': result['total_trades'],
                    'buy_count': result['buy_count'],
                    'sell_count': result['sell_count'],
                    'final_asset': result['final_asset'],
                    'total_return': result['total_return'],
                    # 거래 상세 정보 (거래 없음)
                    'trade_date': None,
                    'action': None,
                    'trade_price': None,
                    'trade_amount': None,
                    'trade_total_value': None,
                    'buy_price': None,
                    'buy_date': None,
                    'profit': None,
                    'profit_percent': None,
                    'volume_a': None,
                    'ma_c': None,
                    'total_asset_after_trade': None,
                    # 실제 잔고 정보
                    'current_krw_balance': current_krw,
                    'current_btc_balance': current_btc,
                    'total_balance': (current_krw + (current_btc * price_at_end)) if (current_krw is not None and current_btc is not None and price_at_end is not None) else None
                }
                new_rows.append(row)
            
            # 새 행들을 DataFrame으로 생성
            new_df = pd.DataFrame(new_rows)
            
            # 파일명에서 조건 문자열 추출
            base_filename = os.path.basename(result_file)
            name_without_ext = os.path.splitext(base_filename)[0]
            
            # 거래 기록이 있는 경우: 거래 날짜별, 시간별로 분할 저장
            if len(new_rows) > 0 and any(row.get('trade_date') is not None for row in new_rows):
                # trade_date가 있는 행만 필터링
                new_df_with_trades = new_df[new_df['trade_date'].notna()].copy()
                
                if len(new_df_with_trades) > 0:
                    # trade_date를 datetime으로 변환
                    new_df_with_trades['_temp_trade_date'] = pd.to_datetime(new_df_with_trades['trade_date'])
                    new_df_with_trades['_temp_date'] = new_df_with_trades['_temp_trade_date'].dt.date
                    new_df_with_trades['_temp_hour'] = new_df_with_trades['_temp_trade_date'].dt.hour
                    
                    saved_files = 0
                    # 거래 날짜별로 그룹화
                    for date, date_group in new_df_with_trades.groupby('_temp_date'):
                        date_folder = pd.Timestamp(date).strftime('%Y-%m-%d')
                        date_dir = os.path.join(self.data_dir, date_folder)
                        
                        # 날짜 폴더 생성
                        if not os.path.exists(date_dir):
                            os.makedirs(date_dir)
                        
                        # 해당 날짜의 데이터를 시간별로 그룹화
                        for hour, hour_group in date_group.groupby('_temp_hour'):
                            # 시간별 데이터프레임 (임시 컬럼 제거)
                            hour_df = hour_group.drop(['_temp_trade_date', '_temp_date', '_temp_hour'], axis=1)
                            
                            # 시간별 파일명 생성 (YYYYMMDD_HH 형식)
                            date_str = pd.Timestamp(date).strftime('%Y%m%d')
                            hour_str = f"{date_str}_{hour:02d}"
                            filename = f"{name_without_ext}_{hour_str}.csv"
                            save_path = os.path.join(date_dir, filename)
                            
                            # 기존 파일이 있으면 로드하고 병합, 없으면 새로 생성
                            if os.path.exists(save_path):
                                try:
                                    existing_df = pd.read_csv(save_path)
                                    if len(existing_df) > 0:
                                        # 기존 데이터와 병합 (중복 제거)
                                        combined_df = pd.concat([existing_df, hour_df], ignore_index=True, sort=False)
                                        # execution_time, trade_date, action을 기준으로 중복 제거
                                        # 이렇게 하면 같은 실행의 같은 거래만 중복 제거되고, 다른 실행의 같은 거래는 누적됨
                                        if 'execution_time' in combined_df.columns and 'trade_date' in combined_df.columns and 'action' in combined_df.columns:
                                            combined_df = combined_df.drop_duplicates(subset=['execution_time', 'trade_date', 'action'], keep='last')
                                        elif 'execution_time' in combined_df.columns and 'trade_date' in combined_df.columns:
                                            combined_df = combined_df.drop_duplicates(subset=['execution_time', 'trade_date'], keep='last')
                                        elif 'trade_date' in combined_df.columns and 'action' in combined_df.columns:
                                            combined_df = combined_df.drop_duplicates(subset=['trade_date', 'action'], keep='last')
                                        elif 'trade_date' in combined_df.columns:
                                            combined_df = combined_df.drop_duplicates(subset=['trade_date'], keep='last')
                                        if 'trade_date' in combined_df.columns:
                                            combined_df = combined_df.sort_values('trade_date')
                                        else:
                                            combined_df = combined_df.sort_values('execution_time')
                                        combined_df.to_csv(save_path, index=False)
                                    else:
                                        hour_df.to_csv(save_path, index=False)
                                except Exception as e:
                                    # 파일 읽기 실패 시 새로 저장
                                    hour_df.to_csv(save_path, index=False)
                            else:
                                hour_df.to_csv(save_path, index=False)
                            
                            saved_files += 1
                    
                    print(f"\n백테스트 결과 저장 완료 (날짜별/시간별 분할): {saved_files}개 파일 생성")
                    print(f"  총 {len(new_df_with_trades)}개 거래 기록")
                else:
                    # 거래 기록이 없지만 실행 정보가 있는 경우: 실행 시간 기준으로 저장
                    date_folder = execution_time.strftime('%Y-%m-%d')
                    date_dir = os.path.join(self.data_dir, date_folder)
                    if not os.path.exists(date_dir):
                        os.makedirs(date_dir)
                    
                    hour_str = execution_time.strftime('%Y%m%d_%H')
                    filename = f"{name_without_ext}_{hour_str}.csv"
                    save_path = os.path.join(date_dir, filename)
                    new_df.to_csv(save_path, index=False)
                    print(f"\n백테스트 결과 저장 완료: {save_path}")
                    print(f"  이번 실행: {len(new_rows)}개 기록 (거래 없음)")
            else:
                # 거래 기록이 없는 경우: 실행 시간 기준으로 저장
                date_folder = execution_time.strftime('%Y-%m-%d')
                date_dir = os.path.join(self.data_dir, date_folder)
                if not os.path.exists(date_dir):
                    os.makedirs(date_dir)
                
                hour_str = execution_time.strftime('%Y%m%d_%H')
                filename = f"{name_without_ext}_{hour_str}.csv"
                save_path = os.path.join(date_dir, filename)
                new_df.to_csv(save_path, index=False)
                print(f"\n백테스트 결과 저장 완료: {save_path}")
                print(f"  이번 실행: {len(new_rows)}개 기록 (거래 없음)")
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def load_backtest_results(self, ticker='BTC', condition_dict=None):
        """
        저장된 백테스트 결과를 로드합니다.
        단일 파일이 없으면 날짜별 폴더에서 시간별 분할 파일들을 찾아 로드하여 병합합니다.

        Parameters:
        - ticker (str): 암호화폐 티커. 기본값 'BTC'
        - condition_dict (dict, optional): 조건 딕셔너리. None이면 self.condition_dict 또는 기본 경로 사용

        Returns:
        - pd.DataFrame or None: 백테스트 결과 데이터프레임. 파일이 없으면 None 반환
        """
        try:
            # 조건 기반 파일 경로 사용
            if condition_dict is None:
                condition_dict = self.condition_dict
            result_file = Config.get_backtest_result_file_path(ticker=ticker, condition_dict=condition_dict)

            # 먼저 단일 파일 확인
            if os.path.exists(result_file):
                df = pd.read_csv(result_file)

                # execution_time을 datetime으로 변환
                if 'execution_time' in df.columns:
                    df['execution_time'] = pd.to_datetime(df['execution_time'])

                # 최신순으로 정렬
                df = df.sort_values('execution_time', ascending=False)

                print(f"백테스트 결과 로드 완료: {result_file}")
                print(f"  총 {len(df)}개 기록")

                return df

            # 단일 파일이 없으면 날짜별 폴더에서 시간별 분할 파일들 찾기
            print("날짜별 폴더에서 백테스트 결과 시간별 분할 파일을 검색합니다...")

            # 파일명에서 조건 부분 추출 (확장자 제외)
            base_filename = os.path.basename(result_file)
            name_without_ext = os.path.splitext(base_filename)[0]

            # 날짜별 폴더에서 해당 조건으로 시작하는 모든 시간별 파일 찾기
            all_dataframes = []

            if os.path.exists(self.data_dir):
                # 날짜별 폴더 목록 가져오기 (YYYY-MM-DD 형식)
                date_folders = []
                for item in os.listdir(self.data_dir):
                    item_path = os.path.join(self.data_dir, item)
                    if os.path.isdir(item_path) and len(item) == 10 and item.count('-') == 2:
                        try:
                            # 날짜 형식 검증
                            datetime.strptime(item, '%Y-%m-%d')
                            date_folders.append(item)
                        except ValueError:
                            continue

                # 날짜순으로 정렬
                date_folders.sort()

                # 각 날짜 폴더에서 시간별 파일 찾기
                for date_folder in date_folders:
                    date_dir = os.path.join(self.data_dir, date_folder)
                    if not os.path.isdir(date_dir):
                        continue

                    # 해당 날짜 폴더의 모든 파일 확인
                    for filename in os.listdir(date_dir):
                        if not filename.endswith('.csv'):
                            continue

                        # 현재 조건의 파일명으로 시작하는지 확인
                        if filename.startswith(name_without_ext):
                            file_path = os.path.join(date_dir, filename)
                            try:
                                file_df = pd.read_csv(file_path)
                                if len(file_df) > 0:
                                    all_dataframes.append(file_df)
                            except Exception as e:
                                print(f"경고: 파일 로드 실패 ({file_path}): {e}")
                                continue

            if len(all_dataframes) == 0:
                print(f"백테스트 결과 파일이 없습니다: {result_file}")
                return None

            # 모든 데이터프레임 병합
            df = pd.concat(all_dataframes, ignore_index=True)

            # execution_time을 datetime으로 변환
            if 'execution_time' in df.columns:
                df['execution_time'] = pd.to_datetime(df['execution_time'])

            # 중복 제거 (execution_time, trade_date, action 기준)
            if 'execution_time' in df.columns and 'trade_date' in df.columns and 'action' in df.columns:
                df = df.drop_duplicates(subset=['execution_time', 'trade_date', 'action'], keep='last')
            elif 'execution_time' in df.columns and 'trade_date' in df.columns:
                df = df.drop_duplicates(subset=['execution_time', 'trade_date'], keep='last')
            elif 'trade_date' in df.columns:
                df = df.drop_duplicates(subset=['trade_date'], keep='last')

            # 최신순으로 정렬
            df = df.sort_values('execution_time', ascending=False)

            print(f"백테스트 결과 로드 완료 (시간별 분할 파일 병합): {len(df)}개 기록")
            print(f"  로드된 파일 수: {len(all_dataframes)}개")

            return df

        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def get_last_backtest_info(self, ticker='BTC', buy_angle_threshold=None,
                              sell_angle_threshold=None, stop_loss_percent=None,
                              min_sell_price=None, price_slippage=None,
                              initial_capital=None, window=None, aspect_ratio=None,
                              interval='24h', condition_dict=None):
        """
        같은 파라미터로 실행된 마지막 백테스트 정보를 조회합니다.
        
        Parameters:
        - ticker (str): 암호화폐 티커. 기본값 'BTC'
        - buy_angle_threshold (float, optional): 매수 조건 각도
        - sell_angle_threshold (float, optional): 매도 조건 각도
        - stop_loss_percent (float, optional): 손절 기준
        - min_sell_price (float, optional): 최소 매도 가격
        - price_slippage (float, optional): 거래 가격 슬리퍼지
        - initial_capital (float, optional): 초기 자본
        - window (int, optional): 추세선 계산 윈도우 크기
        - aspect_ratio (float, optional): 차트 종횡비
        - interval (str): 캔들스틱 간격. 기본값 '24h'
        - condition_dict (dict, optional): 조건 딕셔너리. None이면 self.condition_dict 사용
        
        Returns:
        - dict or None: 마지막 백테스트 정보 (last_trade_date, last_execution_time, last_final_asset).
                       일치하는 기록이 없으면 None 반환
        """
        try:
            # 조건 딕셔너리 사용
            if condition_dict is None:
                condition_dict = self.condition_dict
            df = self.load_backtest_results(ticker=ticker, condition_dict=condition_dict)
            if df is None or len(df) == 0:
                return None
            
            # 파라미터가 모두 None이면 모든 기록 조회 (파라미터 필터링 없음)
            if all(param is None for param in [buy_angle_threshold, sell_angle_threshold,
                                               stop_loss_percent, min_sell_price,
                                               price_slippage, initial_capital,
                                               window, aspect_ratio]):
                # 파라미터 필터링 없이 가장 최신 execution_time의 마지막 trade_date 찾기
                if 'execution_time' not in df.columns:
                    return None
                
                # 가장 최신 execution_time 찾기
                latest_execution_time = df['execution_time'].max()
                latest_df = df[df['execution_time'] == latest_execution_time]
                
                # trade_date가 있는 행 중에서 최신 날짜 찾기
                if 'trade_date' in latest_df.columns:
                    latest_df = latest_df[latest_df['trade_date'].notna()]
                    if len(latest_df) > 0:
                        # trade_date를 datetime으로 변환하여 최신 날짜 찾기
                        latest_df['trade_date_dt'] = pd.to_datetime(latest_df['trade_date'])
                        last_trade_date = latest_df['trade_date_dt'].max()
                        last_final_asset = latest_df.iloc[0]['final_asset'] if 'final_asset' in latest_df.columns else None
                        
                        return {
                            'last_trade_date': last_trade_date,
                            'last_execution_time': latest_execution_time,
                            'last_final_asset': last_final_asset
                        }
            
            # 파라미터로 필터링
            filtered_df = df.copy()
            
            if buy_angle_threshold is not None:
                if 'buy_angle_threshold' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['buy_angle_threshold'] == buy_angle_threshold]
            
            if sell_angle_threshold is not None:
                if 'sell_angle_threshold' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['sell_angle_threshold'] == sell_angle_threshold]
            
            if stop_loss_percent is not None:
                if 'stop_loss_percent' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['stop_loss_percent'] == stop_loss_percent]
            
            if min_sell_price is not None:
                if 'min_sell_price' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['min_sell_price'] == min_sell_price]
            
            if price_slippage is not None:
                if 'price_slippage' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['price_slippage'] == price_slippage]
            
            if initial_capital is not None:
                if 'initial_capital' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['initial_capital'] == initial_capital]
            
            if window is not None:
                if 'window' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['window'] == window]
            
            if aspect_ratio is not None:
                if 'aspect_ratio' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['aspect_ratio'] == aspect_ratio]
            
            if 'interval' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['interval'] == interval]
            
            if len(filtered_df) == 0:
                return None
            
            # 가장 최신 execution_time 찾기
            if 'execution_time' not in filtered_df.columns:
                return None
            
            latest_execution_time = filtered_df['execution_time'].max()
            latest_df = filtered_df[filtered_df['execution_time'] == latest_execution_time]
            
            # trade_date가 있는 행 중에서 최신 날짜 찾기
            if 'trade_date' in latest_df.columns:
                latest_df = latest_df[latest_df['trade_date'].notna()]
                if len(latest_df) > 0:
                    # trade_date를 datetime으로 변환하여 최신 날짜 찾기
                    latest_df['trade_date_dt'] = pd.to_datetime(latest_df['trade_date'])
                    last_trade_date = latest_df['trade_date_dt'].max()
                    last_final_asset = latest_df.iloc[0]['final_asset'] if 'final_asset' in latest_df.columns else None
                    
                    return {
                        'last_trade_date': last_trade_date,
                        'last_execution_time': latest_execution_time,
                        'last_final_asset': last_final_asset
                    }
            
            return None
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise

