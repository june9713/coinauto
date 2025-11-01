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
        
        Returns:
        - pd.DataFrame: 저장된 OHLCV 데이터프레임. 파일이 없으면 None 반환
        """
        try:
            if not os.path.exists(self.data_file):
                print(f"기록 파일이 없습니다: {self.data_file}")
                return None
            
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
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def save_history_to_file(self, df, is_realtime_update=False):
        """
        데이터프레임을 파일에 저장합니다.
        
        Parameters:
        - df (pd.DataFrame): 저장할 OHLCV 데이터프레임
        - is_realtime_update (bool): 실시간 업데이트 여부. True면 날짜별 폴더에 시간별 파일로 저장
        """
        try:
            self.ensure_data_directory()
            
            # 중복 제거 및 정렬
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
            
            if is_realtime_update and len(df) > 0:
                # 실시간 업데이트: 날짜별 폴더에 시간별 파일로 저장
                now = pd.Timestamp.now()
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
                
                # 새로운 데이터만 저장 (이전 파일과 병합하지 않음)
                # 해당 시간대의 데이터만 필터링 (선택사항: 필요시 주석 해제)
                # hour_start = now.replace(minute=0, second=0, microsecond=0)
                # hour_end = hour_start + timedelta(hours=1)
                # hour_df = df[(df.index >= hour_start) & (df.index < hour_end)]
                # if len(hour_df) > 0:
                #     hour_df.to_csv(save_path, index=True)
                # else:
                #     print(f"해당 시간대에 새로운 데이터가 없습니다. 파일 저장 생략.")
                #     return
                
                # 새로운 데이터가 있는 경우에만 저장
                df.to_csv(save_path, index=True)
                print(f"기록 파일 저장 완료 (실시간 업데이트): {save_path}")
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
                            
                            # 시간별 파일로 저장
                            hour_df.to_csv(save_path, index=True)
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
    
    def update_history_data(self, ticker='BTC', interval='24h', start_date='2014-01-01'):
        """
        기록 파일의 마지막 날짜부터 현재까지의 데이터를 다운로드하여 파일에 누적 저장합니다.
        
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
                
                # 마지막 캔들 시간이 현재 시간보다 최신이거나 같으면 업데이트 불필요
                # (시분초까지 비교)
                if last_date >= now:
                    print("기록이 최신 상태입니다. 업데이트 불필요.")
                    return existing_df
                
                # 마지막 캔들 이후부터 데이터 수집
                print(f"누락된 시간부터 다운로드: {last_date.strftime('%Y-%m-%d %H:%M:%S')} ~ {now.strftime('%Y-%m-%d %H:%M:%S')}")
                
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
                
                # 기존 데이터와 병합
                combined_df = pd.concat([existing_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]  # 중복 시 새로운 데이터 우선
                combined_df = combined_df.sort_index()
                
                # 실시간 업데이트: 새로운 데이터만 별도 파일로 저장
                self.save_history_to_file(new_df, is_realtime_update=True)
                
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
                self.save_history_to_file(df)
                
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
                    trade_date_str = trade['date'].strftime('%Y-%m-%d %H:%M:%S')
                else:
                    trade_date = pd.to_datetime(trade['date'])
                    trade_date_str = trade_date.strftime('%Y-%m-%d %H:%M:%S')
                
                # 매수 날짜 형식 변환 (매도 거래인 경우) (초 단위까지 포함)
                buy_date_str = None
                if 'buy_date' in trade and trade['buy_date'] is not None:
                    if isinstance(trade['buy_date'], pd.Timestamp):
                        buy_date_str = trade['buy_date'].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        buy_date = pd.to_datetime(trade['buy_date'])
                        buy_date_str = buy_date.strftime('%Y-%m-%d %H:%M:%S')
                
                # 각 거래 행 데이터 생성
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
            
            # 새로운 업데이트가 있는지 확인 (거래 기록이 있는 경우)
            has_new_updates = len(new_rows) > 0 and any(
                row.get('trade_date') is not None for row in new_rows
            )
            
            # 실시간 업데이트이고 새로운 업데이트가 있는 경우: 날짜별 폴더에 시간별 파일로 저장
            if has_new_updates:
                # 날짜별 폴더 생성
                date_folder = execution_time.strftime('%Y-%m-%d')
                date_dir = os.path.join(self.data_dir, date_folder)
                if not os.path.exists(date_dir):
                    os.makedirs(date_dir)
                
                # 파일명에서 조건 문자열 추출
                base_filename = os.path.basename(result_file)
                # 확장자 제거
                name_without_ext = os.path.splitext(base_filename)[0]
                # 시간별 파일명 생성 (YYYYMMDD_HH 형식)
                hour_str = execution_time.strftime('%Y%m%d_%H')
                filename = f"{name_without_ext}_{hour_str}.csv"
                save_path = os.path.join(date_dir, filename)
                
                # 새로운 데이터만 저장
                new_df.to_csv(save_path, index=False)
                print(f"\n백테스트 결과 저장 완료 (실시간 업데이트): {save_path}")
                print(f"  이번 실행: {len(new_rows)}개 거래 기록")
            else:
                # 초기 다운로드 또는 업데이트 없음: 기존 방식으로 저장 (누적)
                # 기존 파일이 있으면 로드, 없으면 새로 생성
                if os.path.exists(result_file):
                    try:
                        # 파일 크기 확인 (빈 파일 체크)
                        file_size = os.path.getsize(result_file)
                        if file_size > 0:
                            existing_df = pd.read_csv(result_file)
                            # 빈 파일이거나 헤더만 있는 경우 처리
                            if len(existing_df) > 0:
                                print(f"  기존 거래 기록 {len(existing_df)}개 발견, 새 거래 기록 {len(new_rows)}개 추가 중...")
                                # 기존 데이터프레임과 새 데이터프레임을 병합 (컬럼은 자동으로 맞춰짐)
                                combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
                                # 컬럼 순서를 new_rows[0]의 키 순서로 정렬 (일관성 유지)
                                column_order = list(new_rows[0].keys())
                                # 기존에 있던 컬럼만 순서대로 정렬
                                existing_columns = [col for col in column_order if col in combined_df.columns]
                                additional_columns = [col for col in combined_df.columns if col not in column_order]
                                combined_df = combined_df[existing_columns + additional_columns]
                            else:
                                # 헤더만 있는 경우
                                print(f"  헤더만 있는 파일 발견, 새 거래 기록 {len(new_rows)}개 추가 중...")
                                combined_df = new_df
                        else:
                            # 빈 파일인 경우
                            print(f"  빈 파일 발견, 새 거래 기록 {len(new_rows)}개 생성 중...")
                            combined_df = new_df
                    except Exception as e:
                        # 파일 읽기 실패 시 새로 생성
                        err = traceback.format_exc()
                        print(f"경고: 기존 파일 읽기 실패. 새로 생성합니다.")
                        print(f"  오류: {err}")
                        combined_df = new_df
                else:
                    print(f"  새 파일 생성 중... (거래 기록 {len(new_rows)}개)")
                    combined_df = new_df
                
                # CSV 파일로 저장 (기존 파일 덮어쓰기)
                combined_df.to_csv(result_file, index=False)
                
                print(f"\n백테스트 결과 저장 완료: {result_file}")
                print(f"  총 {len(combined_df)}개 거래 기록 (이번 실행: {len(new_rows)}개 거래)")
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def save_qqc_backtest_result(self, df, result, test_start_date, test_end_date,
                                volume_window, ma_window, volume_multiplier,
                                buy_cash_ratio, hold_period, profit_target, stop_loss,
                                price_slippage, initial_capital,
                                ticker='BTC', interval='24h'):
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
                    trade_date_str = trade['date'].strftime('%Y-%m-%d %H:%M:%S')
                else:
                    trade_date = pd.to_datetime(trade['date'])
                    trade_date_str = trade_date.strftime('%Y-%m-%d %H:%M:%S')
                
                # 매수 날짜 형식 변환 (매도 거래인 경우) (초 단위까지 포함)
                buy_date_str = None
                if 'buy_date' in trade and trade['buy_date'] is not None:
                    if isinstance(trade['buy_date'], pd.Timestamp):
                        buy_date_str = trade['buy_date'].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        buy_date = pd.to_datetime(trade['buy_date'])
                        buy_date_str = buy_date.strftime('%Y-%m-%d %H:%M:%S')
                
                # 각 거래 행 데이터 생성
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
                    'total_asset_after_trade': None
                }
                new_rows.append(row)
            
            # 새 행들을 DataFrame으로 생성
            new_df = pd.DataFrame(new_rows)
            
            # 새로운 업데이트가 있는지 확인 (거래 기록이 있는 경우)
            has_new_updates = len(new_rows) > 0 and any(
                row.get('trade_date') is not None for row in new_rows
            )
            
            # 실시간 업데이트이고 새로운 업데이트가 있는 경우: 날짜별 폴더에 시간별 파일로 저장
            if has_new_updates:
                # 날짜별 폴더 생성
                date_folder = execution_time.strftime('%Y-%m-%d')
                date_dir = os.path.join(self.data_dir, date_folder)
                if not os.path.exists(date_dir):
                    os.makedirs(date_dir)
                
                # 파일명에서 조건 문자열 추출
                base_filename = os.path.basename(result_file)
                # 확장자 제거
                name_without_ext = os.path.splitext(base_filename)[0]
                # 시간별 파일명 생성 (YYYYMMDD_HH 형식)
                hour_str = execution_time.strftime('%Y%m%d_%H')
                filename = f"{name_without_ext}_{hour_str}.csv"
                save_path = os.path.join(date_dir, filename)
                
                # 새로운 데이터만 저장
                new_df.to_csv(save_path, index=False)
                print(f"\n백테스트 결과 저장 완료 (실시간 업데이트): {save_path}")
                print(f"  이번 실행: {len(new_rows)}개 거래 기록")
            else:
                # 초기 다운로드 또는 업데이트 없음: 기존 방식으로 저장 (누적)
                # 기존 파일이 있으면 로드, 없으면 새로 생성
                if os.path.exists(result_file):
                    try:
                        # 파일 크기 확인 (빈 파일 체크)
                        file_size = os.path.getsize(result_file)
                        if file_size > 0:
                            existing_df = pd.read_csv(result_file)
                            # 빈 파일이거나 헤더만 있는 경우 처리
                            if len(existing_df) > 0:
                                print(f"  기존 거래 기록 {len(existing_df)}개 발견, 새 거래 기록 {len(new_rows)}개 추가 중...")
                                # 기존 데이터프레임과 새 데이터프레임을 병합 (컬럼은 자동으로 맞춰짐)
                                combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
                                # 컬럼 순서를 new_rows[0]의 키 순서로 정렬 (일관성 유지)
                                column_order = list(new_rows[0].keys())
                                # 기존에 있던 컬럼만 순서대로 정렬
                                existing_columns = [col for col in column_order if col in combined_df.columns]
                                additional_columns = [col for col in combined_df.columns if col not in column_order]
                                combined_df = combined_df[existing_columns + additional_columns]
                            else:
                                # 헤더만 있는 경우
                                print(f"  헤더만 있는 파일 발견, 새 거래 기록 {len(new_rows)}개 추가 중...")
                                combined_df = new_df
                        else:
                            # 빈 파일인 경우
                            print(f"  빈 파일 발견, 새 거래 기록 {len(new_rows)}개 생성 중...")
                            combined_df = new_df
                    except Exception as e:
                        # 파일 읽기 실패 시 새로 생성
                        err = traceback.format_exc()
                        print(f"경고: 기존 파일 읽기 실패. 새로 생성합니다.")
                        print(f"  오류: {err}")
                        combined_df = new_df
                else:
                    print(f"  새 파일 생성 중... (거래 기록 {len(new_rows)}개)")
                    combined_df = new_df
                
                # CSV 파일로 저장 (기존 파일 덮어쓰기)
                combined_df.to_csv(result_file, index=False)
                
                print(f"\n백테스트 결과 저장 완료: {result_file}")
                print(f"  총 {len(combined_df)}개 거래 기록 (이번 실행: {len(new_rows)}개 거래)")
            
        except Exception as e:
            err = traceback.format_exc()
            print("err", err)
            raise
    
    def load_backtest_results(self, ticker='BTC', condition_dict=None):
        """
        저장된 백테스트 결과를 로드합니다.
        
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
            
            if not os.path.exists(result_file):
                print(f"백테스트 결과 파일이 없습니다: {result_file}")
                return None
            
            df = pd.read_csv(result_file)
            
            # execution_time을 datetime으로 변환
            if 'execution_time' in df.columns:
                df['execution_time'] = pd.to_datetime(df['execution_time'])
            
            # 최신순으로 정렬
            df = df.sort_values('execution_time', ascending=False)
            
            print(f"백테스트 결과 로드 완료: {result_file}")
            print(f"  총 {len(df)}개 기록")
            
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

