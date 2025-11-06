"""
데이터 로드 및 전처리
"""
import os
import glob
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_training_data(base_dir, ticker, interval, start_date_str, end_date_str, features):
    """
    dumps2에서 훈련 데이터 로드
    
    Args:
        base_dir: 데이터 기본 디렉토리
        ticker: 코인 티커 (예: 'BTC')
        interval: 시간 간격 (예: '3m')
        start_date_str: 시작 날짜 (YYYY-MM-DD)
        end_date_str: 종료 날짜 (YYYY-MM-DD)
        features: 사용할 피처 리스트
    
    Returns:
        pd.DataFrame: 로드된 데이터
    """
    print(f"데이터 로드 중... ({start_date_str} ~ {end_date_str})")
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    
    all_dfs = []
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        search_path = os.path.join(base_dir, ticker, interval, date_str, '*.csv')
        csv_files = sorted(glob.glob(search_path))
        
        for f in csv_files:
            try:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                all_dfs.append(df)
            except Exception as e:
                print(f"경고: {f} 파일 읽기 오류: {e}")
        
        current_date += timedelta(days=1)
    
    if not all_dfs:
        raise ValueError("로드된 데이터가 없습니다.")
    
    # 모든 DataFrame 병합
    full_df = pd.concat(all_dfs)
    full_df = full_df.sort_index()
    full_df = full_df[~full_df.index.duplicated(keep='first')]
    
    # 필요한 컬럼만 선택
    try:
        selected_df = full_df[features].copy()
        selected_df = selected_df.dropna()
        print(f"데이터 로드 완료. 총 {len(selected_df)}개의 틱(row) 확보.")
        return selected_df
    except KeyError as e:
        print(f"오류: 요청된 피처(컬럼) {e}를 찾을 수 없습니다.")
        print(f"사용 가능한 컬럼: {full_df.columns.tolist()}")
        raise


def split_train_val(data_df, validation_split=0.2, shuffle=False):
    """
    훈련/검증 데이터 분할
    
    Args:
        data_df: 전체 데이터프레임
        validation_split: 검증 데이터 비율
        shuffle: 셔플 여부 (시계열이므로 False 권장)
    
    Returns:
        train_df, val_df: 훈련/검증 데이터프레임
    """
    if shuffle:
        train_df, val_df = train_test_split(
            data_df, 
            test_size=validation_split, 
            shuffle=True
        )
    else:
        # 시계열 데이터는 시간순으로 분할
        split_idx = int(len(data_df) * (1 - validation_split))
        train_df = data_df.iloc[:split_idx].copy()
        val_df = data_df.iloc[split_idx:].copy()
    
    print(f"훈련 데이터: {len(train_df)}개, 검증 데이터: {len(val_df)}개")
    return train_df, val_df


def normalize_features(train_df, val_df, features):
    """
    피처 정규화 (Z-score)
    
    Args:
        train_df: 훈련 데이터프레임
        val_df: 검증 데이터프레임
        features: 정규화할 피처 리스트
    
    Returns:
        scaler: 스케일러 객체
        train_normalized: 정규화된 훈련 데이터
        val_normalized: 정규화된 검증 데이터
        train_original_prices: 원본 가격 데이터 (훈련)
        val_original_prices: 원본 가격 데이터 (검증)
    """
    scaler = StandardScaler()
    
    # 원본 가격 저장 (정규화 전)
    price_cols = ['open', 'high', 'low', 'close']
    train_original_prices = train_df[price_cols].copy() if all(col in train_df.columns for col in price_cols) else None
    val_original_prices = val_df[price_cols].copy() if all(col in val_df.columns for col in price_cols) else None
    
    # 훈련 데이터로 fit
    train_normalized = train_df.copy()
    train_normalized[features] = scaler.fit_transform(train_df[features])
    
    # 검증 데이터는 훈련 데이터의 스케일로 transform
    val_normalized = val_df.copy()
    val_normalized[features] = scaler.transform(val_df[features])
    
    print("피처 정규화 완료.")
    return scaler, train_normalized, val_normalized, train_original_prices, val_original_prices

