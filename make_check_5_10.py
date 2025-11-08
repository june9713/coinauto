"""
기존 dumps 폴더의 CSV 파일들에 check_5_10, PON, diff_s2e, diff_h2l 컬럼을 추가하는 스크립트

- check_5_10: 캔들의 ma5 값이 ma10보다 크면 1.0, 그렇지 않으면 0.0
- PON: 캔들의 종가가 시가보다 크면 1.0, 그렇지 않으면 0.0
- diff_s2e: 종가 - 시가의 값
- diff_h2l: high - low의 값

사용법:
    python make_check_5_10.py
"""
import os
import glob
import pandas as pd
from datetime import datetime


def add_check_5_10_column(df):
    """
    데이터프레임에 check_5_10 컬럼을 추가합니다.

    Parameters:
    - df (pd.DataFrame): OHLCV + MA 데이터프레임

    Returns:
    - pd.DataFrame: check_5_10 컬럼이 추가된 데이터프레임
    """
    try:
        if df is None or len(df) == 0:
            return df

        # ma5와 ma10 컬럼이 있는지 확인
        if 'ma5' not in df.columns or 'ma10' not in df.columns:
            print("  경고: 'ma5' 또는 'ma10' 컬럼이 없어 check_5_10을 계산할 수 없습니다.")
            return df

        # 데이터프레임 복사
        df_copy = df.copy()

        # check_5_10 컬럼 추가: ma5 > ma10이면 1.0, 아니면 0.0
        df_copy['check_5_10'] = (df_copy['ma5'] > df_copy['ma10']).astype(float)

        return df_copy

    except Exception as e:
        print(f"  check_5_10 계산 오류: {e}")
        return df


def add_pon_column(df):
    """
    데이터프레임에 PON 컬럼을 추가합니다.
    종가가 시가보다 크면 1.0, 그렇지 않으면 0.0으로 설정합니다.

    Parameters:
    - df (pd.DataFrame): OHLCV 데이터프레임

    Returns:
    - pd.DataFrame: PON 컬럼이 추가된 데이터프레임
    """
    try:
        if df is None or len(df) == 0:
            return df

        # open과 close 컬럼이 있는지 확인
        if 'open' not in df.columns or 'close' not in df.columns:
            print("  경고: 'open' 또는 'close' 컬럼이 없어 PON을 계산할 수 없습니다.")
            return df

        # 데이터프레임 복사
        df_copy = df.copy()

        # PON 컬럼 추가: close > open이면 1.0, 아니면 0.0
        df_copy['PON'] = (df_copy['close'] > df_copy['open']).astype(float)

        return df_copy

    except Exception as e:
        print(f"  PON 계산 오류: {e}")
        return df


def add_diff_s2e_column(df):
    """
    데이터프레임에 diff_s2e 컬럼을 추가합니다.
    종가 - 시가의 값을 저장합니다.

    Parameters:
    - df (pd.DataFrame): OHLCV 데이터프레임

    Returns:
    - pd.DataFrame: diff_s2e 컬럼이 추가된 데이터프레임
    """
    try:
        if df is None or len(df) == 0:
            return df

        # open과 close 컬럼이 있는지 확인
        if 'open' not in df.columns or 'close' not in df.columns:
            print("  경고: 'open' 또는 'close' 컬럼이 없어 diff_s2e를 계산할 수 없습니다.")
            return df

        # 데이터프레임 복사
        df_copy = df.copy()

        # diff_s2e 컬럼 추가: close - open
        df_copy['diff_s2e'] = (df_copy['close'] - df_copy['open']).astype(float)

        return df_copy

    except Exception as e:
        print(f"  diff_s2e 계산 오류: {e}")
        return df


def add_diff_h2l_column(df):
    """
    데이터프레임에 diff_h2l 컬럼을 추가합니다.
    high - low의 값을 저장합니다.

    Parameters:
    - df (pd.DataFrame): OHLCV 데이터프레임

    Returns:
    - pd.DataFrame: diff_h2l 컬럼이 추가된 데이터프레임
    """
    try:
        if df is None or len(df) == 0:
            return df

        # high와 low 컬럼이 있는지 확인
        if 'high' not in df.columns or 'low' not in df.columns:
            print("  경고: 'high' 또는 'low' 컬럼이 없어 diff_h2l을 계산할 수 없습니다.")
            return df

        # 데이터프레임 복사
        df_copy = df.copy()

        # diff_h2l 컬럼 추가: high - low
        df_copy['diff_h2l'] = (df_copy['high'] - df_copy['low']).astype(float)

        return df_copy

    except Exception as e:
        print(f"  diff_h2l 계산 오류: {e}")
        return df


def process_csv_files(base_dir='./dumps'):
    """
    dumps 폴더의 모든 CSV 파일을 읽어서 check_5_10, PON, diff_s2e, diff_h2l 컬럼을 추가합니다.

    Parameters:
    - base_dir (str): 기본 디렉토리. 기본값 './dumps'
    """
    try:
        print("="*80)
        print("CSV 파일에 check_5_10, PON, diff_s2e, diff_h2l 컬럼 추가 작업 시작")
        print("="*80)
        print(f"기본 디렉토리: {base_dir}\n")

        # 모든 CSV 파일 찾기 (재귀적으로)
        search_pattern = os.path.join(base_dir, '**', '*.csv')
        csv_files = glob.glob(search_pattern, recursive=True)

        if not csv_files:
            print(f"오류: {base_dir} 폴더에서 CSV 파일을 찾을 수 없습니다.")
            return

        print(f"총 {len(csv_files)}개의 CSV 파일을 찾았습니다.\n")

        processed_count = 0
        skipped_count = 0
        error_count = 0

        for idx, csv_file in enumerate(csv_files, 1):
            try:
                # 진행 상황 표시
                if idx % 100 == 0 or idx == 1 or idx == len(csv_files):
                    print(f"[{idx}/{len(csv_files)}] 처리 중: {csv_file}")

                # CSV 파일 읽기
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

                # 인덱스를 datetime으로 변환
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                # 이미 check_5_10, PON, diff_s2e, diff_h2l 컬럼이 있는지 확인하고 제거
                columns_to_drop = []
                if 'check_5_10' in df.columns:
                    columns_to_drop.append('check_5_10')
                if 'PON' in df.columns:
                    columns_to_drop.append('PON')
                if 'diff_s2e' in df.columns:
                    columns_to_drop.append('diff_s2e')
                if 'diff_h2l' in df.columns:
                    columns_to_drop.append('diff_h2l')

                if columns_to_drop:
                    df = df.drop(columns=columns_to_drop)

                # check_5_10 컬럼 추가
                df_updated = add_check_5_10_column(df)

                # PON 컬럼 추가
                df_updated = add_pon_column(df_updated)

                # diff_s2e 컬럼 추가
                df_updated = add_diff_s2e_column(df_updated)

                # diff_h2l 컬럼 추가
                df_updated = add_diff_h2l_column(df_updated)

                # ma5 또는 ma10이 없어서 check_5_10이 추가되지 않은 경우
                if 'check_5_10' not in df_updated.columns:
                    skipped_count += 1
                    continue

                # 파일 저장 (덮어쓰기)
                df_updated.to_csv(csv_file, index=True)
                processed_count += 1

            except Exception as e:
                error_count += 1
                print(f"  오류 ({csv_file}): {e}")
                continue

        print("\n" + "="*80)
        print("작업 완료!")
        print("="*80)
        print(f"처리된 파일: {processed_count}개")
        print(f"건너뛴 파일: {skipped_count}개 (ma5 또는 ma10 컬럼 없음)")
        print(f"오류 발생: {error_count}개")
        print("="*80)

    except Exception as e:
        print(f"오류: {e}")
        raise


def main():
    """
    메인 함수
    """
    try:
        process_csv_files(base_dir='./dumps')

    except Exception as e:
        print(f"오류: {e}")
        raise


if __name__ == '__main__':
    main()
