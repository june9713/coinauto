#!/usr/bin/env python3
"""
CSV 파일의 중복 거래 기록 제거 스크립트

사용법:
    python remove_duplicate_trades.py

기능:
    1. datas 폴더 내의 모든 backtest_results CSV 파일 검색
    2. 각 파일에서 중복 거래 제거 (trade_date + action 기준)
    3. 원본 파일 백업 (.backup 확장자로 저장)
    4. 중복 제거된 데이터로 덮어쓰기
"""

import os
import pandas as pd
import glob
from datetime import datetime


def remove_duplicates_from_csv(csv_path):
    """
    CSV 파일에서 중복 거래 제거

    Parameters:
        csv_path (str): CSV 파일 경로

    Returns:
        dict: 처리 결과 정보
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_path)

        if df is None or len(df) == 0:
            return {
                'success': False,
                'message': 'Empty file',
                'original_count': 0,
                'unique_count': 0,
                'duplicates_removed': 0
            }

        original_count = len(df)

        # trade_date가 있는 행만 처리 (거래 기록이 있는 행)
        if 'trade_date' not in df.columns:
            return {
                'success': False,
                'message': 'No trade_date column',
                'original_count': original_count,
                'unique_count': original_count,
                'duplicates_removed': 0
            }

        # trade_date가 있는 행만 추출
        trades_df = df[df['trade_date'].notna()].copy()
        non_trades_df = df[df['trade_date'].isna()].copy()

        if len(trades_df) == 0:
            return {
                'success': True,
                'message': 'No trades to deduplicate',
                'original_count': original_count,
                'unique_count': original_count,
                'duplicates_removed': 0
            }

        # 중복 제거: trade_date + action 기준으로 첫 번째 행만 유지
        trades_before = len(trades_df)

        # trade_date를 datetime으로 변환 (비교를 위해)
        trades_df['trade_date'] = pd.to_datetime(trades_df['trade_date'])

        # action 컬럼이 없으면 빈 문자열로 대체
        if 'action' not in trades_df.columns:
            trades_df['action'] = ''

        # (trade_date, action) 조합으로 중복 제거 (첫 번째 행 유지)
        trades_df_unique = trades_df.drop_duplicates(subset=['trade_date', 'action'], keep='first')

        trades_after = len(trades_df_unique)
        duplicates_removed = trades_before - trades_after

        # 거래 데이터와 비거래 데이터 병합
        result_df = pd.concat([non_trades_df, trades_df_unique], ignore_index=True)

        # date 컬럼으로 정렬 (있는 경우)
        if 'date' in result_df.columns:
            result_df['date'] = pd.to_datetime(result_df['date'])
            result_df = result_df.sort_values('date').reset_index(drop=True)

        # 원본 파일 백업
        backup_path = csv_path + '.backup'
        os.rename(csv_path, backup_path)

        # 중복 제거된 데이터로 저장
        result_df.to_csv(csv_path, index=False)

        return {
            'success': True,
            'message': 'Success',
            'original_count': original_count,
            'unique_count': len(result_df),
            'duplicates_removed': duplicates_removed,
            'backup_path': backup_path
        }

    except Exception as e:
        return {
            'success': False,
            'message': f'Error: {str(e)}',
            'original_count': 0,
            'unique_count': 0,
            'duplicates_removed': 0
        }


def main():
    """메인 함수"""
    print("=" * 80)
    print("CSV 파일 중복 거래 제거 스크립트")
    print("=" * 80)

    # datas 폴더 내의 모든 backtest_results CSV 파일 검색
    csv_pattern = './datas/**/backtest_results*.csv'
    csv_files = glob.glob(csv_pattern, recursive=True)

    if len(csv_files) == 0:
        print("\n검색된 CSV 파일이 없습니다.")
        print(f"검색 패턴: {csv_pattern}")
        return

    print(f"\n검색된 CSV 파일: {len(csv_files)}개")
    print()

    # 사용자 확인
    print("다음 파일들의 중복 거래를 제거합니다:")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"  {i}. {csv_file}")

    print()
    response = input("계속하시겠습니까? (y/n): ").strip().lower()

    if response != 'y':
        print("취소되었습니다.")
        return

    print()
    print("=" * 80)
    print("처리 시작")
    print("=" * 80)

    total_duplicates = 0
    success_count = 0
    fail_count = 0

    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] 처리 중: {csv_file}")

        result = remove_duplicates_from_csv(csv_file)

        if result['success']:
            success_count += 1
            total_duplicates += result['duplicates_removed']

            print(f"  ✓ 성공")
            print(f"    - 원본 행 수: {result['original_count']}")
            print(f"    - 중복 제거 후: {result['unique_count']}")
            print(f"    - 제거된 중복: {result['duplicates_removed']}개")
            if result['duplicates_removed'] > 0:
                print(f"    - 백업 파일: {result.get('backup_path', 'N/A')}")
        else:
            fail_count += 1
            print(f"  ✗ 실패: {result['message']}")

    print()
    print("=" * 80)
    print("처리 완료")
    print("=" * 80)
    print(f"총 파일 수: {len(csv_files)}개")
    print(f"성공: {success_count}개")
    print(f"실패: {fail_count}개")
    print(f"총 제거된 중복 거래: {total_duplicates}개")
    print()

    if success_count > 0:
        print("주의: 원본 파일은 .backup 확장자로 백업되었습니다.")
        print("      문제가 없다면 백업 파일을 삭제할 수 있습니다.")
        print()
        print("백업 파일 삭제 명령:")
        print("  find ./datas -name '*.backup' -delete")


if __name__ == "__main__":
    main()
