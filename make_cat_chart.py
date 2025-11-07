import os
import glob
import sys
import traceback
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

def create_chart_for_tick(df, tick_index, output_base_dir):
    """
    특정 틱을 중심으로 이전 50개 + 현재 + 이후 50개 (총 101개) 캔들 차트를 생성합니다.

    Args:
        df (pd.DataFrame): 전체 데이터프레임
        tick_index (int): 현재 틱의 인덱스 (0부터 시작)
        output_base_dir (str): 이미지를 저장할 기본 디렉토리 (./cats)

    Returns:
        bool: 성공 여부
    """
    # 101개 캔들을 생성하기 위해 필요한 범위 확인
    # 이전 50개 + 현재 + 이후 50개 = 총 101개
    start_idx = tick_index - 50
    end_idx = tick_index + 50 + 1  # +1은 슬라이싱을 위해

    # 범위 체크
    if start_idx < 0 or end_idx > len(df):
        return False

    # 데이터 추출
    chart_data = df.iloc[start_idx:end_idx].copy()

    # cat01 값 확인
    if 'cat01' not in chart_data.columns:
        return False

    # cat01 값이 모두 유효한지 확인
    if chart_data['cat01'].isna().any():
        return False

    # 현재 틱의 날짜 정보
    current_tick = df.iloc[tick_index]
    current_date = current_tick.name  # 인덱스가 날짜

    # 파일명 생성: dd_{cat01_51}_{cat01_101}.png
    # dd: 현재 캔들의 날짜 (YYYYMMDD_HHMMSS 형식)
    # cat01_51: 51번째 캔들의 cat01 값 (현재 틱, 0-based index로 50)
    # cat01_101: 101번째 캔들의 cat01 값 (마지막 틱, 0-based index로 100)
    if isinstance(current_date, pd.Timestamp):
        dd = current_date.strftime('%Y%m%d_%H%M%S')
    else:
        dd = str(current_date).replace('-', '').replace(':', '').replace(' ', '_')

    cat01_51 = int(chart_data.iloc[50]['cat01'])  # 51번째 캔들 (현재 틱, 0-based index로 50)
    cat01_101 = int(chart_data.iloc[100]['cat01'])  # 101번째 캔들 (마지막 틱, 0-based index로 100)

    # 저장 폴더: ./cats/{cat01_51}/
    output_dir = os.path.join(output_base_dir, str(cat01_51))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{dd}_{cat01_51}_{cat01_101}.png"
    output_path = os.path.join(output_dir, filename)

    # 이미 파일이 존재하면 건너뛰기
    if os.path.exists(output_path):
        return True

    # 차트 생성
    try:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)

        # 1. 캔들스틱 차트
        ax1 = fig.add_subplot(gs[0])

        # 캔들 그리기
        for i in range(len(chart_data)):
            row = chart_data.iloc[i]
            x = i
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']

            # 캔들 색상 (상승: 빨강, 하락: 파랑)
            color = 'red' if close_price >= open_price else 'blue'

            # High-Low 선
            ax1.plot([x, x], [low_price, high_price], color=color, linewidth=0.5)

            # 캔들 몸통
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            ax1.bar(x, body_height, bottom=body_bottom, color=color, width=0.8, alpha=0.7)

        # 51번째 캔들 (현재 틱) 강조 (0-based index로 50)
        ax1.axvline(x=50, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Current Tick')

        # 이동평균선 그리기 (색상과 굵기 개선)
        if 'ma5' in chart_data.columns and chart_data['ma5'].notna().any():
            ax1.plot(range(len(chart_data)), chart_data['ma5'],
                    label='MA5', linewidth=1.5, color='orange', alpha=0.8)
        if 'ma7' in chart_data.columns and chart_data['ma7'].notna().any():
            ax1.plot(range(len(chart_data)), chart_data['ma7'],
                    label='MA7', linewidth=1.5, color='cyan', alpha=0.8)
        if 'ma10' in chart_data.columns and chart_data['ma10'].notna().any():
            ax1.plot(range(len(chart_data)), chart_data['ma10'],
                    label='MA10', linewidth=1.5, color='magenta', alpha=0.8)

        ax1.set_ylabel('Price', fontsize=10)
        ax1.set_title(f'Price Chart - {current_date.strftime("%Y-%m-%d %H:%M:%S") if isinstance(current_date, pd.Timestamp) else current_date}',
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. 거래량 차트
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        colors = ['red' if chart_data.iloc[i]['close'] >= chart_data.iloc[i]['open'] else 'blue'
                  for i in range(len(chart_data))]
        ax2.bar(range(len(chart_data)), chart_data['volume'], color=colors, width=0.8, alpha=0.7)
        ax2.axvline(x=50, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax2.set_ylabel('Volume', fontsize=10)
        ax2.set_title('Volume', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 3. cat01 그래프
        ax3 = fig.add_subplot(gs[2], sharex=ax1)

        ax3.plot(range(len(chart_data)), chart_data['cat01'], marker='o', markersize=3,
                linewidth=1, color='purple', alpha=0.7)
        ax3.axvline(x=50, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Candle Index', fontsize=10)
        ax3.set_ylabel('Category', fontsize=10)
        ax3.set_title('Pattern Category (cat01)', fontsize=10)
        ax3.grid(True, alpha=0.3)

        # X축 라벨 설정
        ax3.set_xlim(-1, len(chart_data))

        # 저장
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        return True

    except Exception as e:
        print(f"  차트 생성 실패 ({filename}): {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        plt.close('all')
        return False


def load_csv_files(csv_files, start_idx, end_idx):
    """
    여러 CSV 파일을 로드하여 연속된 하나의 DataFrame으로 병합합니다.

    Args:
        csv_files (list): 모든 CSV 파일 경로 리스트 (시간순 정렬됨)
        start_idx (int): 로드할 시작 파일 인덱스
        end_idx (int): 로드할 종료 파일 인덱스 (포함)

    Returns:
        pd.DataFrame or None: 병합된 DataFrame (실패 시 None)
    """
    dfs = []

    for i in range(start_idx, end_idx + 1):
        if i < 0 or i >= len(csv_files):
            continue

        try:
            df = pd.read_csv(csv_files[i], index_col=0, parse_dates=True)
            if 'cat01' not in df.columns:
                continue
            dfs.append(df)
        except Exception as e:
            print(f"    경고: {os.path.basename(csv_files[i])} 로드 실패: {e}")
            continue

    if not dfs:
        return None

    # 병합 및 정렬
    merged_df = pd.concat(dfs, axis=0)
    merged_df = merged_df.sort_index()
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

    return merged_df


def filter_csv_files_by_date(csv_files, start_date_str=None, end_date_str=None):
    """
    CSV 파일 경로에서 날짜를 추출하여 지정된 날짜 범위 내의 파일만 필터링합니다.
    
    Args:
        csv_files (list): CSV 파일 경로 리스트
        start_date_str (str, optional): 시작 날짜 (YYYY-MM-DD 형식)
        end_date_str (str, optional): 종료 날짜 (YYYY-MM-DD 형식)
    
    Returns:
        list: 필터링된 CSV 파일 경로 리스트
    """
    if start_date_str is None and end_date_str is None:
        return csv_files
    
    # 날짜 파싱
    start_date = None
    end_date = None
    
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"시작 날짜 형식 오류: {start_date_str} (올바른 형식: YYYY-MM-DD)")
    
    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            # 종료 날짜는 해당 날짜의 23:59:59까지 포함
            end_date = end_date.replace(hour=23, minute=59, second=59)
        except ValueError:
            raise ValueError(f"종료 날짜 형식 오류: {end_date_str} (올바른 형식: YYYY-MM-DD)")
    
    # 날짜 범위 검증
    if start_date and end_date and start_date > end_date:
        raise ValueError(f"시작 날짜({start_date_str})가 종료 날짜({end_date_str})보다 늦습니다.")
    
    # CSV 파일 경로에서 날짜 추출 및 필터링
    filtered_files = []
    
    for csv_path in csv_files:
        # 경로에서 날짜 추출: ./dumps2/BTC/3m/2025-10-05/file.csv
        # 또는 dumps2\BTC\3m\2025-10-05\file.csv
        path_parts = csv_path.replace('\\', '/').split('/')
        
        # 'dumps2' 다음에 오는 부분 찾기
        try:
            dumps2_idx = path_parts.index('dumps2')
            if dumps2_idx + 3 < len(path_parts):
                # dumps2/BTC/3m/YYYY-MM-DD/ 형식
                date_str = path_parts[dumps2_idx + 3]
                
                # 날짜 형식 검증 및 파싱
                try:
                    file_date = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    # 날짜 범위 체크
                    if start_date and file_date < start_date:
                        continue
                    if end_date and file_date > end_date:
                        continue
                    
                    filtered_files.append(csv_path)
                except ValueError:
                    # 날짜 형식이 아닌 폴더명은 건너뜀
                    continue
        except ValueError:
            # 'dumps2' 폴더를 찾을 수 없는 경우 건너뜀
            continue
    
    return filtered_files


def main():
    """메인 함수"""
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(
        description='Pattern Category Chart Generator - 시작날짜와 종료날짜를 지정하여 차트를 생성합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='예시:\n  python make_cat_chart.py\n  python make_cat_chart.py --start-date 2025-10-05\n  python make_cat_chart.py --start-date 2025-10-05 --end-date 2025-10-10'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='시작 날짜 (YYYY-MM-DD 형식, 예: 2025-10-05)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='종료 날짜 (YYYY-MM-DD 형식, 예: 2025-10-10)'
    )
    
    args = parser.parse_args()
    
    # 설정
    DUMPS2_DIR = './dumps2'
    OUTPUT_DIR = './cats'
    TICKER = 'BTC'
    INTERVAL = '3m'
    CONTEXT_FILES = 3  # 이전/이후로 함께 로드할 파일 개수

    print("="*70)
    print("              Pattern Category Chart Generator")
    print("="*70)
    print(f"입력 디렉토리: {DUMPS2_DIR}/{TICKER}/{INTERVAL}")
    print(f"출력 디렉토리: {OUTPUT_DIR}/{{cat01_value}}/")
    print(f"컨텍스트 파일 수: 이전 {CONTEXT_FILES}개 + 현재 + 이후 {CONTEXT_FILES}개")
    
    # 날짜 범위 표시
    if args.start_date or args.end_date:
        date_range = []
        if args.start_date:
            date_range.append(f"시작: {args.start_date}")
        if args.end_date:
            date_range.append(f"종료: {args.end_date}")
        print(f"날짜 범위: {' ~ '.join(date_range)}")
    else:
        print("날짜 범위: 전체")
    
    print("="*70)

    # dumps2 폴더에서 CSV 파일 찾기
    search_pattern = os.path.join(DUMPS2_DIR, TICKER, INTERVAL, '**', '*.csv')
    csv_files = sorted(glob.glob(search_pattern, recursive=True))

    if not csv_files:
        print(f"오류: {DUMPS2_DIR}/{TICKER}/{INTERVAL}/ 에서 CSV 파일을 찾을 수 없습니다.")
        sys.exit(1)

    # 날짜 필터링 적용
    try:
        csv_files = filter_csv_files_by_date(csv_files, args.start_date, args.end_date)
    except ValueError as e:
        print(f"오류: {e}")
        sys.exit(1)
    except Exception as e:
        err = traceback.format_exc()
        print(f"날짜 필터링 오류: {err}")
        raise

    if not csv_files:
        print(f"오류: 지정된 날짜 범위에 해당하는 CSV 파일을 찾을 수 없습니다.")
        sys.exit(1)

    print(f"\n총 {len(csv_files)}개의 CSV 파일을 찾았습니다.\n")

    # 각 CSV 파일 처리 (이전/이후 파일들과 함께)
    total_charts = 0

    for i, csv_path in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] 처리 중: {os.path.basename(csv_path)}")

        # 현재 파일 처리 (이전 N개 + 현재 + 이후 N개 파일과 함께)
        chart_count = process_single_file_with_context(
            csv_files, i - 1, OUTPUT_DIR, CONTEXT_FILES
        )

        if chart_count > 0:
            total_charts += chart_count

        # 진행률 표시
        if i % 10 == 0 or i == len(csv_files):
            print(f"\n진행률: {i}/{len(csv_files)} ({i*100//len(csv_files)}%) - "
                  f"생성된 차트: {total_charts}개\n")

    # 결과 출력
    print("\n" + "="*70)
    print("                         처리 완료")
    print("="*70)
    print(f"총 CSV 파일: {len(csv_files)}개")
    print(f"생성된 차트: {total_charts}개")
    print(f"출력 디렉토리: {OUTPUT_DIR}/{{cat01_value}}/")
    print("="*70)


def process_single_file_with_context(csv_files, file_idx, output_base_dir, context_files):
    """
    단일 파일을 이전/이후 파일들과 함께 처리합니다.

    Args:
        csv_files (list): 모든 CSV 파일 경로 리스트
        file_idx (int): 처리할 파일의 인덱스
        output_base_dir (str): 출력 기본 디렉토리
        context_files (int): 이전/이후로 함께 로드할 파일 개수

    Returns:
        int: 생성된 차트 수
    """
    csv_path = csv_files[file_idx]

    # 현재 파일과 이전 context_files개, 이후 context_files개 파일을 함께 로드
    start_idx = file_idx - context_files
    end_idx = file_idx + context_files

    # 병합된 DataFrame 로드
    merged_df = load_csv_files(csv_files, start_idx, end_idx)

    if merged_df is None or len(merged_df) < 101:
        print(f"  건너뜀: {os.path.basename(csv_path)} (병합된 데이터 부족: {len(merged_df) if merged_df is not None else 0}틱)")
        return 0

    # 현재 파일의 데이터만 추출
    try:
        current_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"  오류: {os.path.basename(csv_path)} 로드 실패: {e}")
        return 0

    if 'cat01' not in current_df.columns:
        print(f"  건너뜀: {os.path.basename(csv_path)} (cat01 컬럼 없음)")
        return 0

    # 현재 파일의 각 틱에 대해 차트 생성
    chart_count = 0

    for tick_date in current_df.index:
        # 병합된 DataFrame에서 현재 틱의 위치 찾기
        try:
            tick_position = merged_df.index.get_loc(tick_date)

            # 이전 50개 + 현재 + 이후 50개 = 총 101개 확인
            if tick_position - 50 < 0 or tick_position + 50 >= len(merged_df):
                continue

            # 차트 생성 (output_base_dir만 전달, 함수 내부에서 cat01 값으로 폴더 생성)
            if create_chart_for_tick(merged_df, tick_position, output_base_dir):
                chart_count += 1

        except KeyError:
            # 현재 틱이 병합된 DataFrame에 없는 경우
            continue
        except Exception as e:
            # 에러는 조용히 무시 (너무 많은 로그 방지)
            continue

    if chart_count > 0:
        print(f"  ✓ {chart_count}개 차트 생성")

    return chart_count


if __name__ == '__main__':
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        traceback.print_exc()
        sys.exit(1)
