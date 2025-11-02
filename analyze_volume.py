import os
import pandas as pd
import glob
from pathlib import Path

# datas 폴더에서 history 파일들 찾기
datas_dir = Path("/home/jay/workspace/python/2025/coinauto/datas")
history_files = list(datas_dir.glob("**/history*.csv"))

# 파일명에서 시간 정보 추출하여 최신순 정렬
history_files.sort(key=lambda x: x.name, reverse=True)

print(f"발견된 history 파일 수: {len(history_files)}")
print(f"가장 최신 파일: {history_files[0].name if history_files else '없음'}\n")

# 최신 파일부터 데이터 읽어서 누적
all_data = []
for file_path in history_files:
    try:
        df = pd.read_csv(file_path)
        # 헤더 제외하고 데이터 추가 (시간 역순으로 추가)
        if len(df) > 1:  # 헤더 제외하고 데이터가 있는 경우
            # 시간순으로 정렬 (오래된 것부터)
            df['datetime'] = pd.to_datetime(df.iloc[:, 0])
            df = df.sort_values('datetime')
            # volume 컬럼 추출 (마지막 컬럼이 volume)
            volumes = df['volume'].tolist()
            timestamps = df['datetime'].tolist()
            # 최신 데이터가 뒤에 오도록 유지
            for ts, vol in zip(timestamps, volumes):
                all_data.append((ts, vol))
            
            print(f"{file_path.name}: {len(volumes)}개 데이터 추가 (총 {len(all_data)}개)")
            
            # 55개 모이면 중단
            if len(all_data) >= 55:
                break
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"파일 읽기 오류 {file_path.name}: {err}")

# 최신 55개만 추출
latest_55 = all_data[-55:] if len(all_data) >= 55 else all_data

print(f"\n총 수집된 데이터: {len(all_data)}개")
print(f"최신 55개 데이터:\n")
print(f"{'번호':<6} {'시간':<25} {'거래량':<15}")
print("-" * 50)

volumes_only = []
for idx, (timestamp, volume) in enumerate(latest_55, 1):
    print(f"{idx:<6} {str(timestamp):<25} {volume:<15}")
    volumes_only.append(volume)

# 평균 계산
if volumes_only:
    avg_volume = sum(volumes_only) / len(volumes_only)
    print("-" * 50)
    print(f"\n거래량 평균: {avg_volume:.8f}")
    print(f"평균 거래량 (과학적 표기): {avg_volume:.2e}")
else:
    print("\n거래량 데이터가 없습니다.")

