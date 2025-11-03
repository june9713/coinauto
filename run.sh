#!/bin/bash
# 워커 수는 환경변수로 설정 가능 (기본값은 run_server.py에서 자동 설정)
# 예: WORKERS=4 PYTHON_GIL=0 python run_server.py

# nohup.out 파일 크기 관리 (1MB = 1048576 bytes)
MAX_SIZE=1048576
NOHUP_FILE="nohup.out"

if [ -f "$NOHUP_FILE" ]; then
    FILE_SIZE=$(stat -f%z "$NOHUP_FILE" 2>/dev/null || stat -c%s "$NOHUP_FILE" 2>/dev/null)
    if [ $? -eq 0 ] && [ "$FILE_SIZE" -gt "$MAX_SIZE" ]; then
        echo "nohup.out 파일 크기가 1MB를 초과했습니다. 오래된 로그를 삭제합니다..."
        # 파일의 마지막 50%만 유지 (약 512KB)
        KEEP_SIZE=$((MAX_SIZE / 2))
        # tail을 사용하여 파일의 마지막 부분만 유지
        tail -c $KEEP_SIZE "$NOHUP_FILE" > "${NOHUP_FILE}.tmp" && mv "${NOHUP_FILE}.tmp" "$NOHUP_FILE"
        echo "nohup.out 파일 크기를 줄였습니다."
    fi
fi

PYTHON_GIL=0 python run_server.py
