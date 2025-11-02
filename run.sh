#!/bin/bash
# 워커 수는 환경변수로 설정 가능 (기본값은 run_server.py에서 자동 설정)
# 예: WORKERS=4 PYTHON_GIL=0 python run_server.py
PYTHON_GIL=0 python run_server.py
