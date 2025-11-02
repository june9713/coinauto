"""
FastAPI 웹 서버를 통한 QQC 백테스트 실시간 모니터링
"""
import os
import traceback
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn

from qqc_main import run_server_mode
from shared_state import shared_state
from config import Config


app = FastAPI(title="QQC 백테스트 모니터링 서버", version="1.0.0")

# CORS 설정 (외부 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용하도록 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 백그라운드 실행 스레드
_background_thread: Optional[threading.Thread] = None


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 백그라운드 작업 시작"""
    global _background_thread
    
    print("="*80)
    print("FastAPI 서버 시작")
    print("="*80)
    
    if _background_thread is None or not _background_thread.is_alive():
        # 백그라운드에서 qqc_main 실행
        _background_thread = threading.Thread(
            target=_run_background_task,
            daemon=True,
            name="QQCBackgroundTask"
        )
        _background_thread.start()
        print("백그라운드 백테스트 작업 시작됨")


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 백그라운드 작업 중지"""
    print("서버 종료 중...")
    shared_state.set_running(False)
    print("백그라운드 작업 중지 요청됨")


def _run_background_task():
    """백그라운드에서 실행될 백테스트 작업"""
    try:
        # QQC 전략 변수 설정 (qqc_main.py의 기본값과 동일)
        run_server_mode(
            start_date=None,
            end_date=None,
            initial_capital=None,  # 상태에서 로드
            price_slippage=1000,
            ticker='BTC',
            interval='3m',
            volume_window=55,
            ma_window=9,
            volume_multiplier=1.4,
            buy_cash_ratio=0.9,
            hold_period=15,
            profit_target=17.6,
            stop_loss=-28.6,
            auto_initialize=True  # 서버 모드에서는 자동 초기화
        )
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        shared_state.set_error(err)
        shared_state.set_running(False)


@app.get("/")
async def root():
    """루트 경로 - HTML 대시보드 제공"""
    html_path = os.path.join(os.getcwd(), 'index.html')
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    else:
        return {
            "message": "QQC 백테스트 모니터링 서버",
            "version": "1.0.0",
            "endpoints": {
                "/api/status": "현재 상태 조회",
                "/api/trades": "거래 기록 조회",
                "/api/results": "백테스트 결과 조회",
                "/api/images/{image_type}": "이미지 조회 (today, 3days, 5days)"
            }
        }


@app.get("/api/status")
async def get_status():
    """현재 상태 조회"""
    try:
        status = shared_state.get_status()
        return JSONResponse(content=status)
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise HTTPException(status_code=500, detail=f"상태 조회 실패: {str(e)}")


@app.get("/api/trades")
async def get_trades():
    """거래 기록 조회"""
    try:
        trades = shared_state.get_trades()
        return JSONResponse(content={
            "trades": trades,
            "count": len(trades)
        })
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise HTTPException(status_code=500, detail=f"거래 기록 조회 실패: {str(e)}")


@app.get("/api/results")
async def get_results():
    """백테스트 결과 조회"""
    try:
        result = shared_state.get_backtest_result()
        if result is None:
            return JSONResponse(content={
                "error": "백테스트 결과가 아직 없습니다.",
                "result": None
            })
        return JSONResponse(content=result)
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise HTTPException(status_code=500, detail=f"결과 조회 실패: {str(e)}")


@app.get("/api/images/{image_type}")
async def get_image(image_type: str):
    """
    이미지 조회
    
    Parameters:
    - image_type: 'today', '3days', '5days'
    """
    try:
        # 이미지 타입과 파일명 매핑
        image_files = {
            'today': 'backtest_result_today.jpg',
            '3days': 'backtest_result_3days.jpg',
            '5days': 'backtest_result_5days.jpg'
        }
        
        if image_type not in image_files:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 이미지 타입: {image_type}. 지원 타입: today, 3days, 5days"
            )
        
        # 공유 상태에서 이미지 경로 조회
        status = shared_state.get_status()
        image_path = status.get('image_paths', {}).get(image_type)
        
        # 경로가 없거나 파일이 존재하지 않으면 기본 경로에서 찾기
        if image_path is None or not os.path.exists(image_path):
            # 기본 이미지 디렉토리에서 찾기
            base_dir = os.path.join(os.getcwd(), 'images')
            image_path = os.path.join(base_dir, image_files[image_type])
        
        # 상대 경로 처리
        if not os.path.isabs(image_path):
            # './' 로 시작하는 경우 제거
            if image_path.startswith('./'):
                image_path = image_path[2:]
            image_path = os.path.join(os.getcwd(), image_path)
        
        # 경로 정규화 (중복 경로 구분자 제거)
        image_path = os.path.normpath(image_path)
        
        # 파일 존재 확인
        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=404,
                detail=f"이미지 파일을 찾을 수 없습니다: {image_path}"
            )
        
        return FileResponse(
            image_path,
            media_type="image/png",
            filename=os.path.basename(image_path)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise HTTPException(status_code=500, detail=f"이미지 조회 실패: {str(e)}")


@app.post("/api/control/stop")
async def stop_background_task():
    """백그라운드 작업 중지"""
    try:
        shared_state.set_running(False)
        return JSONResponse(content={"message": "백그라운드 작업 중지 요청됨"})
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise HTTPException(status_code=500, detail=f"작업 중지 실패: {str(e)}")


@app.post("/api/control/start")
async def start_background_task():
    """백그라운드 작업 시작"""
    try:
        global _background_thread
        
        if _background_thread is not None and _background_thread.is_alive():
            return JSONResponse(content={"message": "이미 실행 중입니다."})
        
        shared_state.set_running(True)
        _background_thread = threading.Thread(
            target=_run_background_task,
            daemon=True,
            name="QQCBackgroundTask"
        )
        _background_thread.start()
        
        return JSONResponse(content={"message": "백그라운드 작업 시작됨"})
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise HTTPException(status_code=500, detail=f"작업 시작 실패: {str(e)}")


if __name__ == "__main__":
    # 서버 실행
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"서버 시작: http://{host}:{port}")
    print(f"API 문서: http://{host}:{port}/docs")
    print(f"대체 문서: http://{host}:{port}/redoc")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

