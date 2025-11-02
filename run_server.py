"""
FastAPI 웹 서버를 통한 QQC 백테스트 실시간 모니터링
"""
import os
import traceback
import threading
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn

from qqc_main import run_server_mode
from shared_state import shared_state
from config import Config
from balance_visualizer import BalanceVisualizer
from condition_manager import ConditionManager

# 블로킹 작업을 처리하기 위한 ThreadPoolExecutor
# CPU 집약적 작업 (그래프 생성 등)을 위한 워커 풀
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="blocking_task")

# 백그라운드 실행 스레드
_background_thread: Optional[threading.Thread] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리 (시작/종료 이벤트)"""
    # Startup
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
    
    yield
    
    # Shutdown
    print("서버 종료 중...")
    shared_state.set_running(False)
    print("백그라운드 작업 중지 요청됨")
    # ThreadPoolExecutor 종료
    _executor.shutdown(wait=False)
    print("ThreadPoolExecutor 종료됨")


app = FastAPI(
    title="QQC 백테스트 모니터링 서버",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정 (외부 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용하도록 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




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
    # 파일 존재 확인을 비동기로 처리
    exists = await asyncio.to_thread(os.path.exists, html_path)
    if exists:
        return FileResponse(html_path, media_type="text/html")
    else:
        return {
            "message": "QQC 백테스트 모니터링 서버",
            "version": "1.0.0",
            "endpoints": {
                "/api/status": "현재 상태 조회",
                "/api/balance": "현재 잔고 조회 (KRW, BTC)",
                "/api/assets_history": "총 자산 변동 이력 조회",
                "/api/trades": "거래 기록 조회",
                "/api/results": "백테스트 결과 조회",
                "/api/logs": "에러 로그 조회",
                "/api/images/{image_type}": "이미지 조회 (today, 3days, 5days)",
                "/api/images/balance_history": "잔고 변동 이력 그래프 이미지",
                "/logs": "로그 페이지"
            }
        }


@app.get("/logs")
async def logs_page():
    """로그 페이지 제공"""
    html_path = os.path.join(os.getcwd(), 'logs.html')
    exists = await asyncio.to_thread(os.path.exists, html_path)
    if exists:
        return FileResponse(html_path, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="로그 페이지를 찾을 수 없습니다.")


@app.get("/api/status")
async def get_status():
    """현재 상태 조회"""
    try:
        # shared_state.get_status()는 lock을 사용하지만 빠르므로 비동기로 처리
        status = await asyncio.to_thread(shared_state.get_status)
        return JSONResponse(content=status)
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise HTTPException(status_code=500, detail=f"상태 조회 실패: {str(e)}")


@app.get("/api/trades")
async def get_trades():
    """거래 기록 조회"""
    try:
        trades = await asyncio.to_thread(shared_state.get_trades)
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
        result = await asyncio.to_thread(shared_state.get_backtest_result)
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


@app.get("/api/logs")
async def get_logs():
    """에러 로그 조회"""
    try:
        logs = await asyncio.to_thread(shared_state.get_error_logs)
        return JSONResponse(content={
            "logs": logs,
            "count": len(logs)
        })
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise HTTPException(status_code=500, detail=f"로그 조회 실패: {str(e)}")


@app.get("/api/balance")
async def get_balance():
    """현재 잔고 조회"""
    try:
        balance = await asyncio.to_thread(shared_state.get_balance)
        if balance is None:
            return JSONResponse(content={
                "error": "잔고 정보가 아직 없습니다.",
                "balance": None
            })
        return JSONResponse(content=balance)
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise HTTPException(status_code=500, detail=f"잔고 조회 실패: {str(e)}")


@app.get("/api/assets_history")
async def get_assets_history():
    """총 자산 변동 이력 조회"""
    try:
        history = await asyncio.to_thread(shared_state.get_total_assets_history)
        return JSONResponse(content={
            "history": history,
            "count": len(history)
        })
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise HTTPException(status_code=500, detail=f"자산 변동 이력 조회 실패: {str(e)}")


@app.get("/api/images/balance_history")
async def get_balance_history_image():
    """잔고 변동 이력 그래프 이미지 조회"""
    try:
        # 현재 조건 로드를 비동기로 처리 (파일 I/O)
        condition = await asyncio.to_thread(ConditionManager.load_condition)

        # 잔고 이력 그래프 생성을 비동기로 처리 (긴 블로킹 작업)
        visualizer = BalanceVisualizer()
        ticker = condition.get('ticker', 'BTC') if condition else 'BTC'
        output_path = './images/balance_history.jpg'
        
        # CPU/IO 집약적 작업을 ThreadPoolExecutor에서 실행
        loop = asyncio.get_event_loop()
        image_path = await loop.run_in_executor(
            _executor,
            visualizer.plot_balance_history,
            ticker,
            condition,
            output_path
        )

        # 파일 존재 확인도 비동기로 처리
        if image_path is None or not await asyncio.to_thread(os.path.exists, image_path):
            raise HTTPException(
                status_code=404,
                detail="잔고 이력 그래프를 생성할 수 없습니다. 데이터가 없거나 생성 중 오류가 발생했습니다."
            )

        return FileResponse(
            image_path,
            media_type="image/jpeg",
            filename="balance_history.jpg"
        )

    except HTTPException:
        raise
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise HTTPException(status_code=500, detail=f"잔고 이력 그래프 조회 실패: {str(e)}")


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
        
        # 공유 상태에서 이미지 경로 조회 (비동기)
        status = await asyncio.to_thread(shared_state.get_status)
        image_path = status.get('image_paths', {}).get(image_type)
        
        # 경로 처리 및 파일 존재 확인을 비동기로 처리
        def resolve_image_path():
            """이미지 경로를 해결하는 동기 함수"""
            resolved_path = image_path
            # 경로가 없거나 파일이 존재하지 않으면 기본 경로에서 찾기
            if resolved_path is None or not os.path.exists(resolved_path):
                # 기본 이미지 디렉토리에서 찾기
                base_dir = os.path.join(os.getcwd(), 'images')
                resolved_path = os.path.join(base_dir, image_files[image_type])
            
            # 상대 경로 처리
            if not os.path.isabs(resolved_path):
                # './' 로 시작하는 경우 제거
                if resolved_path.startswith('./'):
                    resolved_path = resolved_path[2:]
                resolved_path = os.path.join(os.getcwd(), resolved_path)
            
            # 경로 정규화 (중복 경로 구분자 제거)
            resolved_path = os.path.normpath(resolved_path)
            
            # 파일 존재 확인
            if not os.path.exists(resolved_path):
                # HTTPException 대신 None 반환 (비동기 컨텍스트에서 처리)
                return None
            
            return resolved_path
        
        # 경로 해결을 비동기로 실행
        final_image_path = await asyncio.to_thread(resolve_image_path)
        
        # 파일이 없는 경우 에러 처리
        if final_image_path is None:
            # 기본 경로 다시 시도
            base_dir = os.path.join(os.getcwd(), 'images')
            default_path = os.path.join(base_dir, image_files[image_type])
            final_image_path = os.path.normpath(default_path)
            
            # 최종 확인
            if not await asyncio.to_thread(os.path.exists, final_image_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"이미지 파일을 찾을 수 없습니다: {final_image_path}"
                )
        
        return FileResponse(
            final_image_path,
            media_type="image/png",
            filename=os.path.basename(final_image_path)
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
        await asyncio.to_thread(shared_state.set_running, False)
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


@app.post("/api/initial_capital")
async def set_initial_capital(request: dict):
    """초기 자산 설정"""
    try:
        initial_capital = request.get('initial_capital')

        if initial_capital is None:
            raise HTTPException(status_code=400, detail="initial_capital is required")

        if not isinstance(initial_capital, (int, float)) or initial_capital < 0:
            raise HTTPException(status_code=400, detail="initial_capital must be a non-negative number")

        await asyncio.to_thread(shared_state.set_initial_capital, float(initial_capital))

        return JSONResponse(content={
            "message": "초기 자산이 설정되었습니다.",
            "initial_capital": initial_capital
        })
    except HTTPException:
        raise
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise HTTPException(status_code=500, detail=f"초기 자산 설정 실패: {str(e)}")


if __name__ == "__main__":
    # 서버 실행
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    # 워커 수 설정 (환경변수에서 읽거나 기본값 사용)
    # 기본값: CPU 코어 수, 최소 2개, 최대 8개
    import multiprocessing
    workers = int(os.getenv("WORKERS", min(max(multiprocessing.cpu_count(), 2), 8)))
    
    print(f"서버 시작: http://{host}:{port}")
    print(f"워커 수: {workers}")
    print(f"API 문서: http://{host}:{port}/docs")
    print(f"대체 문서: http://{host}:{port}/redoc")
    
    # workers를 사용할 때는 import string을 사용해야 함
    uvicorn.run(
        "run_server:app",  # import string 사용
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )

