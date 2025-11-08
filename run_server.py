"""
FastAPI 웹 서버를 통한 QQC 백테스트 실시간 모니터링
"""
import os
import sys
import signal
import traceback
import threading
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import pandas as pd

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
        # 백그라운드 작업이 시작될 것임을 상태에 즉시 반영
        shared_state.set_running(True)

        # 백그라운드에서 qqc_main 실행
        _background_thread = threading.Thread(
            target=_run_background_task,
            daemon=True,  # 메인 프로세스 종료 시 자동 종료
            name="QQCBackgroundTask"
        )
        _background_thread.start()
        print("백그라운드 백테스트 작업 시작됨 (daemon thread)")

    yield

    # Shutdown
    print("\n" + "="*80)
    print("서버 종료 시작")
    print("="*80)

    # 백그라운드 작업 중지 요청
    shared_state.set_running(False)
    print("백그라운드 작업 중지 요청됨")

    # 백그라운드 스레드가 종료될 때까지 최대 2초 대기
    if _background_thread is not None and _background_thread.is_alive():
        print("백그라운드 스레드 종료 대기 중... (최대 2초)")
        _background_thread.join(timeout=2.0)
        if _background_thread.is_alive():
            print("경고: 백그라운드 스레드가 2초 내에 종료되지 않았습니다.")
            print("      daemon=True 설정으로 인해 메인 프로세스 종료 시 자동 종료됩니다.")
        else:
            print("백그라운드 스레드 정상 종료됨")

    # ThreadPoolExecutor 종료 (실행 중인 작업 완료 대기)
    print("ThreadPoolExecutor 종료 중...")
    _executor.shutdown(wait=False, cancel_futures=True)
    print("ThreadPoolExecutor 종료 완료")

    print("서버 종료 완료")
    print("="*80)


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
            profit_target=0.3,
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
        print("[DEBUG] /api/status 엔드포인트 호출됨")
        status = await asyncio.to_thread(shared_state.get_status)
        print(f"[DEBUG] status 데이터 가져오기 완료. 키: {list(status.keys())}")

        # trade_state가 있는 경우 타입 검증
        if 'trade_state' in status and status['trade_state']:
            print(f"[DEBUG] trade_state 존재. 타입: {type(status['trade_state'])}")
            for key, value in status['trade_state'].items():
                value_type = type(value).__name__
                if isinstance(value, (pd.Timestamp, datetime)):
                    print(f"  [ERROR] trade_state['{key}'] = {value} (type: {value_type}) - JSON 직렬화 불가!")
                else:
                    print(f"  [OK] trade_state['{key}'] = {value} (type: {value_type})")

        print("[DEBUG] JSONResponse 생성 시도...")
        response = JSONResponse(content=status)
        print("[DEBUG] JSONResponse 생성 성공")
        return response
    except Exception as e:
        err = traceback.format_exc()
        print(f"[ERROR] /api/status 실패:\n{err}")

        # 상세한 오류 정보 제공
        error_detail = {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": err
        }
        print(f"[ERROR] 오류 상세: {error_detail}")

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

        # trade_state에서 start_date를 가져와서 경과일수와 일일 평균 수익률 계산
        from trade_state import TradeStateManager
        state = await asyncio.to_thread(TradeStateManager.load_state)

        if state and 'start_date' in state and state['start_date']:
            start_date = pd.to_datetime(state['start_date'])
            now = pd.Timestamp.now()
            elapsed_days = (now - start_date).days

            # 경과일수가 0이면 1로 설정 (같은 날 시작한 경우)
            if elapsed_days == 0:
                elapsed_days = 1

            balance['start_date'] = start_date.isoformat()
            balance['elapsed_days'] = elapsed_days

            # 일일 평균 수익률 계산
            if balance.get('initial_capital') and balance['initial_capital'] > 0:
                total_return_rate = ((balance['total_assets'] / balance['initial_capital']) - 1) * 100
                daily_avg_return = total_return_rate / elapsed_days
                balance['daily_avg_return'] = daily_avg_return

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

        # 작업이 시작될 것임을 상태에 즉시 반영
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

        print("="*80)
        print(f"[TRACE] /api/initial_capital 엔드포인트 호출됨")
        print(f"  요청 데이터: {request}")
        print(f"  initial_capital 값: {initial_capital}")
        print("="*80)

        if initial_capital is None:
            raise HTTPException(status_code=400, detail="initial_capital is required")

        if not isinstance(initial_capital, (int, float)) or initial_capital < 0:
            raise HTTPException(status_code=400, detail="initial_capital must be a non-negative number")

        print(f"[TRACE] shared_state.set_initial_capital({initial_capital:,.0f}) 호출 (API 엔드포인트에서)")
        await asyncio.to_thread(shared_state.set_initial_capital, float(initial_capital))

        print(f"[TRACE] API 응답 반환: 초기 자산 설정 성공")
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


@app.post("/api/start_date")
async def set_start_date(request: dict):
    """시작 날짜 설정"""
    try:
        start_date_str = request.get('start_date')

        if start_date_str is None:
            raise HTTPException(status_code=400, detail="start_date is required")

        # 날짜 형식 검증
        try:
            start_date = pd.to_datetime(start_date_str)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date format. Use ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")

        # trade_state에 start_date 저장
        from trade_state import TradeStateManager
        state = await asyncio.to_thread(TradeStateManager.load_state)

        if state is None:
            raise HTTPException(status_code=404, detail="거래 상태가 초기화되지 않았습니다.")

        state['start_date'] = start_date
        await asyncio.to_thread(TradeStateManager.save_state, state)

        return JSONResponse(content={
            "message": "시작 날짜가 설정되었습니다.",
            "start_date": start_date.isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise HTTPException(status_code=500, detail=f"시작 날짜 설정 실패: {str(e)}")


def cleanup_on_exit(signum=None, frame=None):
    """
    프로세스 종료 시 정리 작업

    Parameters:
    - signum: 시그널 번호 (SIGINT=2, SIGTERM=15)
    - frame: 현재 스택 프레임
    """
    print(f"\n{'='*80}")
    if signum:
        signal_name = signal.Signals(signum).name
        print(f"시그널 수신: {signal_name} ({signum})")
    print("프로세스 종료 정리 작업 시작...")
    print("="*80)

    # 백그라운드 작업 중지
    shared_state.set_running(False)
    print("백그라운드 작업 중지 요청 완료")

    # 백그라운드 스레드 종료 대기
    global _background_thread
    if _background_thread is not None and _background_thread.is_alive():
        print("백그라운드 스레드 종료 대기 중... (최대 2초)")
        _background_thread.join(timeout=2.0)
        if _background_thread.is_alive():
            print("경고: 백그라운드 스레드가 2초 내에 종료되지 않았습니다.")
        else:
            print("백그라운드 스레드 종료 완료")

    # ThreadPoolExecutor 종료
    print("ThreadPoolExecutor 종료 중...")
    _executor.shutdown(wait=False, cancel_futures=True)
    print("ThreadPoolExecutor 종료 완료")

    print("정리 작업 완료")
    print("="*80)

    # 시그널 핸들러로 호출된 경우 프로세스 종료
    if signum:
        sys.exit(0)


if __name__ == "__main__":
    # 시그널 핸들러 등록 (Ctrl+C, kill 등)
    signal.signal(signal.SIGINT, cleanup_on_exit)   # Ctrl+C
    signal.signal(signal.SIGTERM, cleanup_on_exit)  # kill 명령

    # PID를 파일에 기록
    pid = os.getpid()
    pid_file = os.path.join(os.getcwd(), 'pid.txt')
    with open(pid_file, 'w') as f:
        f.write(str(pid))
    print(f"PID {pid}를 {pid_file}에 기록했습니다.")

    # 서버 실행
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    # 백그라운드 작업 때문에 단일 워커만 사용
    # workers > 1이면 각 워커에서 백테스트가 중복 실행됨
    print(f"서버 시작: http://{host}:{port}")
    print(f"워커 수: 1 (백그라운드 작업 중복 방지)")
    print(f"API 문서: http://{host}:{port}/docs")
    print(f"대체 문서: http://{host}:{port}/redoc")

    try:
        # 단일 프로세스 모드로 실행하여 백그라운드 스레드 관리 단순화
        uvicorn.run(
            app,  # 직접 app 객체 사용 (단일 프로세스)
            host=host,
            port=port,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt 감지됨")
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
    finally:
        # 정상 종료 시에도 정리 작업 실행
        cleanup_on_exit()

