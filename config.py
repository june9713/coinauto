"""
설정 관리 모듈
모든 설정 값을 중앙에서 관리
"""
import os
from condition_manager import ConditionManager


class Config:
    """애플리케이션 설정 관리 클래스"""
    
    # 데이터 저장 경로 설정
    DATA_DIR = './datas'
    DATA_FILE = os.path.join(DATA_DIR, 'btc_history.csv')
    
    # 한글 폰트 경로
    FONT_PATH = os.path.join(os.path.dirname(__file__), 'malgun.ttf')
    
    # 기본 백테스트 설정
    DEFAULT_INITIAL_CAPITAL = 1000000  # 초기 자본 (원)
    DEFAULT_BUY_ANGLE_THRESHOLD = 5.4  # 매수 조건 각도
    DEFAULT_SELL_ANGLE_THRESHOLD = -16.7  # 매도 조건 각도
    DEFAULT_STOP_LOSS_PERCENT = -5.0  # 손절 기준 (수익률 %)
    DEFAULT_MIN_SELL_PRICE = 20000  # 최소 매도 가격 (원)
    DEFAULT_PRICE_SLIPPAGE = 1000  # 거래 가격 슬리퍼지 (원)
    DEFAULT_ASPECT_RATIO = 4.0  # 차트 종횡비
    DEFAULT_WINDOW = 14  # 추세선 계산 윈도우 크기
    
    # 데이터 수집 기본 설정
    DEFAULT_TICKER = 'BTC'  # 기본 암호화폐 티커
    DEFAULT_INTERVAL = '3m'  # 기본 캔들스틱 간격
    DEFAULT_START_DATE = '2014-01-01'  # 기본 시작 날짜
    
    # 시각화 설정
    DEFAULT_GRAPH_SAVE_PATH = 'backtest_result.jpg'  # 기본 그래프 저장 경로
    GRAPH_DPI = 60  # 그래프 해상도 (300에서 70으로로 감소)
    
    # 백테스트 결과 저장 설정
    BACKTEST_RESULT_FILE = os.path.join(DATA_DIR, 'backtest_results.csv')  # 백테스트 결과 저장 파일
    
    @classmethod
    def get_data_file_path(cls, ticker=None):
        """
        데이터 파일 경로 반환
        
        Parameters:
        - ticker (str, optional): 암호화폐 티커. None이면 기본값 사용
        
        Returns:
        - str: 데이터 파일 경로
        """
        if ticker:
            return os.path.join(cls.DATA_DIR, f'{ticker.lower()}_history.csv')
        return cls.DATA_FILE
    
    @classmethod
    def get_backtest_result_file_path(cls, ticker=None, condition_dict=None):
        """
        백테스트 결과 파일 경로 반환
        
        Parameters:
        - ticker (str, optional): 암호화폐 티커. None이면 기본값 사용
        - condition_dict (dict, optional): 조건 딕셔너리. None이면 기본 파일명 사용
        
        Returns:
        - str: 백테스트 결과 파일 경로
        """
        if condition_dict is not None:
            # 조건 문자열에 이미 ticker와 interval이 포함되어 있으므로 파일명에는 조건 문자열만 사용
            condition_str = ConditionManager.get_condition_string(condition_dict)
            return os.path.join(cls.DATA_DIR, f'backtest_results_{condition_str}.csv')
        
        if ticker:
            return os.path.join(cls.DATA_DIR, f'{ticker.lower()}_backtest_results.csv')
        return cls.BACKTEST_RESULT_FILE
    
    @classmethod
    def get_data_file_path(cls, ticker=None, condition_dict=None):
        """
        데이터 파일 경로 반환
        
        Parameters:
        - ticker (str, optional): 암호화폐 티커. None이면 기본값 사용
        - condition_dict (dict, optional): 조건 딕셔너리. None이면 기본 파일명 사용
        
        Returns:
        - str: 데이터 파일 경로
        """
        if condition_dict is not None:
            # 조건 문자열에 이미 ticker와 interval이 포함되어 있으므로 파일명에는 조건 문자열만 사용
            condition_str = ConditionManager.get_condition_string(condition_dict)
            return os.path.join(cls.DATA_DIR, f'history_{condition_str}.csv')
        
        if ticker:
            return os.path.join(cls.DATA_DIR, f'{ticker.lower()}_history.csv')
        return cls.DATA_FILE
    
    @classmethod
    def ensure_data_directory(cls):
        """
        데이터 디렉토리가 존재하는지 확인하고, 없으면 생성
        
        Returns:
        - bool: 성공 여부
        """
        try:
            if not os.path.exists(cls.DATA_DIR):
                os.makedirs(cls.DATA_DIR)
                print(f"데이터 폴더 생성: {cls.DATA_DIR}")
            return True
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            print("err", err)
            raise

