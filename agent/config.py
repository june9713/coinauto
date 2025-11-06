"""
하이퍼파라미터 및 설정 관리
"""
import os
import torch

class Config:
    """PPO 거래 에이전트 설정"""
    
    # ==================== 데이터 설정 ====================
    BASE_DIR = './dumps2'  # agent 폴더 내 dumps2
    TICKER = 'BTC'
    INTERVAL = '3m'
    START_DATE = '2025-06-16'
    END_DATE = '2025-11-06'
    VALIDATION_SPLIT = 0.2  # 검증 데이터 비율
    
    # 사용할 피처
    FEATURES = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma7', 'ma10', 'cat01']
    STATE_WINDOW = 20  # 최근 20개 틱을 상태로 사용
    POSITION_INFO_DIM = 6  # 포지션 정보 차원 (보유여부, 포지션비율, 현금비율, 총자산비율, 미실현수익률, 최고점대비비율)
    
    # ==================== 거래 설정 ====================
    INITIAL_CAPITAL = 1_000_000  # 100만원
    TRADING_FEE_RATE = 0.0005  # 0.05% 수수료
    SLIPPAGE_RATE = 0.0  # 슬리피지 (오픈가 기준이므로 0)
    
    # ==================== 네트워크 설정 ====================
    LSTM_HIDDEN_DIM = 64  # LSTM 히든 차원
    LSTM_NUM_LAYERS = 2  # LSTM 레이어 수
    HIDDEN_DIM = 128  # MLP 히든 차원
    NUM_HIDDEN_LAYERS = 2  # MLP 히든 레이어 수
    DROPOUT_RATE = 0.1  # 드롭아웃 비율
    
    # ==================== PPO 설정 ====================
    LEARNING_RATE = 3e-4
    GAMMA = 0.99  # 할인율
    GAE_LAMBDA = 0.95  # GAE 람다
    CLIP_EPSILON = 0.2  # PPO 클립 범위
    ENTROPY_COEF = 0.01  # 엔트로피 계수
    VALUE_COEF = 0.5  # 가치 손실 계수
    MAX_GRAD_NORM = 0.5  # 그라디언트 클리핑
    
    # ==================== 훈련 설정 ====================
    NUM_EPISODES = 100000
    MAX_STEPS_PER_EPISODE = 10000
    BATCH_SIZE = 64
    UPDATE_EPOCHS = 10  # 각 업데이트마다 에폭 수
    MINIBATCH_SIZE = 32  # 미니배치 크기
    
    # ==================== 검증 및 저장 ====================
    VALIDATION_FREQ = 10  # 검증 주기 (에피소드)
    SAVE_FREQ = 50  # 모델 저장 주기 (에피소드)
    LOG_FREQ = 10  # 로그 출력 주기 (에피소드)
    
    # ==================== 경로 설정 ====================
    MODEL_DIR = './rl_models'
    LOG_DIR = './rl_logs'
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')  # 베스트 모델 경로
    
    # ==================== 디바이스 설정 ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== 시드 설정 ====================
    RANDOM_SEED = 42
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)

