"""
PPO 거래 에이전트 훈련 메인 스크립트
"""
import os
import sys
import traceback

# 스크립트 파일의 디렉토리 (agent 폴더)로 작업 디렉토리 변경
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"작업 디렉토리 변경: {os.getcwd()}")

# 상위 디렉토리를 경로에 추가
sys.path.insert(0, os.path.dirname(script_dir))

from agent.config import Config
from agent.data_loader import load_training_data, split_train_val, normalize_features
from agent.environment import TradingEnvironment
from agent.networks import ActorNetwork, CriticNetwork
from agent.ppo_agent import PPOTradingAgent
from agent.trainer import PPOTrainer
from agent.utils import set_seed


def main():
    """메인 함수"""
    try:
        # 설정
        config = Config
        config.create_directories()
        
        # 시드 설정
        set_seed(config.RANDOM_SEED)
        
        print("=" * 50)
        print("PPO 거래 에이전트 훈련 시작")
        print("=" * 50)
        print(f"디바이스: {config.DEVICE}")
        print(f"데이터 디렉토리: {config.BASE_DIR}")
        print(f"티커: {config.TICKER}, 간격: {config.INTERVAL}")
        print(f"기간: {config.START_DATE} ~ {config.END_DATE}")
        print("=" * 50)
        
        # 데이터 로드
        print("\n[1/5] 데이터 로드 중...")
        full_df = load_training_data(
            config.BASE_DIR,
            config.TICKER,
            config.INTERVAL,
            config.START_DATE,
            config.END_DATE,
            config.FEATURES
        )
        
        if full_df.empty:
            print("오류: 데이터가 없습니다.")
            return
        
        # 훈련/검증 분할
        print("\n[2/5] 데이터 분할 중...")
        train_df, val_df = split_train_val(
            full_df,
            validation_split=config.VALIDATION_SPLIT,
            shuffle=False
        )
        
        # 정규화
        print("\n[3/5] 데이터 정규화 중...")
        scaler, train_df_norm, val_df_norm, train_original_prices, val_original_prices = normalize_features(
            train_df, val_df, config.FEATURES
        )
        
        # 환경 생성
        print("\n[4/5] 환경 생성 중...")
        train_env = TradingEnvironment(
            train_df_norm,
            config,
            original_prices=train_original_prices,
            original_df=train_df  # 원본 데이터프레임 전달 (타임스탬프 및 차트 데이터용)
        )
        val_env = TradingEnvironment(
            val_df_norm,
            config,
            original_prices=val_original_prices,
            original_df=val_df  # 원본 데이터프레임 전달
        )
        
        # 상태 차원 계산
        num_features = len(config.FEATURES)
        state_dim = config.STATE_WINDOW * num_features + 6  # 포지션 정보 6개 (기존 4개 + 미실현 수익률 + 최고점 대비 비율)
        action_dim = 4  # 대기, 매수, 홀드, 매도
        
        print(f"상태 차원: {state_dim}")
        print(f"액션 차원: {action_dim}")
        
        # 에이전트 생성
        print("\n[5/5] 에이전트 생성 중...")
        agent = PPOTradingAgent(state_dim, action_dim, config)
        
        # 베스트 모델 로드 (있는 경우)
        if os.path.exists(config.BEST_MODEL_PATH):
            print(f"베스트 모델 로드 중: {config.BEST_MODEL_PATH}")
            agent.load_model(config.BEST_MODEL_PATH)
            print("베스트 모델 로드 완료")
        else:
            print("베스트 모델이 없습니다. 새로 훈련을 시작합니다.")
        
        # 훈련기 생성
        trainer = PPOTrainer(agent, train_env, val_env, config)
        
        # 훈련 시작
        print("\n훈련 시작...")
        trainer.train()
        
        print("\n" + "=" * 50)
        print("훈련 완료!")
        print("=" * 50)
        
    except Exception as e:
        err = traceback.format_exc()
        print("err", err)
        raise


if __name__ == '__main__':
    main()

