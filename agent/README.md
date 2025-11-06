# PPO 기반 강화학습 거래 에이전트

PPO (Proximal Policy Optimization) 알고리즘을 사용한 암호화폐 자동 거래 에이전트입니다.

## 프로젝트 구조

```
agent/
├── __init__.py           # 패키지 초기화
├── config.py             # 하이퍼파라미터 및 설정
├── data_loader.py        # 데이터 로드 및 전처리
├── environment.py        # 거래 환경 클래스
├── networks.py           # Actor-Critic 네트워크
├── ppo_agent.py          # PPO 에이전트 클래스
├── trainer.py            # 훈련 루프 및 로깅
├── utils.py              # 유틸리티 함수
├── train_rl_agent.py     # 메인 실행 스크립트
└── README.md             # 이 파일
```

## 주요 기능

- **4가지 액션**: 대기, 매수, 홀드, 매도
- **액션 제약**: 코인 미보유 시 대기/매수만, 보유 시 홀드/매도만
- **전액 매수/매도**: 초기 자본금 100만원 기준 전액 거래
- **오픈가 기준**: 각 틱의 오픈가로 매수/매도
- **PPO 알고리즘**: 안정적인 정책 기반 강화학습

## 사용 방법

### 1. 데이터 준비

`agent/dumps2` 폴더에 다음 구조로 데이터를 배치합니다:

```
agent/dumps2/
└── BTC/
    └── 3m/
        ├── 2025-06-16/
        │   ├── history_btc_3m_20250616_00.csv
        │   ├── history_btc_3m_20250616_01.csv
        │   └── ...
        ├── 2025-06-17/
        └── ...
```

### 2. 설정 조정

`config.py`에서 하이퍼파라미터를 조정할 수 있습니다:

- 데이터 설정: `BASE_DIR`, `TICKER`, `INTERVAL`, 날짜 범위
- 거래 설정: `INITIAL_CAPITAL`, `TRADING_FEE_RATE`
- 네트워크 설정: `LSTM_HIDDEN_DIM`, `HIDDEN_DIM`
- PPO 설정: `LEARNING_RATE`, `GAMMA`, `CLIP_EPSILON`
- 훈련 설정: `NUM_EPISODES`, `BATCH_SIZE`

### 3. 훈련 실행

```bash
cd agent
python train_rl_agent.py
```

### 4. 결과 확인

- 모델 저장: `rl_models/` 폴더
- 훈련 메트릭: `rl_logs/training_metrics.csv`
- 베스트 모델: `rl_models/best_model_ep*.pth`

## 주요 클래스

### TradingEnvironment

거래 시뮬레이션 환경:
- `reset()`: 환경 초기화
- `step(action)`: 액션 실행 및 다음 상태 반환
- `_get_state()`: 현재 상태 벡터 생성

### PPOTradingAgent

PPO 에이전트:
- `select_action(state)`: 액션 선택
- `update(batch)`: PPO 업데이트
- `save_model(path)`: 모델 저장
- `load_model(path)`: 모델 로드

### PPOTrainer

훈련 관리:
- `train()`: 메인 훈련 루프
- `validate()`: 검증 실행

## 상태 벡터 구성

- 시계열 데이터: 최근 N개 틱의 정규화된 가격/볼륨/기술지표
- 포지션 정보: 보유 여부, 포지션 비율, 현금 비율, 총 자산 비율

## 보상 함수

- 즉시 보상: 포트폴리오 가치 변화율
- 거래 비용: 수수료 반영
- 위험 페널티: 최대 낙폭 50% 초과 시
- 최종 보상: 에피소드 종료 시 총 수익률 보너스

## 하이퍼파라미터 초기값

- 상태 윈도우: 20개 틱
- 초기 자본금: 1,000,000원
- 수수료: 0.05%
- 학습률: 3e-4
- 할인율: 0.99
- PPO 클립 범위: 0.2

## 주의사항

1. **원본 가격 데이터**: 정규화된 데이터와 함께 원본 가격 데이터를 별도로 저장해야 합니다.
2. **메모리 관리**: 대용량 데이터의 경우 배치 크기를 조정하세요.
3. **하이퍼파라미터 튜닝**: 데이터와 목표에 맞게 하이퍼파라미터를 조정하세요.

## 문제 해결

### 데이터 로드 오류
- `dumps2` 폴더 경로 확인
- CSV 파일 형식 확인 (open, high, low, close, volume, ma5, ma7, ma10 컬럼 필요)

### 메모리 부족
- `BATCH_SIZE` 및 `MINIBATCH_SIZE` 감소
- `STATE_WINDOW` 크기 감소

### 학습 불안정
- `LEARNING_RATE` 감소
- `CLIP_EPSILON` 조정
- `ENTROPY_COEF` 증가

