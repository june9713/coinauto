"""
유틸리티 함수
"""
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    샤프 비율 계산
    
    Args:
        returns: 수익률 리스트
        risk_free_rate: 무위험 이자율
    
    Returns:
        sharpe_ratio: 샤프 비율
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = np.array(returns) - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
    
    # 연율화 (일일 수익률 기준으로 가정)
    sharpe_ratio *= np.sqrt(252)  # 연율화
    
    return sharpe_ratio


def calculate_max_drawdown(portfolio_values):
    """
    최대 낙폭 계산
    
    Args:
        portfolio_values: 포트폴리오 가치 리스트
    
    Returns:
        max_drawdown: 최대 낙폭 (0~1)
    """
    if len(portfolio_values) == 0:
        return 0.0
    
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / (peak + 1e-8)
    max_drawdown = np.max(drawdown)
    
    return max_drawdown


def plot_portfolio_value(portfolio_values, save_path=None):
    """
    포트폴리오 가치 시각화
    
    Args:
        portfolio_values: 포트폴리오 가치 리스트
        save_path: 저장 경로 (None이면 표시만)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values)
    plt.xlabel('Step')
    plt.ylabel('Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_action_distribution(actions, save_path=None):
    """
    액션 분포 시각화
    
    Args:
        actions: 액션 리스트
        save_path: 저장 경로 (None이면 표시만)
    """
    action_names = ['대기', '매수', '홀드', '매도']
    action_counts = [actions.count(i) for i in range(4)]
    
    plt.figure(figsize=(8, 6))
    plt.bar(action_names, action_counts)
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.title('Action Distribution')
    plt.grid(True, axis='y')
    
    if save_path:
        plt.savefig(save_path)
        print(f"그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_gae(rewards, values, next_values, dones, gamma, lam):
    """
    Generalized Advantage Estimation (GAE) 계산
    
    Args:
        rewards: 보상 리스트 [T]
        values: 상태 가치 리스트 [T]
        next_values: 다음 상태 가치 리스트 [T]
        dones: 종료 여부 리스트 [T]
        gamma: 할인율
        lam: GAE 람다
    
    Returns:
        advantages: 어드밴티지 [T]
        returns: 반환값 [T]
    """
    T = len(rewards)
    advantages = np.zeros(T)
    returns = np.zeros(T)
    
    gae = 0
    for t in reversed(range(T)):
        if dones[t]:
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + gamma * next_values[t] - values[t]
            gae = delta + gamma * lam * gae
        
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    
    return advantages, returns

