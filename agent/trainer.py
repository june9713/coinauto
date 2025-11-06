"""
훈련 루프 및 로깅
"""
import os
import traceback
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime

from .environment import TradingEnvironment
from .ppo_agent import PPOTradingAgent
from .utils import compute_gae, calculate_sharpe_ratio, calculate_max_drawdown


class PPOTrainer:
    """PPO 훈련 클래스"""
    
    def __init__(self, agent, train_env, val_env, config):
        """
        Args:
            agent: PPO 에이전트
            train_env: 훈련 환경
            val_env: 검증 환경
            config: 설정 객체
        """
        self.agent = agent
        self.train_env = train_env
        self.val_env = val_env
        self.config = config
        
        # 로깅
        self.train_metrics = {
            'episode': [],
            'return': [],
            'portfolio_value': [],
            'total_trades': [],
            'max_drawdown': [],
            'sharpe_ratio': []
        }
        
        self.best_val_return = float('-inf')
        
        # 지난 100회 평균 수익 계산용
        self.recent_returns = deque(maxlen=100)
    
    def train(self):
        """메인 훈련 루프"""
        print("훈련 시작...")
        
        for episode in range(self.config.NUM_EPISODES):
            # 에피소드 실행
            episode_return, episode_info = self._run_episode(self.train_env, training=True)
            
            # 메트릭 기록
            self.train_metrics['episode'].append(episode)
            self.train_metrics['return'].append(episode_return)
            self.train_metrics['portfolio_value'].append(episode_info['final_portfolio_value'])
            self.train_metrics['total_trades'].append(episode_info['total_trades'])
            self.train_metrics['max_drawdown'].append(episode_info['max_drawdown'])
            self.train_metrics['sharpe_ratio'].append(episode_info['sharpe_ratio'])
            
            # 실제 수익률 (초기 자본금 대비)
            profit_rate = episode_info['profit_rate']
            
            # 지난 100회 평균 수익률 계산 (실제 수익률 기준)
            self.recent_returns.append(profit_rate)
            avg_profit_rate_100 = np.mean(self.recent_returns) if len(self.recent_returns) > 0 else 0.0
            
            # 에피소드별 거래 기록 저장
            self._save_episode_deal_history(episode)
            
            # 로그 출력
            if (episode + 1) % self.config.LOG_FREQ == 0:
                print(f"Episode {episode + 1}/{self.config.NUM_EPISODES}")
                print(f"  Portfolio Value: {episode_info['final_portfolio_value']:.0f}")
                print(f"  수익률: {profit_rate*100:.2f}% ({(episode_info['final_portfolio_value'] - self.config.INITIAL_CAPITAL):.0f}원)")
                print(f"  지난 100회 평균 수익률: {avg_profit_rate_100*100:.2f}%")
                print(f"  Total Trades: {episode_info['total_trades']}")
                print(f"  Max Drawdown: {episode_info['max_drawdown']:.4f}")
                print(f"  Sharpe Ratio: {episode_info['sharpe_ratio']:.4f}")
            
            # 검증
            if (episode + 1) % self.config.VALIDATION_FREQ == 0:
                val_return, val_info = self.validate()
                val_profit_rate = val_info['profit_rate']
                print(f"검증 - Portfolio Value: {val_info['final_portfolio_value']:.0f}, 수익률: {val_profit_rate*100:.2f}%")
                
                # 베스트 모델 저장 (검증 수익률 기준)
                if val_profit_rate > self.best_val_return:
                    self.best_val_return = val_profit_rate
                    # 베스트 모델을 항상 같은 경로에 저장 (덮어쓰기)
                    self.agent.save_model(self.config.BEST_MODEL_PATH)
                    print(f"베스트 모델 저장 (검증 수익률: {val_profit_rate*100:.2f}%): {self.config.BEST_MODEL_PATH}")
            
            # 주기적 모델 저장
            if (episode + 1) % self.config.SAVE_FREQ == 0:
                model_path = os.path.join(
                    self.config.MODEL_DIR, 
                    f'model_ep{episode + 1}.pth'
                )
                self.agent.save_model(model_path)
        
        print("훈련 완료!")
        
        # 최종 모델 저장
        final_model_path = os.path.join(self.config.MODEL_DIR, 'final_model.pth')
        self.agent.save_model(final_model_path)
        
        # 메트릭 저장
        self._save_metrics()
        
        # 거래 기록 저장
        self._save_deal_history()
    
    def _run_episode(self, env, training=True):
        """에피소드 실행"""
        state = env.reset()
        
        # 검증 시에는 빠른 실행을 위해 간소화
        if not training:
            return self._run_episode_validation(env)
        
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        step = 0
        while step < self.config.MAX_STEPS_PER_EPISODE:
            # 액션 선택
            action, log_prob, value = self.agent.select_action(state, deterministic=False)
            
            # 액션 실행
            next_state, reward, done, info = env.step(action)
            
            # 데이터 저장
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            
            state = next_state
            step += 1
            
            if done:
                break
        
        # 최종 상태 가치 계산
        if not done:
            _, _, final_value = self.agent.select_action(state, deterministic=True)
        else:
            final_value = 0.0
        
        # GAE 계산
        next_values = values[1:] + [final_value]
        advantages, returns = compute_gae(
            rewards, values, next_values, dones,
            self.config.GAMMA, self.config.GAE_LAMBDA
        )
        
        # 정규화 (선택적)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # 배치 데이터 준비
        batch = {
            'states': np.array(states),
            'actions': np.array(actions),
            'old_log_probs': np.array(log_probs),
            'advantages': advantages,
            'returns': returns
        }
        
        # 텐서 변환
        import torch
        batch['states'] = torch.FloatTensor(batch['states'])
        batch['actions'] = torch.LongTensor(batch['actions'])
        batch['old_log_probs'] = torch.FloatTensor(batch['old_log_probs'])
        batch['advantages'] = torch.FloatTensor(batch['advantages'])
        batch['returns'] = torch.FloatTensor(batch['returns'])
        
        # PPO 업데이트
        loss_dict = self.agent.update(batch)
        
        # 에피소드 정보
        episode_return = sum(rewards)
        final_portfolio_value = info['portfolio_value']
        total_trades = info['total_trades']
        max_drawdown = info['max_drawdown']
        
        # 실제 수익률 계산 (초기 자본금 대비)
        profit_rate = (final_portfolio_value - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL
        
        # 샤프 비율 계산
        returns_array = np.array(rewards)
        sharpe_ratio = calculate_sharpe_ratio(returns_array)
        
        episode_info = {
            'final_portfolio_value': final_portfolio_value,
            'total_trades': total_trades,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_rate': profit_rate  # 실제 수익률 추가
        }
        
        return episode_return, episode_info
    
    def _run_episode_validation(self, env):
        """검증용 에피소드 실행 (빠른 실행)"""
        state = env.reset()
        
        rewards = []
        step = 0
        
        while step < self.config.MAX_STEPS_PER_EPISODE:
            # 액션 선택 (결정적 선택으로 빠르게)
            action, _, _ = self.agent.select_action(state, deterministic=True)
            
            # 액션 실행
            next_state, reward, done, info = env.step(action)
            
            rewards.append(reward)
            state = next_state
            step += 1
            
            if done:
                break
        
        # 에피소드 정보
        episode_return = sum(rewards)
        final_portfolio_value = info['portfolio_value']
        total_trades = info['total_trades']
        max_drawdown = info['max_drawdown']
        
        # 실제 수익률 계산 (초기 자본금 대비)
        profit_rate = (final_portfolio_value - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL
        
        # 샤프 비율 계산
        returns_array = np.array(rewards)
        sharpe_ratio = calculate_sharpe_ratio(returns_array)
        
        episode_info = {
            'final_portfolio_value': final_portfolio_value,
            'total_trades': total_trades,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_rate': profit_rate
        }
        
        return episode_return, episode_info
    
    def validate(self):
        """검증 실행"""
        # 검증 전 거래 기록 초기화 (검증 기록은 저장하지 않음)
        if hasattr(self.val_env, 'trade_history'):
            self.val_env.trade_history = []
        
        return self._run_episode(self.val_env, training=False)
    
    def _save_metrics(self):
        """메트릭 저장"""
        metrics_df = pd.DataFrame(self.train_metrics)
        metrics_path = os.path.join(self.config.LOG_DIR, 'training_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"메트릭 저장: {metrics_path}")
    
    def _save_deal_history(self):
        """거래 기록 저장 (각 에피소드마다 저장)"""
        # 모든 에피소드의 거래 기록을 저장하기 위해 에피소드별로 저장
        # 마지막 에피소드의 거래 기록 저장
        if hasattr(self.train_env, 'trade_history') and len(self.train_env.trade_history) > 0:
            deal_history_dir = os.path.join('./deal_history', self.config.TICKER, self.config.INTERVAL)
            os.makedirs(deal_history_dir, exist_ok=True)
            
            # 현재 날짜와 시간으로 파일명 생성
            now = datetime.now()
            date_str = now.strftime('%Y-%m-%d')
            time_str = now.strftime('%H%M%S')
            
            date_dir = os.path.join(deal_history_dir, date_str)
            os.makedirs(date_dir, exist_ok=True)
            
            filename = f'deal_history_{date_str}_{time_str}.csv'
            filepath = os.path.join(date_dir, filename)
            
            # 거래 기록을 DataFrame으로 변환
            deal_df = pd.DataFrame(self.train_env.trade_history)
            deal_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"거래 기록 저장: {filepath} ({len(deal_df)}개 기록)")
    
    def _save_episode_deal_history(self, episode):
        """에피소드별 거래 기록 저장"""
        if hasattr(self.train_env, 'trade_history') and len(self.train_env.trade_history) > 0:
            deal_history_dir = os.path.join('./deal_history', self.config.TICKER, self.config.INTERVAL)
            os.makedirs(deal_history_dir, exist_ok=True)
            
            # 현재 날짜로 디렉토리 생성
            now = datetime.now()
            date_str = now.strftime('%Y-%m-%d')
            date_dir = os.path.join(deal_history_dir, date_str)
            os.makedirs(date_dir, exist_ok=True)
            
            # 에피소드 번호와 시간으로 파일명 생성
            time_str = now.strftime('%H%M%S')
            filename = f'deal_history_ep{episode+1:04d}_{date_str}_{time_str}.csv'
            filepath = os.path.join(date_dir, filename)
            
            # 거래 기록을 DataFrame으로 변환
            deal_df = pd.DataFrame(self.train_env.trade_history)
            deal_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            # 거래 기록 초기화 (다음 에피소드를 위해)
            self.train_env.trade_history = []

