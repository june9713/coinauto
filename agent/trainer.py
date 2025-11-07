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

        # 배치 학습을 위한 버퍼
        self.experience_buffer = {
            'states': [],
            'actions': [],
            'old_log_probs': [],
            'advantages': [],
            'returns': []
        }
    
    def train(self):
        """메인 훈련 루프 (배치 학습)"""
        print("훈련 시작...")
        print(f"설정: {self.config.EPISODES_PER_UPDATE}개 에피소드마다 {self.config.UPDATE_EPOCHS} 에폭 학습")

        for episode in range(self.config.NUM_EPISODES):
            # 에피소드 실행 (데이터만 수집, 학습은 나중에)
            episode_return, episode_info, episode_batch = self._run_episode_collect(self.train_env)

            # 경험 버퍼에 추가
            self.experience_buffer['states'].extend(episode_batch['states'])
            self.experience_buffer['actions'].extend(episode_batch['actions'])
            self.experience_buffer['old_log_probs'].extend(episode_batch['old_log_probs'])
            self.experience_buffer['advantages'].extend(episode_batch['advantages'])
            self.experience_buffer['returns'].extend(episode_batch['returns'])

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

            # 배치 학습 수행 (N개 에피소드마다)
            if (episode + 1) % self.config.EPISODES_PER_UPDATE == 0:
                print(f"\n[학습 시작] {self.config.EPISODES_PER_UPDATE}개 에피소드 데이터로 학습 중...")
                self._batch_update()
                print(f"[학습 완료] 버퍼 초기화\n")
            
            # 로그 출력
            if (episode + 1) % self.config.LOG_FREQ == 0:
                print(f"Episode {episode + 1}/{self.config.NUM_EPISODES}")
                print(f"  Portfolio Value: {episode_info['final_portfolio_value']:.0f}")
                print(f"  수익률: {profit_rate*100:.2f}% ({(episode_info['final_portfolio_value'] - self.config.INITIAL_CAPITAL):.0f}원)")
                print(f"  지난 100회 평균 수익률: {avg_profit_rate_100*100:.2f}%")
                print(f"  Total Trades: {episode_info['total_trades']}")
                print(f"  Max Drawdown: {episode_info['max_drawdown']:.4f}")
                print(f"  Sharpe Ratio: {episode_info['sharpe_ratio']:.4f}")
                print(f"  Epsilon (탐험률): {self.agent.epsilon:.4f}")
                print(f"  데이터 오프셋: {self.train_env.global_offset}/{self.train_env.total_data_length}")
            
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
    
    def _run_episode_collect(self, env):
        """에피소드 실행 및 데이터 수집 (학습은 나중에)"""
        state = env.reset(use_sliding_window=True)  # 슬라이딩 윈도우 사용
        
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        step = 0
        while step < self.config.MAX_STEPS_PER_EPISODE:
            # 액션 마스크 가져오기
            action_mask = env.get_action_mask()

            # 액션 선택
            action, log_prob, value = self.agent.select_action(state, deterministic=False, action_mask=action_mask)

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
        
        # 에피소드 배치 데이터 (나중에 모아서 학습)
        episode_batch = {
            'states': states,  # 리스트 그대로
            'actions': actions,
            'old_log_probs': log_probs,
            'advantages': advantages.tolist(),  # numpy array를 리스트로
            'returns': returns.tolist()
        }

        # 슬라이딩 윈도우: 에피소드 종료 후 오프셋 업데이트
        env.update_global_offset()

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

        return episode_return, episode_info, episode_batch

    def _batch_update(self):
        """버퍼에 모인 여러 에피소드 데이터로 배치 학습"""
        import torch

        # 버퍼가 비어있으면 스킵
        if len(self.experience_buffer['states']) == 0:
            print("  경고: 버퍼가 비어있습니다")
            return

        print(f"  버퍼 크기: {len(self.experience_buffer['states'])} 스텝")

        # 텐서 변환
        batch = {
            'states': torch.FloatTensor(self.experience_buffer['states']),
            'actions': torch.LongTensor(self.experience_buffer['actions']),
            'old_log_probs': torch.FloatTensor(self.experience_buffer['old_log_probs']),
            'advantages': torch.FloatTensor(self.experience_buffer['advantages']),
            'returns': torch.FloatTensor(self.experience_buffer['returns'])
        }

        # PPO 업데이트 (GPU 집중 사용)
        loss_dict = self.agent.update(batch)
        print(f"  손실 - Actor: {loss_dict['actor_loss']:.4f}, Critic: {loss_dict['critic_loss']:.4f}, Entropy: {loss_dict['entropy']:.4f}")

        # Epsilon 감소 (배치당 한 번)
        self.agent.decay_epsilon()

        # 버퍼 초기화
        self.experience_buffer = {
            'states': [],
            'actions': [],
            'old_log_probs': [],
            'advantages': [],
            'returns': []
        }

    def _run_episode_validation(self, env):
        """검증용 에피소드 실행 (빠른 실행)"""
        state = env.reset(use_sliding_window=False)  # 검증은 항상 처음부터
        
        rewards = []
        step = 0
        
        while step < self.config.MAX_STEPS_PER_EPISODE:
            # 액션 마스크 가져오기
            action_mask = env.get_action_mask()

            # 액션 선택 (결정적 선택으로 빠르게)
            action, _, _ = self.agent.select_action(state, deterministic=True, action_mask=action_mask)

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

        return self._run_episode_validation(self.val_env)
    
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

