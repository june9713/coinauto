"""
PPO 에이전트 클래스
"""
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .networks import ActorNetwork, CriticNetwork


class PPOTradingAgent:
    """PPO 거래 에이전트"""
    
    def __init__(self, state_dim, action_dim, config):
        """
        Args:
            state_dim: 상태 벡터 차원
            action_dim: 액션 공간 크기
            config: 설정 객체
        """
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 네트워크 생성
        self.actor = ActorNetwork(state_dim, action_dim, config).to(config.DEVICE)
        self.critic = CriticNetwork(state_dim, config).to(config.DEVICE)
        
        # 옵티마이저
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.LEARNING_RATE
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.LEARNING_RATE
        )
    
    def select_action(self, state, deterministic=False):
        """
        액션 선택
        
        Args:
            state: 상태 벡터 (numpy array)
            deterministic: 결정적 선택 여부
        
        Returns:
            action: 선택된 액션
            action_log_prob: 액션 로그 확률
            value: 상태 가치
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.DEVICE)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=1).item()
                action_log_prob = torch.log(action_probs[0, action] + 1e-8)
            else:
                dist = Categorical(action_probs)
                action = dist.sample()
                action_log_prob = dist.log_prob(action)
                action = action.item()
        
        return action, action_log_prob.item(), value.item()
    
    def evaluate_actions(self, states, actions):
        """
        액션 평가 (로그 확률, 가치, 엔트로피 계산)
        
        Args:
            states: 상태 배치 [batch_size, state_dim]
            actions: 액션 배치 [batch_size]
        
        Returns:
            action_log_probs: 액션 로그 확률 [batch_size]
            values: 상태 가치 [batch_size, 1]
            entropy: 엔트로피 [batch_size]
        """
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        values = self.critic(states)
        
        return action_log_probs, values, entropy
    
    def update(self, batch):
        """
        PPO 업데이트
        
        Args:
            batch: 배치 데이터 딕셔너리
                - states: [batch_size, state_dim]
                - actions: [batch_size]
                - old_log_probs: [batch_size]
                - advantages: [batch_size]
                - returns: [batch_size]
        
        Returns:
            loss_dict: 손실 정보 딕셔너리
        """
        states = batch['states'].to(self.config.DEVICE)
        actions = batch['actions'].to(self.config.DEVICE)
        old_log_probs = batch['old_log_probs'].to(self.config.DEVICE)
        advantages = batch['advantages'].to(self.config.DEVICE)
        returns = batch['returns'].to(self.config.DEVICE)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # 여러 에폭 업데이트
        for epoch in range(self.config.UPDATE_EPOCHS):
            # 미니배치로 나누기
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.config.MINIBATCH_SIZE):
                end = start + self.config.MINIBATCH_SIZE
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 액션 평가
                action_log_probs, values, entropy = self.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # PPO 클립 손실
                ratio = torch.exp(action_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.CLIP_EPSILON, 
                                   1 + self.config.CLIP_EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic 손실
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # 엔트로피 보너스
                entropy_bonus = -self.config.ENTROPY_COEF * entropy.mean()
                
                # 총 Actor 손실
                total_actor_loss_batch = actor_loss + entropy_bonus
                
                # 업데이트
                self.actor_optimizer.zero_grad()
                total_actor_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 
                    self.config.MAX_GRAD_NORM
                )
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), 
                    self.config.MAX_GRAD_NORM
                )
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
        
        num_updates = self.config.UPDATE_EPOCHS * (len(states) // self.config.MINIBATCH_SIZE + 1)
        
        loss_dict = {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
        
        return loss_dict
    
    def save_model(self, path):
        """모델 저장"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        print(f"모델 저장 완료: {path}")
    
    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.config.DEVICE)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"모델 로드 완료: {path}")

