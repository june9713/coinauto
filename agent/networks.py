"""
Actor-Critic 네트워크 아키텍처
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """정책 네트워크 (Actor)"""
    
    def __init__(self, state_dim, action_dim, config):
        """
        Args:
            state_dim: 상태 벡터 차원
            action_dim: 액션 공간 크기
            config: 설정 객체
        """
        super(ActorNetwork, self).__init__()
        self.config = config
        
        # 시계열 인코더 (LSTM)
        # 상태 벡터는 (STATE_WINDOW * num_features + position_info) 형태
        # 시계열 부분만 LSTM으로 처리
        num_features = len(config.FEATURES)
        lstm_input_dim = num_features
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config.LSTM_HIDDEN_DIM,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT_RATE if config.LSTM_NUM_LAYERS > 1 else 0
        )
        
        # 포지션 정보 차원
        position_info_dim = config.POSITION_INFO_DIM

        # MLP 입력 차원: LSTM 출력 + 포지션 정보
        mlp_input_dim = config.LSTM_HIDDEN_DIM + position_info_dim
        
        # MLP 레이어 구성
        layers = []
        input_dim = mlp_input_dim
        for i in range(config.NUM_HIDDEN_LAYERS):
            layers.append(nn.Linear(input_dim, config.HIDDEN_DIM))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.DROPOUT_RATE))
            input_dim = config.HIDDEN_DIM
        
        self.mlp = nn.Sequential(*layers)
        
        # 출력 레이어 (액션 확률)
        self.action_head = nn.Linear(config.HIDDEN_DIM, action_dim)
    
    def forward(self, state):
        """
        Args:
            state: [batch_size, state_dim] 형태의 상태 벡터
        
        Returns:
            action_probs: [batch_size, action_dim] 형태의 액션 확률
        """
        batch_size = state.shape[0]
        num_features = len(self.config.FEATURES)
        window_size = self.config.STATE_WINDOW
        
        # 상태 벡터 분리: 시계열 부분 + 포지션 정보
        # state = [시계열 데이터 (window_size * num_features), 포지션 정보 (POSITION_INFO_DIM)]
        time_series_end = window_size * num_features
        time_series = state[:, :time_series_end]  # [batch, window_size * num_features]
        position_info = state[:, time_series_end:]  # [batch, POSITION_INFO_DIM]
        
        # 시계열 데이터를 [batch, window_size, num_features]로 reshape
        time_series = time_series.view(batch_size, window_size, num_features)
        
        # LSTM 처리
        lstm_out, _ = self.lstm(time_series)  # [batch, window_size, lstm_hidden]
        lstm_out = lstm_out[:, -1, :]  # 마지막 타임스텝만 사용 [batch, lstm_hidden]
        
        # 포지션 정보와 결합
        combined = torch.cat([lstm_out, position_info], dim=1)  # [batch, lstm_hidden + POSITION_INFO_DIM]
        
        # MLP 처리
        hidden = self.mlp(combined)
        
        # 액션 확률
        action_logits = self.action_head(hidden)
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs


class CriticNetwork(nn.Module):
    """가치 네트워크 (Critic)"""
    
    def __init__(self, state_dim, config):
        """
        Args:
            state_dim: 상태 벡터 차원
            config: 설정 객체
        """
        super(CriticNetwork, self).__init__()
        self.config = config
        
        # 시계열 인코더 (LSTM) - Actor와 동일한 구조
        num_features = len(config.FEATURES)
        lstm_input_dim = num_features
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=config.LSTM_HIDDEN_DIM,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT_RATE if config.LSTM_NUM_LAYERS > 1 else 0
        )
        
        # 포지션 정보 차원
        position_info_dim = config.POSITION_INFO_DIM

        # MLP 입력 차원
        mlp_input_dim = config.LSTM_HIDDEN_DIM + position_info_dim
        
        # MLP 레이어 구성
        layers = []
        input_dim = mlp_input_dim
        for i in range(config.NUM_HIDDEN_LAYERS):
            layers.append(nn.Linear(input_dim, config.HIDDEN_DIM))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.DROPOUT_RATE))
            input_dim = config.HIDDEN_DIM
        
        self.mlp = nn.Sequential(*layers)
        
        # 출력 레이어 (상태 가치)
        self.value_head = nn.Linear(config.HIDDEN_DIM, 1)
    
    def forward(self, state):
        """
        Args:
            state: [batch_size, state_dim] 형태의 상태 벡터
        
        Returns:
            value: [batch_size, 1] 형태의 상태 가치
        """
        batch_size = state.shape[0]
        num_features = len(self.config.FEATURES)
        window_size = self.config.STATE_WINDOW
        
        # 상태 벡터 분리
        time_series_end = window_size * num_features
        time_series = state[:, :time_series_end]
        position_info = state[:, time_series_end:]
        
        # 시계열 데이터 reshape
        time_series = time_series.view(batch_size, window_size, num_features)
        
        # LSTM 처리
        lstm_out, _ = self.lstm(time_series)
        lstm_out = lstm_out[:, -1, :]
        
        # 포지션 정보와 결합
        combined = torch.cat([lstm_out, position_info], dim=1)
        
        # MLP 처리
        hidden = self.mlp(combined)
        
        # 상태 가치
        value = self.value_head(hidden)
        
        return value

