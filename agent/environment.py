"""
거래 환경 클래스
"""
import traceback
import numpy as np
import pandas as pd


class TradingEnvironment:
    """거래 시뮬레이션 환경"""
    
    def __init__(self, data_df, config, original_prices=None, original_df=None):
        """
        Args:
            data_df: 정규화된 데이터프레임
            config: 설정 객체
            original_prices: 원본 가격 데이터 (정규화 전, 선택적)
            original_df: 원본 데이터프레임 (타임스탬프 포함, 선택적)
        """
        self.data_df = data_df.reset_index(drop=True)
        self.config = config
        # 원본 가격 저장 (정규화 전)
        if original_prices is not None:
            self.original_prices = original_prices.reset_index(drop=True)
        else:
            # 원본 가격이 없으면 정규화된 데이터에서 추정 (비권장)
            self.original_prices = None

        # 원본 데이터프레임 저장 (타임스탬프 및 차트 데이터용)
        if original_df is not None:
            self.original_df = original_df.reset_index()
        else:
            self.original_df = None

        # 슬라이딩 윈도우를 위한 전역 오프셋
        self.global_offset = 0
        self.total_data_length = len(self.data_df)

        self.reset()
    
    def reset(self, use_sliding_window=True):
        """
        환경 초기화

        Args:
            use_sliding_window: True면 슬라이딩 윈도우 방식, False면 항상 처음부터 시작
        """
        if use_sliding_window:
            # 슬라이딩 윈도우: 이전 에피소드가 끝난 지점부터 시작
            # 만약 데이터 끝에 도달했으면 처음부터 다시 시작
            if self.global_offset + self.config.MAX_STEPS_PER_EPISODE >= self.total_data_length:
                self.global_offset = 0
                print(f"  [환경] 데이터 끝 도달, 처음부터 재시작 (전체 데이터 길이: {self.total_data_length})")

            self.current_step = self.global_offset + self.config.STATE_WINDOW
        else:
            # 기존 방식: 항상 처음부터
            self.current_step = self.config.STATE_WINDOW

        self.cash = self.config.INITIAL_CAPITAL
        self.coin_amount = 0.0
        self.has_position = False
        self.entry_price = 0.0
        self.total_trades = 0
        self.total_profit = 0.0
        self.initial_portfolio_value = self.config.INITIAL_CAPITAL
        self.max_portfolio_value = self.config.INITIAL_CAPITAL
        self.max_drawdown = 0.0

        # 거래 기록 (모든 액션 기록)
        self.trade_history = []

        # 대기 페널티 추적
        self.consecutive_wait_steps = 0

        # 초기 가격 저장 (Buy & Hold 비교용)
        self.initial_price = self._get_current_price()

        # 초기 상태 반환
        state = self._get_state()
        return state

    def update_global_offset(self):
        """에피소드 종료 후 전역 오프셋 업데이트 (슬라이딩 윈도우용)"""
        episode_length = self.current_step - self.global_offset - self.config.STATE_WINDOW
        self.global_offset += episode_length

        # 데이터 끝에 도달 체크
        if self.global_offset >= self.total_data_length - self.config.MAX_STEPS_PER_EPISODE:
            self.global_offset = 0
    
    def _get_state(self):
        """현재 상태 벡터 생성"""
        if self.current_step < self.config.STATE_WINDOW:
            # 초기 단계: 패딩 사용
            available_data = self.data_df.iloc[:self.current_step + 1]
            padding_needed = self.config.STATE_WINDOW - len(available_data)
            
            if padding_needed > 0:
                # 첫 번째 행을 복사하여 패딩
                first_row = available_data.iloc[0:1]
                padding = pd.concat([first_row] * padding_needed, ignore_index=True)
                window_data = pd.concat([padding, available_data], ignore_index=True)
            else:
                window_data = available_data.iloc[-self.config.STATE_WINDOW:]
        else:
            window_data = self.data_df.iloc[
                self.current_step - self.config.STATE_WINDOW + 1:self.current_step + 1
            ]
        
        # 가격 데이터 추출 (정규화된 값)
        features = []
        for col in self.config.FEATURES:
            if col in window_data.columns:
                values = window_data[col].values
                features.extend(values)
        
        # 포지션 정보
        current_price = self._get_current_price()
        portfolio_value = self._calculate_portfolio_value(current_price)
        
        features.append(1.0 if self.has_position else 0.0)  # 보유 여부
        if portfolio_value > 0:
            features.append(self.coin_amount * current_price / portfolio_value)  # 포지션 비율
        else:
            features.append(0.0)
        features.append(self.cash / self.config.INITIAL_CAPITAL)  # 현금 비율
        features.append(portfolio_value / self.config.INITIAL_CAPITAL)  # 총 자산 비율
        
        # ========== 추가 정보 ==========
        # 미실현 수익률 (매수 가격 대비 현재 가격)
        if self.has_position and self.entry_price > 0:
            unrealized_return = (current_price - self.entry_price) / (self.entry_price + 1e-8)
            features.append(unrealized_return)  # 미실현 수익률
        else:
            features.append(0.0)
        
        # 최고점 대비 현재 가격 비율 (최고점 근처 판단용)
        if self.max_portfolio_value > self.initial_portfolio_value:
            max_price_ratio = (portfolio_value - self.initial_portfolio_value) / (self.max_portfolio_value - self.initial_portfolio_value + 1e-8)
            features.append(max_price_ratio)  # 최고점 대비 비율
        else:
            features.append(0.0)
        
        state = np.array(features, dtype=np.float32)
        return state
    
    def _get_current_price(self):
        """현재 틱의 오픈가 반환 (원본 가격)"""
        if self.original_prices is not None:
            return float(self.original_prices.iloc[self.current_step]['open'])
        else:
            # 원본 가격이 없으면 정규화된 값 사용 (비권장, 스케일러 필요)
            # 임시로 기본값 반환
            return 100000.0  # 임시 값 (실제로는 스케일러로 역변환 필요)
    
    def _get_price_at_step(self, step):
        """특정 스텝의 가격 반환"""
        if self.original_prices is not None and step < len(self.original_prices):
            return float(self.original_prices.iloc[step]['open'])
        else:
            return self._get_current_price()
    
    def _calculate_portfolio_value(self, price):
        """포트폴리오 가치 계산"""
        return self.cash + self.coin_amount * price

    def get_action_mask(self):
        """
        현재 상태에서 유효한 액션 마스크 반환
        Returns:
            mask: [1, 1, 1, 1] 형태의 마스크 (1: 유효, 0: 무효)
                   [대기, 매수, 홀드, 매도]
        """
        if not self.has_position:
            # 코인 미보유: 대기(0), 매수(1)만 가능
            return np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        else:
            # 코인 보유: 홀드(2), 매도(3)만 가능
            return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    
    def step(self, action):
        """
        액션 실행 및 다음 상태 반환

        Args:
            action: 액션 (0: 대기, 1: 매수, 2: 홀드, 3: 매도)

        Returns:
            next_state: 다음 상태
            reward: 보상
            done: 종료 여부
            info: 추가 정보
        """
        # 불가능한 액션 체크
        invalid_action_penalty = 0.0
        if not self.has_position:
            # 코인 미보유 상태에서 홀드나 매도는 불가능
            if action == 2 or action == 3:
                invalid_action_penalty = -0.1
                action = 0  # 대기로 강제 변경
        else:
            # 코인 보유 상태에서 대기나 매수는 불가능
            if action == 0 or action == 1:
                invalid_action_penalty = -0.1
                action = 2  # 홀드로 강제 변경
        
        current_price = self._get_current_price()
        prev_portfolio_value = self._calculate_portfolio_value(current_price)
        
        # 매도 전 정보 저장 (실현 수익 계산용)
        was_selling = False
        realized_return_on_sell = 0.0
        if action == 3 and self.has_position:
            # 매도 전 미실현 수익률 계산
            if self.entry_price > 0:
                unrealized_return_before = (current_price - self.entry_price) / (self.entry_price + 1e-8)
                was_selling = True
        
        # 액션 실행
        if action == 1:  # 매수
            self._apply_buy(current_price)
        elif action == 3:  # 매도
            self._apply_sell(current_price)
            # 매도 후 실현 수익률 가져오기
            if len(self.trade_history) > 0:
                last_trade = self.trade_history[-1]
                if last_trade.get('action') == 'sell' and 'realized_return' in last_trade:
                    realized_return_on_sell = last_trade['realized_return']
        # action == 0 (대기) 또는 action == 2 (홀드)는 아무것도 하지 않음
        
        # 거래 기록 저장 (모든 액션)
        self._record_action(action, current_price, prev_portfolio_value)
        
        # 다음 스텝으로 이동
        self.current_step += 1
        
        # 다음 상태
        next_state = self._get_state()

        # 보상 계산
        new_price = self._get_current_price()
        new_portfolio_value = self._calculate_portfolio_value(new_price)

        # ========== 개선된 보상 설계 ==========

        # 1. 기본 보상: 포트폴리오 가치 변화율
        reward = (new_portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8)

        # 2. 불가능한 액션 페널티
        reward += invalid_action_penalty

        # 3. 대기 페널티 강화 (탐험 유도)
        if action == 0:  # 대기
            self.consecutive_wait_steps += 1
            # 대기에 항상 작은 페널티 부여
            reward -= 0.001
            # 과도한 대기에는 더 큰 페널티
            if self.consecutive_wait_steps > 50:
                reward -= 0.01
        else:
            self.consecutive_wait_steps = 0

        # 4. 매수 인센티브 (초기 탐험 유도)
        if action == 1:  # 매수
            reward += 0.005  # 매수 시도 자체에 작은 보상

        # 5. 홀드 인센티브 (포지션 유지 중 가격 상승 시)
        if action == 2 and self.has_position:  # 홀드
            if reward > 0:  # 가격이 올랐다면
                reward += 0.002  # 작은 보너스

        # 6. 매도 시 실현 수익 보상 (거래 완결 인센티브)
        if was_selling and realized_return_on_sell != 0.0:
            # 실현 수익이 양수면 큰 보상, 음수면 페널티
            reward += realized_return_on_sell * 1.0
        
        # 최대 낙폭 업데이트
        if new_portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = new_portfolio_value

        drawdown = (self.max_portfolio_value - new_portfolio_value) / (self.max_portfolio_value + 1e-8)
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # 위험 페널티 (최대 낙폭이 30% 초과 시)
        if self.max_drawdown > 0.3:
            reward -= 0.05

        # 종료 조건
        done = False
        if self.current_step >= len(self.data_df) - 1:
            done = True
            # 에피소드 종료 시 최종 수익률 보상
            final_return = (new_portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value
            reward += final_return * 2.0  # 최종 수익률에 가중치

            # 샤프 비율 근사 보상 (수익 대비 리스크)
            if self.max_drawdown > 0:
                risk_adjusted_reward = final_return / (self.max_drawdown + 0.1)
                reward += risk_adjusted_reward * 0.5
        
        if self.cash < 1000:  # 자본금이 너무 적으면 종료
            done = True
        
        info = {
            'portfolio_value': new_portfolio_value,
            'cash': self.cash,
            'coin_amount': self.coin_amount,
            'has_position': self.has_position,
            'total_trades': self.total_trades,
            'max_drawdown': self.max_drawdown
        }
        
        return next_state, reward, done, info
    
    def _apply_buy(self, price):
        """매수 실행"""
        if self.has_position:
            return  # 이미 보유 중이면 매수 불가
        
        # 전액 매수
        fee = self.cash * self.config.TRADING_FEE_RATE
        available_cash = self.cash - fee
        
        if available_cash > 0:
            self.coin_amount = available_cash / price
            self.cash = 0.0
            self.has_position = True
            self.entry_price = price
            self.total_trades += 1
            
            self.trade_history.append({
                'step': self.current_step,
                'action': 'buy',
                'price': price,
                'amount': self.coin_amount,
                'fee': fee
            })
    
    def _apply_sell(self, price):
        """매도 실행"""
        if not self.has_position:
            return  # 보유하지 않으면 매도 불가
        
        # 전액 매도
        coin_amount_before = self.coin_amount
        sell_value = self.coin_amount * price
        fee = sell_value * self.config.TRADING_FEE_RATE
        self.cash = sell_value - fee
        self.coin_amount = 0.0
        self.has_position = False
        
        # 수익 계산 (매도 전 coin_amount 사용)
        buy_cost = self.entry_price * coin_amount_before
        profit = sell_value - fee - buy_cost
        realized_return = profit / (buy_cost + 1e-8) if buy_cost > 0 else 0.0  # 실현 수익률
        
        self.total_profit += profit
        self.total_trades += 1
        
        self.trade_history.append({
            'step': self.current_step,
            'action': 'sell',
            'price': price,
            'amount': coin_amount_before,
            'fee': fee,
            'profit': profit,
            'realized_return': realized_return  # 실현 수익률 추가
        })
    
    def _record_action(self, action, price, prev_portfolio_value):
        """액션 기록 저장"""
        action_names = {0: '대기', 1: '매수', 2: '홀드', 3: '매도'}
        action_name = action_names.get(action, '알수없음')
        
        # 현재 시점의 차트 데이터 가져오기
        if self.original_df is not None and self.current_step < len(self.original_df):
            chart_data = self.original_df.iloc[self.current_step]
            # 타임스탬프 가져오기 (reset_index()로 인해 'index' 컬럼이 있거나 원본 인덱스 사용)
            if 'index' in chart_data.index or 'index' in self.original_df.columns:
                timestamp = chart_data.get('index', self.current_step)
            elif len(self.original_df.index) > 0:
                # 원본 인덱스가 타임스탬프인 경우
                timestamp = self.original_df.index[self.current_step]
            else:
                timestamp = self.current_step
            
            open_price = float(chart_data.get('open', price))
            high_price = float(chart_data.get('high', price))
            low_price = float(chart_data.get('low', price))
            close_price = float(chart_data.get('close', price))
            volume = float(chart_data.get('volume', 0.0))
            ma5 = float(chart_data.get('ma5', 0.0))
            ma7 = float(chart_data.get('ma7', 0.0))
            ma10 = float(chart_data.get('ma10', 0.0))
            cat01 = float(chart_data.get('cat01', 0.0))
        else:
            timestamp = self.current_step
            open_price = price
            high_price = price
            low_price = price
            close_price = price
            volume = 0.0
            ma5 = 0.0
            ma7 = 0.0
            ma10 = 0.0
            cat01 = 0.0
        
        # 현재 포트폴리오 가치
        current_portfolio_value = self._calculate_portfolio_value(price)
        
        # 거래 기록 추가
        record = {
            'timestamp': timestamp,
            'step': self.current_step,
            'action': action_name,
            'action_code': action,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'ma5': ma5,
            'ma7': ma7,
            'ma10': ma10,
            'cat01': cat01,
            'price': price,
            'cash': self.cash,
            'coin_amount': self.coin_amount,
            'has_position': self.has_position,
            'portfolio_value': current_portfolio_value,
            'portfolio_change': current_portfolio_value - prev_portfolio_value
        }
        
        self.trade_history.append(record)

