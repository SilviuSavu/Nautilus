"""
Reinforcement Learning Agents for Trading Strategy Optimization
Implementation of DQN, PPO, and A3C algorithms for autonomous trading
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging
import asyncio
from abc import ABC, abstractmethod
import multiprocessing as mp
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class TradingAction:
    """Represents a trading action"""
    action_type: str  # BUY, SELL, HOLD
    quantity: float
    price: float
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class MarketState:
    """Represents the current market state"""
    prices: np.ndarray
    volumes: np.ndarray
    indicators: np.ndarray
    portfolio: Dict[str, float]
    timestamp: datetime
    market_regime: str
    volatility: float


class TradingEnvironment(gym.Env):
    """
    Gymnasium environment for training RL trading agents
    """
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000.0,
                 transaction_cost: float = 0.001, lookback_window: int = 60):
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        # Current position and step
        self.current_step = lookback_window
        self.position = 0.0  # Current position size
        self.portfolio_value = initial_balance
        
        # Define action space: [0: Hold, 1: Buy, 2: Sell]
        # Each action can have different intensities [0.1, 0.5, 1.0]
        self.action_space = spaces.Discrete(9)  # 3 actions × 3 intensities
        
        # Define observation space
        n_features = 20  # price, volume, technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback_window, n_features), 
            dtype=np.float32
        )
        
        # Performance tracking
        self.episode_returns = []
        self.max_drawdown = 0.0
        self.trades_count = 0
        self.winning_trades = 0
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.current_balance = self.initial_balance
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.trades_count = 0
        self.winning_trades = 0
        
        # Reset performance metrics
        self.episode_returns = []
        self.max_drawdown = 0.0
        
        return self._get_observation(), {}
    
    def step(self, action: int):
        """Execute one step in the environment"""
        # Decode action
        action_type = action // 3  # 0: Hold, 1: Buy, 2: Sell
        intensity = (action % 3 + 1) * 0.33  # 0.33, 0.66, 1.0
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Calculate reward before action
        previous_value = self.portfolio_value
        
        # Execute action
        if action_type == 1:  # Buy
            self._execute_buy(current_price, intensity)
        elif action_type == 2:  # Sell
            self._execute_sell(current_price, intensity)
        # Hold: no action needed
        
        # Update portfolio value
        if self.position > 0:
            self.portfolio_value = self.current_balance + (self.position * current_price)
        else:
            self.portfolio_value = self.current_balance
        
        # Calculate reward
        reward = self._calculate_reward(previous_value, self.portfolio_value)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1 or 
                self.portfolio_value <= self.initial_balance * 0.1)
        
        # Get next observation
        observation = self._get_observation() if not done else None
        
        # Calculate info
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'current_balance': self.current_balance,
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
            'trades_count': self.trades_count,
            'win_rate': self.winning_trades / max(self.trades_count, 1)
        }
        
        return observation, reward, done, False, info
    
    def _execute_buy(self, price: float, intensity: float):
        """Execute a buy order"""
        # Calculate maximum position size based on available balance
        max_position = (self.current_balance * intensity) / price
        
        if max_position > 0:
            transaction_cost = max_position * price * self.transaction_cost
            
            if self.current_balance >= (max_position * price + transaction_cost):
                self.position += max_position
                self.current_balance -= (max_position * price + transaction_cost)
                self.trades_count += 1
    
    def _execute_sell(self, price: float, intensity: float):
        """Execute a sell order"""
        if self.position > 0:
            sell_amount = min(self.position, self.position * intensity)
            transaction_cost = sell_amount * price * self.transaction_cost
            
            proceeds = sell_amount * price - transaction_cost
            self.current_balance += proceeds
            self.position -= sell_amount
            self.trades_count += 1
            
            # Check if this was a winning trade
            # Simplified: if we made money, it's winning
            if proceeds > 0:
                self.winning_trades += 1
    
    def _calculate_reward(self, previous_value: float, current_value: float) -> float:
        """Calculate the reward for the current step"""
        # Basic return-based reward
        basic_reward = (current_value - previous_value) / previous_value
        
        # Risk-adjusted reward (penalize high drawdown)
        drawdown_penalty = -abs(min(0, basic_reward)) * 0.5
        
        # Activity bonus (slight preference for action over inaction)
        activity_bonus = 0.001 if self.position > 0 else 0
        
        # Combine components
        total_reward = basic_reward + drawdown_penalty + activity_bonus
        
        return np.clip(total_reward, -1.0, 1.0)
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation"""
        if self.current_step >= len(self.data):
            return np.zeros((self.lookback_window, 20), dtype=np.float32)
        
        # Get historical data
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Create feature matrix
        features = []
        
        for _, row in window_data.iterrows():
            row_features = [
                row['close'], row['open'], row['high'], row['low'], row['volume'],
                row.get('sma_5', 0), row.get('sma_20', 0), row.get('ema_12', 0),
                row.get('rsi', 0), row.get('macd', 0), row.get('bollinger_upper', 0),
                row.get('bollinger_lower', 0), row.get('atr', 0), row.get('vix', 0),
                self.position / 1000,  # Normalized position
                self.current_balance / self.initial_balance,  # Cash ratio
                self.portfolio_value / self.initial_balance,  # Portfolio ratio
                self.trades_count / 100,  # Normalized trade count
                self.winning_trades / max(self.trades_count, 1),  # Win rate
                (self.portfolio_value - self.initial_balance) / self.initial_balance  # Total return
            ]
            features.append(row_features)
        
        # Pad if necessary
        while len(features) < self.lookback_window:
            features.insert(0, [0.0] * 20)
        
        return np.array(features, dtype=np.float32)


class TradingRLAgent(ABC):
    """Abstract base class for reinforcement learning trading agents"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.model = None
        self.training_history = []
        self.performance_metrics = {}
        
    @abstractmethod
    def train(self, env: TradingEnvironment, timesteps: int) -> None:
        """Train the RL agent"""
        pass
    
    @abstractmethod
    def predict(self, state: MarketState) -> TradingAction:
        """Make a trading decision based on current state"""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the trained model"""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load a trained model"""
        pass


class DQNAgent(TradingRLAgent):
    """Deep Q-Network agent for trading"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DQN_Agent", config)
        
        # DQN specific parameters
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.buffer_size = config.get('buffer_size', 1000000)
        self.learning_starts = config.get('learning_starts', 50000)
        self.batch_size = config.get('batch_size', 32)
        self.tau = config.get('tau', 1.0)
        self.gamma = config.get('gamma', 0.99)
        self.exploration_fraction = config.get('exploration_fraction', 0.1)
        self.exploration_initial_eps = config.get('exploration_initial_eps', 1.0)
        self.exploration_final_eps = config.get('exploration_final_eps', 0.05)
    
    def train(self, env: TradingEnvironment, timesteps: int = 100000) -> None:
        """Train the DQN agent"""
        logger.info(f"Training DQN agent for {timesteps} timesteps")
        
        # Create DQN model
        self.model = DQN(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            exploration_fraction=self.exploration_fraction,
            exploration_initial_eps=self.exploration_initial_eps,
            exploration_final_eps=self.exploration_final_eps,
            verbose=1,
            tensorboard_log="./tensorboard_logs/dqn/"
        )
        
        # Train the model
        self.model.learn(
            total_timesteps=timesteps,
            callback=DQNTradingCallback(self)
        )
        
        logger.info("DQN training completed")
    
    def predict(self, state: MarketState) -> TradingAction:
        """Make a trading prediction"""
        if self.model is None:
            logger.warning("Model not trained. Returning HOLD action.")
            return TradingAction("HOLD", 0.0, 0.0, 0.0, {})
        
        # Convert market state to observation format
        obs = self._state_to_observation(state)
        
        # Get action and confidence
        action, _states = self.model.predict(obs, deterministic=True)
        
        # Convert action to trading action
        return self._action_to_trading_action(action, state)
    
    def save_model(self, path: str) -> None:
        """Save the DQN model"""
        if self.model:
            self.model.save(path)
            logger.info(f"DQN model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load the DQN model"""
        self.model = DQN.load(path)
        logger.info(f"DQN model loaded from {path}")
    
    def _state_to_observation(self, state: MarketState) -> np.ndarray:
        """Convert MarketState to observation array"""
        # This is a simplified conversion - in practice, you'd need to
        # match the exact format expected by the trained model
        observation = np.concatenate([
            state.prices[-60:],  # Last 60 prices
            state.volumes[-60:],  # Last 60 volumes
            state.indicators  # Technical indicators
        ])
        return observation.reshape(1, -1)
    
    def _action_to_trading_action(self, action: int, state: MarketState) -> TradingAction:
        """Convert RL action to TradingAction"""
        action_type = action // 3
        intensity = (action % 3 + 1) * 0.33
        
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action_name = action_map.get(action_type, "HOLD")
        
        # Calculate quantity based on portfolio and intensity
        current_price = state.prices[-1]
        quantity = intensity * 1000  # Simplified quantity calculation
        
        return TradingAction(
            action_type=action_name,
            quantity=quantity,
            price=current_price,
            confidence=0.8,  # Default confidence
            metadata={"intensity": intensity, "rl_action": action}
        )


class PPOAgent(TradingRLAgent):
    """Proximal Policy Optimization agent for trading"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("PPO_Agent", config)
        
        # PPO specific parameters
        self.learning_rate = config.get('learning_rate', 0.0003)
        self.n_steps = config.get('n_steps', 2048)
        self.batch_size = config.get('batch_size', 64)
        self.n_epochs = config.get('n_epochs', 10)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.ent_coef = config.get('ent_coef', 0.0)
    
    def train(self, env: TradingEnvironment, timesteps: int = 100000) -> None:
        """Train the PPO agent"""
        logger.info(f"Training PPO agent for {timesteps} timesteps")
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            verbose=1,
            tensorboard_log="./tensorboard_logs/ppo/"
        )
        
        # Train the model
        self.model.learn(
            total_timesteps=timesteps,
            callback=PPOTradingCallback(self)
        )
        
        logger.info("PPO training completed")
    
    def predict(self, state: MarketState) -> TradingAction:
        """Make a trading prediction using PPO"""
        if self.model is None:
            logger.warning("Model not trained. Returning HOLD action.")
            return TradingAction("HOLD", 0.0, 0.0, 0.0, {})
        
        # Convert market state to observation format
        obs = self._state_to_observation(state)
        
        # Get action and confidence
        action, _states = self.model.predict(obs, deterministic=False)
        
        # Convert action to trading action
        return self._action_to_trading_action(action, state)
    
    def save_model(self, path: str) -> None:
        """Save the PPO model"""
        if self.model:
            self.model.save(path)
            logger.info(f"PPO model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load the PPO model"""
        self.model = PPO.load(path)
        logger.info(f"PPO model loaded from {path}")
    
    def _state_to_observation(self, state: MarketState) -> np.ndarray:
        """Convert MarketState to observation array"""
        # Create observation matching training format
        observation = np.array([
            state.prices[-1],  # Current price
            state.volumes[-1],  # Current volume
            *state.indicators,  # Technical indicators
            state.volatility,  # Current volatility
        ])
        return observation.reshape(1, -1)
    
    def _action_to_trading_action(self, action: int, state: MarketState) -> TradingAction:
        """Convert RL action to TradingAction"""
        action_type = action // 3
        intensity = (action % 3 + 1) * 0.33
        
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action_name = action_map.get(action_type, "HOLD")
        
        current_price = state.prices[-1]
        quantity = intensity * 500  # Adjusted quantity for PPO
        
        return TradingAction(
            action_type=action_name,
            quantity=quantity,
            price=current_price,
            confidence=0.75,
            metadata={"intensity": intensity, "rl_action": action}
        )


class A3CAgent(TradingRLAgent):
    """Asynchronous Advantage Actor-Critic agent for trading"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("A3C_Agent", config)
        
        # A3C specific parameters (using A2C as stable-baselines3 implementation)
        self.learning_rate = config.get('learning_rate', 0.0007)
        self.n_steps = config.get('n_steps', 5)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 1.0)
        self.ent_coef = config.get('ent_coef', 0.01)
        self.vf_coef = config.get('vf_coef', 0.25)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
    
    def train(self, env: TradingEnvironment, timesteps: int = 100000) -> None:
        """Train the A3C agent"""
        logger.info(f"Training A3C agent for {timesteps} timesteps")
        
        # Create A2C model (closest to A3C in stable-baselines3)
        self.model = A2C(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            verbose=1,
            tensorboard_log="./tensorboard_logs/a3c/"
        )
        
        # Train the model
        self.model.learn(
            total_timesteps=timesteps,
            callback=A3CTradingCallback(self)
        )
        
        logger.info("A3C training completed")
    
    def predict(self, state: MarketState) -> TradingAction:
        """Make a trading prediction using A3C"""
        if self.model is None:
            logger.warning("Model not trained. Returning HOLD action.")
            return TradingAction("HOLD", 0.0, 0.0, 0.0, {})
        
        # Convert market state to observation format
        obs = self._state_to_observation(state)
        
        # Get action
        action, _states = self.model.predict(obs, deterministic=False)
        
        # Convert action to trading action
        return self._action_to_trading_action(action, state)
    
    def save_model(self, path: str) -> None:
        """Save the A3C model"""
        if self.model:
            self.model.save(path)
            logger.info(f"A3C model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load the A3C model"""
        self.model = A2C.load(path)
        logger.info(f"A3C model loaded from {path}")
    
    def _state_to_observation(self, state: MarketState) -> np.ndarray:
        """Convert MarketState to observation array"""
        observation = np.array([
            state.prices[-1],
            state.volumes[-1],
            *state.indicators[:10],  # First 10 indicators
            state.volatility
        ])
        return observation.reshape(1, -1)
    
    def _action_to_trading_action(self, action: int, state: MarketState) -> TradingAction:
        """Convert RL action to TradingAction"""
        action_type = action // 3
        intensity = (action % 3 + 1) * 0.33
        
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        action_name = action_map.get(action_type, "HOLD")
        
        current_price = state.prices[-1]
        quantity = intensity * 750  # A3C specific quantity
        
        return TradingAction(
            action_type=action_name,
            quantity=quantity,
            price=current_price,
            confidence=0.7,
            metadata={"intensity": intensity, "rl_action": action}
        )


class DQNTradingCallback(BaseCallback):
    """Custom callback for DQN training progress"""
    
    def __init__(self, agent: DQNAgent, verbose=0):
        super().__init__(verbose)
        self.agent = agent
        
    def _on_step(self) -> bool:
        # Log training progress every 1000 steps
        if self.n_calls % 1000 == 0:
            logger.info(f"DQN Training Step: {self.n_calls}")
            
        return True


class PPOTradingCallback(BaseCallback):
    """Custom callback for PPO training progress"""
    
    def __init__(self, agent: PPOAgent, verbose=0):
        super().__init__(verbose)
        self.agent = agent
        
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            logger.info(f"PPO Training Step: {self.n_calls}")
            
        return True


class A3CTradingCallback(BaseCallback):
    """Custom callback for A3C training progress"""
    
    def __init__(self, agent: A3CAgent, verbose=0):
        super().__init__(verbose)
        self.agent = agent
        
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            logger.info(f"A3C Training Step: {self.n_calls}")
            
        return True


class RLAgentManager:
    """Manager for multiple RL agents with ensemble capabilities"""
    
    def __init__(self):
        self.agents: Dict[str, TradingRLAgent] = {}
        self.performance_history = {}
        
    def add_agent(self, agent: TradingRLAgent) -> None:
        """Add an RL agent to the manager"""
        self.agents[agent.name] = agent
        self.performance_history[agent.name] = []
        logger.info(f"Added agent: {agent.name}")
        
    def train_all_agents(self, env: TradingEnvironment, timesteps: int) -> None:
        """Train all registered agents"""
        for name, agent in self.agents.items():
            logger.info(f"Training agent: {name}")
            agent.train(env, timesteps)
            
    def get_ensemble_prediction(self, state: MarketState) -> TradingAction:
        """Get ensemble prediction from all agents"""
        predictions = []
        
        for agent in self.agents.values():
            try:
                prediction = agent.predict(state)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error getting prediction from {agent.name}: {e}")
                
        if not predictions:
            return TradingAction("HOLD", 0.0, 0.0, 0.0, {})
            
        # Simple ensemble: majority vote with confidence weighting
        buy_votes = sum(1 for p in predictions if p.action_type == "BUY")
        sell_votes = sum(1 for p in predictions if p.action_type == "SELL")
        hold_votes = sum(1 for p in predictions if p.action_type == "HOLD")
        
        total_votes = len(predictions)
        
        if buy_votes > sell_votes and buy_votes > hold_votes:
            action_type = "BUY"
            confidence = buy_votes / total_votes
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            action_type = "SELL"
            confidence = sell_votes / total_votes
        else:
            action_type = "HOLD"
            confidence = hold_votes / total_votes
            
        # Average quantity and price
        avg_quantity = np.mean([p.quantity for p in predictions if p.action_type == action_type])
        avg_price = np.mean([p.price for p in predictions])
        
        return TradingAction(
            action_type=action_type,
            quantity=avg_quantity,
            price=avg_price,
            confidence=confidence,
            metadata={"ensemble_size": total_votes, "individual_predictions": predictions}
        )
        
    def evaluate_agent_performance(self, agent_name: str, 
                                  test_env: TradingEnvironment) -> Dict[str, float]:
        """Evaluate individual agent performance"""
        if agent_name not in self.agents:
            logger.error(f"Agent {agent_name} not found")
            return {}
            
        agent = self.agents[agent_name]
        
        # Run evaluation episode
        obs, _ = test_env.reset()
        total_reward = 0.0
        steps = 0
        
        while True:
            # Convert observation to MarketState (simplified)
            state = MarketState(
                prices=obs[:, 0],
                volumes=obs[:, 1],
                indicators=obs[0, 2:],
                portfolio={},
                timestamp=datetime.now(),
                market_regime="unknown",
                volatility=0.02
            )
            
            action = agent.predict(state)
            
            # Convert TradingAction back to environment action
            env_action = self._trading_action_to_env_action(action)
            
            obs, reward, done, truncated, info = test_env.step(env_action)
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
                
        # Calculate performance metrics
        performance = {
            'total_return': info.get('total_return', 0.0),
            'win_rate': info.get('win_rate', 0.0),
            'trades_count': info.get('trades_count', 0),
            'portfolio_value': info.get('portfolio_value', 0.0),
            'avg_reward_per_step': total_reward / max(steps, 1)
        }
        
        self.performance_history[agent_name].append(performance)
        return performance
    
    def _trading_action_to_env_action(self, trading_action: TradingAction) -> int:
        """Convert TradingAction back to environment action integer"""
        action_map = {"HOLD": 0, "BUY": 1, "SELL": 2}
        base_action = action_map.get(trading_action.action_type, 0)
        
        # Estimate intensity from metadata
        intensity = trading_action.metadata.get('intensity', 0.33)
        intensity_level = min(2, int(intensity / 0.34))  # 0, 1, or 2
        
        return base_action * 3 + intensity_level
        
    def get_best_agent(self) -> Optional[str]:
        """Get the name of the best performing agent"""
        if not self.performance_history:
            return None
            
        avg_returns = {}
        for agent_name, history in self.performance_history.items():
            if history:
                avg_returns[agent_name] = np.mean([h['total_return'] for h in history])
                
        return max(avg_returns.items(), key=lambda x: x[1])[0] if avg_returns else None


# Example usage and testing functions
def create_sample_trading_data(n_days: int = 1000) -> pd.DataFrame:
    """Create sample trading data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate synthetic price data with trend and noise
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(15, 0.5, n_days),
        'sma_5': pd.Series(prices).rolling(5).mean().fillna(base_price),
        'sma_20': pd.Series(prices).rolling(20).mean().fillna(base_price),
        'ema_12': pd.Series(prices).ewm(span=12).mean(),
        'rsi': 50 + np.random.normal(0, 15, n_days),
        'macd': np.random.normal(0, 2, n_days),
        'bollinger_upper': [p * 1.02 for p in prices],
        'bollinger_lower': [p * 0.98 for p in prices],
        'atr': np.random.uniform(1, 5, n_days),
        'vix': 15 + np.random.exponential(5, n_days)
    })
    
    return data


async def train_rl_agents_example():
    """Example function to train multiple RL agents"""
    logger.info("Starting RL agents training example")
    
    # Create sample data
    data = create_sample_trading_data(1000)
    
    # Create environment
    env = TradingEnvironment(data)
    
    # Create agent manager
    manager = RLAgentManager()
    
    # Create agents with different configurations
    dqn_config = {
        'learning_rate': 0.0001,
        'buffer_size': 50000,
        'batch_size': 32
    }
    
    ppo_config = {
        'learning_rate': 0.0003,
        'n_steps': 1024,
        'batch_size': 64
    }
    
    a3c_config = {
        'learning_rate': 0.0007,
        'n_steps': 5
    }
    
    # Add agents to manager
    manager.add_agent(DQNAgent(dqn_config))
    manager.add_agent(PPOAgent(ppo_config))
    manager.add_agent(A3CAgent(a3c_config))
    
    # Train all agents
    timesteps = 50000  # Reduced for example
    manager.train_all_agents(env, timesteps)
    
    # Evaluate performance
    test_data = create_sample_trading_data(200)
    test_env = TradingEnvironment(test_data)
    
    for agent_name in manager.agents.keys():
        performance = manager.evaluate_agent_performance(agent_name, test_env)
        logger.info(f"{agent_name} performance: {performance}")
    
    # Get best agent
    best_agent = manager.get_best_agent()
    logger.info(f"Best performing agent: {best_agent}")
    
    # Test ensemble prediction
    sample_state = MarketState(
        prices=np.array([100.0, 101.0, 99.5, 102.0]),
        volumes=np.array([1000000, 1100000, 950000, 1200000]),
        indicators=np.array([50.0, 0.5, 101.0, 99.0, 2.5, 20.0]),
        portfolio={'cash': 10000, 'position': 100},
        timestamp=datetime.now(),
        market_regime='normal',
        volatility=0.02
    )
    
    ensemble_prediction = manager.get_ensemble_prediction(sample_state)
    logger.info(f"Ensemble prediction: {ensemble_prediction}")
    
    logger.info("RL agents training example completed")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    asyncio.run(train_rl_agents_example())