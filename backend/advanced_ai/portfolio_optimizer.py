"""
Deep Reinforcement Learning Portfolio Optimizer
Implementation of advanced portfolio optimization using deep RL and quantum-inspired algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
from abc import ABC, abstractmethod
import cvxpy as cp
from scipy.optimize import minimize, differential_evolution
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Current portfolio state for RL agent"""
    weights: np.ndarray           # Current portfolio weights
    returns: np.ndarray          # Historical returns
    prices: np.ndarray           # Current prices
    volatilities: np.ndarray     # Asset volatilities
    correlations: np.ndarray     # Correlation matrix
    market_features: np.ndarray  # Additional market features
    cash: float                  # Available cash
    timestamp: datetime
    transaction_costs: float = 0.001


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    optimal_weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    confidence_score: float
    rebalancing_frequency: str
    optimization_method: str
    constraints_satisfied: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLAction:
    """Reinforcement learning action for portfolio optimization"""
    weight_changes: np.ndarray   # Change in weights
    rebalance_flag: bool         # Whether to rebalance
    confidence: float            # Action confidence
    metadata: Dict[str, Any] = field(default_factory=dict)


class PortfolioEnvironment:
    """Portfolio optimization environment for RL training"""
    
    def __init__(self, returns_data: pd.DataFrame, initial_capital: float = 1000000.0,
                 transaction_cost: float = 0.001, lookback_window: int = 252):
        
        self.returns_data = returns_data
        self.assets = returns_data.columns.tolist()
        self.num_assets = len(self.assets)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        # Current state
        self.current_step = lookback_window
        self.portfolio_weights = np.ones(self.num_assets) / self.num_assets
        self.portfolio_value = initial_capital
        self.cash = 0.0
        
        # Performance tracking
        self.portfolio_history = []
        self.weight_history = []
        self.returns_history = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_value = initial_capital
        
    def reset(self) -> PortfolioState:
        """Reset the environment to initial state"""
        self.current_step = self.lookback_window
        self.portfolio_weights = np.ones(self.num_assets) / self.num_assets
        self.portfolio_value = self.initial_capital
        self.cash = 0.0
        
        self.portfolio_history = []
        self.weight_history = []
        self.returns_history = []
        self.max_drawdown = 0.0
        self.peak_value = self.initial_capital
        
        return self._get_current_state()
    
    def step(self, action: np.ndarray) -> Tuple[PortfolioState, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        # Decode action (weight changes)
        new_weights = self.portfolio_weights + action
        new_weights = np.clip(new_weights, 0, 1)
        new_weights = new_weights / np.sum(new_weights)  # Normalize
        
        # Calculate transaction costs
        weight_changes = np.abs(new_weights - self.portfolio_weights)
        transaction_costs = np.sum(weight_changes) * self.transaction_cost * self.portfolio_value
        
        # Update weights
        old_weights = self.portfolio_weights.copy()
        self.portfolio_weights = new_weights
        
        # Get returns for this period
        if self.current_step < len(self.returns_data):
            period_returns = self.returns_data.iloc[self.current_step].values
            portfolio_return = np.sum(self.portfolio_weights * period_returns)
        else:
            portfolio_return = 0.0
        
        # Update portfolio value
        self.portfolio_value = self.portfolio_value * (1 + portfolio_return) - transaction_costs
        
        # Track performance
        self.portfolio_history.append(self.portfolio_value)
        self.weight_history.append(self.portfolio_weights.copy())
        self.returns_history.append(portfolio_return)
        
        # Update risk metrics
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_return, transaction_costs, old_weights, new_weights)
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = (self.current_step >= len(self.returns_data) or 
                self.portfolio_value <= self.initial_capital * 0.1)
        
        # Create info dictionary
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return,
            'transaction_costs': transaction_costs,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'weights': self.portfolio_weights.copy()
        }
        
        return self._get_current_state(), reward, done, info
    
    def _get_current_state(self) -> PortfolioState:
        """Get current portfolio state"""
        # Historical returns window
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        hist_returns = self.returns_data.iloc[start_idx:end_idx].values
        
        # Calculate features
        mean_returns = np.mean(hist_returns, axis=0)
        volatilities = np.std(hist_returns, axis=0)
        correlations = np.corrcoef(hist_returns.T)
        
        # Market features
        if len(self.returns_history) >= 10:
            recent_portfolio_returns = np.array(self.returns_history[-10:])
            market_momentum = np.mean(recent_portfolio_returns)
            market_volatility = np.std(recent_portfolio_returns)
        else:
            market_momentum = 0.0
            market_volatility = 0.02
        
        market_features = np.array([
            market_momentum,
            market_volatility,
            self.current_step / len(self.returns_data),  # Time progress
            self.max_drawdown,
            len(self.portfolio_history) / 252 if self.portfolio_history else 0  # Time in market
        ])
        
        return PortfolioState(
            weights=self.portfolio_weights.copy(),
            returns=mean_returns,
            prices=np.ones(self.num_assets),  # Normalized prices
            volatilities=volatilities,
            correlations=correlations,
            market_features=market_features,
            cash=self.cash,
            timestamp=datetime.now(),
            transaction_costs=self.transaction_cost
        )
    
    def _calculate_reward(self, portfolio_return: float, transaction_costs: float,
                         old_weights: np.ndarray, new_weights: np.ndarray) -> float:
        """Calculate reward for the RL agent"""
        # Base reward from return
        return_reward = portfolio_return * 10  # Scale up
        
        # Penalty for transaction costs
        cost_penalty = -transaction_costs / self.portfolio_value * 100
        
        # Risk-adjusted reward (penalize high drawdown)
        risk_penalty = -self.max_drawdown * 5
        
        # Diversification bonus
        weight_entropy = -np.sum(new_weights * np.log(new_weights + 1e-8))
        diversification_bonus = weight_entropy * 0.1
        
        # Total reward
        total_reward = return_reward + cost_penalty + risk_penalty + diversification_bonus
        
        return np.clip(total_reward, -10, 10)  # Clip for stability
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history)
        excess_returns = returns - 0.02/252  # Assume 2% risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


class PortfolioRLNet(nn.Module):
    """Deep RL network for portfolio optimization"""
    
    def __init__(self, state_dim: int, num_assets: int, hidden_dim: int = 512):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        
        # State processing network
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Portfolio weights network (Actor)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_assets),
            nn.Softmax(dim=-1)
        )
        
        # Value function network (Critic)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Weight change network (for continuous action space)
        self.weight_change_net = nn.Sequential(
            nn.Linear(hidden_dim + num_assets, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_assets),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Uncertainty estimation
        self.uncertainty_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state: torch.Tensor, current_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the RL network"""
        # Encode state
        state_encoding = self.state_encoder(state)
        
        # Generate portfolio weights (direct)
        portfolio_weights = self.actor(state_encoding)
        
        # Generate weight changes (delta)
        combined_input = torch.cat([state_encoding, current_weights], dim=-1)
        weight_changes = self.weight_change_net(combined_input) * 0.1  # Limit change magnitude
        
        # Value estimation
        state_value = self.critic(state_encoding)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_net(state_encoding)
        
        return {
            'portfolio_weights': portfolio_weights,
            'weight_changes': weight_changes,
            'state_value': state_value,
            'uncertainty': uncertainty
        }


class DeepRLPortfolioOptimizer:
    """Deep reinforcement learning portfolio optimizer"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        self.num_assets = config.get('num_assets', 10)
        self.hidden_dim = config.get('hidden_dim', 512)
        self.learning_rate = config.get('learning_rate', 0.0003)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        
        # Calculate state dimension
        # weights + returns + volatilities + correlations + market_features
        correlation_size = self.num_assets * (self.num_assets - 1) // 2
        self.state_dim = self.num_assets * 3 + correlation_size + 5  # market features
        
        # Initialize networks
        self.main_net = PortfolioRLNet(self.state_dim, self.num_assets, self.hidden_dim)
        self.target_net = PortfolioRLNet(self.state_dim, self.num_assets, self.hidden_dim)
        
        # Copy weights to target network
        self.target_net.load_state_dict(self.main_net.state_dict())
        
        # Move to device
        self.main_net.to(self.device)
        self.target_net.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.main_net.parameters(), lr=self.learning_rate)
        
        # Training state
        self.training_history = []
        self.is_trained = False
        
    async def optimize_portfolio(self, state: PortfolioState) -> OptimizationResult:
        """Optimize portfolio using deep RL"""
        try:
            # Convert state to tensor
            state_tensor = self._state_to_tensor(state)
            current_weights = torch.FloatTensor(state.weights).unsqueeze(0).to(self.device)
            
            # Forward pass through network
            with torch.no_grad():
                outputs = self.main_net(state_tensor, current_weights)
            
            # Extract optimal weights
            if self.config.get('use_weight_changes', True):
                # Use weight changes approach
                weight_changes = outputs['weight_changes'].cpu().numpy()[0]
                optimal_weights = state.weights + weight_changes
                optimal_weights = np.clip(optimal_weights, 0, 1)
                optimal_weights = optimal_weights / np.sum(optimal_weights)
            else:
                # Use direct weights approach
                optimal_weights = outputs['portfolio_weights'].cpu().numpy()[0]
            
            # Calculate expected performance
            expected_return = np.sum(optimal_weights * state.returns) * 252  # Annualized
            expected_volatility = self._calculate_portfolio_volatility(optimal_weights, state)
            sharpe_ratio = (expected_return - 0.02) / expected_volatility if expected_volatility > 0 else 0
            
            # Calculate confidence
            uncertainty = outputs['uncertainty'].cpu().numpy()[0][0]
            confidence_score = 1 - uncertainty
            
            # Estimate other metrics
            var_95 = self._calculate_var(optimal_weights, state, confidence_level=0.95)
            max_drawdown = self._estimate_max_drawdown(optimal_weights, state)
            
            return OptimizationResult(
                optimal_weights=optimal_weights,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                confidence_score=confidence_score,
                rebalancing_frequency='daily',
                optimization_method='Deep RL',
                constraints_satisfied=True,
                metadata={
                    'model_uncertainty': uncertainty,
                    'state_value': outputs['state_value'].cpu().numpy()[0][0]
                }
            )
            
        except Exception as e:
            logger.error(f"Error in RL optimization: {e}")
            # Return equal-weight fallback
            equal_weights = np.ones(self.num_assets) / self.num_assets
            return OptimizationResult(
                optimal_weights=equal_weights,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                var_95=0.0,
                confidence_score=0.0,
                rebalancing_frequency='daily',
                optimization_method='Equal Weight (Fallback)',
                constraints_satisfied=True,
                metadata={'error': str(e)}
            )
    
    async def train_optimizer(self, returns_data: pd.DataFrame, 
                            episodes: int = 1000) -> Dict[str, Any]:
        """Train the deep RL optimizer"""
        logger.info(f"Training Deep RL optimizer for {episodes} episodes")
        
        # Create environment
        env = PortfolioEnvironment(returns_data)
        
        training_history = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            while True:
                # Convert state to tensor
                state_tensor = self._state_to_tensor(state)
                current_weights = torch.FloatTensor(state.weights).unsqueeze(0).to(self.device)
                
                # Get action from network
                outputs = self.main_net(state_tensor, current_weights)
                
                # Extract action (weight changes)
                weight_changes = outputs['weight_changes'].cpu().numpy()[0]
                
                # Add exploration noise
                if episode < episodes * 0.8:  # Exploration phase
                    noise = np.random.normal(0, 0.05, size=weight_changes.shape)
                    weight_changes += noise
                
                # Take step in environment
                next_state, reward, done, info = env.step(weight_changes)
                
                # Store transition for training
                if episode > 10:  # Start training after some episodes
                    loss = self._train_step(state, weight_changes, reward, next_state, done)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            # Update target network
            if episode % 10 == 0:
                self._soft_update_target()
            
            # Record training metrics
            episode_metrics = {
                'episode': episode,
                'reward': episode_reward,
                'length': episode_length,
                'final_value': info.get('portfolio_value', 0),
                'sharpe_ratio': info.get('sharpe_ratio', 0),
                'max_drawdown': info.get('max_drawdown', 0)
            }
            training_history.append(episode_metrics)
            
            # Logging
            if episode % 100 == 0:
                logger.info(f"Episode {episode}: Reward: {episode_reward:.4f}, "
                          f"Sharpe: {info.get('sharpe_ratio', 0):.4f}")
        
        self.training_history = training_history
        self.is_trained = True
        
        logger.info("Deep RL optimizer training completed")
        return {
            'training_history': training_history,
            'final_reward': episode_reward,
            'final_sharpe': info.get('sharpe_ratio', 0)
        }
    
    def _state_to_tensor(self, state: PortfolioState) -> torch.Tensor:
        """Convert PortfolioState to tensor"""
        # Flatten correlation matrix (upper triangle)
        correlation_flat = state.correlations[np.triu_indices_from(state.correlations, k=1)]
        
        # Combine all features
        state_vector = np.concatenate([
            state.weights,
            state.returns,
            state.volatilities,
            correlation_flat,
            state.market_features
        ])
        
        return torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
    
    def _train_step(self, state: PortfolioState, action: np.ndarray, 
                   reward: float, next_state: PortfolioState, done: bool) -> float:
        """Single training step"""
        # Convert to tensors
        state_tensor = self._state_to_tensor(state)
        next_state_tensor = self._state_to_tensor(next_state)
        current_weights = torch.FloatTensor(state.weights).unsqueeze(0).to(self.device)
        next_weights = torch.FloatTensor(next_state.weights).unsqueeze(0).to(self.device)
        
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)
        
        # Current Q values
        current_outputs = self.main_net(state_tensor, current_weights)
        current_q = current_outputs['state_value']
        
        # Target Q values
        with torch.no_grad():
            next_outputs = self.target_net(next_state_tensor, next_weights)
            target_q = reward_tensor + self.gamma * next_outputs['state_value'] * (1 - done_tensor)
        
        # Calculate losses
        q_loss = F.mse_loss(current_q, target_q)
        
        # Policy loss (actor-critic style)
        predicted_weights = current_outputs['portfolio_weights']
        actual_new_weights = torch.FloatTensor(next_state.weights).unsqueeze(0).to(self.device)
        policy_loss = F.mse_loss(predicted_weights, actual_new_weights)
        
        # Total loss
        total_loss = q_loss + 0.1 * policy_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def _soft_update_target(self):
        """Soft update of target network"""
        for target_param, main_param in zip(self.target_net.parameters(), self.main_net.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1 - self.tau) * target_param.data)
    
    def _calculate_portfolio_volatility(self, weights: np.ndarray, state: PortfolioState) -> float:
        """Calculate portfolio volatility"""
        cov_matrix = state.correlations * np.outer(state.volatilities, state.volatilities)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        return np.sqrt(portfolio_variance) * np.sqrt(252)  # Annualized
    
    def _calculate_var(self, weights: np.ndarray, state: PortfolioState, 
                      confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        portfolio_return = np.sum(weights * state.returns)
        portfolio_vol = self._calculate_portfolio_volatility(weights, state) / np.sqrt(252)  # Daily vol
        
        # Assume normal distribution
        from scipy.stats import norm
        var = -norm.ppf(1 - confidence_level) * portfolio_vol + portfolio_return
        return var
    
    def _estimate_max_drawdown(self, weights: np.ndarray, state: PortfolioState) -> float:
        """Estimate maximum drawdown"""
        # Simple estimation based on volatility
        portfolio_vol = self._calculate_portfolio_volatility(weights, state)
        estimated_max_dd = portfolio_vol * 0.5  # Rough heuristic
        return min(estimated_max_dd, 0.5)  # Cap at 50%


class QuantumInspiredOptimizer:
    """Quantum-inspired portfolio optimization using variational algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_assets = config.get('num_assets', 10)
        self.num_qubits = config.get('num_qubits', self.num_assets)
        self.num_layers = config.get('num_layers', 3)
        self.learning_rate = config.get('learning_rate', 0.1)
        
        # Quantum-inspired parameters
        self.theta = np.random.uniform(0, 2*np.pi, size=(self.num_layers, self.num_qubits))
        
    async def optimize_portfolio(self, state: PortfolioState) -> OptimizationResult:
        """Optimize portfolio using quantum-inspired algorithms"""
        try:
            logger.info("Running quantum-inspired portfolio optimization")
            
            # Create optimization problem
            returns = state.returns
            cov_matrix = state.correlations * np.outer(state.volatilities, state.volatilities)
            
            # Quantum-inspired optimization using differential evolution
            bounds = [(0, 1) for _ in range(self.num_assets)]
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            
            # Objective function (negative Sharpe ratio)
            def objective(weights):
                portfolio_return = np.sum(weights * returns)
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                portfolio_std = np.sqrt(portfolio_variance)
                
                if portfolio_std == 0:
                    return -portfolio_return
                
                sharpe_ratio = (portfolio_return - 0.02/252) / portfolio_std
                
                # Add quantum-inspired regularization
                quantum_penalty = self._quantum_penalty(weights)
                
                return -(sharpe_ratio - 0.01 * quantum_penalty)
            
            # Use differential evolution (quantum-inspired global optimization)
            result = differential_evolution(
                objective, bounds, 
                maxiter=100, 
                popsize=15,
                seed=42,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                optimal_weights = optimal_weights / np.sum(optimal_weights)  # Normalize
            else:
                logger.warning("Quantum optimization failed, using equal weights")
                optimal_weights = np.ones(self.num_assets) / self.num_assets
            
            # Calculate performance metrics
            expected_return = np.sum(optimal_weights * returns) * 252
            portfolio_variance = np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))
            expected_volatility = np.sqrt(portfolio_variance) * np.sqrt(252)
            sharpe_ratio = (expected_return - 0.02) / expected_volatility if expected_volatility > 0 else 0
            
            # Quantum-inspired confidence calculation
            confidence_score = self._calculate_quantum_confidence(optimal_weights, state)
            
            # Risk metrics
            var_95 = self._calculate_var(optimal_weights, returns, np.sqrt(portfolio_variance))
            max_drawdown = self._estimate_max_drawdown(optimal_weights, expected_volatility)
            
            return OptimizationResult(
                optimal_weights=optimal_weights,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                confidence_score=confidence_score,
                rebalancing_frequency='weekly',
                optimization_method='Quantum-Inspired',
                constraints_satisfied=True,
                metadata={
                    'optimization_success': result.success if 'result' in locals() else False,
                    'quantum_penalty': self._quantum_penalty(optimal_weights),
                    'iterations': result.nit if 'result' in locals() and result.success else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Error in quantum-inspired optimization: {e}")
            # Fallback to equal weights
            equal_weights = np.ones(self.num_assets) / self.num_assets
            return OptimizationResult(
                optimal_weights=equal_weights,
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                var_95=0.0,
                confidence_score=0.0,
                rebalancing_frequency='weekly',
                optimization_method='Equal Weight (Fallback)',
                constraints_satisfied=True,
                metadata={'error': str(e)}
            )
    
    def _quantum_penalty(self, weights: np.ndarray) -> float:
        """Calculate quantum-inspired penalty term"""
        # Simulate quantum entanglement penalty for concentration
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        max_entropy = np.log(len(weights))
        normalized_entropy = entropy / max_entropy
        
        # Penalty for low diversification (quantum coherence loss)
        penalty = 1 - normalized_entropy
        return penalty
    
    def _calculate_quantum_confidence(self, weights: np.ndarray, state: PortfolioState) -> float:
        """Calculate confidence using quantum-inspired metrics"""
        # Quantum-inspired confidence based on weight stability
        weight_variance = np.var(weights)
        diversification = -np.sum(weights * np.log(weights + 1e-8))
        
        # Combine metrics (simulate quantum measurement confidence)
        base_confidence = 1 / (1 + weight_variance * 10)
        diversification_bonus = diversification / np.log(len(weights))
        
        confidence = base_confidence * 0.7 + diversification_bonus * 0.3
        return np.clip(confidence, 0, 1)
    
    def _calculate_var(self, weights: np.ndarray, returns: np.ndarray, volatility: float) -> float:
        """Calculate Value at Risk"""
        portfolio_return = np.sum(weights * returns)
        from scipy.stats import norm
        var_95 = -norm.ppf(0.05) * volatility + portfolio_return
        return var_95
    
    def _estimate_max_drawdown(self, weights: np.ndarray, volatility: float) -> float:
        """Estimate maximum drawdown"""
        return volatility * 0.6  # Heuristic based on volatility


class PortfolioOptimizerManager:
    """Manager for multiple portfolio optimization approaches"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimizers = {}
        
        # Initialize optimizers
        if config.get('use_deep_rl', True):
            self.optimizers['deep_rl'] = DeepRLPortfolioOptimizer(
                config.get('rl_config', {})
            )
        
        if config.get('use_quantum_inspired', True):
            self.optimizers['quantum'] = QuantumInspiredOptimizer(
                config.get('quantum_config', {})
            )
        
        # Traditional optimizers for comparison
        self.traditional_methods = ['mean_variance', 'risk_parity', 'equal_weight']
        
    async def optimize_portfolio_ensemble(self, state: PortfolioState) -> List[OptimizationResult]:
        """Run ensemble of optimization methods"""
        results = []
        
        # Run advanced optimizers
        for name, optimizer in self.optimizers.items():
            try:
                result = await optimizer.optimize_portfolio(state)
                result.optimization_method = f"{result.optimization_method} ({name})"
                results.append(result)
                logger.info(f"Completed {name} optimization: Sharpe {result.sharpe_ratio:.3f}")
            except Exception as e:
                logger.error(f"Error in {name} optimization: {e}")
        
        # Run traditional methods
        for method in self.traditional_methods:
            try:
                result = await self._traditional_optimization(state, method)
                results.append(result)
                logger.info(f"Completed {method} optimization: Sharpe {result.sharpe_ratio:.3f}")
            except Exception as e:
                logger.error(f"Error in {method} optimization: {e}")
        
        # Sort by Sharpe ratio
        results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        
        return results
    
    async def _traditional_optimization(self, state: PortfolioState, method: str) -> OptimizationResult:
        """Traditional portfolio optimization methods"""
        n_assets = len(state.returns)
        
        if method == 'equal_weight':
            optimal_weights = np.ones(n_assets) / n_assets
        
        elif method == 'mean_variance':
            # Use cvxpy for mean-variance optimization
            w = cp.Variable(n_assets)
            returns = state.returns
            cov_matrix = state.correlations * np.outer(state.volatilities, state.volatilities)
            
            # Objective: maximize return - risk_aversion * variance
            risk_aversion = self.config.get('risk_aversion', 1.0)
            objective = cp.Maximize(returns.T @ w - 0.5 * risk_aversion * cp.quad_form(w, cov_matrix))
            
            # Constraints
            constraints = [cp.sum(w) == 1, w >= 0]
            
            # Solve
            prob = cp.Problem(objective, constraints)
            prob.solve()
            
            if prob.status in ['optimal', 'optimal_inaccurate']:
                optimal_weights = w.value
            else:
                optimal_weights = np.ones(n_assets) / n_assets
        
        elif method == 'risk_parity':
            # Risk parity optimization
            def risk_parity_objective(weights):
                cov_matrix = state.correlations * np.outer(state.volatilities, state.volatilities)
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                
                # Risk contributions
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                risk_contrib = weights * marginal_contrib
                
                # Minimize sum of squared deviations from equal risk
                target_risk = portfolio_vol / n_assets
                return np.sum((risk_contrib - target_risk) ** 2)
            
            # Optimization
            bounds = [(0, 1) for _ in range(n_assets)]
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            
            result = minimize(
                risk_parity_objective,
                x0=np.ones(n_assets) / n_assets,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP'
            )
            
            if result.success:
                optimal_weights = result.x
            else:
                optimal_weights = np.ones(n_assets) / n_assets
        
        # Calculate performance metrics
        expected_return = np.sum(optimal_weights * state.returns) * 252
        cov_matrix = state.correlations * np.outer(state.volatilities, state.volatilities)
        portfolio_variance = np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))
        expected_volatility = np.sqrt(portfolio_variance) * np.sqrt(252)
        sharpe_ratio = (expected_return - 0.02) / expected_volatility if expected_volatility > 0 else 0
        
        # Risk metrics
        portfolio_std = np.sqrt(portfolio_variance)
        from scipy.stats import norm
        var_95 = -norm.ppf(0.05) * portfolio_std + np.sum(optimal_weights * state.returns)
        max_drawdown = expected_volatility * 0.5  # Heuristic
        
        return OptimizationResult(
            optimal_weights=optimal_weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            confidence_score=0.8,  # Default confidence for traditional methods
            rebalancing_frequency='monthly',
            optimization_method=method.replace('_', ' ').title(),
            constraints_satisfied=True,
            metadata={'method': method}
        )
    
    def get_best_optimizer_result(self, results: List[OptimizationResult]) -> OptimizationResult:
        """Get the best optimization result based on risk-adjusted returns"""
        if not results:
            raise ValueError("No optimization results available")
        
        # Score each result
        scored_results = []
        for result in results:
            score = (
                result.sharpe_ratio * 0.4 +
                (1 - result.max_drawdown) * 0.2 +
                result.confidence_score * 0.2 +
                (result.expected_return / (result.expected_volatility + 1e-8)) * 0.2
            )
            scored_results.append((score, result))
        
        # Return the highest scored result
        return max(scored_results, key=lambda x: x[0])[1]


# Example usage and testing
async def demo_portfolio_optimization():
    """Demonstrate portfolio optimization capabilities"""
    logger.info("Starting portfolio optimization demo")
    
    # Generate sample returns data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n_assets = 5
    assets = [f'ASSET_{i+1}' for i in range(n_assets)]
    
    # Generate correlated returns
    returns_data = pd.DataFrame(
        np.random.multivariate_normal(
            mean=[0.0005, 0.0003, 0.0004, 0.0002, 0.0006],  # Different expected returns
            cov=[[0.0004, 0.0001, 0.0002, 0.0001, 0.0001],
                 [0.0001, 0.0003, 0.0001, 0.0001, 0.0000],
                 [0.0002, 0.0001, 0.0005, 0.0002, 0.0001],
                 [0.0001, 0.0001, 0.0002, 0.0002, 0.0000],
                 [0.0001, 0.0000, 0.0001, 0.0000, 0.0006]],
            size=len(dates)
        ),
        index=dates,
        columns=assets
    )
    
    # Configuration
    config = {
        'use_deep_rl': True,
        'use_quantum_inspired': True,
        'rl_config': {
            'num_assets': n_assets,
            'hidden_dim': 256,
            'learning_rate': 0.0003,
            'use_weight_changes': True
        },
        'quantum_config': {
            'num_assets': n_assets,
            'num_layers': 3
        },
        'risk_aversion': 1.0
    }
    
    # Initialize manager
    manager = PortfolioOptimizerManager(config)
    
    # Create sample portfolio state
    recent_returns = returns_data.tail(252).mean().values
    recent_vol = returns_data.tail(252).std().values
    correlation_matrix = returns_data.tail(252).corr().values
    
    sample_state = PortfolioState(
        weights=np.ones(n_assets) / n_assets,
        returns=recent_returns,
        prices=np.ones(n_assets),
        volatilities=recent_vol,
        correlations=correlation_matrix,
        market_features=np.array([0.001, 0.02, 0.5, 0.05, 1.0]),
        cash=0.0,
        timestamp=datetime.now()
    )
    
    # Train Deep RL optimizer (quick demo)
    if manager.optimizers.get('deep_rl'):
        logger.info("Training Deep RL optimizer...")
        training_results = await manager.optimizers['deep_rl'].train_optimizer(
            returns_data, episodes=100  # Reduced for demo
        )
        logger.info(f"Training completed. Final Sharpe: {training_results['final_sharpe']:.3f}")
    
    # Run ensemble optimization
    logger.info("Running ensemble optimization...")
    results = await manager.optimize_portfolio_ensemble(sample_state)
    
    # Display results
    logger.info(f"\nOptimization Results ({len(results)} methods):")
    logger.info("=" * 80)
    
    for i, result in enumerate(results):
        logger.info(f"{i+1}. {result.optimization_method}")
        logger.info(f"   Sharpe Ratio: {result.sharpe_ratio:.4f}")
        logger.info(f"   Expected Return: {result.expected_return:.2%}")
        logger.info(f"   Expected Volatility: {result.expected_volatility:.2%}")
        logger.info(f"   Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"   VaR (95%): {result.var_95:.4f}")
        logger.info(f"   Confidence: {result.confidence_score:.4f}")
        logger.info(f"   Weights: {result.optimal_weights}")
        logger.info()
    
    # Get best result
    best_result = manager.get_best_optimizer_result(results)
    logger.info(f"Best Optimizer: {best_result.optimization_method}")
    logger.info(f"Best Sharpe Ratio: {best_result.sharpe_ratio:.4f}")
    
    logger.info("Portfolio optimization demo completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_portfolio_optimization())