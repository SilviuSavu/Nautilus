#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

"""
ML-Based Routing Optimization for Enhanced MessageBus.

Provides intelligent message routing optimization using machine learning algorithms
to dynamically adjust priorities and routing decisions based on:
- Historical performance data
- Market volatility conditions  
- Trading session characteristics
- Cross-venue arbitrage opportunities
- Latency-sensitive event detection
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
from nautilus_trader.infrastructure.messagebus.config import MessagePriority


class MarketRegime(Enum):
    """Market regime classifications for adaptive routing."""
    NORMAL = "normal"          # Standard market conditions
    VOLATILE = "volatile"      # High volatility periods
    TRENDING = "trending"      # Strong directional movement
    RANGING = "ranging"        # Sideways market movement
    NEWS_EVENT = "news_event"  # Economic news or events
    ILLIQUID = "illiquid"      # Low liquidity conditions
    PRE_MARKET = "pre_market"  # Pre-market trading hours
    POST_MARKET = "post_market" # After-hours trading


class RouteOptimizationLevel(Enum):
    """Optimization levels for different trading strategies."""
    CONSERVATIVE = "conservative"    # Low-risk, stable routing
    BALANCED = "balanced"           # Balanced risk/performance
    AGGRESSIVE = "aggressive"       # High-performance, latency-optimized
    ARBITRAGE = "arbitrage"         # Cross-venue arbitrage focused
    SCALPING = "scalping"          # High-frequency scalping
    SWING = "swing"                # Swing trading optimized


@dataclass
class RoutingMetrics:
    """Metrics for routing performance analysis."""
    total_messages: int = 0
    avg_latency_ms: float = 0.0
    success_rate: float = 0.0
    throughput_msg_per_sec: float = 0.0
    priority_distribution: Dict[MessagePriority, int] = None
    error_rate: float = 0.0
    queue_depth_avg: float = 0.0
    adaptive_adjustments: int = 0
    
    def __post_init__(self):
        if self.priority_distribution is None:
            self.priority_distribution = defaultdict(int)


@dataclass
class MarketConditions:
    """Current market condition indicators."""
    volatility_percentile: float = 50.0  # 0-100 percentile
    volume_ratio: float = 1.0            # Relative to avg volume
    spread_ratio: float = 1.0             # Relative to avg spread
    momentum_score: float = 0.0           # -1 to 1 trend strength
    liquidity_score: float = 1.0          # 0-1 liquidity measure
    news_impact_score: float = 0.0        # 0-1 news event impact
    session_phase: str = "main"           # market session phase
    cross_venue_spread: float = 0.0       # arbitrage opportunity measure


class MLRoutingOptimizer:
    """
    Machine Learning-based routing optimizer for Enhanced MessageBus.
    
    Uses reinforcement learning and predictive models to optimize:
    - Dynamic priority adjustment based on market conditions
    - Intelligent load balancing across message queues  
    - Predictive latency-sensitive event detection
    - Cross-venue arbitrage opportunity routing
    - Adaptive performance optimization
    """
    
    def __init__(self, 
                 optimization_level: RouteOptimizationLevel = RouteOptimizationLevel.BALANCED,
                 learning_rate: float = 0.01,
                 memory_size: int = 10000,
                 update_interval_seconds: int = 30):
        """
        Initialize ML routing optimizer.
        
        Args:
            optimization_level: Trading strategy optimization focus
            learning_rate: ML model learning rate
            memory_size: Historical data memory size
            update_interval_seconds: Model update frequency
        """
        self.optimization_level = optimization_level
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.update_interval = update_interval_seconds
        
        # ML Model Components
        self._priority_weights = self._initialize_priority_weights()
        self._routing_history = deque(maxlen=memory_size)
        self._performance_history = deque(maxlen=memory_size)
        self._market_state_history = deque(maxlen=memory_size)
        
        # Current state tracking
        self._current_metrics = RoutingMetrics()
        self._current_market_conditions = MarketConditions()
        self._current_regime = MarketRegime.NORMAL
        
        # Adaptive parameters
        self._priority_multipliers = {
            MessagePriority.CRITICAL: 1.0,
            MessagePriority.HIGH: 1.0,
            MessagePriority.NORMAL: 1.0,
            MessagePriority.LOW: 1.0
        }
        
        # Pattern detection
        self._latency_patterns = defaultdict(list)
        self._arbitrage_opportunities = deque(maxlen=1000)
        self._volatility_events = deque(maxlen=500)
        
        # Learning components
        self._q_table = defaultdict(lambda: defaultdict(float))  # Q-learning table
        self._state_action_counts = defaultdict(lambda: defaultdict(int))
        
        # Performance tracking
        self._last_update_time = time.time()
        self._total_optimizations = 0
        self._successful_predictions = 0
        self._total_predictions = 0
        
        # Logger
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Background update task
        self._update_task: Optional[asyncio.Task] = None
        self._is_learning_enabled = True
        
        self._logger.info(f"ML Routing Optimizer initialized with {optimization_level.value} level")
    
    def _initialize_priority_weights(self) -> Dict[MessagePriority, float]:
        """Initialize priority weights based on optimization level."""
        base_weights = {
            MessagePriority.CRITICAL: 10.0,
            MessagePriority.HIGH: 5.0,
            MessagePriority.NORMAL: 1.0,
            MessagePriority.LOW: 0.5
        }
        
        if self.optimization_level == RouteOptimizationLevel.AGGRESSIVE:
            return {
                MessagePriority.CRITICAL: 20.0,
                MessagePriority.HIGH: 10.0,
                MessagePriority.NORMAL: 1.0,
                MessagePriority.LOW: 0.1
            }
        elif self.optimization_level == RouteOptimizationLevel.ARBITRAGE:
            return {
                MessagePriority.CRITICAL: 50.0,  # Arbitrage opportunities
                MessagePriority.HIGH: 15.0,
                MessagePriority.NORMAL: 1.0,
                MessagePriority.LOW: 0.1
            }
        elif self.optimization_level == RouteOptimizationLevel.SCALPING:
            return {
                MessagePriority.CRITICAL: 25.0,
                MessagePriority.HIGH: 20.0,  # High-freq data
                MessagePriority.NORMAL: 1.0,
                MessagePriority.LOW: 0.05
            }
        else:
            return base_weights
    
    def start_learning(self) -> None:
        """Start the ML optimization background task."""
        if self._update_task is None or self._update_task.done():
            self._update_task = asyncio.create_task(self._learning_loop())
            self._logger.info("ML routing optimization learning started")
    
    def stop_learning(self) -> None:
        """Stop the ML optimization background task."""
        self._is_learning_enabled = False
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
        self._logger.info("ML routing optimization learning stopped")
    
    async def _learning_loop(self) -> None:
        """Background learning and optimization loop."""
        while self._is_learning_enabled:
            try:
                await asyncio.sleep(self.update_interval)
                await self._update_models()
                await self._optimize_routing_parameters()
                self._total_optimizations += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _update_models(self) -> None:
        """Update ML models with recent performance data."""
        if len(self._performance_history) < 10:
            return  # Need minimum data for meaningful updates
        
        # Update Q-learning model
        self._update_q_learning()
        
        # Update priority multipliers based on performance
        self._update_priority_multipliers()
        
        # Detect and learn from patterns
        self._detect_latency_patterns()
        self._detect_arbitrage_patterns()
        
        # Update market regime classification
        self._update_market_regime()
        
        self._logger.debug("ML models updated")
    
    def _update_q_learning(self) -> None:
        """Update Q-learning table with recent experiences."""
        if len(self._routing_history) < 2:
            return
        
        # Simple Q-learning update for routing decisions
        for i in range(len(self._routing_history) - 1):
            current_state = self._get_state_key(self._routing_history[i])
            action = self._routing_history[i].get('action', 'default')
            reward = self._calculate_reward(self._routing_history[i])
            next_state = self._get_state_key(self._routing_history[i + 1])
            
            # Q-learning update rule
            current_q = self._q_table[current_state][action]
            max_next_q = max(self._q_table[next_state].values()) if self._q_table[next_state] else 0
            
            new_q = current_q + self.learning_rate * (reward + 0.9 * max_next_q - current_q)
            self._q_table[current_state][action] = new_q
            
            self._state_action_counts[current_state][action] += 1
    
    def _get_state_key(self, routing_data: Dict[str, Any]) -> str:
        """Generate state key for Q-learning."""
        regime = routing_data.get('market_regime', 'normal')
        volatility_bucket = int(routing_data.get('volatility', 50) / 20)  # 0-4
        volume_bucket = int(min(routing_data.get('volume_ratio', 1.0), 3.0))  # 0-3
        
        return f"{regime}_{volatility_bucket}_{volume_bucket}"
    
    def _calculate_reward(self, routing_data: Dict[str, Any]) -> float:
        """Calculate reward for reinforcement learning."""
        # Reward based on achieved vs target latency
        target_latency = routing_data.get('target_latency_ms', 10.0)
        actual_latency = routing_data.get('actual_latency_ms', 10.0)
        
        latency_reward = max(0, 2.0 - (actual_latency / target_latency))
        
        # Reward based on throughput
        throughput = routing_data.get('throughput', 1000)
        throughput_reward = min(1.0, throughput / 10000)  # Normalize to 10k msg/sec
        
        # Penalty for errors
        error_penalty = -5.0 * routing_data.get('error_rate', 0.0)
        
        return latency_reward + throughput_reward + error_penalty
    
    def _update_priority_multipliers(self) -> None:
        """Update priority multipliers based on recent performance."""
        if len(self._performance_history) < 20:
            return
        
        # Calculate performance metrics for each priority level
        priority_performance = defaultdict(list)
        
        for perf_data in list(self._performance_history)[-20:]:
            for priority, latency in perf_data.get('priority_latencies', {}).items():
                priority_performance[priority].append(latency)
        
        # Adjust multipliers based on performance
        for priority in MessagePriority:
            if priority in priority_performance:
                avg_latency = np.mean(priority_performance[priority])
                target_latency = self._get_target_latency(priority)
                
                if avg_latency > target_latency * 1.5:
                    # Increase priority if consistently slow
                    self._priority_multipliers[priority] *= 1.1
                elif avg_latency < target_latency * 0.8:
                    # Decrease priority if consistently fast (can handle more load)
                    self._priority_multipliers[priority] *= 0.95
                
                # Clamp values to reasonable ranges
                self._priority_multipliers[priority] = max(0.1, min(10.0, self._priority_multipliers[priority]))
    
    def _get_target_latency(self, priority: MessagePriority) -> float:
        """Get target latency for priority level."""
        targets = {
            MessagePriority.CRITICAL: 0.5,   # 0.5ms
            MessagePriority.HIGH: 2.0,       # 2ms
            MessagePriority.NORMAL: 10.0,    # 10ms
            MessagePriority.LOW: 50.0        # 50ms
        }
        return targets.get(priority, 10.0)
    
    def _detect_latency_patterns(self) -> None:
        """Detect patterns in latency data for predictive routing."""
        current_time = time.time()
        
        for routing_data in list(self._routing_history)[-100:]:
            timestamp = routing_data.get('timestamp', current_time)
            latency = routing_data.get('actual_latency_ms', 0)
            topic = routing_data.get('topic', 'unknown')
            
            # Store latency patterns by topic
            self._latency_patterns[topic].append({
                'timestamp': timestamp,
                'latency': latency,
                'hour': int((timestamp % 86400) / 3600)  # Hour of day
            })
            
            # Keep only recent patterns
            self._latency_patterns[topic] = self._latency_patterns[topic][-50:]
    
    def _detect_arbitrage_patterns(self) -> None:
        """Detect cross-venue arbitrage opportunity patterns."""
        if len(self._routing_history) < 10:
            return
        
        # Look for correlated price movements across venues
        venue_prices = defaultdict(list)
        
        for routing_data in list(self._routing_history)[-50:]:
            if 'price_data' in routing_data:
                venue = routing_data.get('venue', 'unknown')
                price = routing_data.get('price_data', {}).get('price', 0)
                timestamp = routing_data.get('timestamp', time.time())
                
                venue_prices[venue].append({
                    'price': price,
                    'timestamp': timestamp
                })
        
        # Identify arbitrage opportunities
        if len(venue_prices) >= 2:
            self._identify_arbitrage_opportunities(venue_prices)
    
    def _identify_arbitrage_opportunities(self, venue_prices: Dict[str, List[Dict]]) -> None:
        """Identify and store arbitrage opportunities."""
        venues = list(venue_prices.keys())
        current_time = time.time()
        
        for i, venue_a in enumerate(venues):
            for venue_b in venues[i+1:]:
                if len(venue_prices[venue_a]) > 0 and len(venue_prices[venue_b]) > 0:
                    price_a = venue_prices[venue_a][-1]['price']
                    price_b = venue_prices[venue_b][-1]['price']
                    
                    spread = abs(price_a - price_b) / min(price_a, price_b) if min(price_a, price_b) > 0 else 0
                    
                    if spread > 0.001:  # 0.1% spread threshold
                        self._arbitrage_opportunities.append({
                            'timestamp': current_time,
                            'venue_a': venue_a,
                            'venue_b': venue_b,
                            'spread': spread,
                            'price_a': price_a,
                            'price_b': price_b
                        })
    
    def _update_market_regime(self) -> None:
        """Update current market regime classification."""
        conditions = self._current_market_conditions
        
        # Simple rule-based regime classification
        if conditions.news_impact_score > 0.7:
            self._current_regime = MarketRegime.NEWS_EVENT
        elif conditions.volatility_percentile > 80:
            self._current_regime = MarketRegime.VOLATILE
        elif conditions.liquidity_score < 0.3:
            self._current_regime = MarketRegime.ILLIQUID
        elif abs(conditions.momentum_score) > 0.7:
            self._current_regime = MarketRegime.TRENDING
        elif conditions.volume_ratio < 0.5:
            if conditions.session_phase == "pre":
                self._current_regime = MarketRegime.PRE_MARKET
            elif conditions.session_phase == "post":
                self._current_regime = MarketRegime.POST_MARKET
            else:
                self._current_regime = MarketRegime.RANGING
        else:
            self._current_regime = MarketRegime.NORMAL
    
    async def _optimize_routing_parameters(self) -> None:
        """Optimize routing parameters based on current conditions."""
        # Adjust parameters based on market regime
        regime_adjustments = self._get_regime_adjustments()
        
        for priority in MessagePriority:
            base_multiplier = self._priority_multipliers[priority]
            regime_factor = regime_adjustments.get(priority, 1.0)
            
            # Apply regime-based adjustment
            new_multiplier = base_multiplier * regime_factor
            self._priority_multipliers[priority] = max(0.05, min(50.0, new_multiplier))
        
        self._logger.debug(f"Routing optimized for regime: {self._current_regime.value}")
    
    def _get_regime_adjustments(self) -> Dict[MessagePriority, float]:
        """Get priority adjustments based on current market regime."""
        adjustments = {
            MarketRegime.NORMAL: {
                MessagePriority.CRITICAL: 1.0,
                MessagePriority.HIGH: 1.0,
                MessagePriority.NORMAL: 1.0,
                MessagePriority.LOW: 1.0
            },
            MarketRegime.VOLATILE: {
                MessagePriority.CRITICAL: 2.0,    # Critical events more important
                MessagePriority.HIGH: 1.5,
                MessagePriority.NORMAL: 1.0,
                MessagePriority.LOW: 0.5
            },
            MarketRegime.NEWS_EVENT: {
                MessagePriority.CRITICAL: 3.0,    # News requires immediate attention
                MessagePriority.HIGH: 2.0,
                MessagePriority.NORMAL: 1.0,
                MessagePriority.LOW: 0.2
            },
            MarketRegime.ILLIQUID: {
                MessagePriority.CRITICAL: 1.5,
                MessagePriority.HIGH: 1.2,
                MessagePriority.NORMAL: 1.0,
                MessagePriority.LOW: 1.2         # Low priority data more valuable when illiquid
            },
            MarketRegime.ARBITRAGE: {
                MessagePriority.CRITICAL: 5.0,    # Arbitrage opportunities critical
                MessagePriority.HIGH: 3.0,
                MessagePriority.NORMAL: 1.0,
                MessagePriority.LOW: 0.1
            }
        }
        
        return adjustments.get(self._current_regime, adjustments[MarketRegime.NORMAL])
    
    def optimize_message_priority(self, topic: str, data: Any, current_priority: MessagePriority) -> MessagePriority:
        """
        Optimize message priority using ML predictions.
        
        Args:
            topic: Message topic
            data: Message data
            current_priority: Current assigned priority
            
        Returns:
            Optimized message priority
        """
        # Record prediction attempt
        self._total_predictions += 1
        
        try:
            # Get ML-based priority recommendation
            predicted_priority = self._predict_optimal_priority(topic, data, current_priority)
            
            # Apply regime-based adjustments
            final_priority = self._apply_regime_adjustment(predicted_priority)
            
            # Record routing decision
            self._record_routing_decision(topic, data, current_priority, final_priority)
            
            return final_priority
            
        except Exception as e:
            self._logger.error(f"Error in priority optimization: {e}")
            return current_priority
    
    def _predict_optimal_priority(self, topic: str, data: Any, current_priority: MessagePriority) -> MessagePriority:
        """Predict optimal priority using ML models."""
        # Extract features
        features = self._extract_features(topic, data)
        state_key = self._get_current_state_key()
        
        # Get Q-learning recommendation
        action_values = self._q_table.get(state_key, {})
        
        if action_values:
            # Choose action with highest Q-value (with some exploration)
            if np.random.random() > 0.1:  # 90% exploitation
                best_action = max(action_values, key=action_values.get)
            else:  # 10% exploration
                best_action = np.random.choice(list(MessagePriority))
        else:
            best_action = current_priority
        
        # Convert action to priority
        if isinstance(best_action, str):
            try:
                predicted_priority = MessagePriority(best_action)
            except ValueError:
                predicted_priority = current_priority
        else:
            predicted_priority = best_action
        
        return predicted_priority
    
    def _extract_features(self, topic: str, data: Any) -> Dict[str, float]:
        """Extract features from message for ML prediction."""
        features = {
            'topic_hash': hash(topic) % 1000 / 1000.0,
            'data_size': len(str(data)) / 1000.0,  # Normalize data size
            'volatility': self._current_market_conditions.volatility_percentile / 100.0,
            'volume_ratio': min(self._current_market_conditions.volume_ratio, 3.0) / 3.0,
            'spread_ratio': min(self._current_market_conditions.spread_ratio, 3.0) / 3.0,
            'momentum': (self._current_market_conditions.momentum_score + 1.0) / 2.0,  # -1 to 1 -> 0 to 1
            'liquidity': self._current_market_conditions.liquidity_score,
            'time_of_day': (time.time() % 86400) / 86400.0,  # 0-1 for time of day
        }
        
        # Add topic-specific features
        if 'tick' in topic.lower():
            features['is_tick'] = 1.0
        elif 'trade' in topic.lower():
            features['is_trade'] = 1.0
        elif 'book' in topic.lower():
            features['is_book'] = 1.0
        elif 'arbitrage' in topic.lower():
            features['is_arbitrage'] = 1.0
        
        return features
    
    def _get_current_state_key(self) -> str:
        """Get current state key for Q-learning."""
        regime = self._current_regime.value
        volatility_bucket = int(self._current_market_conditions.volatility_percentile / 20)
        volume_bucket = int(min(self._current_market_conditions.volume_ratio, 3.0))
        
        return f"{regime}_{volatility_bucket}_{volume_bucket}"
    
    def _apply_regime_adjustment(self, priority: MessagePriority) -> MessagePriority:
        """Apply market regime-based priority adjustments."""
        multiplier = self._priority_multipliers.get(priority, 1.0)
        regime_adjustments = self._get_regime_adjustments()
        regime_factor = regime_adjustments.get(priority, 1.0)
        
        final_multiplier = multiplier * regime_factor
        
        # Map multiplier to priority levels
        if final_multiplier >= 5.0:
            return MessagePriority.CRITICAL
        elif final_multiplier >= 2.0:
            return MessagePriority.HIGH
        elif final_multiplier >= 0.5:
            return MessagePriority.NORMAL
        else:
            return MessagePriority.LOW
    
    def _record_routing_decision(self, topic: str, data: Any, original_priority: MessagePriority, 
                               final_priority: MessagePriority) -> None:
        """Record routing decision for learning."""
        current_time = time.time()
        
        routing_record = {
            'timestamp': current_time,
            'topic': topic,
            'original_priority': original_priority.value,
            'final_priority': final_priority.value,
            'market_regime': self._current_regime.value,
            'volatility': self._current_market_conditions.volatility_percentile,
            'volume_ratio': self._current_market_conditions.volume_ratio,
            'action': final_priority.value,
            'features': self._extract_features(topic, data)
        }
        
        self._routing_history.append(routing_record)
    
    def update_market_conditions(self, conditions: MarketConditions) -> None:
        """Update current market conditions for ML optimization."""
        self._current_market_conditions = conditions
        self._update_market_regime()
        
        # Record market state for learning
        self._market_state_history.append({
            'timestamp': time.time(),
            'conditions': conditions,
            'regime': self._current_regime.value
        })
    
    def record_performance_metrics(self, metrics: RoutingMetrics) -> None:
        """Record performance metrics for ML learning."""
        current_time = time.time()
        
        performance_record = {
            'timestamp': current_time,
            'metrics': metrics,
            'market_regime': self._current_regime.value,
            'priority_latencies': {}  # Will be populated by specific implementations
        }
        
        self._performance_history.append(performance_record)
        self._current_metrics = metrics
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get current optimization statistics."""
        prediction_accuracy = (
            self._successful_predictions / self._total_predictions 
            if self._total_predictions > 0 else 0.0
        )
        
        return {
            'optimization_level': self.optimization_level.value,
            'current_regime': self._current_regime.value,
            'total_optimizations': self._total_optimizations,
            'total_predictions': self._total_predictions,
            'prediction_accuracy': prediction_accuracy,
            'priority_multipliers': dict(self._priority_multipliers),
            'q_table_size': len(self._q_table),
            'routing_history_size': len(self._routing_history),
            'arbitrage_opportunities_detected': len(self._arbitrage_opportunities),
            'learning_enabled': self._is_learning_enabled,
            'last_update': self._last_update_time
        }
    
    def reset_learning(self) -> None:
        """Reset learning state for fresh start."""
        self._q_table.clear()
        self._state_action_counts.clear()
        self._routing_history.clear()
        self._performance_history.clear()
        self._market_state_history.clear()
        self._latency_patterns.clear()
        self._arbitrage_opportunities.clear()
        self._volatility_events.clear()
        
        self._priority_multipliers = self._initialize_priority_weights()
        self._total_optimizations = 0
        self._successful_predictions = 0
        self._total_predictions = 0
        
        self._logger.info("ML routing optimizer learning state reset")


class AdvancedPatternMatcher:
    """
    Advanced pattern matching engine with ML-enhanced topic routing.
    
    Provides sophisticated pattern matching beyond basic glob patterns:
    - Semantic similarity matching
    - Dynamic pattern learning
    - Context-aware routing
    - Performance-optimized pattern trees
    """
    
    def __init__(self):
        self.pattern_tree = {}
        self.semantic_cache = {}
        self.pattern_performance = defaultdict(lambda: {'hits': 0, 'avg_latency': 0.0})
        self.learned_patterns = set()
        
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        
    def add_pattern(self, pattern: str, callback: Any, priority: MessagePriority = MessagePriority.NORMAL) -> None:
        """Add a pattern to the matching engine."""
        # Build optimized pattern tree for fast matching
        self._build_pattern_tree(pattern, callback, priority)
        
    def match_topic(self, topic: str) -> List[Tuple[Any, MessagePriority]]:
        """Match topic against all patterns and return callbacks with priorities."""
        matches = []
        
        # Direct tree traversal for performance
        matches.extend(self._traverse_pattern_tree(topic))
        
        # Semantic matching for learned patterns
        matches.extend(self._semantic_match(topic))
        
        return matches
        
    def _build_pattern_tree(self, pattern: str, callback: Any, priority: MessagePriority) -> None:
        """Build optimized pattern tree structure."""
        # Convert glob patterns to regex-like tree structure
        parts = pattern.split('.')
        current_node = self.pattern_tree
        
        for i, part in enumerate(parts):
            if part not in current_node:
                current_node[part] = {'children': {}, 'callbacks': []}
            
            # If this is the final part, add the callback
            if i == len(parts) - 1:
                current_node[part]['callbacks'].append({
                    'callback': callback,
                    'priority': priority,
                    'pattern': pattern,
                    'performance': {'hits': 0, 'total_latency': 0.0}
                })
            
            current_node = current_node[part]['children']
    
    def _traverse_pattern_tree(self, topic: str) -> List[Tuple[Any, MessagePriority]]:
        """Traverse pattern tree for matches."""
        matches = []
        topic_parts = topic.split('.')
        
        def traverse_node(node: Dict, topic_parts: List[str], part_index: int = 0):
            if part_index >= len(topic_parts):
                # Check if this node has callbacks
                if 'callbacks' in node:
                    for callback_info in node['callbacks']:
                        matches.append((callback_info['callback'], callback_info['priority']))
                        # Update performance metrics
                        callback_info['performance']['hits'] += 1
                return
            
            current_part = topic_parts[part_index]
            
            # Check for exact matches
            if current_part in node:
                traverse_node(node[current_part]['children'], topic_parts, part_index + 1)
            
            # Check for wildcard matches
            if '*' in node:
                traverse_node(node['*']['children'], topic_parts, part_index + 1)
            
            # Check for multi-level wildcard
            if '**' in node:
                # Can match current and all remaining parts
                for i in range(part_index, len(topic_parts) + 1):
                    traverse_node(node['**']['children'], topic_parts, i)
        
        if self.pattern_tree:
            traverse_node(self.pattern_tree, topic_parts)
        
        return matches
    
    def _semantic_match(self, topic: str) -> List[Tuple[Any, MessagePriority]]:
        """Perform semantic similarity matching."""
        matches = []
        
        # Use simple semantic rules for topic similarity
        semantic_rules = {
            'trade': ['execution', 'fill', 'order'],
            'market_data': ['tick', 'quote', 'book', 'depth'],
            'risk': ['limit', 'breach', 'violation'],
            'position': ['portfolio', 'balance', 'exposure'],
            'arbitrage': ['spread', 'opportunity', 'cross_venue']
        }
        
        topic_lower = topic.lower()
        
        for category, related_terms in semantic_rules.items():
            if any(term in topic_lower for term in related_terms):
                # Check if we have learned patterns for this category
                for learned_pattern in self.learned_patterns:
                    if category in learned_pattern.lower():
                        # Apply higher priority for semantic matches
                        semantic_priority = MessagePriority.HIGH if 'critical' in topic_lower else MessagePriority.NORMAL
                        matches.append((learned_pattern, semantic_priority))
        
        return matches
    
    def learn_pattern(self, topic: str, performance_score: float) -> None:
        """Learn new patterns from high-performing topics."""
        if performance_score > 0.8:  # High performance threshold
            # Extract pattern from successful topic
            pattern_candidate = self._extract_pattern(topic)
            if pattern_candidate and pattern_candidate not in self.learned_patterns:
                self.learned_patterns.add(pattern_candidate)
                self._logger.info(f"Learned new pattern: {pattern_candidate}")
    
    def _extract_pattern(self, topic: str) -> str:
        """Extract a generalizable pattern from a specific topic."""
        parts = topic.split('.')
        
        # Simple pattern extraction - replace specific symbols/IDs with wildcards
        pattern_parts = []
        for part in parts:
            # If part looks like a symbol (all caps, 3-8 chars), replace with wildcard
            if part.isupper() and 3 <= len(part) <= 8:
                pattern_parts.append('*')
            # If part looks like an ID (contains numbers), replace with wildcard  
            elif any(char.isdigit() for char in part):
                pattern_parts.append('*')
            else:
                pattern_parts.append(part)
        
        return '.'.join(pattern_parts)
    
    def get_pattern_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all patterns."""
        performance = {}
        
        def collect_performance(node: Dict, path: str = ""):
            if 'callbacks' in node:
                for callback_info in node['callbacks']:
                    pattern = callback_info['pattern']
                    perf = callback_info['performance']
                    if perf['hits'] > 0:
                        avg_latency = perf['total_latency'] / perf['hits']
                        performance[pattern] = {
                            'hits': perf['hits'],
                            'avg_latency_ms': avg_latency,
                            'priority': callback_info['priority'].value
                        }
            
            if 'children' in node:
                for child_key, child_node in node['children'].items():
                    new_path = f"{path}.{child_key}" if path else child_key
                    collect_performance(child_node, new_path)
        
        collect_performance(self.pattern_tree)
        return performance
    
    def optimize_patterns(self) -> None:
        """Optimize pattern matching based on performance data."""
        performance = self.get_pattern_performance()
        
        # Identify slow patterns and suggest optimizations
        slow_patterns = []
        for pattern, metrics in performance.items():
            if metrics['avg_latency_ms'] > 10.0 and metrics['hits'] > 10:
                slow_patterns.append(pattern)
        
        if slow_patterns:
            self._logger.warning(f"Slow patterns detected: {slow_patterns}")
            # Could implement pattern restructuring here
    
    def clear_cache(self) -> None:
        """Clear semantic matching cache."""
        self.semantic_cache.clear()
        self._logger.debug("Pattern matcher cache cleared")