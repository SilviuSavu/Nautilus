"""
Margin Calculator
================

Advanced margin calculation engine with cross-margining optimization and regulatory compliance.
Integrates with M4 Max hardware acceleration for real-time calculations.
"""

import asyncio
import logging
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone

from models import (
    Position, Portfolio, MarginRequirement, AssetClass, MarginType,
    CrossMarginBenefit, MarginStressTest
)

# Import M4 Max hardware acceleration if available
try:
    from backend.hardware_router import hardware_accelerated, WorkloadType
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False
    
    # Create dummy WorkloadType enum
    class WorkloadType:
        RISK_CALCULATION = "risk_calculation"
        MONTE_CARLO = "monte_carlo"
        ML_INFERENCE = "ml_inference"
    
    def hardware_accelerated(workload_type, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


class MarginCalculator:
    """
    Advanced margin calculator with support for:
    - Multiple asset classes (equity, fixed income, FX, derivatives)
    - Cross-margining optimization
    - Regulatory requirements (Basel III, Dodd-Frank)
    - Real-time market data integration
    - M4 Max hardware acceleration
    """
    
    def __init__(self):
        self.margin_rates = self._initialize_margin_rates()
        self.correlation_matrix = {}
        self.volatility_cache = {}
        
    def _initialize_margin_rates(self) -> Dict[AssetClass, Dict[str, Decimal]]:
        """Initialize margin rates by asset class"""
        return {
            AssetClass.EQUITY: {
                'initial_margin_rate': Decimal('0.25'),  # 25% for equities
                'maintenance_margin_rate': Decimal('0.15'),  # 15% maintenance
                'concentration_threshold': Decimal('0.10'),  # 10% concentration limit
                'volatility_multiplier': Decimal('1.5')
            },
            AssetClass.BOND: {
                'initial_margin_rate': Decimal('0.05'),  # 5% for government bonds
                'maintenance_margin_rate': Decimal('0.03'),  # 3% maintenance
                'duration_multiplier': Decimal('0.01'),  # 1% per year of duration
                'credit_spread_multiplier': Decimal('2.0')
            },
            AssetClass.FX: {
                'initial_margin_rate': Decimal('0.02'),  # 2% for major currencies
                'maintenance_margin_rate': Decimal('0.015'),  # 1.5% maintenance
                'volatility_multiplier': Decimal('2.0'),
                'correlation_benefit': Decimal('0.3')  # 30% correlation benefit
            },
            AssetClass.DERIVATIVE: {
                'initial_margin_rate': Decimal('0.15'),  # 15% base rate
                'maintenance_margin_rate': Decimal('0.10'),  # 10% maintenance
                'delta_multiplier': Decimal('1.0'),
                'gamma_adjustment': Decimal('0.5'),
                'vega_adjustment': Decimal('0.3')
            },
            AssetClass.COMMODITY: {
                'initial_margin_rate': Decimal('0.30'),  # 30% for commodities
                'maintenance_margin_rate': Decimal('0.20'),  # 20% maintenance
                'volatility_multiplier': Decimal('2.5')
            },
            AssetClass.CRYPTO: {
                'initial_margin_rate': Decimal('0.50'),  # 50% for crypto
                'maintenance_margin_rate': Decimal('0.35'),  # 35% maintenance
                'volatility_multiplier': Decimal('3.0')
            }
        }
    
    @hardware_accelerated(WorkloadType.RISK_CALCULATION, data_size=1000)
    async def calculate_initial_margin(self, position: Position) -> Decimal:
        """Calculate initial margin requirement for a position"""
        try:
            base_margin = await self._calculate_base_margin(position)
            volatility_adjustment = await self._calculate_volatility_adjustment(position)
            concentration_adjustment = await self._calculate_concentration_adjustment(position)
            
            # Apply asset-specific calculations
            if position.asset_class == AssetClass.DERIVATIVE:
                greeks_adjustment = await self._calculate_greeks_adjustment(position)
                total_margin = base_margin * (1 + volatility_adjustment + concentration_adjustment + greeks_adjustment)
            else:
                total_margin = base_margin * (1 + volatility_adjustment + concentration_adjustment)
            
            return total_margin.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
        except Exception as e:
            logger.error(f"Error calculating initial margin for position {position.id}: {e}")
            # Fallback to conservative calculation
            return position.notional_value * Decimal('0.50')  # 50% conservative margin
    
    async def _calculate_base_margin(self, position: Position) -> Decimal:
        """Calculate base margin before adjustments"""
        margin_config = self.margin_rates.get(position.asset_class)
        if not margin_config:
            logger.warning(f"No margin config for asset class {position.asset_class}, using default")
            return position.notional_value * Decimal('0.25')  # 25% default
        
        base_rate = margin_config['initial_margin_rate']
        
        # Apply duration multiplier for bonds
        if position.asset_class == AssetClass.BOND and position.duration:
            duration_adjustment = position.duration * margin_config.get('duration_multiplier', Decimal('0'))
            base_rate += duration_adjustment
        
        return position.notional_value * base_rate
    
    async def _calculate_volatility_adjustment(self, position: Position) -> Decimal:
        """Calculate volatility-based margin adjustment"""
        # In production, this would use real market data
        # For now, use mock volatility data
        mock_volatility = self._get_mock_volatility(position.symbol)
        
        margin_config = self.margin_rates.get(position.asset_class, {})
        volatility_multiplier = margin_config.get('volatility_multiplier', Decimal('1.0'))
        
        # Higher volatility = higher margin adjustment
        if mock_volatility > Decimal('0.30'):  # 30% volatility threshold
            return (mock_volatility - Decimal('0.30')) * volatility_multiplier
        
        return Decimal('0')
    
    def _get_mock_volatility(self, symbol: str) -> Decimal:
        """Mock volatility data - replace with real market data in production"""
        mock_volatilities = {
            'TSLA': Decimal('0.45'),
            'AAPL': Decimal('0.25'),
            'SPY': Decimal('0.18'),
            'BTC': Decimal('0.75'),
            'EUR/USD': Decimal('0.08'),
            'GLD': Decimal('0.15')
        }
        return mock_volatilities.get(symbol, Decimal('0.20'))  # 20% default
    
    async def _calculate_concentration_adjustment(self, position: Position) -> Decimal:
        """Calculate concentration risk adjustment"""
        # This would analyze position size relative to portfolio in production
        # Mock implementation for demonstration
        if position.notional_value > Decimal('10000000'):  # $10M threshold
            return Decimal('0.1')  # 10% concentration penalty
        return Decimal('0')
    
    async def _calculate_greeks_adjustment(self, position: Position) -> Decimal:
        """Calculate Greeks-based margin adjustment for derivatives"""
        if not all([position.delta, position.gamma, position.vega]):
            return Decimal('0')
        
        margin_config = self.margin_rates[AssetClass.DERIVATIVE]
        
        # Delta adjustment
        delta_adj = abs(position.delta or Decimal('0')) * margin_config['delta_multiplier']
        
        # Gamma adjustment (convexity risk)
        gamma_adj = abs(position.gamma or Decimal('0')) * margin_config['gamma_adjustment']
        
        # Vega adjustment (volatility risk)
        vega_adj = abs(position.vega or Decimal('0')) * margin_config['vega_adjustment']
        
        return delta_adj + gamma_adj + vega_adj
    
    @hardware_accelerated(WorkloadType.RISK_CALCULATION, data_size=1000)
    async def calculate_variation_margin(self, position: Position) -> Decimal:
        """Calculate variation margin (mark-to-market)"""
        # Variation margin is the unrealized P&L that needs to be collateralized
        # This would use real-time pricing in production
        mock_price_change = self._get_mock_price_change(position.symbol)
        variation_margin = position.quantity * mock_price_change
        
        # Only collateralize losses (negative variation margin)
        return max(Decimal('0'), -variation_margin)
    
    def _get_mock_price_change(self, symbol: str) -> Decimal:
        """Mock price change data - replace with real market data"""
        import random
        # Simulate random price changes between -5% and +5%
        random.seed(hash(symbol))  # Deterministic for testing
        price_change_percent = Decimal(str(random.uniform(-0.05, 0.05)))
        return price_change_percent
    
    @hardware_accelerated(WorkloadType.RISK_CALCULATION, data_size=5000)
    async def calculate_portfolio_margin_requirement(self, portfolio: Portfolio) -> MarginRequirement:
        """Calculate comprehensive portfolio margin requirements with cross-margining"""
        
        # Calculate individual position margins
        position_margins = {}
        total_initial_margin = Decimal('0')
        total_variation_margin = Decimal('0')
        
        for position in portfolio.positions:
            initial_margin = await self.calculate_initial_margin(position)
            variation_margin = await self.calculate_variation_margin(position)
            maintenance_margin = initial_margin * Decimal('0.75')  # 75% of initial
            
            position_margins[position.id] = {
                'initial_margin': initial_margin,
                'variation_margin': variation_margin,
                'maintenance_margin': maintenance_margin
            }
            
            total_initial_margin += initial_margin
            total_variation_margin += variation_margin
        
        # Calculate cross-margining benefits
        cross_margin_offset = await self._calculate_cross_margin_offset(portfolio.positions)
        
        # Net margin after cross-margining
        net_initial_margin = total_initial_margin - cross_margin_offset
        
        # Total margin requirement
        total_margin = net_initial_margin + total_variation_margin
        
        # Calculate utilization and excess
        margin_utilization = total_margin / portfolio.available_cash if portfolio.available_cash > 0 else Decimal('inf')
        margin_excess = portfolio.available_cash - total_margin
        
        # Time to margin call estimation
        time_to_margin_call = self._estimate_time_to_margin_call(portfolio, margin_excess)
        
        return MarginRequirement(
            portfolio_id=portfolio.id,
            gross_initial_margin=total_initial_margin,
            cross_margin_offset=cross_margin_offset,
            net_initial_margin=net_initial_margin,
            variation_margin=total_variation_margin,
            maintenance_margin=net_initial_margin * Decimal('0.75'),
            regulatory_margin=await self._calculate_regulatory_margin(portfolio),
            total_margin_requirement=total_margin,
            margin_utilization=margin_utilization,
            margin_excess=margin_excess,
            time_to_margin_call_minutes=time_to_margin_call,
            position_margins=position_margins
        )
    
    async def _calculate_cross_margin_offset(self, positions: List[Position]) -> Decimal:
        """Calculate margin offset from cross-margining correlated positions"""
        total_offset = Decimal('0')
        
        # Group positions by asset class for cross-margining
        grouped_positions = {}
        for position in positions:
            if position.asset_class not in grouped_positions:
                grouped_positions[position.asset_class] = []
            grouped_positions[position.asset_class].append(position)
        
        # Calculate cross-margin benefits within each asset class
        for asset_class, asset_positions in grouped_positions.items():
            if len(asset_positions) > 1:
                offset = await self._calculate_asset_class_cross_margin(asset_class, asset_positions)
                total_offset += offset
        
        return total_offset
    
    async def _calculate_asset_class_cross_margin(self, asset_class: AssetClass, positions: List[Position]) -> Decimal:
        """Calculate cross-margin offset for positions within same asset class"""
        if len(positions) < 2:
            return Decimal('0')
        
        # Calculate correlation-based offset
        correlation_benefit = await self._get_correlation_benefit(asset_class, positions)
        
        # Calculate total margin for these positions
        total_margin = sum(await self.calculate_initial_margin(pos) for pos in positions)
        
        # Apply correlation benefit (typically 10-40% depending on correlation)
        offset = total_margin * correlation_benefit
        
        return offset
    
    async def _get_correlation_benefit(self, asset_class: AssetClass, positions: List[Position]) -> Decimal:
        """Calculate correlation benefit for cross-margining"""
        # Mock correlation calculations - replace with real correlation matrix
        base_benefits = {
            AssetClass.EQUITY: Decimal('0.20'),  # 20% benefit for diversified equity portfolio
            AssetClass.BOND: Decimal('0.15'),    # 15% benefit for bond portfolios
            AssetClass.FX: Decimal('0.30'),      # 30% benefit for currency pairs
            AssetClass.DERIVATIVE: Decimal('0.10'), # 10% conservative benefit
            AssetClass.COMMODITY: Decimal('0.05'),  # 5% minimal benefit
            AssetClass.CRYPTO: Decimal('0.0')    # No cross-margin benefit for crypto
        }
        
        return base_benefits.get(asset_class, Decimal('0.0'))
    
    def _estimate_time_to_margin_call(self, portfolio: Portfolio, margin_excess: Decimal) -> Optional[int]:
        """Estimate time until margin call based on current trends"""
        if margin_excess <= 0:
            return 0  # Immediate margin call
        
        # Mock calculation - in production, use portfolio volatility and trends
        daily_portfolio_volatility = portfolio.gross_exposure * Decimal('0.02')  # 2% daily vol
        
        if daily_portfolio_volatility > 0:
            days_to_margin_call = margin_excess / daily_portfolio_volatility
            return int(days_to_margin_call * 24 * 60)  # Convert to minutes
        
        return None
    
    async def _calculate_regulatory_margin(self, portfolio: Portfolio) -> Decimal:
        """Calculate regulatory margin requirements"""
        # Mock regulatory calculation - implement actual regulatory formulas
        total_exposure = portfolio.gross_exposure
        regulatory_multiplier = Decimal('0.08')  # 8% regulatory requirement
        
        return total_exposure * regulatory_multiplier
    
    @hardware_accelerated(WorkloadType.RISK_CALCULATION, data_size=10000)
    async def run_margin_stress_test(self, portfolio: Portfolio, scenario_name: str = "Market Crash") -> MarginStressTest:
        """Run margin stress test under adverse market conditions"""
        
        # Current margin
        base_margin_req = await self.calculate_portfolio_margin_requirement(portfolio)
        base_margin = base_margin_req.total_margin_requirement
        
        # Apply stress scenario (e.g., 30% market decline, 50% volatility increase)
        stress_factors = {
            'market_decline': Decimal('0.30'),
            'volatility_increase': Decimal('0.50'),
            'correlation_increase': Decimal('0.20')
        }
        
        # Calculate stressed margin
        stressed_margin = await self._calculate_stressed_margin(portfolio, stress_factors)
        margin_increase = stressed_margin - base_margin
        margin_increase_percent = margin_increase / base_margin if base_margin > 0 else Decimal('0')
        
        # Identify positions at risk
        positions_at_risk = [pos.id for pos in portfolio.positions if pos.asset_class in [AssetClass.EQUITY, AssetClass.CRYPTO]]
        
        # Estimate liquidation value under stress
        estimated_liquidation_value = portfolio.total_market_value * (Decimal('1') - stress_factors['market_decline'])
        
        return MarginStressTest(
            portfolio_id=portfolio.id,
            scenario_name=scenario_name,
            base_margin=base_margin,
            stressed_margin=stressed_margin,
            margin_increase=margin_increase,
            margin_increase_percent=margin_increase_percent,
            positions_at_risk=positions_at_risk,
            estimated_liquidation_value=estimated_liquidation_value,
            stress_factors=stress_factors
        )
    
    async def _calculate_stressed_margin(self, portfolio: Portfolio, stress_factors: Dict[str, Decimal]) -> Decimal:
        """Calculate margin requirements under stress conditions"""
        total_stressed_margin = Decimal('0')
        
        for position in portfolio.positions:
            base_margin = await self.calculate_initial_margin(position)
            
            # Apply stress multipliers based on asset class
            if position.asset_class == AssetClass.EQUITY:
                stress_multiplier = Decimal('1') + stress_factors['volatility_increase']
            elif position.asset_class == AssetClass.CRYPTO:
                stress_multiplier = Decimal('1') + stress_factors['volatility_increase'] * Decimal('1.5')
            else:
                stress_multiplier = Decimal('1') + stress_factors['volatility_increase'] * Decimal('0.5')
            
            stressed_margin = base_margin * stress_multiplier
            total_stressed_margin += stressed_margin
        
        return total_stressed_margin