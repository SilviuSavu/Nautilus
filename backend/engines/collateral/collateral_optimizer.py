"""
Collateral Optimizer
===================

Advanced collateral optimization engine for maximizing capital efficiency through:
- Cross-margining optimization across asset classes
- Portfolio-level margin netting
- Intelligent collateral allocation
- M4 Max hardware acceleration for complex optimization problems
"""

import asyncio
import logging
import numpy as np
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass

from models import (
    Position, Portfolio, AssetClass, CrossMarginBenefit,
    MarginRequirement
)

# Import M4 Max hardware acceleration
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


@dataclass
class OptimizationResult:
    """Result of collateral optimization"""
    original_margin: Decimal
    optimized_margin: Decimal
    margin_savings: Decimal
    capital_efficiency_improvement: Decimal
    cross_margin_benefits: List[CrossMarginBenefit]
    optimization_method: str
    computation_time_ms: Optional[float] = None


class CollateralOptimizer:
    """
    Advanced collateral optimization engine that maximizes capital efficiency through:
    - Cross-asset correlation analysis
    - Portfolio-level margin netting
    - Intelligent position clustering
    - Real-time optimization with M4 Max acceleration
    """
    
    def __init__(self):
        self.correlation_matrix = self._initialize_correlation_matrix()
        self.optimization_cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        
    def _initialize_correlation_matrix(self) -> Dict[str, Dict[str, Decimal]]:
        """Initialize correlation matrix for cross-margining calculations"""
        # Mock correlation matrix - in production, use real market data
        return {
            'EQUITY_EQUITY': {
                'SPY_AAPL': Decimal('0.75'),
                'SPY_MSFT': Decimal('0.70'),
                'AAPL_MSFT': Decimal('0.65'),
                'TECH_FINANCE': Decimal('0.45')
            },
            'BOND_BOND': {
                'UST_2Y_10Y': Decimal('0.85'),
                'UST_CORP': Decimal('0.65'),
                'GOVT_GOVT': Decimal('0.90')
            },
            'FX_FX': {
                'EUR_GBP': Decimal('0.70'),
                'USD_CHF': Decimal('-0.65'),  # Negative correlation
                'EUR_JPY': Decimal('0.45')
            },
            'CROSS_ASSET': {
                'EQUITY_BOND': Decimal('-0.25'),  # Flight to quality
                'EQUITY_GOLD': Decimal('-0.15'),
                'BOND_GOLD': Decimal('0.05')
            }
        }
    
    @hardware_accelerated(WorkloadType.ML_INFERENCE, data_size=5000)
    async def optimize_portfolio_margin(self, portfolio: Portfolio) -> OptimizationResult:
        """
        Optimize portfolio margin requirements through advanced cross-margining
        and portfolio-level netting strategies
        """
        start_time = datetime.now()
        
        try:
            # Calculate original margin without optimization
            from margin_calculator import MarginCalculator
            calculator = MarginCalculator()
            original_margin_req = await calculator.calculate_portfolio_margin_requirement(portfolio)
            original_margin = original_margin_req.total_margin_requirement
            
            # Apply optimization strategies
            cross_margin_benefits = await self._optimize_cross_margining(portfolio.positions)
            
            # Calculate optimized margin
            total_cross_margin_offset = sum(benefit.cross_margin_offset for benefit in cross_margin_benefits)
            optimized_margin = original_margin - total_cross_margin_offset
            
            # Ensure margin doesn't go below regulatory minimums
            regulatory_minimum = original_margin * Decimal('0.50')  # 50% minimum
            optimized_margin = max(optimized_margin, regulatory_minimum)
            
            margin_savings = original_margin - optimized_margin
            capital_efficiency_improvement = (margin_savings / original_margin * 100) if original_margin > 0 else Decimal('0')
            
            computation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return OptimizationResult(
                original_margin=original_margin,
                optimized_margin=optimized_margin,
                margin_savings=margin_savings,
                capital_efficiency_improvement=capital_efficiency_improvement,
                cross_margin_benefits=cross_margin_benefits,
                optimization_method="advanced_cross_margining",
                computation_time_ms=computation_time
            )
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio margin: {e}")
            # Return no optimization if error occurs
            return OptimizationResult(
                original_margin=Decimal('0'),
                optimized_margin=Decimal('0'),
                margin_savings=Decimal('0'),
                capital_efficiency_improvement=Decimal('0'),
                cross_margin_benefits=[],
                optimization_method="fallback"
            )
    
    async def _optimize_cross_margining(self, positions: List[Position]) -> List[CrossMarginBenefit]:
        """Optimize cross-margining across all positions"""
        cross_margin_benefits = []
        
        # Group positions by asset class
        asset_class_groups = self._group_positions_by_asset_class(positions)
        
        # Calculate within-asset-class cross-margining
        for asset_class, asset_positions in asset_class_groups.items():
            if len(asset_positions) > 1:
                benefit = await self._calculate_within_asset_class_benefit(asset_class, asset_positions)
                if benefit:
                    cross_margin_benefits.append(benefit)
        
        # Calculate cross-asset-class margining
        cross_asset_benefit = await self._calculate_cross_asset_benefit(asset_class_groups)
        if cross_asset_benefit:
            cross_margin_benefits.append(cross_asset_benefit)
        
        return cross_margin_benefits
    
    def _group_positions_by_asset_class(self, positions: List[Position]) -> Dict[AssetClass, List[Position]]:
        """Group positions by asset class for optimization"""
        groups = {}
        for position in positions:
            if position.asset_class not in groups:
                groups[position.asset_class] = []
            groups[position.asset_class].append(position)
        return groups
    
    @hardware_accelerated(WorkloadType.ML_INFERENCE, data_size=1000)
    async def _calculate_within_asset_class_benefit(self, asset_class: AssetClass, positions: List[Position]) -> Optional[CrossMarginBenefit]:
        """Calculate cross-margin benefit within same asset class"""
        if len(positions) < 2:
            return None
        
        # Calculate gross margin for all positions
        from margin_calculator import MarginCalculator
        calculator = MarginCalculator()
        
        gross_margin = Decimal('0')
        for position in positions:
            position_margin = await calculator.calculate_initial_margin(position)
            gross_margin += position_margin
        
        # Calculate correlation-based offset
        correlation_benefit = await self._get_asset_class_correlation_benefit(asset_class, positions)
        cross_margin_offset = gross_margin * correlation_benefit
        net_margin = gross_margin - cross_margin_offset
        
        offset_percentage = (cross_margin_offset / gross_margin * 100) if gross_margin > 0 else Decimal('0')
        
        return CrossMarginBenefit(
            asset_class=asset_class,
            position_ids=[pos.id for pos in positions],
            correlation_coefficient=correlation_benefit,
            gross_margin=gross_margin,
            cross_margin_offset=cross_margin_offset,
            net_margin=net_margin,
            offset_percentage=offset_percentage,
            calculation_method=f"within_{asset_class.value}_correlation"
        )
    
    async def _get_asset_class_correlation_benefit(self, asset_class: AssetClass, positions: List[Position]) -> Decimal:
        """Calculate correlation benefit for positions within same asset class"""
        
        # Base correlation benefits by asset class
        base_benefits = {
            AssetClass.EQUITY: await self._calculate_equity_correlation_benefit(positions),
            AssetClass.BOND: await self._calculate_bond_correlation_benefit(positions),
            AssetClass.FX: await self._calculate_fx_correlation_benefit(positions),
            AssetClass.DERIVATIVE: await self._calculate_derivative_correlation_benefit(positions),
            AssetClass.COMMODITY: Decimal('0.05'),  # 5% minimal benefit
            AssetClass.CRYPTO: Decimal('0.0')  # No correlation benefit for crypto
        }
        
        return base_benefits.get(asset_class, Decimal('0.0'))
    
    async def _calculate_equity_correlation_benefit(self, positions: List[Position]) -> Decimal:
        """Calculate correlation benefit for equity positions"""
        # Analyze sector diversification
        sectors = set(pos.sector for pos in positions if pos.sector)
        
        if len(sectors) >= 3:
            return Decimal('0.25')  # 25% benefit for well-diversified portfolio
        elif len(sectors) == 2:
            return Decimal('0.15')  # 15% benefit for some diversification
        else:
            return Decimal('0.05')  # 5% minimal benefit for single sector
    
    async def _calculate_bond_correlation_benefit(self, positions: List[Position]) -> Decimal:
        """Calculate correlation benefit for bond positions"""
        # Analyze duration and credit quality diversity
        durations = [pos.duration for pos in positions if pos.duration]
        
        if len(durations) >= 2:
            duration_spread = max(durations) - min(durations)
            if duration_spread > Decimal('3.0'):  # More than 3 years duration spread
                return Decimal('0.20')  # 20% benefit for duration diversification
            else:
                return Decimal('0.10')  # 10% benefit for some duration spread
        
        return Decimal('0.05')  # 5% minimal benefit
    
    async def _calculate_fx_correlation_benefit(self, positions: List[Position]) -> Decimal:
        """Calculate correlation benefit for FX positions"""
        # FX pairs often have strong correlations that can be exploited
        if len(positions) >= 2:
            return Decimal('0.30')  # 30% benefit for FX cross-margining
        return Decimal('0.0')
    
    async def _calculate_derivative_correlation_benefit(self, positions: List[Position]) -> Decimal:
        """Calculate correlation benefit for derivative positions"""
        # Conservative approach for derivatives due to complexity
        return Decimal('0.10')  # 10% benefit
    
    @hardware_accelerated(WorkloadType.ML_INFERENCE, data_size=2000)
    async def _calculate_cross_asset_benefit(self, asset_class_groups: Dict[AssetClass, List[Position]]) -> Optional[CrossMarginBenefit]:
        """Calculate cross-margining benefit across different asset classes"""
        if len(asset_class_groups) < 2:
            return None
        
        # Focus on major cross-asset correlations
        total_cross_offset = Decimal('0')
        all_positions = []
        total_gross_margin = Decimal('0')
        
        from margin_calculator import MarginCalculator
        calculator = MarginCalculator()
        
        # Calculate gross margin across all asset classes
        for positions in asset_class_groups.values():
            all_positions.extend(positions)
            for position in positions:
                position_margin = await calculator.calculate_initial_margin(position)
                total_gross_margin += position_margin
        
        # Apply cross-asset correlation benefits
        if AssetClass.EQUITY in asset_class_groups and AssetClass.BOND in asset_class_groups:
            # Equity-Bond negative correlation benefit (flight to quality)
            equity_margin = sum(await calculator.calculate_initial_margin(pos) for pos in asset_class_groups[AssetClass.EQUITY])
            bond_margin = sum(await calculator.calculate_initial_margin(pos) for pos in asset_class_groups[AssetClass.BOND])
            
            # 10% offset for equity-bond negative correlation
            cross_offset = min(equity_margin, bond_margin) * Decimal('0.10')
            total_cross_offset += cross_offset
        
        if AssetClass.EQUITY in asset_class_groups and AssetClass.COMMODITY in asset_class_groups:
            # Equity-Commodity correlation benefit
            equity_margin = sum(await calculator.calculate_initial_margin(pos) for pos in asset_class_groups[AssetClass.EQUITY])
            commodity_margin = sum(await calculator.calculate_initial_margin(pos) for pos in asset_class_groups[AssetClass.COMMODITY])
            
            # 5% offset for equity-commodity correlation
            cross_offset = min(equity_margin, commodity_margin) * Decimal('0.05')
            total_cross_offset += cross_offset
        
        if total_cross_offset > 0:
            net_margin = total_gross_margin - total_cross_offset
            offset_percentage = (total_cross_offset / total_gross_margin * 100) if total_gross_margin > 0 else Decimal('0')
            
            return CrossMarginBenefit(
                asset_class=AssetClass.EQUITY,  # Placeholder - this is cross-asset
                position_ids=[pos.id for pos in all_positions],
                correlation_coefficient=Decimal('0.15'),  # Average cross-asset correlation
                gross_margin=total_gross_margin,
                cross_margin_offset=total_cross_offset,
                net_margin=net_margin,
                offset_percentage=offset_percentage,
                calculation_method="cross_asset_correlation"
            )
        
        return None
    
    @hardware_accelerated(WorkloadType.ML_INFERENCE, data_size=10000)
    async def optimize_collateral_allocation(self, portfolio: Portfolio, available_collateral: Dict[str, Decimal]) -> Dict[str, Any]:
        """
        Optimize allocation of different types of collateral to minimize margin requirements
        and maximize capital efficiency
        """
        
        # Collateral hierarchy (preference order)
        collateral_hierarchy = {
            'cash_usd': {'haircut': Decimal('0.0'), 'preference': 1},
            'cash_eur': {'haircut': Decimal('0.02'), 'preference': 2},
            'treasury_bills': {'haircut': Decimal('0.01'), 'preference': 3},
            'government_bonds': {'haircut': Decimal('0.04'), 'preference': 4},
            'corporate_bonds': {'haircut': Decimal('0.08'), 'preference': 5},
            'equity_securities': {'haircut': Decimal('0.15'), 'preference': 6}
        }
        
        # Calculate margin requirement
        from margin_calculator import MarginCalculator
        calculator = MarginCalculator()
        margin_req = await calculator.calculate_portfolio_margin_requirement(portfolio)
        required_collateral = margin_req.total_margin_requirement
        
        # Optimize collateral allocation
        allocation = {}
        remaining_requirement = required_collateral
        
        # Allocate collateral in preference order
        for collateral_type, config in sorted(collateral_hierarchy.items(), key=lambda x: x[1]['preference']):
            if collateral_type in available_collateral and remaining_requirement > 0:
                available_amount = available_collateral[collateral_type]
                haircut = config['haircut']
                
                # Calculate effective collateral value after haircut
                effective_value = available_amount * (Decimal('1') - haircut)
                
                # Allocate up to remaining requirement
                allocated_amount = min(effective_value, remaining_requirement)
                
                if allocated_amount > 0:
                    # Convert back to nominal amount considering haircut
                    nominal_amount = allocated_amount / (Decimal('1') - haircut)
                    allocation[collateral_type] = {
                        'nominal_amount': nominal_amount,
                        'effective_value': allocated_amount,
                        'haircut': haircut
                    }
                    remaining_requirement -= allocated_amount
        
        return {
            'total_margin_requirement': required_collateral,
            'collateral_allocation': allocation,
            'remaining_shortfall': remaining_requirement,
            'total_haircut_cost': required_collateral - sum(alloc['effective_value'] for alloc in allocation.values()),
            'optimization_efficiency': ((required_collateral - remaining_requirement) / required_collateral * 100) if required_collateral > 0 else Decimal('0')
        }
    
    async def suggest_portfolio_rebalancing(self, portfolio: Portfolio, target_margin_reduction: Decimal) -> Dict[str, Any]:
        """
        Suggest portfolio rebalancing to achieve target margin reduction while
        maintaining investment objectives
        """
        
        from margin_calculator import MarginCalculator
        calculator = MarginCalculator()
        
        # Current margin requirement
        current_margin_req = await calculator.calculate_portfolio_margin_requirement(portfolio)
        current_margin = current_margin_req.total_margin_requirement
        target_margin = current_margin - target_margin_reduction
        
        # Analyze positions by margin efficiency (return per unit of margin)
        position_efficiency = []
        
        for position in portfolio.positions:
            position_margin = await calculator.calculate_initial_margin(position)
            
            # Mock return calculation - replace with real performance data
            expected_return = self._get_mock_expected_return(position)
            
            margin_efficiency = expected_return / position_margin if position_margin > 0 else Decimal('0')
            
            position_efficiency.append({
                'position_id': position.id,
                'symbol': position.symbol,
                'current_margin': position_margin,
                'expected_return': expected_return,
                'margin_efficiency': margin_efficiency,
                'notional_value': position.notional_value
            })
        
        # Sort by margin efficiency (lowest efficiency first for potential reduction)
        position_efficiency.sort(key=lambda x: x['margin_efficiency'])
        
        # Suggest rebalancing
        suggestions = []
        margin_saved = Decimal('0')
        
        for pos_data in position_efficiency:
            if margin_saved >= target_margin_reduction:
                break
                
            # Suggest reducing position by 25%
            reduction_percent = Decimal('0.25')
            margin_reduction = pos_data['current_margin'] * reduction_percent
            
            suggestions.append({
                'action': 'reduce',
                'position_id': pos_data['position_id'],
                'symbol': pos_data['symbol'],
                'reduction_percent': float(reduction_percent * 100),
                'margin_saved': margin_reduction,
                'efficiency_score': pos_data['margin_efficiency']
            })
            
            margin_saved += margin_reduction
        
        return {
            'current_margin': current_margin,
            'target_margin': target_margin,
            'target_reduction': target_margin_reduction,
            'achievable_reduction': margin_saved,
            'rebalancing_suggestions': suggestions,
            'success_probability': min(Decimal('1.0'), margin_saved / target_margin_reduction) if target_margin_reduction > 0 else Decimal('1.0')
        }
    
    def _get_mock_expected_return(self, position: Position) -> Decimal:
        """Mock expected return calculation - replace with real alpha models"""
        # Mock returns based on asset class
        mock_returns = {
            AssetClass.EQUITY: Decimal('0.08'),     # 8% expected return
            AssetClass.BOND: Decimal('0.03'),       # 3% expected return
            AssetClass.FX: Decimal('0.02'),         # 2% expected return
            AssetClass.COMMODITY: Decimal('0.05'),  # 5% expected return
            AssetClass.DERIVATIVE: Decimal('0.10'), # 10% expected return
            AssetClass.CRYPTO: Decimal('0.15')      # 15% expected return (high risk)
        }
        
        base_return = mock_returns.get(position.asset_class, Decimal('0.05'))
        return base_return * position.notional_value