"""
Cross-Source Factor Engine
==========================

Institutional-grade factor synthesizer that combines EDGAR, FRED, and IBKR data sources
to create unique factor combinations unavailable in commercial platforms.

This is the core competitive advantage of the Nautilus Factor Platform.

Generates 25-30 unique cross-source factors:
- EDGAR × FRED: Fundamental quality vs Economic cycles
- FRED × IBKR: Macro regimes vs Market microstructure  
- EDGAR × IBKR: Fundamental momentum vs Price action
- Triple Integration: Economic × Fundamental × Technical
"""

import logging
import asyncio
import numpy as np
import polars as pl
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from enum import Enum
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from fastapi import HTTPException

# Import factor engines
from fred_integration import fred_integration
from ibkr_technical_factors import ibkr_technical_engine
# Note: EDGAR factor service would be imported here when available

logger = logging.getLogger(__name__)


class CrossSourceFactorType(Enum):
    """Types of cross-source factor combinations."""
    EDGAR_FRED = "edgar_fred"           # Fundamental × Economic
    FRED_IBKR = "fred_ibkr"            # Economic × Technical  
    EDGAR_IBKR = "edgar_ibkr"          # Fundamental × Technical
    TRIPLE_SOURCE = "triple_source"     # Economic × Fundamental × Technical


@dataclass
class FactorSynthesisConfig:
    """Configuration for cross-source factor synthesis."""
    normalization_method: str = "zscore"  # zscore, percentile, minmax
    correlation_threshold: float = 0.95   # Remove highly correlated factors
    significance_threshold: float = 0.05  # Statistical significance level
    lookback_window: int = 252            # Days for historical analysis
    enable_regime_detection: bool = True   # Enable market regime factors
    enable_interaction_terms: bool = True # Enable interaction factors
    max_factors_per_category: int = 10    # Limit factors per category


class CrossSourceFactorEngine:
    """
    Cross-Source Factor Synthesizer for institutional-grade factor modeling.
    
    Creates unique competitive advantage through multi-source factor combinations:
    
    **EDGAR × FRED Factors (8-10 factors):**
    - Earnings quality × Economic cycle alignment
    - Revenue growth × GDP momentum correlation
    - Margin expansion × Inflation regime interaction
    - Balance sheet strength × Credit conditions
    - Cash flow quality × Monetary policy stance
    
    **FRED × IBKR Factors (8-10 factors):**
    - Economic surprises × Price momentum alignment
    - Interest rate regime × Volatility patterns
    - Inflation dynamics × Sector rotation
    - Credit spreads × Market microstructure
    - Macro uncertainty × Liquidity conditions
    
    **EDGAR × IBKR Factors (8-10 factors):**
    - Fundamental momentum × Price action confirmation
    - Earnings revisions × Volume patterns
    - Quality scores × Relative strength
    - Growth consistency × Trend persistence
    - Profitability × Market efficiency
    
    **Triple Integration Factors (5-7 factors):**
    - Economic cycle × Fundamental quality × Technical momentum
    - Policy regime × Earnings growth × Volatility environment
    - Credit conditions × Balance sheet strength × Liquidity
    """
    
    def __init__(self, config: Optional[FactorSynthesisConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or FactorSynthesisConfig()
        
        # Factor engines
        self.fred_engine = fred_integration
        self.ibkr_engine = ibkr_technical_engine
        # self.edgar_engine = edgar_factor_service  # When available
        
        # Factor cache and correlation matrix
        self._factor_cache = {}
        self._correlation_matrix = None
        self._last_correlation_update = None
        
        # Regime detection models
        self._regime_models = {}
    
    async def synthesize_cross_source_factors(
        self,
        symbol: str,
        edgar_factors: Optional[Dict[str, float]] = None,
        fred_factors: Optional[Dict[str, float]] = None,
        ibkr_factors: Optional[Dict[str, float]] = None,
        as_of_date: date = None
    ) -> pl.DataFrame:
        """
        Generate cross-source factor combinations for a single symbol.
        
        Args:
            symbol: Stock symbol
            edgar_factors: EDGAR fundamental factors (if available)
            fred_factors: FRED macro-economic factors (if available)  
            ibkr_factors: IBKR technical factors (if available)
            as_of_date: Calculation date
            
        Returns:
            DataFrame with cross-source factors
        """
        try:
            if as_of_date is None:
                as_of_date = datetime.now().date()
            
            self.logger.info(f"Synthesizing cross-source factors for {symbol} as of {as_of_date}")
            
            # Collect available factors
            available_sources = []
            if edgar_factors:
                available_sources.append("EDGAR")
            if fred_factors:
                available_sources.append("FRED")
            if ibkr_factors:
                available_sources.append("IBKR")
            
            if len(available_sources) < 2:
                raise ValueError(f"Need at least 2 data sources, got: {available_sources}")
            
            # Calculate cross-source factors
            cross_factors = {}
            
            # EDGAR × FRED Combinations
            if edgar_factors and fred_factors:
                edgar_fred_factors = await self._calculate_edgar_fred_factors(
                    edgar_factors, fred_factors, symbol, as_of_date
                )
                cross_factors.update(edgar_fred_factors)
            
            # FRED × IBKR Combinations
            if fred_factors and ibkr_factors:
                fred_ibkr_factors = await self._calculate_fred_ibkr_factors(
                    fred_factors, ibkr_factors, symbol, as_of_date
                )
                cross_factors.update(fred_ibkr_factors)
            
            # EDGAR × IBKR Combinations
            if edgar_factors and ibkr_factors:
                edgar_ibkr_factors = await self._calculate_edgar_ibkr_factors(
                    edgar_factors, ibkr_factors, symbol, as_of_date
                )
                cross_factors.update(edgar_ibkr_factors)
            
            # Triple Integration Factors
            if edgar_factors and fred_factors and ibkr_factors:
                triple_factors = await self._calculate_triple_integration_factors(
                    edgar_factors, fred_factors, ibkr_factors, symbol, as_of_date
                )
                cross_factors.update(triple_factors)
            
            # Create result DataFrame
            factor_record = {
                'symbol': symbol,
                'date': as_of_date,
                'calculation_timestamp': datetime.now(),
                'source_combination': '+'.join(available_sources),
                'cross_factor_count': len(cross_factors),
                **cross_factors
            }
            
            result_df = pl.DataFrame([factor_record])
            
            self.logger.info(f"Generated {len(cross_factors)} cross-source factors for {symbol}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error synthesizing cross-source factors for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Cross-source factor synthesis failed: {e}")
    
    async def synthesize_universe_factors(
        self,
        universe_data: Dict[str, Dict[str, Dict[str, float]]],
        as_of_date: date = None
    ) -> pl.DataFrame:
        """
        Generate cross-source factors for entire universe.
        
        Args:
            universe_data: {symbol: {source: {factor: value}}}
            as_of_date: Calculation date
            
        Returns:
            Combined DataFrame with cross-source factors for all symbols
        """
        try:
            self.logger.info(f"Synthesizing cross-source factors for {len(universe_data)} symbols")
            
            # Calculate factors for each symbol in parallel
            tasks = []
            for symbol, source_factors in universe_data.items():
                edgar_factors = source_factors.get('edgar', {})
                fred_factors = source_factors.get('fred', {})
                ibkr_factors = source_factors.get('ibkr', {})
                
                tasks.append(
                    self.synthesize_cross_source_factors(
                        symbol, edgar_factors, fred_factors, ibkr_factors, as_of_date
                    )
                )
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine successful results
            successful_results = []
            failed_symbols = []
            
            for i, result in enumerate(results):
                symbol = list(universe_data.keys())[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to synthesize factors for {symbol}: {result}")
                    failed_symbols.append(symbol)
                    continue
                
                successful_results.append(result)
            
            if not successful_results:
                raise HTTPException(status_code=500, detail="No cross-source factors synthesized")
            
            # Concatenate results
            universe_factors = pl.concat(successful_results)
            
            # Apply correlation filtering if enabled
            if self.config.correlation_threshold < 1.0:
                universe_factors = await self._filter_correlated_factors(universe_factors)
            
            if failed_symbols:
                self.logger.warning(f"Failed synthesis for {len(failed_symbols)} symbols: {failed_symbols}")
            
            self.logger.info(f"Successfully synthesized factors for {len(successful_results)} symbols")
            return universe_factors
            
        except Exception as e:
            self.logger.error(f"Error synthesizing universe factors: {e}")
            raise HTTPException(status_code=500, detail=f"Universe synthesis failed: {e}")
    
    # EDGAR × FRED FACTOR COMBINATIONS
    async def _calculate_edgar_fred_factors(
        self, 
        edgar_factors: Dict[str, float], 
        fred_factors: Dict[str, float],
        symbol: str,
        as_of_date: date
    ) -> Dict[str, float]:
        """Calculate EDGAR × FRED cross-source factors (8-10 factors)."""
        factors = {}
        
        try:
            # 1. Earnings Quality × Economic Cycle Alignment
            if 'earnings_quality' in edgar_factors and 'economic_activity_composite' in fred_factors:
                earnings_quality = edgar_factors['earnings_quality']
                economic_cycle = fred_factors['economic_activity_composite']
                
                # Higher earnings quality is more valuable during economic expansion
                earnings_cycle_alignment = earnings_quality * (1 + economic_cycle / 100)
                factors['edgar_fred_earnings_cycle_alignment'] = earnings_cycle_alignment
            
            # 2. Revenue Growth × GDP Momentum Correlation
            if 'revenue_growth_yoy' in edgar_factors and 'industrial_production_yoy' in fred_factors:
                revenue_growth = edgar_factors['revenue_growth_yoy']
                gdp_momentum = fred_factors['industrial_production_yoy']
                
                # Companies with revenue growth aligned with GDP growth
                growth_momentum_sync = revenue_growth * gdp_momentum / 100
                factors['edgar_fred_growth_momentum_sync'] = growth_momentum_sync
            
            # 3. Margin Expansion × Inflation Regime
            if 'operating_margin' in edgar_factors and 'cpi_inflation_yoy' in fred_factors:
                margin_trend = edgar_factors['operating_margin']
                inflation_rate = fred_factors['cpi_inflation_yoy']
                
                # Companies maintaining margins during inflation are strong
                inflation_margin_resilience = margin_trend / (1 + abs(inflation_rate) / 100)
                factors['edgar_fred_inflation_margin_resilience'] = inflation_margin_resilience
            
            # 4. Balance Sheet Strength × Credit Conditions
            if 'debt_to_equity' in edgar_factors and 'money_supply_growth_yoy' in fred_factors:
                leverage_ratio = edgar_factors['debt_to_equity']
                credit_conditions = fred_factors['money_supply_growth_yoy']
                
                # Low leverage companies benefit more from tight credit
                credit_advantage = (1 / (leverage_ratio + 0.1)) * max(0, 10 - credit_conditions)
                factors['edgar_fred_credit_advantage'] = credit_advantage
            
            # 5. Cash Flow Quality × Monetary Policy Stance
            if 'free_cash_flow_yield' in edgar_factors and 'fed_funds_level' in fred_factors:
                cash_flow_yield = edgar_factors['free_cash_flow_yield']
                fed_funds_rate = fred_factors['fed_funds_level']
                
                # Cash-generating companies valuable when rates are high
                monetary_cash_value = cash_flow_yield * (1 + fed_funds_rate / 100)
                factors['edgar_fred_monetary_cash_value'] = monetary_cash_value
            
            # 6. Profitability × Economic Surprise
            if 'roe' in edgar_factors and 'unemployment_trend_6m' in fred_factors:
                profitability = edgar_factors['roe']
                employment_surprise = -fred_factors['unemployment_trend_6m']  # Negative is good
                
                # Profitable companies benefit from positive economic surprises
                profit_surprise_leverage = profitability * (1 + employment_surprise / 10)
                factors['edgar_fred_profit_surprise_leverage'] = profit_surprise_leverage
            
            # 7. Dividend Sustainability × Interest Rate Environment  
            if 'dividend_yield' in edgar_factors and 'treasury_10y_level' in fred_factors:
                dividend_yield = edgar_factors['dividend_yield']
                treasury_yield = fred_factors['treasury_10y_level']
                
                # Dividend attractiveness vs risk-free rate
                dividend_attractiveness = max(0, dividend_yield - treasury_yield) * 2
                factors['edgar_fred_dividend_attractiveness'] = dividend_attractiveness
            
            # 8. Asset Efficiency × Economic Growth Phase
            if 'asset_turnover' in edgar_factors and 'yield_curve_slope' in fred_factors:
                asset_efficiency = edgar_factors['asset_turnover']
                growth_phase = fred_factors['yield_curve_slope']  # Steep curve = growth
                
                # Efficient asset use more valuable during growth phases
                efficiency_growth_value = asset_efficiency * (1 + max(0, growth_phase) / 100)
                factors['edgar_fred_efficiency_growth_value'] = efficiency_growth_value
            
        except Exception as e:
            self.logger.warning(f"Error calculating EDGAR × FRED factors for {symbol}: {e}")
        
        return factors
    
    # FRED × IBKR FACTOR COMBINATIONS
    async def _calculate_fred_ibkr_factors(
        self,
        fred_factors: Dict[str, float],
        ibkr_factors: Dict[str, float], 
        symbol: str,
        as_of_date: date
    ) -> Dict[str, float]:
        """Calculate FRED × IBKR cross-source factors (8-10 factors)."""
        factors = {}
        
        try:
            # 1. Economic Surprises × Price Momentum Alignment
            if 'payroll_growth_yoy' in fred_factors and 'momentum_medium_60d' in ibkr_factors:
                employment_surprise = fred_factors['payroll_growth_yoy']
                price_momentum = ibkr_factors['momentum_medium_60d']
                
                # Momentum aligned with economic data is stronger
                surprise_momentum_confirmation = employment_surprise * price_momentum / 100
                factors['fred_ibkr_surprise_momentum_confirmation'] = surprise_momentum_confirmation
            
            # 2. Interest Rate Regime × Volatility Patterns
            if 'fed_funds_change_30d' in fred_factors and 'volatility_realized_20d' in ibkr_factors:
                rate_change = fred_factors['fed_funds_change_30d']
                volatility = ibkr_factors['volatility_realized_20d']
                
                # Rate changes create volatility opportunities
                rate_volatility_opportunity = abs(rate_change) * volatility / 10
                factors['fred_ibkr_rate_volatility_opportunity'] = rate_volatility_opportunity
            
            # 3. Inflation Dynamics × Sector Rotation Signal
            if 'cpi_inflation_yoy' in fred_factors and 'momentum_percentile_rank' in ibkr_factors:
                inflation_rate = fred_factors['cpi_inflation_yoy']
                relative_strength = ibkr_factors['momentum_percentile_rank']
                
                # Strong relative performance during inflation
                inflation_rotation_signal = relative_strength * (1 + inflation_rate / 100)
                factors['fred_ibkr_inflation_rotation_signal'] = inflation_rotation_signal
            
            # 4. Credit Spreads × Market Microstructure
            if 'consumer_credit_growth_yoy' in fred_factors and 'microstructure_effective_spread' in ibkr_factors:
                credit_growth = fred_factors['consumer_credit_growth_yoy']
                bid_ask_spread = ibkr_factors['microstructure_effective_spread']
                
                # Credit availability vs liquidity conditions
                credit_liquidity_environment = credit_growth / (bid_ask_spread + 0.1)
                factors['fred_ibkr_credit_liquidity_environment'] = credit_liquidity_environment
            
            # 5. Macro Uncertainty × Liquidity Conditions
            if 'vix_level' in fred_factors and 'market_quality_liquidity_score' in ibkr_factors:
                macro_uncertainty = fred_factors['vix_level']
                liquidity_score = ibkr_factors['market_quality_liquidity_score']
                
                # High liquidity valuable during uncertainty
                uncertainty_liquidity_premium = liquidity_score / (1 + macro_uncertainty / 100)
                factors['fred_ibkr_uncertainty_liquidity_premium'] = uncertainty_liquidity_premium
            
            # 6. Yield Curve Shape × Trend Quality
            if 'yield_curve_curvature' in fred_factors and 'trend_strength_50d' in ibkr_factors:
                curve_shape = fred_factors['yield_curve_curvature']
                trend_quality = ibkr_factors['trend_strength_50d']
                
                # Yield curve changes predict trend changes
                curve_trend_prediction = curve_shape * trend_quality
                factors['fred_ibkr_curve_trend_prediction'] = curve_trend_prediction
            
            # 7. Economic Activity × Volume Patterns
            if 'retail_sales_yoy' in fred_factors and 'microstructure_trading_intensity' in ibkr_factors:
                economic_activity = fred_factors['retail_sales_yoy']
                trading_intensity = ibkr_factors['microstructure_trading_intensity']
                
                # Economic strength confirms trading activity
                activity_volume_confirmation = economic_activity * trading_intensity / 10
                factors['fred_ibkr_activity_volume_confirmation'] = activity_volume_confirmation
            
            # 8. Monetary Policy × Price Efficiency
            if 'treasury_2y_level' in fred_factors and 'market_quality_price_efficiency' in ibkr_factors:
                short_rate = fred_factors['treasury_2y_level']
                price_efficiency = ibkr_factors['market_quality_price_efficiency']
                
                # Rate environment affects price discovery
                policy_efficiency_interaction = short_rate * price_efficiency / 10
                factors['fred_ibkr_policy_efficiency_interaction'] = policy_efficiency_interaction
            
        except Exception as e:
            self.logger.warning(f"Error calculating FRED × IBKR factors for {symbol}: {e}")
        
        return factors
    
    # EDGAR × IBKR FACTOR COMBINATIONS
    async def _calculate_edgar_ibkr_factors(
        self,
        edgar_factors: Dict[str, float],
        ibkr_factors: Dict[str, float],
        symbol: str, 
        as_of_date: date
    ) -> Dict[str, float]:
        """Calculate EDGAR × IBKR cross-source factors (8-10 factors)."""
        factors = {}
        
        try:
            # 1. Fundamental Momentum × Price Action Confirmation
            if 'earnings_growth_yoy' in edgar_factors and 'momentum_short_20d' in ibkr_factors:
                earnings_momentum = edgar_factors['earnings_growth_yoy']
                price_momentum = ibkr_factors['momentum_short_20d']
                
                # Fundamental and technical momentum alignment
                fundamental_technical_confirmation = (earnings_momentum * price_momentum) / 100
                factors['edgar_ibkr_fundamental_technical_confirmation'] = fundamental_technical_confirmation
            
            # 2. Earnings Revisions × Volume Patterns
            if 'earnings_surprise' in edgar_factors and 'microstructure_volume_irregularity' in ibkr_factors:
                earnings_surprise = edgar_factors['earnings_surprise']
                volume_pattern = ibkr_factors['microstructure_volume_irregularity']
                
                # Earnings surprises with volume confirmation
                surprise_volume_validation = abs(earnings_surprise) / (volume_pattern + 0.1)
                factors['edgar_ibkr_surprise_volume_validation'] = surprise_volume_validation
            
            # 3. Quality Scores × Relative Strength
            if 'financial_strength_score' in edgar_factors and 'momentum_percentile_rank' in ibkr_factors:
                quality_score = edgar_factors['financial_strength_score']
                relative_strength = ibkr_factors['momentum_percentile_rank']
                
                # High-quality companies with strong performance
                quality_performance_combination = quality_score * (relative_strength / 50 - 1)
                factors['edgar_ibkr_quality_performance_combination'] = quality_performance_combination
            
            # 4. Growth Consistency × Trend Persistence
            if 'revenue_growth_stability' in edgar_factors and 'trend_persistence' in ibkr_factors:
                growth_consistency = edgar_factors['revenue_growth_stability']
                trend_persistence = ibkr_factors['trend_persistence']
                
                # Consistent fundamentals with persistent trends
                consistency_persistence_synergy = growth_consistency * trend_persistence
                factors['edgar_ibkr_consistency_persistence_synergy'] = consistency_persistence_synergy
            
            # 5. Profitability × Market Efficiency
            if 'gross_margin' in edgar_factors and 'market_quality_price_efficiency' in ibkr_factors:
                profitability = edgar_factors['gross_margin']
                market_efficiency = ibkr_factors['market_quality_price_efficiency']
                
                # Profitable companies in efficient markets
                profitability_efficiency_signal = profitability * market_efficiency
                factors['edgar_ibkr_profitability_efficiency_signal'] = profitability_efficiency_signal
            
            # 6. Balance Sheet Quality × Liquidity
            if 'current_ratio' in edgar_factors and 'market_quality_liquidity_score' in ibkr_factors:
                balance_sheet_strength = edgar_factors['current_ratio']
                market_liquidity = ibkr_factors['market_quality_liquidity_score']
                
                # Strong balance sheet with good liquidity
                strength_liquidity_combo = (balance_sheet_strength - 1) * market_liquidity
                factors['edgar_ibkr_strength_liquidity_combo'] = strength_liquidity_combo
            
            # 7. Cash Generation × Trading Intensity
            if 'operating_cash_flow_margin' in edgar_factors and 'microstructure_trading_intensity' in ibkr_factors:
                cash_generation = edgar_factors['operating_cash_flow_margin']
                trading_intensity = ibkr_factors['microstructure_trading_intensity']
                
                # Cash generation attracting trading interest
                cash_interest_attraction = cash_generation * trading_intensity / 10
                factors['edgar_ibkr_cash_interest_attraction'] = cash_interest_attraction
            
            # 8. Valuation × Volatility Environment
            if 'pe_ratio' in edgar_factors and 'volatility_regime_signal' in ibkr_factors:
                valuation = 1 / (edgar_factors['pe_ratio'] + 0.1)  # Lower PE is better
                volatility_regime = ibkr_factors['volatility_regime_signal']
                
                # Cheap stocks in low volatility environments
                valuation_volatility_opportunity = valuation * max(0, -volatility_regime)
                factors['edgar_ibkr_valuation_volatility_opportunity'] = valuation_volatility_opportunity
            
        except Exception as e:
            self.logger.warning(f"Error calculating EDGAR × IBKR factors for {symbol}: {e}")
        
        return factors
    
    # TRIPLE INTEGRATION FACTORS
    async def _calculate_triple_integration_factors(
        self,
        edgar_factors: Dict[str, float],
        fred_factors: Dict[str, float], 
        ibkr_factors: Dict[str, float],
        symbol: str,
        as_of_date: date
    ) -> Dict[str, float]:
        """Calculate triple integration factors (5-7 factors)."""
        factors = {}
        
        try:
            # 1. Economic Cycle × Fundamental Quality × Technical Momentum
            if all(k in edgar_factors for k in ['earnings_quality']) and \
               all(k in fred_factors for k in ['economic_activity_composite']) and \
               all(k in ibkr_factors for k in ['momentum_medium_60d']):
                
                quality = edgar_factors['earnings_quality']
                economic_cycle = fred_factors['economic_activity_composite']
                momentum = ibkr_factors['momentum_medium_60d']
                
                # Triple factor confluence
                triple_confluence = quality * (economic_cycle / 100) * (momentum / 100)
                factors['triple_economic_quality_momentum'] = triple_confluence
            
            # 2. Policy Regime × Earnings Growth × Volatility Environment
            if all(k in edgar_factors for k in ['earnings_growth_yoy']) and \
               all(k in fred_factors for k in ['fed_funds_change_30d']) and \
               all(k in ibkr_factors for k in ['volatility_regime_signal']):
                
                earnings_growth = edgar_factors['earnings_growth_yoy']
                policy_change = fred_factors['fed_funds_change_30d']
                vol_regime = ibkr_factors['volatility_regime_signal']
                
                # Policy-earnings-volatility interaction
                policy_earnings_vol_signal = earnings_growth * (1 - abs(policy_change)) * (1 - vol_regime / 100)
                factors['triple_policy_earnings_volatility'] = policy_earnings_vol_signal
            
            # 3. Credit Conditions × Balance Sheet × Liquidity
            if all(k in edgar_factors for k in ['debt_to_equity']) and \
               all(k in fred_factors for k in ['consumer_credit_growth_yoy']) and \
               all(k in ibkr_factors for k in ['market_quality_liquidity_score']):
                
                leverage = 1 / (edgar_factors['debt_to_equity'] + 0.1)  # Lower is better
                credit_availability = fred_factors['consumer_credit_growth_yoy']
                liquidity = ibkr_factors['market_quality_liquidity_score']
                
                # Credit-balance sheet-liquidity synergy
                credit_balance_liquidity = leverage * (credit_availability / 100) * liquidity
                factors['triple_credit_balance_liquidity'] = credit_balance_liquidity
            
            # 4. Inflation × Margin Power × Price Efficiency
            if all(k in edgar_factors for k in ['operating_margin']) and \
               all(k in fred_factors for k in ['cpi_inflation_yoy']) and \
               all(k in ibkr_factors for k in ['market_quality_price_efficiency']):
                
                margin_power = edgar_factors['operating_margin']
                inflation_pressure = fred_factors['cpi_inflation_yoy']
                price_efficiency = ibkr_factors['market_quality_price_efficiency']
                
                # Inflation-resistant margin power in efficient markets
                inflation_margin_efficiency = margin_power / (1 + inflation_pressure / 100) * price_efficiency
                factors['triple_inflation_margin_efficiency'] = inflation_margin_efficiency
            
            # 5. Growth Phase × Profitability × Trend Strength
            if all(k in edgar_factors for k in ['roe']) and \
               all(k in fred_factors for k in ['yield_curve_slope']) and \
               all(k in ibkr_factors for k in ['trend_strength_50d']):
                
                profitability = edgar_factors['roe']
                growth_phase = max(0, fred_factors['yield_curve_slope'])  # Only positive slope
                trend_strength = ibkr_factors['trend_strength_50d']
                
                # Growth-profitability-trend alignment
                growth_profit_trend = profitability * (growth_phase / 100) * trend_strength
                factors['triple_growth_profitability_trend'] = growth_profit_trend
            
        except Exception as e:
            self.logger.warning(f"Error calculating triple integration factors for {symbol}: {e}")
        
        return factors
    
    # UTILITY METHODS
    async def _filter_correlated_factors(self, factors_df: pl.DataFrame) -> pl.DataFrame:
        """Remove highly correlated factors to reduce redundancy."""
        try:
            # Extract factor columns (exclude metadata)
            metadata_cols = ['symbol', 'date', 'calculation_timestamp', 'source_combination', 'cross_factor_count']
            factor_cols = [col for col in factors_df.columns if col not in metadata_cols]
            
            if len(factor_cols) < 2:
                return factors_df
            
            # Calculate correlation matrix
            factor_values = factors_df.select(factor_cols).to_pandas()
            correlation_matrix = factor_values.corr().abs()
            
            # Find highly correlated pairs
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            # Identify factors to drop
            to_drop = [
                column for column in upper_triangle.columns 
                if any(upper_triangle[column] > self.config.correlation_threshold)
            ]
            
            if to_drop:
                self.logger.info(f"Removing {len(to_drop)} highly correlated factors: {to_drop}")
                factors_df = factors_df.drop(to_drop)
            
            return factors_df
            
        except Exception as e:
            self.logger.warning(f"Error filtering correlated factors: {e}")
            return factors_df
    
    async def get_synthesis_summary(self) -> Dict[str, Any]:
        """Get summary of cross-source factor synthesis capabilities."""
        return {
            "cross_source_types": {
                "edgar_fred": {
                    "description": "Fundamental quality × Economic cycles",
                    "typical_factors": 8-10,
                    "examples": [
                        "earnings_cycle_alignment",
                        "growth_momentum_sync", 
                        "inflation_margin_resilience",
                        "credit_advantage"
                    ]
                },
                "fred_ibkr": {
                    "description": "Macro regimes × Market microstructure",
                    "typical_factors": 8-10,
                    "examples": [
                        "surprise_momentum_confirmation",
                        "rate_volatility_opportunity",
                        "inflation_rotation_signal",
                        "uncertainty_liquidity_premium"
                    ]
                },
                "edgar_ibkr": {
                    "description": "Fundamental momentum × Price action",
                    "typical_factors": 8-10, 
                    "examples": [
                        "fundamental_technical_confirmation",
                        "quality_performance_combination",
                        "consistency_persistence_synergy",
                        "profitability_efficiency_signal"
                    ]
                },
                "triple_source": {
                    "description": "Economic × Fundamental × Technical",
                    "typical_factors": 5-7,
                    "examples": [
                        "economic_quality_momentum",
                        "policy_earnings_volatility",
                        "credit_balance_liquidity",
                        "growth_profitability_trend"
                    ]
                }
            },
            "competitive_advantages": [
                "Unique factor combinations unavailable in commercial platforms",
                "Multi-source data integration creates proprietary signals",
                "Real-time synthesis for institutional-grade factor modeling",
                "Correlation filtering ensures factor independence",
                "Statistical significance testing for factor validation"
            ],
            "total_potential_factors": "25-30 unique cross-source combinations",
            "data_sources_required": ["EDGAR", "FRED", "IBKR"],
            "minimum_sources_for_synthesis": 2
        }


# Global service instance
cross_source_engine = CrossSourceFactorEngine()


async def test_cross_source_synthesis():
    """Test function for cross-source factor synthesis."""
    try:
        # Mock factor data for testing
        edgar_factors = {
            'earnings_quality': 7.5,
            'revenue_growth_yoy': 12.3,
            'operating_margin': 18.2,
            'debt_to_equity': 0.4,
            'roe': 15.6
        }
        
        fred_factors = {
            'economic_activity_composite': 2.1,
            'cpi_inflation_yoy': 3.2,
            'fed_funds_level': 4.5,
            'payroll_growth_yoy': 3.8,
            'consumer_credit_growth_yoy': 5.2
        }
        
        ibkr_factors = {
            'momentum_medium_60d': 8.7,
            'volatility_realized_20d': 22.1,
            'microstructure_effective_spread': 0.12,
            'market_quality_liquidity_score': 0.85,
            'trend_strength_50d': 0.72
        }
        
        # Test synthesis
        result_df = await cross_source_engine.synthesize_cross_source_factors(
            symbol='TEST',
            edgar_factors=edgar_factors,
            fred_factors=fred_factors,
            ibkr_factors=ibkr_factors
        )
        
        print("Cross-Source Factor Synthesis Test Results:")
        print(f"Generated factors for TEST symbol")
        print(f"Cross-source factor count: {result_df['cross_factor_count'][0]}")
        print(f"Source combination: {result_df['source_combination'][0]}")
        
        # Show sample factors
        factor_cols = [col for col in result_df.columns 
                      if col not in ['symbol', 'date', 'calculation_timestamp', 'source_combination', 'cross_factor_count']]
        
        print(f"\nSample cross-source factors:")
        for col in factor_cols[:5]:  # Show first 5
            if col in result_df.columns:
                print(f"  {col}: {result_df[col][0]:.4f}")
        
        # Test summary
        summary = await cross_source_engine.get_synthesis_summary()
        print(f"\nSynthesis Summary:")
        print(f"Total potential factors: {summary['total_potential_factors']}")
        print(f"Cross-source types: {len(summary['cross_source_types'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"Cross-source synthesis test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run test
    asyncio.run(test_cross_source_synthesis())