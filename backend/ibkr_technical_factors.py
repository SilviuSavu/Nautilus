"""
IBKR Technical Factor Engine
===========================

Technical factor computation engine for Interactive Brokers market data.
Generates 15-20 technical factors including momentum, volatility, microstructure, and market quality indicators.

Part of Phase 2: Cross-Source Factor Engine Implementation
"""

import logging
import asyncio
import numpy as np
import polars as pl
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from enum import Enum
import talib
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class FactorCategory(Enum):
    """Technical factor categories."""
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    MICROSTRUCTURE = "microstructure" 
    MARKET_QUALITY = "market_quality"
    TREND = "trend"


@dataclass
class TechnicalFactorConfig:
    """Configuration for technical factor calculation."""
    lookback_periods: Dict[str, int]
    volatility_windows: List[int]
    momentum_windows: List[int]
    microstructure_enabled: bool = True
    trend_analysis_enabled: bool = True


class IBKRTechnicalFactorEngine:
    """
    IBKR Technical Factor Engine for institutional-grade factor modeling.
    
    Generates 15-20 technical factors from IBKR market data:
    - Momentum factors (5): Multi-timeframe momentum, relative strength
    - Volatility factors (4): Realized vol, GARCH forecasts, volatility risk premium
    - Microstructure factors (4): Bid-ask spreads, order flow, trade size analysis
    - Market Quality factors (3): Liquidity, market depth, price efficiency
    - Trend factors (4): Trend strength, direction, persistence, quality
    """
    
    def __init__(self, config: Optional[TechnicalFactorConfig] = None):
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.config = config or TechnicalFactorConfig(
            lookback_periods={
                'short': 20,    # 1 month
                'medium': 60,   # 3 months  
                'long': 252,    # 1 year
                'ultra_short': 5  # 1 week
            },
            volatility_windows=[10, 20, 60],
            momentum_windows=[5, 20, 60, 120],
            microstructure_enabled=True,
            trend_analysis_enabled=True
        )
        
        # Factor cache for performance
        self._factor_cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def calculate_technical_factors(
        self,
        symbol: str,
        market_data: pl.DataFrame,
        as_of_date: date = None
    ) -> pl.DataFrame:
        """
        Calculate comprehensive technical factors for a single symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            market_data: DataFrame with OHLCV data + bid/ask/volume details
            as_of_date: Calculation date (defaults to latest date in data)
            
        Returns:
            DataFrame with technical factor scores
        """
        try:
            if as_of_date is None:
                as_of_date = market_data['date'].max()
            
            self.logger.info(f"Calculating technical factors for {symbol} as of {as_of_date}")
            
            # Validate input data
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_columns if col not in market_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Sort data by date
            market_data = market_data.sort('date')
            
            # Filter data up to calculation date
            data = market_data.filter(pl.col('date') <= as_of_date)
            
            if len(data) < self.config.lookback_periods['long']:
                self.logger.warning(f"Insufficient data for {symbol}: {len(data)} days")
            
            # Calculate factor categories
            factors = {}
            
            # 1. Momentum Factors (5 factors)
            momentum_factors = await self._calculate_momentum_factors(data, symbol)
            factors.update(momentum_factors)
            
            # 2. Volatility Factors (4 factors)
            volatility_factors = await self._calculate_volatility_factors(data, symbol)
            factors.update(volatility_factors)
            
            # 3. Microstructure Factors (4 factors)
            if self.config.microstructure_enabled and self._has_microstructure_data(data):
                microstructure_factors = await self._calculate_microstructure_factors(data, symbol)
                factors.update(microstructure_factors)
            
            # 4. Market Quality Factors (3 factors)
            market_quality_factors = await self._calculate_market_quality_factors(data, symbol)
            factors.update(market_quality_factors)
            
            # 5. Trend Factors (4 factors)
            if self.config.trend_analysis_enabled:
                trend_factors = await self._calculate_trend_factors(data, symbol)
                factors.update(trend_factors)
            
            # Create result DataFrame
            factor_record = {
                'symbol': symbol,
                'date': as_of_date,
                'calculation_timestamp': datetime.now(),
                'data_points': len(data),
                **factors
            }
            
            result_df = pl.DataFrame([factor_record])
            
            self.logger.info(f"Calculated {len(factors)} technical factors for {symbol}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical factors for {symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Technical factor calculation failed: {e}")
    
    async def calculate_universe_factors(
        self,
        universe_data: Dict[str, pl.DataFrame],
        as_of_date: date = None
    ) -> pl.DataFrame:
        """
        Calculate technical factors for entire universe in parallel.
        
        Args:
            universe_data: Dictionary mapping symbols to market data DataFrames
            as_of_date: Calculation date
            
        Returns:
            Combined DataFrame with factors for all symbols
        """
        try:
            self.logger.info(f"Calculating technical factors for {len(universe_data)} symbols")
            
            # Calculate factors for each symbol in parallel
            tasks = [
                self.calculate_technical_factors(symbol, data, as_of_date)
                for symbol, data in universe_data.items()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine successful results
            successful_results = []
            failed_symbols = []
            
            for i, result in enumerate(results):
                symbol = list(universe_data.keys())[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to calculate factors for {symbol}: {result}")
                    failed_symbols.append(symbol)
                    continue
                
                successful_results.append(result)
            
            if not successful_results:
                raise HTTPException(status_code=500, detail="No factors calculated successfully")
            
            # Concatenate all results
            universe_factors = pl.concat(successful_results)
            
            if failed_symbols:
                self.logger.warning(f"Failed to calculate factors for {len(failed_symbols)} symbols: {failed_symbols}")
            
            self.logger.info(f"Successfully calculated factors for {len(successful_results)} symbols")
            return universe_factors
            
        except Exception as e:
            self.logger.error(f"Error calculating universe factors: {e}")
            raise HTTPException(status_code=500, detail=f"Universe factor calculation failed: {e}")
    
    # MOMENTUM FACTORS
    async def _calculate_momentum_factors(self, data: pl.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate momentum-based factors (5 factors)."""
        factors = {}
        
        try:
            closes = data['close'].to_numpy()
            volumes = data['volume'].to_numpy()
            
            if len(closes) < self.config.lookback_periods['medium']:
                return factors
            
            # 1. Multi-timeframe momentum
            for period_name, period in self.config.lookback_periods.items():
                if len(closes) > period and period > 0:
                    momentum = (closes[-1] / closes[-period - 1] - 1) * 100
                    factors[f'momentum_{period_name}_{period}d'] = momentum
            
            # 2. Volume-weighted momentum (VWAP-based)
            if len(data) >= 20:
                # Calculate VWAP over last 20 days
                typical_price = (data['high'] + data['low'] + data['close']) / 3
                vwap_20d = (typical_price * data['volume']).tail(20).sum() / data['volume'].tail(20).sum()
                vwap_momentum = (closes[-1] / vwap_20d - 1) * 100
                factors['momentum_vwap_20d'] = vwap_momentum
            
            # 3. RSI-based momentum
            if len(closes) >= 14:
                rsi = talib.RSI(closes, timeperiod=14)
                if not np.isnan(rsi[-1]):
                    # Convert RSI to momentum signal (-50 to +50 scale)
                    factors['momentum_rsi_signal'] = (rsi[-1] - 50)
            
            # 4. Rate of Change acceleration
            if len(closes) >= 30:
                roc_10 = talib.ROC(closes, timeperiod=10)
                roc_20 = talib.ROC(closes, timeperiod=20)
                if not (np.isnan(roc_10[-1]) or np.isnan(roc_20[-1])):
                    momentum_acceleration = roc_10[-1] - roc_20[-1]
                    factors['momentum_acceleration'] = momentum_acceleration
            
            # 5. Relative strength vs market (placeholder - would need market index data)
            # For now, use price momentum percentile vs historical range
            if len(closes) >= 252:
                current_price = closes[-1]
                yearly_range = closes[-252:]
                percentile_rank = (current_price > yearly_range).sum() / len(yearly_range) * 100
                factors['momentum_percentile_rank'] = percentile_rank
            
        except Exception as e:
            self.logger.warning(f"Error calculating momentum factors for {symbol}: {e}")
        
        return factors
    
    # VOLATILITY FACTORS  
    async def _calculate_volatility_factors(self, data: pl.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate volatility-based factors (4 factors)."""
        factors = {}
        
        try:
            closes = data['close'].to_numpy()
            highs = data['high'].to_numpy()
            lows = data['low'].to_numpy()
            
            if len(closes) < 20:
                return factors
            
            # Calculate returns
            returns = np.diff(np.log(closes))
            
            # 1. Realized volatility (multiple windows)
            for window in self.config.volatility_windows:
                if len(returns) >= window:
                    window_returns = returns[-window:]
                    realized_vol = np.std(window_returns) * np.sqrt(252) * 100  # Annualized %
                    factors[f'volatility_realized_{window}d'] = realized_vol
            
            # 2. GARCH volatility forecast (simplified)
            if len(returns) >= 60:
                # Simple EWMA volatility forecast
                lambda_decay = 0.94
                weights = np.array([(lambda_decay ** i) for i in range(30)])
                weights = weights / weights.sum()
                recent_returns = returns[-30:]
                ewma_variance = np.sum(weights * (recent_returns ** 2))
                garch_vol = np.sqrt(ewma_variance * 252) * 100
                factors['volatility_garch_forecast'] = garch_vol
            
            # 3. Volatility risk premium (ATR-based)
            if len(highs) >= 14:
                atr = talib.ATR(highs, lows, closes, timeperiod=14)
                if not np.isnan(atr[-1]):
                    # ATR as percentage of price
                    atr_percent = (atr[-1] / closes[-1]) * 100
                    factors['volatility_atr_percent'] = atr_percent
            
            # 4. Volatility regime (high/low vol environment)
            if len(returns) >= 120:
                current_vol = np.std(returns[-20:]) * np.sqrt(252) * 100
                historical_vol = np.std(returns[-120:]) * np.sqrt(252) * 100
                vol_regime = (current_vol / historical_vol - 1) * 100
                factors['volatility_regime_signal'] = vol_regime
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility factors for {symbol}: {e}")
        
        return factors
    
    # MICROSTRUCTURE FACTORS
    async def _calculate_microstructure_factors(self, data: pl.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate microstructure factors (4 factors)."""
        factors = {}
        
        try:
            # Check if microstructure data is available
            has_bid_ask = 'bid' in data.columns and 'ask' in data.columns
            has_trade_size = 'trade_size' in data.columns
            
            if not has_bid_ask:
                # Use high-low spread as proxy for bid-ask spread
                highs = data['high'].to_numpy()
                lows = data['low'].to_numpy()
                closes = data['close'].to_numpy()
                
                # 1. Effective spread (High-Low proxy)
                if len(data) >= 20:
                    hl_spreads = (highs - lows) / closes * 100  # As percentage
                    avg_spread = np.mean(hl_spreads[-20:])
                    factors['microstructure_effective_spread'] = avg_spread
                
                # 2. Price impact (using high-low range vs volume)
                volumes = data['volume'].to_numpy()
                if len(data) >= 20:
                    price_ranges = (highs - lows) / closes
                    volume_ratios = volumes / np.mean(volumes[-60:] if len(volumes) >= 60 else volumes)
                    # Higher volume should reduce price impact
                    price_impact = np.mean(price_ranges[-20:] / (volume_ratios[-20:] + 0.1))
                    factors['microstructure_price_impact'] = price_impact * 100
                
            else:
                # Use actual bid-ask data
                bids = data['bid'].to_numpy()
                asks = data['ask'].to_numpy()
                closes = data['close'].to_numpy()
                
                # 1. Bid-ask spread
                spreads = (asks - bids) / ((asks + bids) / 2) * 100
                recent_spread = np.mean(spreads[-20:] if len(spreads) >= 20 else spreads)
                factors['microstructure_bid_ask_spread'] = recent_spread
                
                # 2. Quote midpoint vs price efficiency
                midpoints = (bids + asks) / 2
                price_efficiency = np.std((closes - midpoints) / midpoints) * 100
                factors['microstructure_price_efficiency'] = price_efficiency
            
            # 3. Volume pattern analysis
            volumes = data['volume'].to_numpy()
            if len(volumes) >= 20:
                # Volume irregularity (coefficient of variation)
                volume_cv = np.std(volumes[-20:]) / np.mean(volumes[-20:])
                factors['microstructure_volume_irregularity'] = volume_cv
            
            # 4. Trading intensity
            if len(data) >= 10:
                # Average daily turnover
                daily_turnover = data['volume'] * data['close']
                avg_turnover = daily_turnover.tail(10).mean()
                # Normalize by market cap proxy (price * recent volume)
                turnover_intensity = avg_turnover / (data['close'].tail(1)[0] * data['volume'].tail(10).mean())
                factors['microstructure_trading_intensity'] = turnover_intensity
            
        except Exception as e:
            self.logger.warning(f"Error calculating microstructure factors for {symbol}: {e}")
        
        return factors
    
    # MARKET QUALITY FACTORS
    async def _calculate_market_quality_factors(self, data: pl.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate market quality factors (3 factors)."""
        factors = {}
        
        try:
            closes = data['close'].to_numpy()
            volumes = data['volume'].to_numpy()
            highs = data['high'].to_numpy()
            lows = data['low'].to_numpy()
            
            # 1. Liquidity score (Amihud illiquidity measure)
            if len(data) >= 20:
                returns = np.abs(np.diff(closes) / closes[:-1])
                dollar_volumes = volumes[1:] * closes[1:]
                # Avoid division by zero
                illiquidity_ratios = returns / (dollar_volumes + 1e-10)
                amihud_illiquidity = np.mean(illiquidity_ratios[-20:]) * 1e6  # Scale up
                # Convert to liquidity score (higher is better)
                factors['market_quality_liquidity_score'] = 1 / (amihud_illiquidity + 1e-6)
            
            # 2. Market depth proxy (volume vs volatility)
            if len(data) >= 30:
                returns = np.diff(np.log(closes))
                vol_30d = np.std(returns[-30:]) * np.sqrt(252)
                avg_volume_30d = np.mean(volumes[-30:])
                # Higher volume with lower volatility indicates better depth
                depth_score = avg_volume_30d / (vol_30d + 1e-6)
                factors['market_quality_depth_score'] = depth_score / 1e6  # Scale down
            
            # 3. Price efficiency (random walk measure)
            if len(closes) >= 60:
                returns = np.diff(np.log(closes))
                # Variance ratio test (simplified)
                returns_1d = returns[-60:]
                returns_5d = np.array([np.sum(returns_1d[i:i+5]) for i in range(0, len(returns_1d)-4, 5)])
                
                if len(returns_5d) > 5:
                    var_ratio = (np.var(returns_5d) / 5) / (np.var(returns_1d) + 1e-10)
                    # Closer to 1 indicates more efficient pricing (random walk)
                    efficiency_score = 1 - abs(var_ratio - 1)
                    factors['market_quality_price_efficiency'] = max(0, efficiency_score)
            
        except Exception as e:
            self.logger.warning(f"Error calculating market quality factors for {symbol}: {e}")
        
        return factors
    
    # TREND FACTORS
    async def _calculate_trend_factors(self, data: pl.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate trend analysis factors (4 factors)."""
        factors = {}
        
        try:
            closes = data['close'].to_numpy()
            
            if len(closes) < 50:
                return factors
            
            # 1. Trend strength (R-squared of linear regression)
            if len(closes) >= 50:
                x = np.arange(len(closes[-50:]))
                y = closes[-50:]
                correlation = np.corrcoef(x, y)[0, 1]
                trend_strength = correlation ** 2  # R-squared
                factors['trend_strength_50d'] = trend_strength
            
            # 2. Trend direction (slope of linear regression)
            if len(closes) >= 20:
                x = np.arange(20)
                y = closes[-20:]
                slope = np.polyfit(x, y, 1)[0]
                # Normalize by price level
                trend_direction = (slope / closes[-1]) * 100
                factors['trend_direction_20d'] = trend_direction
            
            # 3. Trend persistence (moving average convergence)
            if len(closes) >= 50:
                ma_10 = talib.SMA(closes, timeperiod=10)
                ma_20 = talib.SMA(closes, timeperiod=20)
                ma_50 = talib.SMA(closes, timeperiod=50)
                
                if not (np.isnan(ma_10[-1]) or np.isnan(ma_20[-1]) or np.isnan(ma_50[-1])):
                    # Check if short-term > medium-term > long-term (uptrend persistence)
                    uptrend_signal = (ma_10[-1] > ma_20[-1]) and (ma_20[-1] > ma_50[-1])
                    downtrend_signal = (ma_10[-1] < ma_20[-1]) and (ma_20[-1] < ma_50[-1])
                    
                    if uptrend_signal:
                        trend_persistence = 1.0
                    elif downtrend_signal:
                        trend_persistence = -1.0
                    else:
                        trend_persistence = 0.0
                    
                    factors['trend_persistence'] = trend_persistence
            
            # 4. Trend quality (consistency of direction)
            if len(closes) >= 30:
                # Count days where price is above/below trend line
                returns = np.diff(closes[-30:])
                positive_days = (returns > 0).sum()
                trend_consistency = (positive_days / len(returns) - 0.5) * 2  # Scale to [-1, 1]
                factors['trend_quality'] = trend_consistency
            
        except Exception as e:
            self.logger.warning(f"Error calculating trend factors for {symbol}: {e}")
        
        return factors
    
    # UTILITY METHODS
    def _has_microstructure_data(self, data: pl.DataFrame) -> bool:
        """Check if microstructure data (bid/ask) is available."""
        return 'bid' in data.columns and 'ask' in data.columns
    
    async def get_factor_summary(self) -> Dict[str, Any]:
        """Get summary of available factors and their categories."""
        return {
            "total_factors": "15-20",
            "categories": {
                "momentum": {
                    "count": 5,
                    "factors": [
                        "momentum_short/medium/long_term",
                        "momentum_vwap_based", 
                        "momentum_rsi_signal",
                        "momentum_acceleration",
                        "momentum_percentile_rank"
                    ]
                },
                "volatility": {
                    "count": 4, 
                    "factors": [
                        "volatility_realized_multiple_windows",
                        "volatility_garch_forecast",
                        "volatility_atr_percent", 
                        "volatility_regime_signal"
                    ]
                },
                "microstructure": {
                    "count": 4,
                    "factors": [
                        "microstructure_effective_spread",
                        "microstructure_price_impact",
                        "microstructure_volume_irregularity",
                        "microstructure_trading_intensity"
                    ]
                },
                "market_quality": {
                    "count": 3,
                    "factors": [
                        "market_quality_liquidity_score",
                        "market_quality_depth_score", 
                        "market_quality_price_efficiency"
                    ]
                },
                "trend": {
                    "count": 4,
                    "factors": [
                        "trend_strength",
                        "trend_direction",
                        "trend_persistence",
                        "trend_quality"
                    ]
                }
            },
            "data_requirements": {
                "minimum_history_days": 252,
                "required_columns": ["date", "open", "high", "low", "close", "volume"],
                "optional_columns": ["bid", "ask", "trade_size"]
            }
        }


# Global service instance
ibkr_technical_engine = IBKRTechnicalFactorEngine()


async def test_ibkr_technical_factors():
    """Test function for IBKR technical factor engine."""
    try:
        # Generate sample market data
        import random
        from datetime import datetime, timedelta
        
        dates = [datetime.now().date() - timedelta(days=i) for i in range(300, 0, -1)]
        base_price = 100.0
        
        sample_data = []
        for i, date in enumerate(dates):
            # Random walk with trend
            change = random.gauss(0, 0.02)
            base_price *= (1 + change)
            
            high = base_price * (1 + abs(random.gauss(0, 0.01)))
            low = base_price * (1 - abs(random.gauss(0, 0.01)))
            volume = random.randint(1000000, 5000000)
            
            sample_data.append({
                'date': date,
                'open': base_price,
                'high': high,
                'low': low, 
                'close': base_price,
                'volume': volume
            })
        
        df = pl.DataFrame(sample_data)
        
        # Test factor calculation
        factors_df = await ibkr_technical_engine.calculate_technical_factors('TEST', df)
        
        print("IBKR Technical Factor Engine Test Results:")
        print(f"Calculated factors for TEST symbol")
        print(f"Factor count: {len(factors_df.columns) - 4}")  # Exclude metadata columns
        print(factors_df.head())
        
        # Test factor summary
        summary = await ibkr_technical_engine.get_factor_summary()
        print(f"\nFactor Summary:")
        print(f"Total factor categories: {len(summary['categories'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"IBKR technical factor test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run test
    asyncio.run(test_ibkr_technical_factors())