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
Example strategy demonstrating the integration of FRED economic data with NautilusTrader.

This strategy subscribes to various economic indicators from the Federal Reserve Economic
Data (FRED) API and demonstrates how macroeconomic factors can be incorporated into
trading decision-making processes.

The strategy monitors key economic indicators such as:
- GDP (Gross Domestic Product)
- UNRATE (Unemployment Rate)
- FEDFUNDS (Federal Funds Rate)
- CPIAUCSL (Consumer Price Index)

The strategy demonstrates:
1. How to subscribe to economic data feeds
2. How to process economic indicators in trading logic
3. How to combine multiple economic signals
4. Basic risk management based on economic conditions

This is a demonstration strategy and should not be used for live trading without
proper testing and validation.
"""

from decimal import Decimal
from typing import Any

from nautilus_trader.adapters.fred.data import EconomicData
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import DataType
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.trading.strategy import Strategy


class EconomicAwareStrategy(Strategy):
    """
    A demonstration strategy that incorporates FRED economic data.
    
    This strategy monitors economic indicators and demonstrates how they
    can be integrated into trading decision-making processes.
    
    Parameters
    ----------
    config : dict[str, Any], optional
        The configuration for the strategy.
        
    """
    
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        
        # Economic indicators to monitor
        self.economic_indicators = {
            "GDP": "Gross Domestic Product",
            "UNRATE": "Unemployment Rate", 
            "FEDFUNDS": "Federal Funds Rate",
            "CPIAUCSL": "Consumer Price Index",
            "DGS10": "10-Year Treasury Rate",
        }
        
        # Store latest economic data
        self.latest_economic_data: dict[str, EconomicData] = {}
        
        # Economic trend indicators
        self.gdp_trend = "neutral"
        self.unemployment_trend = "neutral"
        self.inflation_trend = "neutral"
        self.interest_rate_trend = "neutral"
        
        # Risk settings based on economic conditions
        self.economic_risk_multiplier = Decimal("1.0")
        
    def on_start(self) -> None:
        """
        Actions to be performed when the strategy is started.
        """
        self.log.info("Starting Economic Aware Strategy")
        
        # Subscribe to economic data for each indicator
        for series_id, description in self.economic_indicators.items():
            instrument_id = InstrumentId(Symbol(series_id), Venue("FRED"))
            
            # Create data type for economic data subscription
            data_type = DataType(
                EconomicData,
                metadata={
                    "instrument_id": instrument_id,
                    "update_interval": 3600,  # Check hourly for updates
                    "limit": 50,  # Get recent data points
                }
            )
            
            # Subscribe to the economic data
            self.subscribe_data(data_type, ClientId("FRED"))
            
            self.log.info(f"Subscribed to {series_id}: {description}")
    
    def on_stop(self) -> None:
        """
        Actions to be performed when the strategy is stopped.
        """
        self.log.info("Stopping Economic Aware Strategy")
        
        # Unsubscribe from all economic data
        for series_id in self.economic_indicators.keys():
            instrument_id = InstrumentId(Symbol(series_id), Venue("FRED"))
            data_type = DataType(
                EconomicData,
                metadata={"instrument_id": instrument_id}
            )
            self.unsubscribe_data(data_type, ClientId("FRED"))
    
    def on_data(self, data: Data) -> None:
        """
        Actions to be performed when data is received.
        
        Parameters
        ----------
        data : Data
            The data received.
            
        """
        if isinstance(data, EconomicData):
            self._process_economic_data(data)
    
    def _process_economic_data(self, data: EconomicData) -> None:
        """
        Process incoming economic data and update trading signals.
        
        Parameters
        ----------
        data : EconomicData
            The economic data to process.
            
        """
        series_id = data.series_id
        value = float(data.value)
        
        self.log.info(
            f"Received {series_id} data: {value} {data.units} "
            f"(Frequency: {data.frequency})"
        )
        
        # Store the latest data
        self.latest_economic_data[series_id] = data
        
        # Process specific economic indicators
        if series_id == "GDP":
            self._process_gdp_data(data)
        elif series_id == "UNRATE":
            self._process_unemployment_data(data)
        elif series_id == "FEDFUNDS":
            self._process_fed_funds_data(data)
        elif series_id == "CPIAUCSL":
            self._process_inflation_data(data)
        elif series_id == "DGS10":
            self._process_treasury_data(data)
            
        # Update overall economic assessment
        self._update_economic_assessment()
        
    def _process_gdp_data(self, data: EconomicData) -> None:
        """
        Process GDP data and determine economic growth trend.
        
        Parameters
        ----------
        data : EconomicData
            The GDP data.
            
        """
        gdp_value = float(data.value)
        
        # Simple trend analysis (in practice, you would compare with historical data)
        if gdp_value > 25000:  # Example threshold in billions
            self.gdp_trend = "strong_growth"
            self.log.info(f"Strong GDP growth detected: ${gdp_value:.1f}B")
        elif gdp_value > 24000:
            self.gdp_trend = "moderate_growth"
            self.log.info(f"Moderate GDP growth: ${gdp_value:.1f}B")
        else:
            self.gdp_trend = "weak_growth"
            self.log.warning(f"Weak GDP growth: ${gdp_value:.1f}B")
    
    def _process_unemployment_data(self, data: EconomicData) -> None:
        """
        Process unemployment data.
        
        Parameters
        ----------
        data : EconomicData
            The unemployment data.
            
        """
        unemployment_rate = float(data.value)
        
        if unemployment_rate < 4.0:
            self.unemployment_trend = "low"
            self.log.info(f"Low unemployment: {unemployment_rate:.1f}%")
        elif unemployment_rate < 6.0:
            self.unemployment_trend = "moderate"
            self.log.info(f"Moderate unemployment: {unemployment_rate:.1f}%")
        else:
            self.unemployment_trend = "high"
            self.log.warning(f"High unemployment: {unemployment_rate:.1f}%")
    
    def _process_fed_funds_data(self, data: EconomicData) -> None:
        """
        Process Federal Funds Rate data.
        
        Parameters
        ----------
        data : EconomicData
            The Fed Funds rate data.
            
        """
        fed_funds_rate = float(data.value)
        
        if fed_funds_rate > 4.0:
            self.interest_rate_trend = "high"
            self.log.info(f"High interest rates: {fed_funds_rate:.2f}%")
        elif fed_funds_rate > 2.0:
            self.interest_rate_trend = "moderate"
            self.log.info(f"Moderate interest rates: {fed_funds_rate:.2f}%")
        else:
            self.interest_rate_trend = "low"
            self.log.info(f"Low interest rates: {fed_funds_rate:.2f}%")
    
    def _process_inflation_data(self, data: EconomicData) -> None:
        """
        Process inflation (CPI) data.
        
        Parameters
        ----------
        data : EconomicData
            The CPI data.
            
        """
        # CPI is an index, so we'd typically look at year-over-year change
        # For demonstration, we'll use a simple threshold approach
        cpi_value = float(data.value)
        
        if cpi_value > 280:  # Example threshold
            self.inflation_trend = "high"
            self.log.warning(f"High inflation indicated by CPI: {cpi_value:.1f}")
        elif cpi_value > 260:
            self.inflation_trend = "moderate"
            self.log.info(f"Moderate inflation: CPI {cpi_value:.1f}")
        else:
            self.inflation_trend = "low"
            self.log.info(f"Low inflation: CPI {cpi_value:.1f}")
    
    def _process_treasury_data(self, data: EconomicData) -> None:
        """
        Process 10-year Treasury rate data.
        
        Parameters
        ----------
        data : EconomicData
            The Treasury rate data.
            
        """
        treasury_rate = float(data.value)
        self.log.info(f"10-Year Treasury Rate: {treasury_rate:.2f}%")
        
        # Treasury rates can indicate economic sentiment and inflation expectations
        if treasury_rate > 4.0:
            self.log.info("High Treasury rates - potential economic strength or inflation concerns")
        elif treasury_rate < 2.0:
            self.log.info("Low Treasury rates - potential economic weakness or deflation risk")
    
    def _update_economic_assessment(self) -> None:
        """
        Update overall economic assessment and adjust risk parameters.
        """
        # Determine overall economic condition
        positive_indicators = 0
        total_indicators = 0
        
        if self.gdp_trend == "strong_growth":
            positive_indicators += 1
        elif self.gdp_trend == "moderate_growth":
            positive_indicators += 0.5
        total_indicators += 1
        
        if self.unemployment_trend == "low":
            positive_indicators += 1
        elif self.unemployment_trend == "moderate":
            positive_indicators += 0.5
        total_indicators += 1
        
        # Calculate economic strength score
        if total_indicators > 0:
            economic_strength = positive_indicators / total_indicators
        else:
            economic_strength = 0.5  # Neutral
        
        # Adjust risk multiplier based on economic conditions
        if economic_strength > 0.7:
            self.economic_risk_multiplier = Decimal("1.2")  # Increase risk in strong economy
            self.log.info("Strong economic conditions - increasing risk appetite")
        elif economic_strength < 0.3:
            self.economic_risk_multiplier = Decimal("0.7")  # Reduce risk in weak economy  
            self.log.warning("Weak economic conditions - reducing risk appetite")
        else:
            self.economic_risk_multiplier = Decimal("1.0")  # Neutral risk
            self.log.info("Neutral economic conditions - maintaining normal risk")
        
        # Log current economic assessment
        self.log.info(
            f"Economic Assessment - GDP: {self.gdp_trend}, "
            f"Unemployment: {self.unemployment_trend}, "
            f"Inflation: {self.inflation_trend}, "
            f"Interest Rates: {self.interest_rate_trend}, "
            f"Risk Multiplier: {self.economic_risk_multiplier}"
        )
    
    def get_economic_signal(self) -> str:
        """
        Get the current overall economic signal.
        
        Returns
        -------
        str
            The economic signal ("bullish", "bearish", or "neutral").
            
        """
        # Simple logic to determine overall economic signal
        bullish_factors = 0
        bearish_factors = 0
        
        if self.gdp_trend == "strong_growth":
            bullish_factors += 2
        elif self.gdp_trend == "moderate_growth":
            bullish_factors += 1
        elif self.gdp_trend == "weak_growth":
            bearish_factors += 1
            
        if self.unemployment_trend == "low":
            bullish_factors += 1
        elif self.unemployment_trend == "high":
            bearish_factors += 1
            
        if self.inflation_trend == "high":
            bearish_factors += 1
        elif self.inflation_trend == "low":
            bullish_factors += 1
            
        if bullish_factors > bearish_factors + 1:
            return "bullish"
        elif bearish_factors > bullish_factors + 1:
            return "bearish"
        else:
            return "neutral"
    
    def request_economic_data(self, series_id: str, limit: int = 100) -> None:
        """
        Request historical economic data for a specific series.
        
        Parameters
        ----------
        series_id : str
            The FRED series ID to request.
        limit : int, default 100
            The number of data points to request.
            
        """
        instrument_id = InstrumentId(Symbol(series_id), Venue("FRED"))
        
        data_type = DataType(
            EconomicData,
            metadata={
                "instrument_id": instrument_id,
                "limit": limit,
                "sort_order": "desc",  # Most recent first
            }
        )
        
        # Request the data
        self.request_data(data_type, ClientId("FRED"))
        
        self.log.info(f"Requested {limit} data points for {series_id}")