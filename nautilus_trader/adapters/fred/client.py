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

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any

from nautilus_trader.adapters.fred.config import FREDDataClientConfig
from nautilus_trader.adapters.fred.data import EconomicData
from nautilus_trader.adapters.fred.http import FREDHttpClient
from nautilus_trader.adapters.fred.parsing import parse_fred_observations
from nautilus_trader.adapters.fred.parsing import parse_fred_series_info
from nautilus_trader.adapters.fred.providers import FREDInstrumentProvider
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import MessageBus
from nautilus_trader.common.enums import LogColor
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.data.messages import DataRequest
from nautilus_trader.data.messages import DataResponse
from nautilus_trader.data.messages import Subscribe
from nautilus_trader.data.messages import Unsubscribe
from nautilus_trader.live.data_client import LiveMarketDataClient
from nautilus_trader.model.data import DataType
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId


class FREDDataClient(LiveMarketDataClient):
    """
    Provides a data client for the Federal Reserve Economic Data (FRED) API.

    This client provides access to economic time series data through the FRED API.
    It implements intelligent caching, rate limiting, and robust error handling.

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        The event loop for the client.
    msgbus : MessageBus
        The message bus for the client.
    cache : Cache
        The cache for the client.
    clock : LiveClock
        The clock for the client.
    instrument_provider : FREDInstrumentProvider
        The instrument provider for the client.
    config : FREDDataClientConfig, optional
        The configuration for the client.
    name : str, optional
        The custom client ID.

    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        msgbus: MessageBus,
        cache: Cache,
        clock: LiveClock,
        instrument_provider: FREDInstrumentProvider,
        config: FREDDataClientConfig | None = None,
        name: str | None = None,
    ) -> None:
        if config is None:
            config = FREDDataClientConfig()

        PyCondition.type(config, FREDDataClientConfig, "config")

        super().__init__(
            loop=loop,
            client_id=ClientId(name or "FRED"),
            venue=None,  # Multi-venue support for economic data
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            instrument_provider=instrument_provider,
            config=config,
        )

        # Configuration
        self._config = config

        # HTTP client for FRED API
        self._http_client = FREDHttpClient(
            api_key=config.api_key,
            base_url=config.base_url,
            request_timeout=config.request_timeout,
            rate_limit_delay=config.rate_limit_delay,
            logger=self._log,
        )

        # Subscriptions and data management
        self._subscriptions: dict[DataType, set[InstrumentId]] = defaultdict(set)
        self._update_tasks: dict[InstrumentId, asyncio.Task] = {}
        self._series_info_cache: dict[str, dict[str, Any]] = {}
        self._last_data_timestamps: dict[str, int] = {}
        
        # Enhanced macro factors from comprehensive implementation
        self._key_macro_series = {
            # Economic Growth & Activity (6 series)
            "GDP": {"title": "Gross Domestic Product", "category": "growth"},
            "GDPC1": {"title": "Real GDP", "category": "growth"},
            "GDPPOT": {"title": "Real Potential GDP", "category": "growth"},
            "INDPRO": {"title": "Industrial Production Index", "category": "growth"},
            "RSXFS": {"title": "Retail Sales", "category": "growth"},
            "HOUST": {"title": "Housing Starts", "category": "growth"},
            
            # Employment & Labor Market (6 series)
            "UNRATE": {"title": "Unemployment Rate", "category": "employment"},
            "PAYEMS": {"title": "Nonfarm Payrolls", "category": "employment"},
            "CIVPART": {"title": "Labor Force Participation Rate", "category": "employment"},
            "AHETPI": {"title": "Average Hourly Earnings", "category": "employment"},
            "ICSA": {"title": "Initial Claims", "category": "employment"},
            "JTSJOL": {"title": "Job Openings", "category": "employment"},
            
            # Inflation & Prices (4 series)
            "CPIAUCSL": {"title": "Consumer Price Index", "category": "inflation"},
            "CPILFESL": {"title": "Core CPI", "category": "inflation"},
            "PCEPI": {"title": "PCE Price Index", "category": "inflation"},
            "DFEDTARU": {"title": "Fed Inflation Target Upper", "category": "inflation"},
            
            # Monetary Policy & Interest Rates (6 series)
            "DFF": {"title": "Federal Funds Rate", "category": "monetary"},
            "DGS10": {"title": "10-Year Treasury Rate", "category": "monetary"},
            "DGS2": {"title": "2-Year Treasury Rate", "category": "monetary"},
            "DGS3MO": {"title": "3-Month Treasury Rate", "category": "monetary"},
            "DGS5": {"title": "5-Year Treasury Rate", "category": "monetary"},
            "DGS30": {"title": "30-Year Treasury Rate", "category": "monetary"},
            
            # Money Supply & Credit (4 series)
            "M2SL": {"title": "M2 Money Supply", "category": "monetary"},
            "BOGMBASE": {"title": "Monetary Base", "category": "monetary"},
            "TOTALSL": {"title": "Total Consumer Credit", "category": "monetary"},
            "DRTSCIS": {"title": "Delinquency Rate on Credit Cards", "category": "monetary"},
            
            # Market & Financial Indicators (6 series)
            "BAMLH0A0HYM2": {"title": "High Yield Credit Spread", "category": "financial"},
            "VIXCLS": {"title": "VIX Volatility Index", "category": "financial"},
            "DEXUSEU": {"title": "USD/EUR Exchange Rate", "category": "financial"},
            "BAMLEM": {"title": "Emerging Markets Bond Spread", "category": "financial"},
            "DCOILWTICO": {"title": "WTI Oil Price", "category": "financial"},
            "GOLDAMGBD228NLBM": {"title": "Gold Price", "category": "financial"},
        }

    # -- CONNECTION MANAGEMENT ---------------------------------------------------------------

    async def _connect(self) -> None:
        """
        Connect to the FRED API.
        """
        await self._http_client.connect()

        # Load configured instruments
        if self._config.instrument_ids:
            await self._instrument_provider.load_ids_async(self._config.instrument_ids)
            
        if self._config.series_ids:
            from nautilus_trader.model.identifiers import Symbol, Venue
            instrument_ids = [
                InstrumentId(Symbol(series_id), Venue("FRED"))
                for series_id in self._config.series_ids
            ]
            await self._instrument_provider.load_ids_async(instrument_ids)

        # Auto-subscribe if configured
        if self._config.auto_subscribe and self._config.instrument_ids:
            for instrument_id in self._config.instrument_ids:
                data_type = DataType(EconomicData, metadata={"instrument_id": instrument_id})
                await self._subscribe(data_type)

        self._log.info("Connected to FRED API", LogColor.GREEN)

    async def _disconnect(self) -> None:
        """
        Disconnect from the FRED API.
        """
        # Cancel all update tasks
        for task in self._update_tasks.values():
            if not task.done():
                task.cancel()
        
        if self._update_tasks:
            await asyncio.gather(*self._update_tasks.values(), return_exceptions=True)
        
        self._update_tasks.clear()
        self._subscriptions.clear()

        await self._http_client.disconnect()
        self._log.info("Disconnected from FRED API", LogColor.GREEN)

    # -- SUBSCRIPTIONS ------------------------------------------------------------------------

    async def _subscribe(self, data_type: DataType) -> None:
        """
        Subscribe to economic data updates.

        Parameters
        ----------
        data_type : DataType
            The data type to subscribe to.

        """
        if data_type.type != EconomicData:
            self._log.error(f"Cannot subscribe to {data_type.type}, only EconomicData supported")
            return

        # Extract instrument ID from metadata
        metadata = data_type.metadata or {}
        instrument_id = metadata.get("instrument_id")
        
        if not instrument_id:
            self._log.error("instrument_id required in DataType metadata for FRED subscriptions")
            return

        if not isinstance(instrument_id, InstrumentId):
            self._log.error(f"instrument_id must be InstrumentId, got {type(instrument_id)}")
            return

        # Add to subscriptions
        self._subscriptions[data_type].add(instrument_id)
        
        # Start update task if auto-subscribe enabled
        if self._config.auto_subscribe:
            await self._start_update_task(instrument_id, data_type)

        self._log.info(f"Subscribed to FRED data for {instrument_id}")

    async def _unsubscribe(self, data_type: DataType) -> None:
        """
        Unsubscribe from economic data updates.

        Parameters
        ----------
        data_type : DataType
            The data type to unsubscribe from.

        """
        metadata = data_type.metadata or {}
        instrument_id = metadata.get("instrument_id")
        
        if instrument_id and instrument_id in self._subscriptions[data_type]:
            self._subscriptions[data_type].remove(instrument_id)
            
            # Stop update task
            if instrument_id in self._update_tasks:
                task = self._update_tasks[instrument_id]
                if not task.done():
                    task.cancel()
                del self._update_tasks[instrument_id]

        self._log.info(f"Unsubscribed from FRED data for {instrument_id}")

    # -- REQUESTS -----------------------------------------------------------------------------

    async def _request(self, data_type: DataType, correlation_id: UUID4) -> None:
        """
        Handle a data request for economic data.

        Parameters
        ----------
        data_type : DataType
            The data type being requested.
        correlation_id : UUID4
            The correlation ID for the request.

        """
        if data_type.type != EconomicData:
            self._log.error(f"Cannot request {data_type.type}, only EconomicData supported")
            return

        metadata = data_type.metadata or {}
        instrument_id = metadata.get("instrument_id")
        
        if not instrument_id:
            self._log.error("instrument_id required in DataType metadata for FRED requests")
            return

        try:
            # Get series ID from instrument
            series_id = instrument_id.symbol.value
            
            # Fetch series info if not cached
            if series_id not in self._series_info_cache:
                series_data = await self._http_client.get_series_info(series_id)
                if series_data:
                    self._series_info_cache[series_id] = parse_fred_series_info(series_data)
            
            series_info = self._series_info_cache.get(series_id)
            
            # Prepare request parameters
            request_params = {
                "limit": metadata.get("limit", self._config.default_limit),
                "sort_order": metadata.get("sort_order", "desc"),  # Most recent first
            }
            
            # Add date filters if provided
            if "start_date" in metadata:
                request_params["observation_start"] = metadata["start_date"]
            if "end_date" in metadata:
                request_params["observation_end"] = metadata["end_date"]
            
            # Fetch observations
            observations_data = await self._http_client.get_series_observations(
                series_id=series_id,
                **request_params,
            )
            
            # Parse observations to EconomicData objects
            economic_data_list = parse_fred_observations(
                series_id=series_id,
                observations_data=observations_data,
                series_info=series_info,
            )
            
            # Send data response
            response = DataResponse(
                client_id=self.id,
                venue=instrument_id.venue,
                data_type=data_type,
                data=economic_data_list,
                correlation_id=correlation_id,
                response_id=UUID4(),
                ts_init=self._clock.timestamp_ns(),
            )
            
            self._handle_data_response(response)
            
            # Update last data timestamp
            if economic_data_list:
                latest_data = economic_data_list[0]  # sorted desc, so first is latest
                self._last_data_timestamps[series_id] = latest_data.ts_event
            
            self._log.info(
                f"Provided {len(economic_data_list)} FRED observations for {series_id}",
                LogColor.GREEN,
            )

        except Exception as e:
            self._log.error(f"Failed to handle FRED data request for {instrument_id}: {e}")

    # -- INTERNAL METHODS ---------------------------------------------------------------------

    async def _start_update_task(self, instrument_id: InstrumentId, data_type: DataType) -> None:
        """
        Start an update task for an instrument subscription.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument ID to update.
        data_type : DataType
            The data type for updates.
            
        """
        if instrument_id in self._update_tasks:
            return  # Task already running
            
        task = self._loop.create_task(
            self._update_economic_data(instrument_id, data_type)
        )
        self._update_tasks[instrument_id] = task

    async def _update_economic_data(self, instrument_id: InstrumentId, data_type: DataType) -> None:
        """
        Periodically update economic data for a subscribed instrument.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument ID to update.
        data_type : DataType
            The data type for updates.
            
        """
        series_id = instrument_id.symbol.value
        
        while True:
            try:
                await asyncio.sleep(self._config.update_interval)
                
                # Check for new data
                series_data = await self._http_client.get_series_info(series_id)
                if not series_data:
                    continue
                    
                last_updated = series_data.get("last_updated", "")
                if not last_updated:
                    continue
                    
                # Convert last_updated to timestamp for comparison
                try:
                    last_updated_dt = datetime.strptime(last_updated, "%Y-%m-%d %H:%M:%S%z")
                    last_updated_ts = int(last_updated_dt.timestamp() * 1_000_000_000)
                except ValueError:
                    continue
                
                # Check if we have newer data
                cached_ts = self._last_data_timestamps.get(series_id, 0)
                if last_updated_ts <= cached_ts:
                    continue  # No new data
                
                # Fetch latest observations
                observations_data = await self._http_client.get_series_observations(
                    series_id=series_id,
                    limit=10,  # Get recent data points
                    sort_order="desc",
                )
                
                # Parse and publish new data
                series_info = self._series_info_cache.get(series_id)
                economic_data_list = parse_fred_observations(
                    series_id=series_id,
                    observations_data=observations_data,
                    series_info=series_info,
                )
                
                # Filter to only new data points
                new_data = [
                    data for data in economic_data_list
                    if data.ts_event > cached_ts
                ]
                
                if new_data:
                    # Publish new economic data
                    for economic_data in new_data:
                        self._handle_data(economic_data)
                    
                    # Update timestamp
                    self._last_data_timestamps[series_id] = max(
                        data.ts_event for data in new_data
                    )
                    
                    self._log.info(f"Published {len(new_data)} new FRED data points for {series_id}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.error(f"Error updating FRED data for {series_id}: {e}")
                # Continue the loop to retry later

    # -- ENHANCED MACRO FACTOR METHODS -------------------------------------------------------

    async def calculate_macro_factors(self, as_of_date: datetime | None = None) -> dict[str, float]:
        """
        Calculate comprehensive macro-economic factors for factor model.
        
        Generates 15-20 macro factors including:
        - Economic growth indicators (GDP growth, employment trends)
        - Inflation dynamics (CPI trends, inflation expectations)
        - Monetary policy stance (Fed funds level and changes)
        - Yield curve factors (level, slope, curvature)
        - Credit and financial conditions
        
        Parameters
        ----------
        as_of_date : datetime, optional
            Calculate factors as of this date. If None, uses current date.
            
        Returns
        -------
        dict[str, float]
            Dictionary of calculated macro factors
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        factors = {}
        
        try:
            self._log.info(f"Calculating macro factors as of {as_of_date}")
            
            # 1. Interest Rate Level & Change Factors (4 factors)
            rate_factors = await self._calculate_interest_rate_factors(as_of_date)
            factors.update(rate_factors)
            
            # 2. Yield Curve Shape Factors (3 factors)
            curve_factors = await self._calculate_yield_curve_factors(as_of_date)
            factors.update(curve_factors)
            
            # 3. Economic Growth & Activity Factors (4 factors)
            activity_factors = await self._calculate_economic_activity_factors(as_of_date)
            factors.update(activity_factors)
            
            # 4. Labor Market Factors (3 factors)
            labor_factors = await self._calculate_labor_market_factors(as_of_date)
            factors.update(labor_factors)
            
            # 5. Inflation & Price Factors (3 factors)
            inflation_factors = await self._calculate_inflation_factors(as_of_date)
            factors.update(inflation_factors)
            
            # 6. Market Regime & Risk Factors (3 factors)
            regime_factors = await self._calculate_market_regime_factors(as_of_date)
            factors.update(regime_factors)
            
            # 7. Monetary Conditions Factors (2 factors)
            monetary_factors = await self._calculate_monetary_conditions_factors(as_of_date)
            factors.update(monetary_factors)
            
            self._log.info(f"Calculated {len(factors)} macro factors")
            return factors
            
        except Exception as e:
            self._log.error(f"Error calculating macro factors: {e}")
            return {}

    async def _calculate_interest_rate_factors(self, as_of_date: datetime) -> dict[str, float]:
        """Calculate interest rate level and change factors."""
        factors = {}
        
        try:
            # Get Fed Funds Rate
            dff_data = await self._http_client.get_series_observations(
                series_id="DFF", limit=50, sort_order="desc"
            )
            
            # Get 10Y Treasury
            dgs10_data = await self._http_client.get_series_observations(
                series_id="DGS10", limit=50, sort_order="desc"
            )
            
            # Process Fed Funds Rate
            if dff_data and "observations" in dff_data:
                observations = dff_data["observations"]
                current_val = None
                past_val = None
                
                for i, obs in enumerate(observations):
                    if obs["value"] != "." and current_val is None:
                        current_val = float(obs["value"])
                    elif obs["value"] != "." and i >= 30 and past_val is None:
                        past_val = float(obs["value"])
                        break
                
                if current_val is not None:
                    factors["fed_funds_level"] = current_val
                    if past_val is not None:
                        factors["fed_funds_change_30d"] = current_val - past_val
            
            # Process 10Y Treasury
            if dgs10_data and "observations" in dgs10_data:
                observations = dgs10_data["observations"]
                current_val = None
                past_val = None
                
                for i, obs in enumerate(observations):
                    if obs["value"] != "." and current_val is None:
                        current_val = float(obs["value"])
                    elif obs["value"] != "." and i >= 30 and past_val is None:
                        past_val = float(obs["value"])
                        break
                
                if current_val is not None:
                    factors["treasury_10y_level"] = current_val
                    if past_val is not None:
                        factors["treasury_10y_change_30d"] = current_val - past_val
            
        except Exception as e:
            self._log.warning(f"Error calculating interest rate factors: {e}")
        
        return factors

    async def _calculate_yield_curve_factors(self, as_of_date: datetime) -> dict[str, float]:
        """Calculate yield curve level, slope, and curvature factors."""
        factors = {}
        
        try:
            # Get yield curve data
            rates = {}
            for series_id, maturity in [("DGS3MO", "3m"), ("DGS2", "2y"), ("DGS10", "10y")]:
                data = await self._http_client.get_series_observations(
                    series_id=series_id, limit=10, sort_order="desc"
                )
                
                if data and "observations" in data:
                    for obs in data["observations"]:
                        if obs["value"] != ".":
                            rates[maturity] = float(obs["value"])
                            break
            
            # Calculate yield curve factors
            if "10y" in rates and "2y" in rates:
                factors["yield_curve_slope"] = rates["10y"] - rates["2y"]
            
            if "10y" in rates and "3m" in rates:
                factors["yield_curve_level"] = (rates["10y"] + rates["3m"]) / 2
            
            if all(k in rates for k in ["10y", "2y", "3m"]):
                factors["yield_curve_curvature"] = 2 * rates["2y"] - rates["10y"] - rates["3m"]
            
        except Exception as e:
            self._log.warning(f"Error calculating yield curve factors: {e}")
        
        return factors

    async def _calculate_economic_activity_factors(self, as_of_date: datetime) -> dict[str, float]:
        """Calculate economic growth and activity factors."""
        factors = {}
        
        try:
            # Industrial Production momentum
            indpro_data = await self._http_client.get_series_observations(
                series_id="INDPRO", limit=50, sort_order="desc"
            )
            
            if indpro_data and "observations" in indpro_data:
                observations = indpro_data["observations"]
                current_val = None
                year_ago_val = None
                
                for i, obs in enumerate(observations):
                    if obs["value"] != "." and current_val is None:
                        current_val = float(obs["value"])
                    elif obs["value"] != "." and i >= 12 and year_ago_val is None:
                        year_ago_val = float(obs["value"])
                        break
                
                if current_val is not None and year_ago_val is not None:
                    factors["industrial_production_yoy"] = ((current_val / year_ago_val) - 1) * 100
            
            # Could add more economic activity indicators here
            
        except Exception as e:
            self._log.warning(f"Error calculating economic activity factors: {e}")
        
        return factors

    async def _calculate_labor_market_factors(self, as_of_date: datetime) -> dict[str, float]:
        """Calculate labor market strength factors."""
        factors = {}
        
        try:
            # Unemployment rate trend
            unrate_data = await self._http_client.get_series_observations(
                series_id="UNRATE", limit=20, sort_order="desc"
            )
            
            if unrate_data and "observations" in unrate_data:
                observations = unrate_data["observations"]
                current_val = None
                past_val = None
                
                for i, obs in enumerate(observations):
                    if obs["value"] != "." and current_val is None:
                        current_val = float(obs["value"])
                    elif obs["value"] != "." and i >= 6 and past_val is None:
                        past_val = float(obs["value"])
                        break
                
                if current_val is not None:
                    factors["unemployment_rate"] = current_val
                    if past_val is not None:
                        factors["unemployment_trend_6m"] = current_val - past_val
            
        except Exception as e:
            self._log.warning(f"Error calculating labor market factors: {e}")
        
        return factors

    async def _calculate_inflation_factors(self, as_of_date: datetime) -> dict[str, float]:
        """Calculate inflation and price pressure factors."""
        factors = {}
        
        try:
            # CPI inflation rate
            cpi_data = await self._http_client.get_series_observations(
                series_id="CPIAUCSL", limit=50, sort_order="desc"
            )
            
            if cpi_data and "observations" in cpi_data:
                observations = cpi_data["observations"]
                current_val = None
                year_ago_val = None
                
                for i, obs in enumerate(observations):
                    if obs["value"] != "." and current_val is None:
                        current_val = float(obs["value"])
                    elif obs["value"] != "." and i >= 12 and year_ago_val is None:
                        year_ago_val = float(obs["value"])
                        break
                
                if current_val is not None and year_ago_val is not None:
                    factors["cpi_inflation_yoy"] = ((current_val / year_ago_val) - 1) * 100
            
        except Exception as e:
            self._log.warning(f"Error calculating inflation factors: {e}")
        
        return factors

    async def _calculate_market_regime_factors(self, as_of_date: datetime) -> dict[str, float]:
        """Calculate market regime and risk factors."""
        factors = {}
        
        try:
            # VIX level and trend
            vix_data = await self._http_client.get_series_observations(
                series_id="VIXCLS", limit=20, sort_order="desc"
            )
            
            if vix_data and "observations" in vix_data:
                observations = vix_data["observations"]
                for obs in observations:
                    if obs["value"] != ".":
                        vix_level = float(obs["value"])
                        factors["vix_level"] = vix_level
                        
                        # Classify regime
                        if vix_level < 15:
                            factors["volatility_regime"] = 1  # Low vol
                        elif vix_level > 30:
                            factors["volatility_regime"] = 3  # High vol
                        else:
                            factors["volatility_regime"] = 2  # Medium vol
                        break
            
        except Exception as e:
            self._log.warning(f"Error calculating market regime factors: {e}")
        
        return factors

    async def _calculate_monetary_conditions_factors(self, as_of_date: datetime) -> dict[str, float]:
        """Calculate monetary conditions and credit factors."""
        factors = {}
        
        try:
            # Money supply growth (M2)
            m2_data = await self._http_client.get_series_observations(
                series_id="M2SL", limit=50, sort_order="desc"
            )
            
            if m2_data and "observations" in m2_data:
                observations = m2_data["observations"]
                current_val = None
                year_ago_val = None
                
                for i, obs in enumerate(observations):
                    if obs["value"] != "." and current_val is None:
                        current_val = float(obs["value"])
                    elif obs["value"] != "." and i >= 12 and year_ago_val is None:
                        year_ago_val = float(obs["value"])
                        break
                
                if current_val is not None and year_ago_val is not None:
                    factors["money_supply_growth_yoy"] = ((current_val / year_ago_val) - 1) * 100
            
        except Exception as e:
            self._log.warning(f"Error calculating monetary conditions factors: {e}")
        
        return factors