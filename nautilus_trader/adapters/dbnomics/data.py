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

from decimal import Decimal

import dbnomics
import pandas as pd
from tenacity import RetryError

from nautilus_trader.adapters.dbnomics.config import DBnomicsDataClientConfig
from nautilus_trader.adapters.dbnomics.core import DBNOMICS_VENUE
from nautilus_trader.adapters.dbnomics.errors import DBnomicsConnectionError
from nautilus_trader.adapters.dbnomics.errors import DBnomicsDataError
from nautilus_trader.adapters.dbnomics.errors import DBnomicsRateLimitError
from nautilus_trader.adapters.dbnomics.types import DBnomicsTimeSeriesData
from nautilus_trader.common.enums import LogColor
from nautilus_trader.data.messages import RequestData
from nautilus_trader.data.messages import SubscribeData
from nautilus_trader.data.messages import UnsubscribeData
from nautilus_trader.live.data_client import LiveDataClient
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId


class DBnomicsDataClient(LiveDataClient):
    """
    Provides a data client for DBnomics economic time series data.
    
    This client fetches economic and statistical data from the dbnomics.world API
    and transforms it into NautilusTrader data types for use in trading strategies.
    """

    def __init__(
        self,
        loop,
        client_id: ClientId,
        config: DBnomicsDataClientConfig,
        msgbus,
        cache,
        clock,
    ) -> None:
        super().__init__(
            loop=loop,
            client_id=client_id,
            venue=DBNOMICS_VENUE,
            msgbus=msgbus,
            cache=cache,
            clock=clock,
            config=config,
        )
        
        self._config = config
        self._subscriptions: dict[InstrumentId, dict] = {}

    async def _connect(self) -> None:
        """
        Connect to the DBnomics API.
        """
        self._log.info("Connecting to DBnomics API", LogColor.BLUE)
        
        # Test connection by fetching a simple series
        try:
            test_df = dbnomics.fetch_series(
                series_ids=["IMF/IFS/A.AD.BOP_BP6_GG."],
                max_nb_series=1,
                api_base_url=self._config.api_base_url,
                timeout=self._config.timeout,
            )
            if test_df is not None and not test_df.empty:
                self._log.info("Successfully connected to DBnomics API", LogColor.GREEN)
            else:
                self._log.warning("DBnomics API connection test returned empty data")
        except dbnomics.FetchError as e:
            error_msg = f"DBnomics API connection failed: {e}"
            self._log.error(error_msg)
            raise DBnomicsConnectionError(error_msg) from e
        except RetryError as e:
            error_msg = f"DBnomics API connection retries exhausted: {e}"
            self._log.error(error_msg)
            raise DBnomicsRateLimitError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error connecting to DBnomics API: {e}"
            self._log.error(error_msg)
            raise DBnomicsConnectionError(error_msg) from e

    async def _disconnect(self) -> None:
        """
        Disconnect from the DBnomics API.
        """
        self._log.info("Disconnecting from DBnomics API", LogColor.BLUE)
        self._subscriptions.clear()
        self._log.info("Disconnected from DBnomics API", LogColor.GREEN)

    def reset(self) -> None:
        """
        Reset the data client.
        """
        self._subscriptions.clear()
        super().reset()

    def dispose(self) -> None:
        """
        Dispose of the data client.
        """
        self._subscriptions.clear()
        super().dispose()

    # -- SUBSCRIPTIONS ----------------------------------------------------------------------------

    async def _subscribe(self, command: SubscribeData) -> None:
        """
        Subscribe to a DBnomics data feed.
        
        For economic data, subscription means setting up periodic fetching
        of the latest data points for the requested series.
        """
        data_type = command.data_type
        
        if data_type.type != DBnomicsTimeSeriesData:
            self._log.error(f"Cannot subscribe to unsupported data type {data_type}")
            return
            
        # Extract series information from metadata
        metadata = data_type.metadata or {}
        instrument_id = metadata.get('instrument_id')
        
        if not instrument_id:
            self._log.error("No instrument_id provided in subscription metadata")
            return
            
        self._subscriptions[instrument_id] = {
            'command': command,
            'metadata': metadata,
        }
        
        # Fetch initial data for the series
        await self._fetch_series_data(instrument_id, metadata)
        
        self._log.info(f"Subscribed to DBnomics series {instrument_id}")

    async def _unsubscribe(self, command: UnsubscribeData) -> None:
        """
        Unsubscribe from a DBnomics data feed.
        """
        data_type = command.data_type
        metadata = data_type.metadata or {}
        instrument_id = metadata.get('instrument_id')
        
        if instrument_id in self._subscriptions:
            del self._subscriptions[instrument_id]
            self._log.info(f"Unsubscribed from DBnomics series {instrument_id}")

    # -- REQUESTS ---------------------------------------------------------------------------------

    async def _request(self, request: RequestData) -> None:
        """
        Handle data requests from DBnomics.
        """
        data_type = request.data_type
        
        if data_type.type != DBnomicsTimeSeriesData:
            self._log.error(f"Cannot request unsupported data type {data_type}")
            return
            
        metadata = data_type.metadata or {}
        instrument_id = metadata.get('instrument_id')
        
        if not instrument_id:
            self._log.error("No instrument_id provided in request metadata")
            return
            
        await self._fetch_series_data(instrument_id, metadata, request.correlation_id)

    async def _fetch_series_data(
        self,
        instrument_id: InstrumentId,
        metadata: dict,
        correlation_id: str | None = None,
    ) -> None:
        """
        Fetch time series data for a specific instrument.
        """
        try:
            # Convert instrument ID back to DBnomics series format
            symbol_str = str(instrument_id.symbol)
            parts = symbol_str.split('-', 2)
            
            if len(parts) != 3:
                self._log.error(f"Invalid instrument symbol format: {symbol_str}")
                return
                
            provider_code, dataset_code, series_code = parts
            series_id = f"{provider_code}/{dataset_code}/{series_code}"
            
            # Extract additional parameters from metadata
            start_date = metadata.get('start_date')
            end_date = metadata.get('end_date') 
            filters = metadata.get('filters')
            
            # Fetch the data
            self._log.info(f"Fetching DBnomics series: {series_id}")
            
            df = dbnomics.fetch_series(
                series_ids=[series_id],
                max_nb_series=self._config.max_nb_series,
                api_base_url=self._config.api_base_url,
                editor_api_base_url=self._config.editor_api_base_url,
                filters=filters,
                timeout=self._config.timeout,
            )
            
            if df is None or df.empty:
                self._log.warning(f"No data returned for series {series_id}")
                return
                
            # Process and publish data points
            await self._process_and_publish_data(df, instrument_id, correlation_id)
            
        except dbnomics.FetchError as e:
            error_msg = f"Failed to fetch series {series_id}: {e}"
            self._log.error(error_msg)
            raise DBnomicsDataError(error_msg) from e
        except dbnomics.TooManySeries as e:
            error_msg = f"Too many series requested for {series_id}: {e}"
            self._log.error(error_msg)
            raise DBnomicsDataError(error_msg) from e
        except RetryError as e:
            error_msg = f"Rate limit exceeded fetching {series_id}: {e}"
            self._log.error(error_msg)
            raise DBnomicsRateLimitError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error fetching series data for {instrument_id}: {e}"
            self._log.error(error_msg)
            raise DBnomicsDataError(error_msg) from e

    async def _process_and_publish_data(
        self,
        df: pd.DataFrame,
        instrument_id: InstrumentId,
        correlation_id: str | None = None,
    ) -> None:
        """
        Process DBnomics dataframe and publish data points.
        """
        data_points = []
        
        for _, row in df.iterrows():
            try:
                # Skip rows with missing values
                if pd.isna(row.get('value')):
                    continue
                    
                timestamp = pd.to_datetime(row['period'])
                value = Decimal(str(row['value']))
                
                data_point = DBnomicsTimeSeriesData(
                    instrument_id=instrument_id,
                    timestamp=timestamp,
                    value=value,
                    series_code=row.get('series_code', ''),
                    provider_code=row.get('provider_code', ''),
                    dataset_code=row.get('dataset_code', ''),
                    frequency=row.get('freq', ''),
                    unit=row.get('unit', ''),
                    ts_init=timestamp.value,
                )
                
                data_points.append(data_point)
                
            except Exception as e:
                self._log.warning(f"Failed to process data row: {e}")
                continue
        
        # Publish all data points
        for data_point in data_points:
            self._handle_data(data_point, correlation_id)
        
        self._log.info(f"Published {len(data_points)} data points for {instrument_id}")

    def _handle_data(self, data: DBnomicsTimeSeriesData, correlation_id: str | None = None) -> None:
        """
        Handle and route data through the message bus.
        """
        self._msgbus.publish(
            topic=f"data.{data.__class__.__name__}.{data.instrument_id}",
            msg=data,
        )
        
        if correlation_id:
            # Send response for data request
            self._msgbus.response(correlation_id, data)