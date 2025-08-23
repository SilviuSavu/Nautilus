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

import dbnomics
from tenacity import RetryError

from nautilus_trader.adapters.dbnomics.core import DBNOMICS_VENUE
from nautilus_trader.adapters.dbnomics.errors import DBnomicsDataError
from nautilus_trader.adapters.dbnomics.errors import DBnomicsRateLimitError
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.instruments import GenericInstrument


class DBnomicsInstrumentProvider(InstrumentProvider):
    """
    Provides instruments from DBnomics economic data series.
    
    Each DBnomics series is represented as a GenericInstrument with the 
    series ID as the symbol and additional metadata.
    """

    def __init__(
        self,
        max_nb_series: int = 50,
        api_base_url: str | None = None,
        timeout: int = 30,
    ) -> None:
        super().__init__(venue=DBNOMICS_VENUE)
        self._max_nb_series = max_nb_series
        self._api_base_url = api_base_url
        self._timeout = timeout

    async def load_all_async(
        self,
        filters: dict | None = None,
    ) -> None:
        """
        Load all available series from specified providers and datasets.
        
        Parameters
        ----------
        filters : dict, optional
            Filters to apply. Expected format:
            {
                'providers': ['IMF', 'OECD'],
                'datasets': {'IMF': ['CPI'], 'OECD': ['QNA']},
                'dimensions': {'geo': ['FR', 'DE']}
            }
        """
        if filters is None:
            self._log.warning("No filters provided - cannot load all series without constraints")
            return
            
        providers = filters.get('providers', [])
        datasets = filters.get('datasets', {})
        dimensions = filters.get('dimensions')
        
        for provider in providers:
            provider_datasets = datasets.get(provider, [])
            for dataset in provider_datasets:
                try:
                    df = dbnomics.fetch_series(
                        provider_code=provider,
                        dataset_code=dataset, 
                        dimensions=dimensions,
                        max_nb_series=self._max_nb_series,
                        api_base_url=self._api_base_url,
                        timeout=self._timeout,
                    )
                    
                    await self._process_series_dataframe(df)
                    
                except dbnomics.FetchError as e:
                    self._log.error(f"Failed to fetch data from {provider}/{dataset}: {e}")
                except dbnomics.TooManySeries as e:
                    self._log.error(f"Too many series requested from {provider}/{dataset}: {e}")
                except RetryError as e:
                    self._log.error(f"Rate limit exceeded for {provider}/{dataset}: {e}")
                except Exception as e:
                    self._log.error(f"Unexpected error loading series from {provider}/{dataset}: {e}")

    async def load_ids_async(
        self,
        instrument_ids: list[InstrumentId],
        filters: dict | None = None,
    ) -> None:
        """
        Load specific series by their instrument IDs.
        """
        # Convert instrument IDs back to DBnomics series IDs
        series_ids = []
        for instrument_id in instrument_ids:
            # Expect format like "IMF-CPI-M.FR.PCPIEC_WT" -> "IMF/CPI/M.FR.PCPIEC_WT"
            symbol_str = str(instrument_id.symbol)
            series_id = symbol_str.replace('-', '/', 2)  # Only first 2 hyphens
            series_ids.append(series_id)
        
        try:
            df = dbnomics.fetch_series(
                series_ids=series_ids,
                max_nb_series=self._max_nb_series,
                api_base_url=self._api_base_url,
                timeout=self._timeout,
            )
            
            await self._process_series_dataframe(df)
            
        except dbnomics.FetchError as e:
            self._log.error(f"Failed to fetch series {series_ids}: {e}")
        except dbnomics.TooManySeries as e:
            self._log.error(f"Too many series requested {series_ids}: {e}")
        except RetryError as e:
            self._log.error(f"Rate limit exceeded for series {series_ids}: {e}")
        except Exception as e:
            self._log.error(f"Unexpected error loading series {series_ids}: {e}")

    async def load_async(
        self,
        instrument_id: InstrumentId,
        filters: dict | None = None,
    ) -> None:
        """
        Load a single series by its instrument ID.
        """
        await self.load_ids_async([instrument_id], filters)

    async def _process_series_dataframe(self, df) -> None:
        """
        Process a DBnomics dataframe and create instruments.
        """
        if df is None or df.empty:
            return
            
        # Group by series to get metadata
        for series_code in df['series_code'].unique():
            series_df = df[df['series_code'] == series_code].iloc[0]
            
            # Create instrument ID from series components
            provider_code = series_df.get('provider_code', 'UNKNOWN')
            dataset_code = series_df.get('dataset_code', 'UNKNOWN') 
            
            # Convert series ID to symbol format: "IMF/CPI/M.FR.PCPIEC_WT" -> "IMF-CPI-M.FR.PCPIEC_WT"
            symbol_str = f"{provider_code}-{dataset_code}-{series_code}"
            symbol = Symbol(symbol_str)
            instrument_id = InstrumentId(symbol, DBNOMICS_VENUE)
            
            # Create generic instrument
            instrument = GenericInstrument(
                instrument_id=instrument_id,
                native_symbol=Symbol(series_code),
                currency=None,  # Economic data typically doesn't have currency
                price_precision=8,  # Use high precision for economic data
                size_precision=0,
                price_increment=None,
                size_increment=None,
                lot_size=None,
                max_quantity=None,
                min_quantity=None,
                max_price=None,
                min_price=None,
                margin_init=None,
                margin_maint=None,
                maker_fee=None,
                taker_fee=None,
                ts_event=0,
                ts_init=0,
                info={
                    'provider_code': provider_code,
                    'dataset_code': dataset_code,
                    'series_name': series_df.get('series_name', ''),
                    'frequency': series_df.get('frequency', ''),
                    'unit': series_df.get('unit', ''),
                    'last_update': str(series_df.get('last_update', '')),
                },
            )
            
            self.add(instrument)