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

import pandas as pd

from nautilus_trader.core.data import Data
from nautilus_trader.model.identifiers import InstrumentId


class DBnomicsTimeSeriesData(Data):
    """
    Represents a DBnomics time series data point.
    
    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument identifier for this series.
    timestamp : pd.Timestamp
        The timestamp of the data point.
    value : Decimal
        The value of the time series at this timestamp.
    series_code : str
        The original DBnomics series code.
    provider_code : str
        The data provider code.
    dataset_code : str
        The dataset code.
    frequency : str, optional
        The frequency of the time series (e.g., 'M', 'Q', 'A').
    unit : str, optional
        The unit of measurement.
    """
    
    def __init__(
        self,
        instrument_id: InstrumentId,
        timestamp: pd.Timestamp,
        value: Decimal,
        series_code: str,
        provider_code: str,
        dataset_code: str,
        frequency: str | None = None,
        unit: str | None = None,
        ts_init: int | None = None,
    ) -> None:
        super().__init__(ts_init=ts_init or timestamp.value)
        
        self.instrument_id = instrument_id
        self.timestamp = timestamp
        self.value = value
        self.series_code = series_code
        self.provider_code = provider_code  
        self.dataset_code = dataset_code
        self.frequency = frequency
        self.unit = unit
    
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"instrument_id={self.instrument_id}, "
            f"timestamp={self.timestamp}, "
            f"value={self.value}, "
            f"series_code='{self.series_code}', "
            f"provider_code='{self.provider_code}', "
            f"dataset_code='{self.dataset_code}'"
            f")"
        )