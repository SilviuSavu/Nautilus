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

from nautilus_trader.core.data import Data
from nautilus_trader.core.datetime import unix_nanos_to_iso8601
from nautilus_trader.model.custom import customdataclass
from nautilus_trader.model.identifiers import InstrumentId


@customdataclass
class EconomicData(Data):
    """
    Represents economic time series data from the FRED API.
    
    This data type encapsulates economic indicators such as GDP, unemployment rates,
    inflation data, interest rates, and other macroeconomic time series.
    
    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument ID associated with this economic data series.
    series_id : str
        The FRED series ID (e.g., "GDP", "UNRATE", "CPIAUCSL").
    value : Decimal
        The economic indicator value.
    units : str
        The units of measurement for the value.
    frequency : str
        The frequency of the data series (e.g., "Monthly", "Quarterly", "Annual").
    seasonal_adjustment : str
        The seasonal adjustment status (e.g., "Seasonally Adjusted", "Not Seasonally Adjusted").
    last_updated : int
        The timestamp (Unix nanoseconds) when the data was last updated by FRED.
    release_date : int
        The timestamp (Unix nanoseconds) when the data was officially released.
    
    """
    
    instrument_id: InstrumentId
    series_id: str
    value: Decimal
    units: str = ""
    frequency: str = ""  
    seasonal_adjustment: str = ""
    last_updated: int = 0
    release_date: int = 0

    def __repr__(self) -> str:
        return (
            f"EconomicData(series_id={self.series_id}, "
            f"instrument_id={self.instrument_id}, "
            f"value={self.value}, "
            f"units={self.units}, "
            f"frequency={self.frequency}, "
            f"ts_event={unix_nanos_to_iso8601(self.ts_event)}, "
            f"ts_init={unix_nanos_to_iso8601(self.ts_init)})"
        )

    @classmethod
    def create(
        cls,
        instrument_id: InstrumentId,
        series_id: str,
        value: Decimal | float | int | str,
        units: str = "",
        frequency: str = "",
        seasonal_adjustment: str = "",
        ts_event: int = 0,
        ts_init: int = 0,
        last_updated: int = 0,
        release_date: int = 0,
    ) -> "EconomicData":
        """
        Create an EconomicData instance with convenient type conversion.
        
        Parameters
        ----------
        instrument_id : InstrumentId
            The instrument ID.
        series_id : str
            The FRED series ID.
        value : Decimal | float | int | str
            The economic value (will be converted to Decimal).
        units : str, optional
            The units of measurement.
        frequency : str, optional
            The data frequency.
        seasonal_adjustment : str, optional
            The seasonal adjustment status.
        ts_event : int, optional
            The event timestamp in Unix nanoseconds.
        ts_init : int, optional
            The initialization timestamp in Unix nanoseconds.
        last_updated : int, optional
            The last updated timestamp in Unix nanoseconds.
        release_date : int, optional
            The release date timestamp in Unix nanoseconds.
            
        Returns
        -------
        EconomicData
        
        """
        if not isinstance(value, Decimal):
            if isinstance(value, str):
                # Handle potential "." or empty string values from FRED
                if value == "." or value == "":
                    value = Decimal("0")
                else:
                    value = Decimal(value)
            else:
                value = Decimal(str(value))
                
        return cls(
            ts_event=ts_event,
            ts_init=ts_init,
            instrument_id=instrument_id,
            series_id=series_id,
            value=value,
            units=units,
            frequency=frequency,
            seasonal_adjustment=seasonal_adjustment,
            last_updated=last_updated,
            release_date=release_date,
        )
        
    @property
    def is_valid_value(self) -> bool:
        """
        Check if the economic data value is valid (not NaN or missing).
        
        Returns
        -------
        bool
            True if the value is valid, False otherwise.
            
        """
        return self.value != Decimal("0") or self.units != ""
        
    def to_dict(self) -> dict:
        """
        Convert the economic data to a dictionary representation.
        
        Returns
        -------
        dict
            The dictionary representation.
            
        """
        return {
            "instrument_id": str(self.instrument_id),
            "series_id": self.series_id,
            "value": float(self.value),
            "units": self.units,
            "frequency": self.frequency,
            "seasonal_adjustment": self.seasonal_adjustment,
            "ts_event": self.ts_event,
            "ts_init": self.ts_init,
            "last_updated": self.last_updated,
            "release_date": self.release_date,
        }