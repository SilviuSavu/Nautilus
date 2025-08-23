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

from datetime import datetime
from decimal import Decimal
from typing import Any

from nautilus_trader.adapters.fred.data import EconomicData
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AssetClass
from nautilus_trader.model.enums import InstrumentClass
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Currency
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity


def parse_fred_series_info(series_data: dict[str, Any]) -> dict[str, str]:
    """
    Parse FRED series information into a standardized format.
    
    Parameters
    ----------
    series_data : dict[str, Any]
        The raw series data from FRED API.
        
    Returns
    -------
    dict[str, str]
        The parsed series information.
        
    """
    return {
        "id": series_data.get("id", ""),
        "title": series_data.get("title", ""),
        "units": series_data.get("units", ""),
        "units_short": series_data.get("units_short", ""),
        "frequency": series_data.get("frequency", ""),
        "frequency_short": series_data.get("frequency_short", ""),
        "seasonal_adjustment": series_data.get("seasonal_adjustment", ""),
        "seasonal_adjustment_short": series_data.get("seasonal_adjustment_short", ""),
        "last_updated": series_data.get("last_updated", ""),
        "popularity": str(series_data.get("popularity", 0)),
        "notes": series_data.get("notes", ""),
    }


def parse_fred_observations(
    series_id: str,
    observations_data: dict[str, Any],
    series_info: dict[str, str] | None = None,
) -> list[EconomicData]:
    """
    Parse FRED observations data into EconomicData objects.
    
    Parameters
    ----------
    series_id : str
        The FRED series ID.
    observations_data : dict[str, Any]
        The raw observations data from FRED API.
    series_info : dict[str, str], optional
        Additional series metadata.
        
    Returns
    -------
    list[EconomicData]
        The parsed economic data objects.
        
    """
    observations = observations_data.get("observations", [])
    economic_data_list = []
    
    # Get series metadata
    units = series_info.get("units", "") if series_info else ""
    frequency = series_info.get("frequency", "") if series_info else ""
    seasonal_adjustment = series_info.get("seasonal_adjustment", "") if series_info else ""
    
    # Create instrument ID
    venue = Venue("FRED")
    symbol = Symbol(series_id)
    instrument_id = InstrumentId(symbol, venue)
    
    for obs in observations:
        try:
            # Parse date
            date_str = obs.get("date", "")
            if not date_str:
                continue
                
            # Convert date to timestamp
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                ts_event = dt_to_unix_nanos(date_obj)
            except ValueError:
                continue
            
            # Parse value
            value_str = obs.get("value", ".")
            if value_str == "." or value_str == "":
                # FRED uses "." for missing values
                continue
                
            # Parse last updated timestamp
            last_updated_str = obs.get("realtime_end", "")
            last_updated = 0
            if last_updated_str:
                try:
                    last_updated_dt = datetime.strptime(last_updated_str, "%Y-%m-%d")
                    last_updated = dt_to_unix_nanos(last_updated_dt)
                except ValueError:
                    pass
            
            # Create EconomicData object
            economic_data = EconomicData.create(
                instrument_id=instrument_id,
                series_id=series_id,
                value=value_str,
                units=units,
                frequency=frequency,
                seasonal_adjustment=seasonal_adjustment,
                ts_event=ts_event,
                ts_init=ts_event,
                last_updated=last_updated,
                release_date=ts_event,  # Use observation date as release date
            )
            
            economic_data_list.append(economic_data)
            
        except Exception:
            # Skip malformed observations
            continue
    
    return economic_data_list


def create_fred_instrument(series_info: dict[str, str]) -> Instrument:
    """
    Create a Nautilus Instrument from FRED series information.
    
    For economic data, we create a synthetic equity instrument since economic
    indicators don't fit traditional instrument categories. The instrument
    serves as a container for the economic time series data.
    
    Parameters
    ----------
    series_info : dict[str, str]
        The FRED series information.
        
    Returns
    -------
    Instrument
        The Nautilus instrument.
        
    """
    series_id = series_info["id"]
    title = series_info.get("title", series_id)
    
    # Create instrument components
    venue = Venue("FRED")
    symbol = Symbol(series_id)
    instrument_id = InstrumentId(symbol, venue)
    
    # Use appropriate currency based on series context
    # Most US economic data is conceptually in USD
    currency = USD
    
    # Create a synthetic equity instrument for economic data
    # This allows the economic data to be treated as a tradable asset
    # in backtesting and strategy contexts
    instrument = Equity(
        instrument_id=instrument_id,
        raw_symbol=Symbol(series_id),
        currency=currency,
        price_precision=8,  # High precision for economic data
        price_increment=Price.from_str("0.00000001"),
        multiplier=Quantity.from_int(1),
        lot_size=Quantity.from_int(1),
        isin=None,
        margin_init=Decimal("0.0"),
        margin_maint=Decimal("0.0"),
        maker_fee=Decimal("0.0"),
        taker_fee=Decimal("0.0"),
        ts_event=0,
        ts_init=0,
        info={"title": title, "type": "economic_indicator"},
    )
    
    return instrument


def get_popular_fred_series() -> dict[str, str]:
    """
    Get a dictionary of popular FRED economic series.
    
    Returns
    -------
    dict[str, str]
        A mapping of series ID to description.
        
    """
    return {
        # GDP and Growth
        "GDP": "Gross Domestic Product",
        "GDPC1": "Real Gross Domestic Product",
        "GDPPOT": "Real Potential Gross Domestic Product",
        
        # Employment and Labor
        "UNRATE": "Unemployment Rate",
        "CIVPART": "Labor Force Participation Rate",
        "PAYEMS": "All Employees, Total Nonfarm",
        "EMRATIO": "Employment-Population Ratio",
        
        # Inflation and Prices
        "CPIAUCSL": "Consumer Price Index for All Urban Consumers: All Items",
        "CPILFESL": "Consumer Price Index for All Urban Consumers: All Items Less Food and Energy",
        "PCEPI": "Personal Consumption Expenditures: Chain-type Price Index",
        "PCEPILFE": "Personal Consumption Expenditures Excluding Food and Energy (Core PCE Price Index)",
        
        # Interest Rates
        "FEDFUNDS": "Federal Funds Effective Rate",
        "DGS10": "10-Year Treasury Constant Maturity Rate",
        "DGS2": "2-Year Treasury Constant Maturity Rate",
        "DGS1MO": "1-Month Treasury Constant Maturity Rate",
        "TB3MS": "3-Month Treasury Bill: Secondary Market Rate",
        
        # Money Supply
        "M1SL": "M1 Money Stock",
        "M2SL": "M2 Money Stock",
        "BASE": "St. Louis Adjusted Monetary Base",
        
        # Housing
        "HOUST": "Housing Starts: Total: New Privately Owned Housing Units Started",
        "CSUSHPISA": "S&P/Case-Shiller U.S. National Home Price Index",
        "MORTGAGE30US": "30-Year Fixed Rate Mortgage Average in the United States",
        
        # Consumer and Business
        "RSAFS": "Advance Retail Sales: Retail Trade",
        "INDPRO": "Industrial Production Index",
        "UMCSENT": "University of Michigan: Consumer Sentiment",
        "DEXUSEU": "U.S. / Euro Foreign Exchange Rate",
        
        # International
        "DEXJPUS": "Japan / U.S. Foreign Exchange Rate",
        "DEXCHUS": "China / U.S. Foreign Exchange Rate",
        "DEXCAUS": "Canada / U.S. Foreign Exchange Rate",
    }


def validate_fred_series_id(series_id: str) -> bool:
    """
    Validate a FRED series ID format.
    
    Parameters
    ----------
    series_id : str
        The series ID to validate.
        
    Returns
    -------
    bool
        True if the series ID is valid format, False otherwise.
        
    """
    if not series_id or not isinstance(series_id, str):
        return False
        
    # FRED series IDs are typically uppercase alphanumeric with some special characters
    # They can contain letters, numbers, and limited special characters
    if len(series_id) < 1 or len(series_id) > 50:
        return False
        
    # Basic character validation - allow alphanumeric and common special chars
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    return all(c in allowed_chars for c in series_id.upper())


def format_economic_value(value: Decimal, units: str) -> str:
    """
    Format an economic value for display based on its units.
    
    Parameters
    ----------
    value : Decimal
        The economic value.
    units : str
        The units description from FRED.
        
    Returns
    -------
    str
        The formatted value string.
        
    """
    # Convert to float for formatting
    val = float(value)
    
    units_lower = units.lower()
    
    if "percent" in units_lower or "rate" in units_lower:
        return f"{val:.2f}%"
    elif "billions" in units_lower:
        return f"${val:,.1f}B"
    elif "millions" in units_lower:
        return f"${val:,.1f}M"
    elif "thousands" in units_lower:
        return f"${val:,.1f}K"
    elif "index" in units_lower:
        return f"{val:.2f}"
    elif "$" in units or "dollar" in units_lower:
        return f"${val:,.2f}"
    else:
        return f"{val:,.4f}"