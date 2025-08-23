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
DBnomics adapter for economic and statistical time series data.

The DBnomics adapter provides access to economic data from various statistical offices,
central banks and international organizations through the dbnomics.world API.
"""

from nautilus_trader.adapters.dbnomics.config import DBnomicsDataClientConfig
from nautilus_trader.adapters.dbnomics.data import DBnomicsDataClient
from nautilus_trader.adapters.dbnomics.factories import DBnomicsInstrumentProviderFactory
from nautilus_trader.adapters.dbnomics.factories import DBnomicsLiveDataClientFactory
from nautilus_trader.adapters.dbnomics.providers import DBnomicsInstrumentProvider
from nautilus_trader.adapters.dbnomics.types import DBnomicsTimeSeriesData

__all__ = [
    "DBnomicsDataClient",
    "DBnomicsDataClientConfig", 
    "DBnomicsInstrumentProvider",
    "DBnomicsInstrumentProviderFactory",
    "DBnomicsLiveDataClientFactory",
    "DBnomicsTimeSeriesData",
]