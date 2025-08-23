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

from nautilus_trader.adapters.dbnomics.core import DBNOMICS_VENUE
from nautilus_trader.config import LiveDataClientConfig
from nautilus_trader.config import PositiveInt
from nautilus_trader.model.identifiers import Venue


class DBnomicsDataClientConfig(LiveDataClientConfig, frozen=True):
    """
    Configuration for ``DBnomicsDataClient`` instances.
    
    Parameters
    ----------
    venue : Venue, default DBNOMICS_VENUE
        The venue for the client.
    max_nb_series : PositiveInt, default 50
        Maximum number of series to fetch in a single request.
    api_base_url : str, optional
        Custom API base URL for DBnomics API.
        If ``None`` then will use default dbnomics.world API.
    editor_api_base_url : str, optional  
        Custom editor API base URL for filters.
        If ``None`` then will use default editor.nomics.world API.
    timeout : PositiveInt, default 30
        Request timeout in seconds.
    default_providers : list[str], optional
        Default data providers to search when none specified.
        Common providers: ['IMF', 'OECD', 'ECB', 'EUROSTAT', 'BIS'].
    instrument_id_mappings : dict[str, str], optional
        Custom mappings from series IDs to instrument identifiers.
    auto_retry : bool, default True
        Whether to automatically retry failed requests.
    """

    venue: Venue = DBNOMICS_VENUE
    max_nb_series: PositiveInt = 50
    api_base_url: str | None = None
    editor_api_base_url: str | None = None
    timeout: PositiveInt = 30
    default_providers: list[str] | None = None
    instrument_id_mappings: dict[str, str] | None = None
    auto_retry: bool = True