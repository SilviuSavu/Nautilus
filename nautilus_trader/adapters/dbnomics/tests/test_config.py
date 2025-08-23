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

import pytest

from nautilus_trader.adapters.dbnomics.config import DBnomicsDataClientConfig
from nautilus_trader.adapters.dbnomics.core import DBNOMICS_VENUE


class TestDBnomicsDataClientConfig:
    """
    Test cases for DBnomicsDataClientConfig.
    """

    def test_default_configuration(self):
        """Test default configuration values."""
        config = DBnomicsDataClientConfig()
        
        assert config.venue == DBNOMICS_VENUE
        assert config.max_nb_series == 50
        assert config.api_base_url is None
        assert config.editor_api_base_url is None
        assert config.timeout == 30
        assert config.default_providers is None
        assert config.instrument_id_mappings is None
        assert config.auto_retry is True

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = DBnomicsDataClientConfig(
            max_nb_series=100,
            api_base_url="https://custom.api.url",
            editor_api_base_url="https://custom.editor.url",
            timeout=60,
            default_providers=["IMF", "OECD"],
            instrument_id_mappings={"IMF/CPI/A.FR": "EUR_CPI"},
            auto_retry=False,
        )
        
        assert config.max_nb_series == 100
        assert config.api_base_url == "https://custom.api.url"
        assert config.editor_api_base_url == "https://custom.editor.url"
        assert config.timeout == 60
        assert config.default_providers == ["IMF", "OECD"]
        assert config.instrument_id_mappings == {"IMF/CPI/A.FR": "EUR_CPI"}
        assert config.auto_retry is False

    def test_configuration_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            DBnomicsDataClientConfig(max_nb_series=0)
            
        with pytest.raises(ValueError):
            DBnomicsDataClientConfig(timeout=-1)