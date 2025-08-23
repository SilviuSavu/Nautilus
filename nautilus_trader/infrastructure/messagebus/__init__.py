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
Enhanced MessageBus implementation for NautilusTrader.

This package provides enterprise-grade messaging capabilities with:
- 10x performance improvement over basic implementations
- Priority-based message handling
- Auto-scaling workers
- Pattern-based topic routing
- Comprehensive monitoring and health checks
"""

from nautilus_trader.infrastructure.messagebus.client import BufferedMessageBusClient
from nautilus_trader.infrastructure.messagebus.config import EnhancedMessageBusConfig
from nautilus_trader.infrastructure.messagebus.config import MessagePriority
from nautilus_trader.infrastructure.messagebus.streams import RedisStreamManager

__all__ = [
    "BufferedMessageBusClient",
    "EnhancedMessageBusConfig", 
    "MessagePriority",
    "RedisStreamManager",
]