"""
Collateral Management Engine
===========================

Mission-critical collateral and margin management system for leveraged trading operations.

Key Features:
- Real-time margin monitoring and alerts
- Cross-margining optimization for capital efficiency
- Regulatory capital compliance (Basel III, Dodd-Frank)
- M4 Max hardware acceleration for real-time calculations
- Integration with existing Risk Engine and hardware routing
"""

from collateral_engine import CollateralManagementEngine
from margin_calculator import MarginCalculator, MarginRequirement
from collateral_optimizer import CollateralOptimizer, CrossMarginOptimizer
from margin_monitor import RealTimeMarginMonitor, MarginAlert
from regulatory_calculator import RegulatoryCapitalCalculator
from models import *
from routes import router as collateral_router

__all__ = [
    'CollateralManagementEngine',
    'MarginCalculator',
    'MarginRequirement', 
    'CollateralOptimizer',
    'CrossMarginOptimizer',
    'RealTimeMarginMonitor',
    'MarginAlert',
    'RegulatoryCapitalCalculator',
    'collateral_router'
]