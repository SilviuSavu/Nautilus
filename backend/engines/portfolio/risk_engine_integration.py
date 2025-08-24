#!/usr/bin/env python3
"""
Risk Engine Integration Service
Enhanced integration with institutional Risk Engine for portfolio analysis
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import httpx
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RiskEngineResponse:
    success: bool
    analysis_id: str
    portfolio_risk_score: float
    risk_metrics: Dict[str, float]
    recommendations: List[str]
    computation_time_ms: float

class RiskEngineIntegration:
    """Integration service for Enhanced Risk Engine communication"""
    
    def __init__(self, risk_engine_url: str = "http://risk-engine:8200"):
        self.risk_engine_url = risk_engine_url
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.integration_active = False
        
    async def initialize(self) -> bool:
        """Initialize connection to Risk Engine"""
        try:
            # Test connection to Risk Engine
            response = await self.http_client.get(f"{self.risk_engine_url}/api/v1/enhanced-risk/health")
            if response.status_code == 200:
                self.integration_active = True
                logger.info("Risk Engine integration initialized successfully")
                return True
            else:
                logger.warning(f"Risk Engine health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to connect to Risk Engine: {e}")
            return False
    
    async def submit_portfolio_for_analysis(self, portfolio_data: Dict[str, Any]) -> Optional[RiskEngineResponse]:
        """Submit portfolio to Enhanced Risk Engine for institutional analysis"""
        if not self.integration_active:
            logger.warning("Risk Engine integration not active")
            return None
            
        try:
            start_time = time.time()
            
            # Prepare portfolio data for Risk Engine
            risk_request = {
                "portfolio_id": portfolio_data.get("portfolio_id"),
                "portfolio_value": portfolio_data.get("total_value", 0),
                "positions": self._format_positions_for_risk_engine(portfolio_data.get("positions", {})),
                "benchmark": portfolio_data.get("benchmark", "SPY"),
                "analysis_type": "institutional_comprehensive",
                "request_timestamp": datetime.now().isoformat()
            }
            
            # Submit to Enhanced Risk Engine hybrid processor
            response = await self.http_client.post(
                f"{self.risk_engine_url}/api/v1/enhanced-risk/hybrid/submit",
                json=risk_request
            )
            
            if response.status_code == 200:
                result = response.json()
                computation_time = (time.time() - start_time) * 1000
                
                return RiskEngineResponse(
                    success=True,
                    analysis_id=result.get("analysis_id", f"risk_{int(time.time())}"),
                    portfolio_risk_score=result.get("risk_score", 0.5),
                    risk_metrics=result.get("risk_metrics", {}),
                    recommendations=result.get("recommendations", []),
                    computation_time_ms=computation_time
                )
            else:
                logger.error(f"Risk Engine analysis failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting to Risk Engine: {e}")
            return None
    
    async def get_enhanced_risk_dashboard(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get enhanced risk dashboard from Risk Engine"""
        if not self.integration_active:
            return None
            
        try:
            dashboard_request = {
                "portfolio_id": portfolio_id,
                "dashboard_type": "Portfolio Risk",
                "include_charts": True,
                "format": "json"
            }
            
            response = await self.http_client.post(
                f"{self.risk_engine_url}/api/v1/enhanced-risk/dashboard/generate",
                json=dashboard_request
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Risk dashboard generation failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting risk dashboard: {e}")
            return None
    
    async def run_institutional_backtest(self, backtest_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run institutional-grade backtest using Risk Engine's VectorBT integration"""
        if not self.integration_active:
            return None
            
        try:
            # Use Enhanced Risk Engine's VectorBT backtesting
            response = await self.http_client.post(
                f"{self.risk_engine_url}/api/v1/enhanced-risk/backtest/run",
                json={
                    "portfolio_data": backtest_config.get("portfolio_data"),
                    "start_date": backtest_config.get("start_date"),
                    "end_date": backtest_config.get("end_date"),
                    "benchmark": backtest_config.get("benchmark", "SPY"),
                    "analysis_depth": "comprehensive",
                    "use_gpu_acceleration": True
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Institutional backtest failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error running institutional backtest: {e}")
            return None
    
    async def get_alpha_signals(self, portfolio_id: str, symbols: List[str]) -> Optional[Dict[str, Any]]:
        """Get AI-generated alpha signals from Risk Engine's Qlib integration"""
        if not self.integration_active:
            return None
            
        try:
            alpha_request = {
                "portfolio_id": portfolio_id,
                "symbols": symbols,
                "signal_type": "alpha_generation",
                "time_horizon": "1_month",
                "use_neural_engine": True
            }
            
            response = await self.http_client.post(
                f"{self.risk_engine_url}/api/v1/enhanced-risk/alpha/generate",
                json=alpha_request
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Alpha signal generation failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting alpha signals: {e}")
            return None
    
    async def store_portfolio_timeseries(self, portfolio_id: str, timeseries_data: Dict[str, Any]) -> bool:
        """Store portfolio time-series data using Risk Engine's ArcticDB integration"""
        if not self.integration_active:
            return False
            
        try:
            storage_request = {
                "symbol": f"portfolio_{portfolio_id}",
                "data": timeseries_data,
                "metadata": {
                    "data_type": "portfolio_performance",
                    "portfolio_id": portfolio_id,
                    "stored_at": datetime.now().isoformat()
                }
            }
            
            response = await self.http_client.post(
                f"{self.risk_engine_url}/api/v1/enhanced-risk/data/store",
                json=storage_request
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error storing portfolio time-series: {e}")
            return False
    
    async def retrieve_portfolio_history(self, portfolio_id: str, start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
        """Retrieve portfolio history using Risk Engine's ArcticDB"""
        if not self.integration_active:
            return None
            
        try:
            symbol = f"portfolio_{portfolio_id}"
            params = {
                "start_date": start_date,
                "end_date": end_date
            }
            
            response = await self.http_client.get(
                f"{self.risk_engine_url}/api/v1/enhanced-risk/data/retrieve/{symbol}",
                params=params
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving portfolio history: {e}")
            return None
    
    async def calculate_xva_adjustments(self, derivatives_positions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Calculate XVA adjustments for derivatives using Risk Engine's ORE gateway"""
        if not self.integration_active:
            return None
            
        try:
            xva_request = {
                "positions": derivatives_positions,
                "calculation_date": datetime.now().isoformat(),
                "include_cva": True,
                "include_dva": True,
                "include_fva": True,
                "include_kva": True
            }
            
            response = await self.http_client.post(
                f"{self.risk_engine_url}/api/v1/enhanced-risk/xva/calculate",
                json=xva_request
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error calculating XVA adjustments: {e}")
            return None
    
    def _format_positions_for_risk_engine(self, positions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format portfolio positions for Risk Engine consumption"""
        formatted_positions = []
        
        for symbol, position in positions.items():
            formatted_position = {
                "symbol": symbol,
                "quantity": position.get("quantity", 0),
                "market_value": position.get("market_value", 0),
                "weight": position.get("weight", 0),
                "avg_cost": position.get("avg_cost", 0),
                "unrealized_pnl": position.get("unrealized_pnl", 0),
                "sector": position.get("sector"),
                "currency": position.get("currency", "USD")
            }
            formatted_positions.append(formatted_position)
        
        return formatted_positions
    
    async def close(self):
        """Close HTTP client connection"""
        await self.http_client.aclose()

class PortfolioRiskMonitor:
    """Real-time portfolio risk monitoring service"""
    
    def __init__(self, risk_integration: RiskEngineIntegration):
        self.risk_integration = risk_integration
        self.monitoring_active = False
        self.risk_alerts: List[Dict[str, Any]] = []
        
    async def start_monitoring(self, portfolio_id: str, risk_thresholds: Dict[str, float]):
        """Start real-time risk monitoring for portfolio"""
        self.monitoring_active = True
        logger.info(f"Started risk monitoring for portfolio {portfolio_id}")
        
        # In production, this would continuously monitor portfolio risk metrics
        # For now, simulate periodic risk checks
        while self.monitoring_active:
            try:
                # Check risk metrics every 5 minutes
                await asyncio.sleep(300)
                
                # Get current risk analysis from Risk Engine
                # This would be implemented based on real-time data feeds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def stop_monitoring(self):
        """Stop risk monitoring"""
        self.monitoring_active = False
        logger.info("Risk monitoring stopped")
    
    def get_risk_alerts(self) -> List[Dict[str, Any]]:
        """Get current risk alerts"""
        return self.risk_alerts