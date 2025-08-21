"""
EDGAR Data Client
================

Live data client for SEC EDGAR API integration with NautilusTrader.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from uuid import UUID

from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.providers import InstrumentProvider
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.live.data_client import LiveMarketDataClient
from nautilus_trader.model.data import CustomData, DataType
from nautilus_trader.model.identifiers import ClientId, InstrumentId

from edgar_connector.api_client import EDGARAPIClient
from edgar_connector.config import EDGARConfig, EDGARDataClientConfig
from edgar_connector.data_types import (
    FilingData,
    CompanyFacts,
    SECFiling,
    FilingType,
    EDGARSubscription
)
from edgar_connector.instrument_provider import EDGARInstrumentProvider
from edgar_connector.utils import XBRLParser, normalize_cik

logger = logging.getLogger(__name__)


class EDGARDataClient(LiveMarketDataClient):
    """
    Live data client for SEC EDGAR API.
    
    This client provides streaming access to SEC filings and financial data,
    integrating with NautilusTrader's data architecture.
    """
    
    def __init__(
        self,
        client_id: ClientId,
        config: EDGARConfig,
        data_config: EDGARDataClientConfig,
        clock: LiveClock,
        instrument_provider: Optional[EDGARInstrumentProvider] = None,
    ):
        super().__init__(
            client_id=client_id,
            clock=clock,
            instrument_provider=instrument_provider,
        )
        
        self.config = config
        self.data_config = data_config
        
        # API client
        self.api_client = EDGARAPIClient(config)
        
        # Subscription management
        self._subscriptions: Dict[DataType, Set[EDGARSubscription]] = {}
        self._subscription_tasks: Dict[str, asyncio.Task] = {}
        
        # Data processing
        self.xbrl_parser = XBRLParser()
        
        # State
        self._is_connected = False
        self._last_filing_check = None
        
        logger.info(f"EDGAR Data Client initialized with client_id: {client_id}")
    
    # LiveMarketDataClient implementation
    
    async def _connect(self) -> None:
        """Connect to EDGAR API."""
        logger.info("Connecting to SEC EDGAR API...")
        
        try:
            # Health check
            if not await self.api_client.health_check():
                raise Exception("EDGAR API health check failed")
            
            # Initialize instrument provider if needed
            if self._instrument_provider:
                await self._instrument_provider.load_all_async()
            
            self._is_connected = True
            logger.info("Successfully connected to SEC EDGAR API")
            
            # Start subscription monitoring if enabled
            if self.data_config.auto_subscribe_filings:
                self._start_filing_monitor()
            
        except Exception as e:
            logger.error(f"Failed to connect to EDGAR API: {e}")
            raise
    
    async def _disconnect(self) -> None:
        """Disconnect from EDGAR API."""
        logger.info("Disconnecting from SEC EDGAR API...")
        
        # Stop all subscription tasks
        for task_id, task in self._subscription_tasks.items():
            logger.info(f"Cancelling subscription task: {task_id}")
            task.cancel()
            
        self._subscription_tasks.clear()
        self._subscriptions.clear()
        
        # Close API client
        await self.api_client.close()
        
        self._is_connected = False
        logger.info("Disconnected from SEC EDGAR API")
    
    def _reset(self) -> None:
        """Reset client state."""
        self._subscriptions.clear()
        self._subscription_tasks.clear()
        self._last_filing_check = None
        self._is_connected = False
        logger.info("EDGAR Data Client reset")
    
    async def _subscribe(self, data_type: DataType) -> None:
        """Subscribe to EDGAR data type."""
        logger.info(f"Subscribing to data type: {data_type}")
        
        if not self._is_connected:
            logger.warning("Cannot subscribe - client not connected")
            return
        
        # Parse subscription from data type metadata
        subscription = self._parse_subscription_from_data_type(data_type)
        
        if data_type not in self._subscriptions:
            self._subscriptions[data_type] = set()
        
        self._subscriptions[data_type].add(subscription)
        
        # Start subscription task
        task_id = f"{data_type}_{len(self._subscription_tasks)}"
        task = asyncio.create_task(self._subscription_worker(data_type, subscription))
        self._subscription_tasks[task_id] = task
        
        logger.info(f"Started subscription task: {task_id}")
    
    async def _unsubscribe(self, data_type: DataType) -> None:
        """Unsubscribe from EDGAR data type."""
        logger.info(f"Unsubscribing from data type: {data_type}")
        
        # Remove subscriptions
        if data_type in self._subscriptions:
            del self._subscriptions[data_type]
        
        # Cancel related tasks
        tasks_to_cancel = [
            (task_id, task) for task_id, task in self._subscription_tasks.items()
            if task_id.startswith(str(data_type))
        ]
        
        for task_id, task in tasks_to_cancel:
            logger.info(f"Cancelling subscription task: {task_id}")
            task.cancel()
            del self._subscription_tasks[task_id]
    
    async def _request(self, data_type: DataType, correlation_id: UUID4) -> None:
        """Handle data request."""
        logger.info(f"Handling data request: {data_type}, correlation_id: {correlation_id}")
        
        try:
            if not self._is_connected:
                logger.warning("Cannot handle request - client not connected")
                return
            
            # Parse request parameters
            subscription = self._parse_subscription_from_data_type(data_type)
            
            # Fetch data based on subscription type
            if subscription.cik:
                await self._fetch_company_data(subscription.cik, correlation_id)
            elif subscription.ticker:
                # Resolve ticker to CIK
                if self._instrument_provider:
                    cik = self._instrument_provider.resolve_ticker_to_cik(subscription.ticker)
                    if cik:
                        await self._fetch_company_data(cik, correlation_id)
                    else:
                        logger.warning(f"Could not resolve ticker {subscription.ticker} to CIK")
                else:
                    cik = await self.api_client.resolve_ticker_to_cik(subscription.ticker)
                    if cik:
                        await self._fetch_company_data(cik, correlation_id)
            else:
                # Fetch recent filings for all companies
                await self._fetch_recent_filings(correlation_id)
        
        except Exception as e:
            logger.error(f"Error handling data request: {e}")
    
    # EDGAR-specific methods
    
    def _parse_subscription_from_data_type(self, data_type: DataType) -> EDGARSubscription:
        """Parse subscription parameters from DataType metadata."""
        # Extract parameters from data type
        # This is a simplified implementation - in practice you'd have
        # more sophisticated parameter parsing
        
        metadata = getattr(data_type, 'metadata', {})
        
        return EDGARSubscription(
            cik=metadata.get('cik'),
            ticker=metadata.get('ticker'),
            filing_types=[FilingType(ft) for ft in metadata.get('filing_types', ['10-K', '10-Q'])],
            include_amendments=metadata.get('include_amendments', False),
            max_age_days=metadata.get('max_age_days', 365)
        )
    
    async def _subscription_worker(
        self,
        data_type: DataType,
        subscription: EDGARSubscription
    ) -> None:
        """Worker task for handling subscriptions."""
        logger.info(f"Starting subscription worker for {data_type}")
        
        try:
            while True:
                # Check for new filings periodically
                if subscription.cik:
                    await self._check_company_filings(subscription.cik, data_type)
                elif subscription.ticker:
                    # Resolve ticker to CIK
                    cik = None
                    if self._instrument_provider:
                        cik = self._instrument_provider.resolve_ticker_to_cik(subscription.ticker)
                    else:
                        cik = await self.api_client.resolve_ticker_to_cik(subscription.ticker)
                    
                    if cik:
                        await self._check_company_filings(cik, data_type)
                
                # Wait for next check
                await asyncio.sleep(self.data_config.subscription_check_interval)
                
        except asyncio.CancelledError:
            logger.info(f"Subscription worker cancelled for {data_type}")
        except Exception as e:
            logger.error(f"Error in subscription worker for {data_type}: {e}")
    
    async def _check_company_filings(self, cik: str, data_type: DataType) -> None:
        """Check for new filings for a specific company."""
        try:
            normalized_cik = normalize_cik(cik)
            
            # Get company submissions
            submissions_data = await self.api_client.get_submissions(normalized_cik)
            
            if not submissions_data:
                return
            
            # Process recent filings
            filings = submissions_data.get("filings", {}).get("recent", {})
            
            if not filings:
                return
            
            # Extract filing data
            forms = filings.get("form", [])
            filing_dates = filings.get("filingDate", [])
            accession_numbers = filings.get("accessionNumber", [])
            
            # Create filing data objects for recent filings
            cutoff_date = datetime.now() - timedelta(days=self.data_config.max_filing_age_days)
            
            for i, (form, date_str, accession) in enumerate(
                zip(forms, filing_dates, accession_numbers)
            ):
                try:
                    filing_date = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    # Skip old filings
                    if filing_date < cutoff_date:
                        continue
                    
                    # Skip if not in requested filing types
                    subscription = self._parse_subscription_from_data_type(data_type)
                    if form not in [ft.value for ft in subscription.filing_types]:
                        continue
                    
                    # Create filing data
                    filing_data = FilingData(
                        filing_type=FilingType(form),
                        cik=normalized_cik,
                        company_name=submissions_data.get("name", ""),
                        accession_number=accession,
                        filing_date=filing_date
                    )
                    
                    # Send to message bus with error handling
                    try:
                        self._handle_data(filing_data, UUID4())
                        logger.debug(f"Successfully sent filing data to message bus: {form} for {normalized_cik}")
                    except Exception as e:
                        logger.error(f"Failed to send filing data to message bus: {e}")
                        # Consider implementing dead letter queue or retry logic
                        continue
                    
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error processing filing {i}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error checking filings for CIK {cik}: {e}")
    
    async def _fetch_company_data(self, cik: str, correlation_id: UUID4) -> None:
        """Fetch comprehensive company data."""
        try:
            normalized_cik = normalize_cik(cik)
            
            # Fetch company facts
            if self.data_config.parse_financial_data:
                facts_data = await self.api_client.get_company_facts(normalized_cik)
                
                if facts_data:
                    # Parse financial data
                    parsed_facts = self.xbrl_parser.parse_company_facts(facts_data)
                    
                    company_facts = CompanyFacts(
                        cik=normalized_cik,
                        company_name=facts_data.get("entityName", ""),
                        facts=parsed_facts
                    )
                    
                    # Send to message bus with error handling
                    try:
                        self._handle_data(company_facts, correlation_id)
                        logger.debug(f"Successfully sent company facts to message bus for {normalized_cik}")
                    except Exception as e:
                        logger.error(f"Failed to send company facts to message bus: {e}")
                        # Consider implementing retry logic or fallback storage
            
            # Fetch recent submissions
            submissions_data = await self.api_client.get_submissions(normalized_cik)
            
            if submissions_data:
                # Process filings similar to _check_company_filings
                await self._process_submissions_data(submissions_data, correlation_id)
        
        except Exception as e:
            logger.error(f"Error fetching company data for CIK {cik}: {e}")
    
    async def _fetch_recent_filings(self, correlation_id: UUID4) -> None:
        """Fetch recent filings from all monitored companies."""
        # This is a placeholder - SEC doesn't provide a direct "recent filings" API
        # In practice, you'd maintain a list of companies to monitor and check their submissions
        logger.info("Fetching recent filings (placeholder implementation)")
        
        # For demonstration, we'll check a few major companies
        major_ciks = ["0000320193", "0001652044", "0000789019"]  # Apple, Google, Microsoft
        
        for cik in major_ciks:
            try:
                await self._fetch_company_data(cik, correlation_id)
            except Exception as e:
                logger.error(f"Error fetching data for CIK {cik}: {e}")
    
    async def _process_submissions_data(
        self,
        submissions_data: Dict,
        correlation_id: UUID4
    ) -> None:
        """Process submissions data and create filing objects."""
        try:
            filings = submissions_data.get("filings", {}).get("recent", {})
            
            if not filings:
                return
            
            forms = filings.get("form", [])
            filing_dates = filings.get("filingDate", [])
            accession_numbers = filings.get("accessionNumber", [])
            
            cutoff_date = datetime.now() - timedelta(days=self.data_config.max_filing_age_days)
            
            for form, date_str, accession in zip(forms, filing_dates, accession_numbers):
                try:
                    filing_date = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    if filing_date < cutoff_date:
                        continue
                    
                    filing_data = FilingData(
                        filing_type=FilingType(form),
                        cik=submissions_data.get("cik", ""),
                        company_name=submissions_data.get("name", ""),
                        accession_number=accession,
                        filing_date=filing_date
                    )
                    
                    try:
                        self._handle_data(filing_data, correlation_id)
                        logger.debug(f"Successfully sent filing data to message bus: {form}")
                    except Exception as e:
                        logger.error(f"Failed to send filing data to message bus: {e}")
                        continue
                
                except (ValueError, KeyError):
                    continue
        
        except Exception as e:
            logger.error(f"Error processing submissions data: {e}")
    
    def _start_filing_monitor(self) -> None:
        """Start background task to monitor for new filings."""
        if not self.data_config.auto_subscribe_filings:
            return
        
        task_id = "filing_monitor"
        task = asyncio.create_task(self._filing_monitor_worker())
        self._subscription_tasks[task_id] = task
        
        logger.info("Started filing monitor task")
    
    async def _filing_monitor_worker(self) -> None:
        """Background worker to monitor for new filings."""
        try:
            while True:
                logger.debug("Checking for new filings...")
                
                # This would implement logic to check for new filings
                # across all monitored companies
                
                self._last_filing_check = datetime.now()
                
                # Wait for next check
                await asyncio.sleep(self.data_config.subscription_check_interval)
        
        except asyncio.CancelledError:
            logger.info("Filing monitor worker cancelled")
        except Exception as e:
            logger.error(f"Error in filing monitor worker: {e}")
    
    # Public API methods
    
    async def request_company_facts(self, cik: str) -> Optional[CompanyFacts]:
        """Request company facts for specific CIK."""
        try:
            normalized_cik = normalize_cik(cik)
            facts_data = await self.api_client.get_company_facts(normalized_cik)
            
            if facts_data:
                parsed_facts = self.xbrl_parser.parse_company_facts(facts_data)
                
                return CompanyFacts(
                    cik=normalized_cik,
                    company_name=facts_data.get("entityName", ""),
                    facts=parsed_facts
                )
        
        except Exception as e:
            logger.error(f"Error requesting company facts for CIK {cik}: {e}")
        
        return None
    
    async def request_company_filings(
        self,
        cik: str,
        filing_types: Optional[List[FilingType]] = None
    ) -> List[FilingData]:
        """Request filings for specific company."""
        try:
            normalized_cik = normalize_cik(cik)
            submissions_data = await self.api_client.get_submissions(normalized_cik)
            
            if not submissions_data:
                return []
            
            filings = submissions_data.get("filings", {}).get("recent", {})
            results = []
            
            forms = filings.get("form", [])
            filing_dates = filings.get("filingDate", [])
            accession_numbers = filings.get("accessionNumber", [])
            
            for form, date_str, accession in zip(forms, filing_dates, accession_numbers):
                try:
                    # Filter by filing types if specified
                    if filing_types and form not in [ft.value for ft in filing_types]:
                        continue
                    
                    filing_date = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    filing_data = FilingData(
                        filing_type=FilingType(form),
                        cik=normalized_cik,
                        company_name=submissions_data.get("name", ""),
                        accession_number=accession,
                        filing_date=filing_date
                    )
                    
                    results.append(filing_data)
                
                except (ValueError, KeyError):
                    continue
            
            return results
        
        except Exception as e:
            logger.error(f"Error requesting filings for CIK {cik}: {e}")
            return []
    
    def get_subscription_count(self) -> int:
        """Get number of active subscriptions."""
        return len(self._subscriptions)
    
    def get_subscription_status(self) -> Dict[str, int]:
        """Get subscription status."""
        return {
            "active_subscriptions": len(self._subscriptions),
            "running_tasks": len(self._subscription_tasks),
            "is_connected": self._is_connected
        }