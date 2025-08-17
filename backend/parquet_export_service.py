"""
Parquet Export Service for NautilusTrader Compatibility

This service bridges our live trading PostgreSQL system with NautilusTrader's
research-focused Parquet file system, enabling the best of both approaches:

🎯 Our Implementation: Optimized for live trading and web applications
🎯 NautilusTrader: Optimized for research and backtesting with file-based performance

Key Features:
- Export live trading data to Parquet format
- Maintain nanosecond precision and data quality standards
- Support all NautilusTrader data types (ticks, quotes, bars)
- Efficient batch export with compression
- Automatic file organization by venue/instrument/date
"""

import asyncio
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass, asdict
import asyncpg

from historical_data_service import HistoricalDataService, HistoricalDataQuery


@dataclass
class ParquetExportConfig:
    """Configuration for Parquet export operations"""
    output_directory: Path
    compression: str = 'snappy'  # 'snappy', 'gzip', 'brotli', 'lz4'
    batch_size: int = 100000
    partition_by: List[str] = None  # ['venue', 'date']
    include_raw_data: bool = False
    nautilus_format: bool = True  # Use NautilusTrader-compatible schema


class ParquetExportService:
    """
    Service for exporting live trading data to Parquet format for NautilusTrader compatibility.
    
    This enables our live trading system to seamlessly integrate with NautilusTrader's
    research and backtesting capabilities while maintaining our real-time performance.
    """
    
    def __init__(
        self,
        historical_service: HistoricalDataService,
        export_config: ParquetExportConfig,
    ):
        self.logger = logging.getLogger(__name__)
        self.historical_service = historical_service
        self.config = export_config
        self._ensure_output_directory()
        
    def _ensure_output_directory(self) -> None:
        """Ensure output directory exists"""
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        for subdir in ['ticks', 'quotes', 'bars', 'instruments']:
            (self.config.output_directory / subdir).mkdir(exist_ok=True)
            
    def _get_output_path(
        self, 
        data_type: str, 
        venue: str, 
        instrument_id: str, 
        date: datetime,
        timeframe: Optional[str] = None
    ) -> Path:
        """Generate output file path with proper organization"""
        date_str = date.strftime('%Y-%m-%d')
        
        if data_type == 'bar' and timeframe:
            filename = f"{venue}_{instrument_id}_{timeframe}_{date_str}.parquet"
        else:
            filename = f"{venue}_{instrument_id}_{date_str}.parquet"
            
        return self.config.output_directory / data_type / venue / filename
        
    async def export_ticks(
        self,
        venue: str,
        instrument_id: str,
        start_time: datetime,
        end_time: datetime,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Export tick data to Parquet format"""
        
        if not output_path:
            output_path = self._get_output_path('ticks', venue, instrument_id, start_time)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            query = HistoricalDataQuery(
                venue=venue,
                instrument_id=instrument_id,
                data_type='tick',
                start_time=start_time,
                end_time=end_time,
                limit=None
            )
            
            tick_data = await self.historical_service.query_ticks(query)
            
            if not tick_data:
                self.logger.warning(f"No tick data found for {venue}:{instrument_id}")
                return {"status": "no_data", "records": 0}
                
            # Convert to NautilusTrader compatible format
            if self.config.nautilus_format:
                df = self._convert_ticks_to_nautilus_format(tick_data, venue, instrument_id)
                schema = self._get_nautilus_tick_schema()
            else:
                df = pd.DataFrame(tick_data)
                schema = None
                
            # Write to Parquet
            table = pa.Table.from_pandas(df, schema=schema)
            pq.write_table(
                table, 
                output_path,
                compression=self.config.compression,
                use_dictionary=True,
                row_group_size=self.config.batch_size
            )
            
            file_size = output_path.stat().st_size
            
            self.logger.info(
                f"Exported {len(tick_data)} ticks to {output_path} "
                f"({file_size / 1024 / 1024:.2f} MB)"
            )
            
            return {
                "status": "success",
                "records": len(tick_data),
                "file_path": str(output_path),
                "file_size_mb": file_size / 1024 / 1024,
                "compression": self.config.compression
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export tick data: {e}")
            return {"status": "error", "error": str(e)}
            
    async def export_quotes(
        self,
        venue: str,
        instrument_id: str,
        start_time: datetime,
        end_time: datetime,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Export quote data to Parquet format"""
        
        if not output_path:
            output_path = self._get_output_path('quotes', venue, instrument_id, start_time)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            query = HistoricalDataQuery(
                venue=venue,
                instrument_id=instrument_id,
                data_type='quote',
                start_time=start_time,
                end_time=end_time,
                limit=None
            )
            
            quote_data = await self.historical_service.query_quotes(query)
            
            if not quote_data:
                self.logger.warning(f"No quote data found for {venue}:{instrument_id}")
                return {"status": "no_data", "records": 0}
                
            # Convert to NautilusTrader compatible format
            if self.config.nautilus_format:
                df = self._convert_quotes_to_nautilus_format(quote_data, venue, instrument_id)
                schema = self._get_nautilus_quote_schema()
            else:
                df = pd.DataFrame(quote_data)
                schema = None
                
            # Write to Parquet
            table = pa.Table.from_pandas(df, schema=schema)
            pq.write_table(
                table, 
                output_path,
                compression=self.config.compression,
                use_dictionary=True,
                row_group_size=self.config.batch_size
            )
            
            file_size = output_path.stat().st_size
            
            self.logger.info(
                f"Exported {len(quote_data)} quotes to {output_path} "
                f"({file_size / 1024 / 1024:.2f} MB)"
            )
            
            return {
                "status": "success",
                "records": len(quote_data),
                "file_path": str(output_path),
                "file_size_mb": file_size / 1024 / 1024,
                "compression": self.config.compression
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export quote data: {e}")
            return {"status": "error", "error": str(e)}
            
    async def export_bars(
        self,
        venue: str,
        instrument_id: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Export bar data to Parquet format"""
        
        if not output_path:
            output_path = self._get_output_path('bars', venue, instrument_id, start_time, timeframe)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            query = HistoricalDataQuery(
                venue=venue,
                instrument_id=instrument_id,
                data_type='bar',
                start_time=start_time,
                end_time=end_time,
                timeframe=timeframe,
                limit=None
            )
            
            bar_data = await self.historical_service.query_bars(query)
            
            if not bar_data:
                self.logger.warning(f"No bar data found for {venue}:{instrument_id}:{timeframe}")
                return {"status": "no_data", "records": 0}
                
            # Convert to NautilusTrader compatible format
            if self.config.nautilus_format:
                df = self._convert_bars_to_nautilus_format(bar_data, venue, instrument_id, timeframe)
                schema = self._get_nautilus_bar_schema()
            else:
                df = pd.DataFrame(bar_data)
                schema = None
                
            # Write to Parquet
            table = pa.Table.from_pandas(df, schema=schema)
            pq.write_table(
                table, 
                output_path,
                compression=self.config.compression,
                use_dictionary=True,
                row_group_size=self.config.batch_size
            )
            
            file_size = output_path.stat().st_size
            
            self.logger.info(
                f"Exported {len(bar_data)} bars ({timeframe}) to {output_path} "
                f"({file_size / 1024 / 1024:.2f} MB)"
            )
            
            return {
                "status": "success",
                "records": len(bar_data),
                "file_path": str(output_path),
                "file_size_mb": file_size / 1024 / 1024,
                "compression": self.config.compression,
                "timeframe": timeframe
            }
            
        except Exception as e:
            self.logger.error(f"Failed to export bar data: {e}")
            return {"status": "error", "error": str(e)}
            
    def _convert_ticks_to_nautilus_format(
        self, 
        tick_data: List[Dict], 
        venue: str, 
        instrument_id: str
    ) -> pd.DataFrame:
        """Convert tick data to NautilusTrader format"""
        
        records = []
        for tick in tick_data:
            record = {
                'venue': venue,
                'instrument_id': instrument_id,
                'ts_event': tick['timestamp_ns'],
                'ts_init': tick['timestamp_ns'],
                'price': float(tick['price']),
                'size': float(tick['size']),
                'aggressor_side': self._convert_side_to_nautilus(tick.get('side')),
                'trade_id': tick.get('trade_id', ''),
                'sequence_num': tick.get('sequence_num', 0)
            }
            records.append(record)
            
        return pd.DataFrame(records)
        
    def _convert_quotes_to_nautilus_format(
        self, 
        quote_data: List[Dict], 
        venue: str, 
        instrument_id: str
    ) -> pd.DataFrame:
        """Convert quote data to NautilusTrader format"""
        
        records = []
        for quote in quote_data:
            record = {
                'venue': venue,
                'instrument_id': instrument_id,
                'ts_event': quote['timestamp_ns'],
                'ts_init': quote['timestamp_ns'],
                'bid_price': float(quote['bid_price']),
                'ask_price': float(quote['ask_price']),
                'bid_size': float(quote['bid_size']),
                'ask_size': float(quote['ask_size']),
                'sequence_num': quote.get('sequence_num', 0)
            }
            records.append(record)
            
        return pd.DataFrame(records)
        
    def _convert_bars_to_nautilus_format(
        self, 
        bar_data: List[Dict], 
        venue: str, 
        instrument_id: str,
        timeframe: str
    ) -> pd.DataFrame:
        """Convert bar data to NautilusTrader format"""
        
        records = []
        for bar in bar_data:
            record = {
                'venue': venue,
                'instrument_id': instrument_id,
                'bar_type': timeframe,
                'ts_event': bar['timestamp_ns'],
                'ts_init': bar['timestamp_ns'],
                'open': float(bar['open_price']),
                'high': float(bar['high_price']),
                'low': float(bar['low_price']),
                'close': float(bar['close_price']),
                'volume': float(bar['volume'])
            }
            records.append(record)
            
        return pd.DataFrame(records)
        
    def _convert_side_to_nautilus(self, side: Optional[str]) -> int:
        """Convert trade side to NautilusTrader format"""
        if side == 'BUY':
            return 1
        elif side == 'SELL':
            return 2
        else:
            return 0  # No aggressor
            
    def _get_nautilus_tick_schema(self) -> pa.Schema:
        """Get PyArrow schema for NautilusTrader tick format"""
        return pa.schema([
            ('venue', pa.string()),
            ('instrument_id', pa.string()),
            ('ts_event', pa.int64()),
            ('ts_init', pa.int64()),
            ('price', pa.float64()),
            ('size', pa.float64()),
            ('aggressor_side', pa.int8()),
            ('trade_id', pa.string()),
            ('sequence_num', pa.int64())
        ])
        
    def _get_nautilus_quote_schema(self) -> pa.Schema:
        """Get PyArrow schema for NautilusTrader quote format"""
        return pa.schema([
            ('venue', pa.string()),
            ('instrument_id', pa.string()),
            ('ts_event', pa.int64()),
            ('ts_init', pa.int64()),
            ('bid_price', pa.float64()),
            ('ask_price', pa.float64()),
            ('bid_size', pa.float64()),
            ('ask_size', pa.float64()),
            ('sequence_num', pa.int64())
        ])
        
    def _get_nautilus_bar_schema(self) -> pa.Schema:
        """Get PyArrow schema for NautilusTrader bar format"""
        return pa.schema([
            ('venue', pa.string()),
            ('instrument_id', pa.string()),
            ('bar_type', pa.string()),
            ('ts_event', pa.int64()),
            ('ts_init', pa.int64()),
            ('open', pa.float64()),
            ('high', pa.float64()),
            ('low', pa.float64()),
            ('close', pa.float64()),
            ('volume', pa.float64())
        ])
        
    async def export_daily_batch(
        self, 
        date: datetime,
        venues: Optional[List[str]] = None,
        instrument_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Export a full day's data for specified venues/instruments"""
        
        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)
        
        results = {
            "date": date.strftime('%Y-%m-%d'),
            "exports": [],
            "summary": {"total_files": 0, "total_records": 0, "total_size_mb": 0}
        }
        
        # Get available instruments from database if not specified
        if not venues or not instrument_ids:
            # Query database for active instruments in the date range
            # This would need implementation based on your needs
            pass
            
        # Export data for each combination
        for venue in venues or ['IB']:
            for instrument_id in instrument_ids or ['AAPL.NASDAQ', 'EUR.USD']:
                
                # Export ticks
                tick_result = await self.export_ticks(venue, instrument_id, start_time, end_time)
                if tick_result['status'] == 'success':
                    results['exports'].append({
                        'type': 'ticks',
                        'venue': venue,
                        'instrument_id': instrument_id,
                        **tick_result
                    })
                    results['summary']['total_records'] += tick_result['records']
                    results['summary']['total_size_mb'] += tick_result['file_size_mb']
                    
                # Export quotes
                quote_result = await self.export_quotes(venue, instrument_id, start_time, end_time)
                if quote_result['status'] == 'success':
                    results['exports'].append({
                        'type': 'quotes',
                        'venue': venue,
                        'instrument_id': instrument_id,
                        **quote_result
                    })
                    results['summary']['total_records'] += quote_result['records']
                    results['summary']['total_size_mb'] += quote_result['file_size_mb']
                    
                # Export bars for common timeframes
                for timeframe in ['1m', '5m', '15m', '1h', '4h', '1d']:
                    bar_result = await self.export_bars(
                        venue, instrument_id, timeframe, start_time, end_time
                    )
                    if bar_result['status'] == 'success':
                        results['exports'].append({
                            'type': 'bars',
                            'venue': venue,
                            'instrument_id': instrument_id,
                            'timeframe': timeframe,
                            **bar_result
                        })
                        results['summary']['total_records'] += bar_result['records']
                        results['summary']['total_size_mb'] += bar_result['file_size_mb']
                        
        results['summary']['total_files'] = len(results['exports'])
        
        self.logger.info(
            f"Daily batch export completed for {date.strftime('%Y-%m-%d')}: "
            f"{results['summary']['total_files']} files, "
            f"{results['summary']['total_records']} records, "
            f"{results['summary']['total_size_mb']:.2f} MB"
        )
        
        return results
        
    async def create_nautilus_catalog(
        self,
        catalog_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Create a catalog file for NautilusTrader data discovery"""
        
        if not catalog_path:
            catalog_path = self.config.output_directory / "catalog.json"
            
        catalog = {
            "created_at": datetime.utcnow().isoformat(),
            "format_version": "1.0",
            "description": "Live trading data exported for NautilusTrader compatibility",
            "compression": self.config.compression,
            "venues": {},
            "instruments": {},
            "data_types": ["ticks", "quotes", "bars"],
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
        }
        
        # Scan exported files to build catalog
        for data_type in ["ticks", "quotes", "bars"]:
            data_dir = self.config.output_directory / data_type
            if data_dir.exists():
                for venue_dir in data_dir.iterdir():
                    if venue_dir.is_dir():
                        venue = venue_dir.name
                        if venue not in catalog["venues"]:
                            catalog["venues"][venue] = []
                            
                        for parquet_file in venue_dir.glob("*.parquet"):
                            file_info = {
                                "filename": parquet_file.name,
                                "data_type": data_type,
                                "venue": venue,
                                "size_mb": parquet_file.stat().st_size / 1024 / 1024,
                                "modified": datetime.fromtimestamp(
                                    parquet_file.stat().st_mtime
                                ).isoformat()
                            }
                            
                            # Extract instrument and date from filename
                            parts = parquet_file.stem.split('_')
                            if len(parts) >= 3:
                                instrument = parts[1]
                                file_info["instrument_id"] = instrument
                                
                                if instrument not in catalog["instruments"]:
                                    catalog["instruments"][instrument] = {
                                        "venue": venue,
                                        "data_types": []
                                    }
                                    
                                if data_type not in catalog["instruments"][instrument]["data_types"]:
                                    catalog["instruments"][instrument]["data_types"].append(data_type)
                                    
                            catalog["venues"][venue].append(file_info)
                            
        # Write catalog file
        import json
        with open(catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2, default=str)
            
        self.logger.info(f"Created NautilusTrader catalog at {catalog_path}")
        
        return {
            "status": "success",
            "catalog_path": str(catalog_path),
            "venues": len(catalog["venues"]),
            "instruments": len(catalog["instruments"]),
            "total_files": sum(len(files) for files in catalog["venues"].values())
        }


# Global parquet export service instance
import os
from pathlib import Path

# Configuration
export_directory = Path(os.getenv("PARQUET_EXPORT_DIR", "/tmp/nautilus_exports"))
export_config = ParquetExportConfig(
    output_directory=export_directory,
    compression='snappy',
    batch_size=100000,
    nautilus_format=True
)

# Initialize service (requires historical_data_service to be connected)
from historical_data_service import historical_data_service
parquet_export_service = ParquetExportService(
    historical_service=historical_data_service,
    export_config=export_config
)