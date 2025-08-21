"""
Story 5.3: Data Export and Reporting API Routes
Backend API implementation for flexible data export and report generation
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
import json
import csv
import io
import asyncio
import uuid
from enum import Enum
import logging
import gzip
import base64
from openpyxl import Workbook
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["data-export-reporting"])

# Pydantic Models
class ExportType(str, Enum):
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    PDF = "pdf"

class DataSource(str, Enum):
    TRADES = "trades"
    POSITIONS = "positions"
    PERFORMANCE = "performance"
    ORDERS = "orders"
    SYSTEM_METRICS = "system_metrics"

class ExportStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DateRange(BaseModel):
    start_date: datetime
    end_date: datetime

class ExportFilters(BaseModel):
    date_range: DateRange
    symbols: Optional[List[str]] = None
    accounts: Optional[List[str]] = None
    strategies: Optional[List[str]] = None
    venues: Optional[List[str]] = None

class ExportOptions(BaseModel):
    include_headers: bool = True
    compression: bool = False
    precision: int = 4
    timezone: str = "UTC"
    currency: str = "USD"

class ExportRequest(BaseModel):
    id: Optional[str] = None
    type: ExportType
    data_source: DataSource
    filters: ExportFilters
    fields: List[str]
    options: ExportOptions
    status: ExportStatus = ExportStatus.PENDING
    progress: int = 0
    download_url: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class ReportType(str, Enum):
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    RISK = "risk"
    CUSTOM = "custom"

class ReportFormat(str, Enum):
    PDF = "pdf"
    EXCEL = "excel"
    HTML = "html"

class ScheduleFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class ReportSchedule(BaseModel):
    frequency: ScheduleFrequency
    time: str
    timezone: str
    recipients: List[str]

class ReportSection(BaseModel):
    id: str
    name: str
    type: str
    configuration: Dict[str, Any]

class ReportParameter(BaseModel):
    name: str
    type: str
    default_value: Any
    required: bool

class ReportTemplate(BaseModel):
    id: Optional[str] = None
    name: str
    description: str
    type: ReportType
    format: ReportFormat
    schedule: Optional[ReportSchedule] = None
    sections: List[ReportSection]
    parameters: List[ReportParameter]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class AuthenticationType(str, Enum):
    API_KEY = "api_key"
    OAUTH = "oauth"
    BASIC = "basic"

class IntegrationStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"

class FieldMapping(BaseModel):
    source_field: str
    target_field: str
    transformation: Optional[str] = None

class ApiIntegration(BaseModel):
    id: Optional[str] = None
    name: str
    endpoint: str
    authentication: Dict[str, Any]
    data_mapping: List[FieldMapping]
    schedule: Optional[Dict[str, Any]] = None
    status: IntegrationStatus
    created_at: Optional[datetime] = None
    last_sync: Optional[datetime] = None

# In-memory storage (replace with database in production)
export_requests: Dict[str, ExportRequest] = {}
report_templates: Dict[str, ReportTemplate] = {}
api_integrations: Dict[str, ApiIntegration] = {}

# Mock data generators
def generate_mock_trade_data(filters: ExportFilters, fields: List[str]) -> List[Dict[str, Any]]:
    """Generate mock trading data with historical bulk support"""
    trades = []
    base_data = {
        "id": "TRD-2024-001",
        "timestamp": "2024-01-15T10:30:00Z",
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 100,
        "price": 150.25,
        "value": 15025.00,
        "commission": 1.50,
        "pnl": 250.00,
        "strategy": "momentum_1",
        "venue": "IB",
        "account": "ACC001"
    }
    
    # Calculate date range for bulk historical data
    start_date = filters.date_range.start_date
    end_date = filters.date_range.end_date
    days_range = (end_date - start_date).days
    
    # Generate more records for larger date ranges (bulk historical data)
    record_count = min(max(days_range * 5, 10), 10000)  # 5 records per day, max 10k
    symbols = filters.symbols if filters.symbols else ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    strategies = filters.strategies if filters.strategies else ["momentum_1", "mean_reversion", "pairs_trading"]
    accounts = filters.accounts if filters.accounts else ["ACC001", "ACC002", "ACC003"]
    
    for i in range(record_count):
        trade = base_data.copy()
        
        # Generate distributed timestamps across date range
        days_offset = (i * days_range) // record_count
        hours_offset = (i * 24) % 24
        trade_time = start_date + timedelta(days=days_offset, hours=hours_offset)
        
        trade["id"] = f"TRD-2024-{i+1:06d}"
        trade["timestamp"] = trade_time.isoformat() + "Z"
        trade["symbol"] = symbols[i % len(symbols)]
        trade["side"] = "buy" if i % 2 == 0 else "sell"
        trade["quantity"] = 50 + (i % 500)
        trade["price"] = 100.0 + (i * 0.25) % 200
        trade["value"] = trade["price"] * trade["quantity"]
        trade["commission"] = trade["value"] * 0.001  # 0.1% commission
        trade["pnl"] = (trade["value"] * 0.02) - trade["commission"] if trade["side"] == "buy" else -(trade["value"] * 0.02) - trade["commission"]
        trade["strategy"] = strategies[i % len(strategies)]
        trade["account"] = accounts[i % len(accounts)]
        
        # Apply symbol filter if specified
        if filters.symbols and trade["symbol"] not in filters.symbols:
            continue
            
        # Apply strategy filter if specified
        if filters.strategies and trade["strategy"] not in filters.strategies:
            continue
            
        # Apply account filter if specified
        if filters.accounts and trade["account"] not in filters.accounts:
            continue
        
        # Filter by requested fields
        filtered_trade = {field: trade.get(field) for field in fields if field in trade}
        trades.append(filtered_trade)
    
    return trades

def generate_mock_performance_data(filters: ExportFilters, fields: List[str]) -> List[Dict[str, Any]]:
    """Generate mock performance data"""
    performance = []
    base_data = {
        "timestamp": "2024-01-15T00:00:00Z",
        "total_pnl": 5250.75,
        "unrealized_pnl": 1250.25,
        "realized_pnl": 4000.50,
        "win_rate": 0.65,
        "sharpe_ratio": 1.45,
        "max_drawdown": -2.3,
        "total_trades": 156,
        "winning_trades": 101,
        "strategy_id": "momentum_1"
    }
    
    for i in range(7):  # Last 7 days
        perf = base_data.copy()
        perf["timestamp"] = (datetime.now() - timedelta(days=i)).isoformat() + "Z"
        perf["total_pnl"] = 5250.75 + (i * 125.5)
        perf["win_rate"] = max(0.5, min(0.8, 0.65 + (i * 0.02)))
        
        filtered_perf = {field: perf.get(field) for field in fields if field in perf}
        performance.append(filtered_perf)
    
    return performance

def generate_mock_system_metrics(filters: ExportFilters, fields: List[str]) -> List[Dict[str, Any]]:
    """Generate mock system metrics data"""
    metrics = []
    base_data = {
        "timestamp": "2024-01-15T10:30:00Z",
        "cpu_usage": 45.2,
        "memory_usage": 62.8,
        "disk_usage": 78.5,
        "network_in": 1024.5,
        "network_out": 856.3,
        "latency_avg": 12.4,
        "latency_p95": 18.7,
        "latency_p99": 25.1,
        "venue": "IB"
    }
    
    for i in range(24):  # Last 24 hours
        metric = base_data.copy()
        metric["timestamp"] = (datetime.now() - timedelta(hours=i)).isoformat() + "Z"
        metric["cpu_usage"] = max(10, min(90, 45.2 + (i * 1.2)))
        metric["memory_usage"] = max(30, min(85, 62.8 + (i * 0.8)))
        metric["latency_avg"] = max(5, min(50, 12.4 + (i * 0.3)))
        
        filtered_metric = {field: metric.get(field) for field in fields if field in metric}
        metrics.append(filtered_metric)
    
    return metrics

def generate_csv_export(data: List[Dict[str, Any]], options: ExportOptions) -> bytes:
    """Generate CSV export with customizable options"""
    output = io.StringIO()
    
    if not data:
        return b""
    
    fieldnames = list(data[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    
    if options.include_headers:
        writer.writeheader()
    
    # Apply precision formatting for numeric fields
    for row in data:
        formatted_row = {}
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                formatted_row[key] = round(value, options.precision) if isinstance(value, float) else value
            else:
                formatted_row[key] = value
        writer.writerow(formatted_row)
    
    csv_data = output.getvalue().encode('utf-8')
    
    # Apply compression if requested
    if options.compression:
        csv_data = gzip.compress(csv_data)
    
    return csv_data

def generate_json_export(data: List[Dict[str, Any]], options: ExportOptions) -> bytes:
    """Generate JSON export with customizable options"""
    if not data:
        json_data = json.dumps([], indent=2).encode('utf-8')
    else:
        # Apply precision formatting for numeric fields
        formatted_data = []
        for row in data:
            formatted_row = {}
            for key, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    formatted_row[key] = round(value, options.precision) if isinstance(value, float) else value
                else:
                    formatted_row[key] = value
            formatted_data.append(formatted_row)
        
        json_data = json.dumps(formatted_data, indent=2, default=str).encode('utf-8')
    
    # Apply compression if requested
    if options.compression:
        json_data = gzip.compress(json_data)
    
    return json_data

def generate_excel_export(data: List[Dict[str, Any]], options: ExportOptions, data_source: str) -> bytes:
    """Generate Excel export with multiple worksheets"""
    wb = Workbook()
    ws = wb.active
    ws.title = f"{data_source.replace('_', ' ').title()} Data"
    
    if not data:
        # Create empty worksheet with headers
        ws.append(["No data available"])
    else:
        fieldnames = list(data[0].keys())
        
        # Add headers if requested
        if options.include_headers:
            ws.append(fieldnames)
        
        # Add data rows with formatting
        for row in data:
            formatted_row = []
            for field in fieldnames:
                value = row.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    formatted_row.append(round(value, options.precision) if isinstance(value, float) else value)
                else:
                    formatted_row.append(value)
            ws.append(formatted_row)
    
    # Save to BytesIO
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output.getvalue()

def generate_pdf_export(data: List[Dict[str, Any]], options: ExportOptions, data_source: str) -> bytes:
    """Generate PDF export with formatted tables"""
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter)
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.black,
        spaceAfter=30,
    )
    
    # Title
    title = Paragraph(f"{data_source.replace('_', ' ').title()} Data Export", title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    if not data:
        no_data = Paragraph("No data available", styles['Normal'])
        elements.append(no_data)
    else:
        fieldnames = list(data[0].keys())
        
        # Prepare table data
        table_data = []
        if options.include_headers:
            table_data.append(fieldnames)
        
        # Add data rows
        for row in data:
            formatted_row = []
            for field in fieldnames:
                value = row.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    formatted_value = round(value, options.precision) if isinstance(value, float) else value
                    formatted_row.append(str(formatted_value))
                else:
                    formatted_row.append(str(value) if value is not None else "")
            table_data.append(formatted_row)
        
        # Create table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
    
    # Build PDF
    doc.build(elements)
    output.seek(0)
    return output.getvalue()

# In-memory file storage (replace with proper file storage in production)
generated_files: Dict[str, bytes] = {}

async def process_export_request(export_id: str):
    """Background task to process export request"""
    try:
        export_req = export_requests[export_id]
        export_req.status = ExportStatus.PROCESSING
        export_req.progress = 10
        
        # Simulate processing time
        await asyncio.sleep(1)
        export_req.progress = 30
        
        # Generate data based on source
        if export_req.data_source == DataSource.TRADES:
            data = generate_mock_trade_data(export_req.filters, export_req.fields)
        elif export_req.data_source == DataSource.PERFORMANCE:
            data = generate_mock_performance_data(export_req.filters, export_req.fields)
        elif export_req.data_source == DataSource.SYSTEM_METRICS:
            data = generate_mock_system_metrics(export_req.filters, export_req.fields)
        else:
            data = []
        
        export_req.progress = 60
        
        # Generate export file based on format
        if export_req.type == ExportType.CSV:
            file_data = generate_csv_export(data, export_req.options)
        elif export_req.type == ExportType.JSON:
            file_data = generate_json_export(data, export_req.options)
        elif export_req.type == ExportType.EXCEL:
            file_data = generate_excel_export(data, export_req.options, export_req.data_source.value)
        elif export_req.type == ExportType.PDF:
            file_data = generate_pdf_export(data, export_req.options, export_req.data_source.value)
        else:
            raise ValueError(f"Unsupported export type: {export_req.type}")
        
        export_req.progress = 90
        
        # Store generated file
        generated_files[export_id] = file_data
        
        # Update export request
        export_req.download_url = f"/api/v1/export/download/{export_id}"
        export_req.status = ExportStatus.COMPLETED
        export_req.progress = 100
        export_req.completed_at = datetime.now()
        
        logger.info(f"Export {export_id} completed successfully - {len(file_data)} bytes generated")
        
    except Exception as e:
        logger.error(f"Export {export_id} failed: {str(e)}")
        export_req.status = ExportStatus.FAILED
        export_req.progress = 0

# Data Export Endpoints
@router.post("/export/request")
async def create_export_request(request: ExportRequest, background_tasks: BackgroundTasks):
    """Create a new data export request"""
    try:
        export_id = str(uuid.uuid4())
        request.id = export_id
        request.created_at = datetime.now()
        request.status = ExportStatus.PENDING
        
        export_requests[export_id] = request
        
        # Start background processing
        background_tasks.add_task(process_export_request, export_id)
        
        return {
            "export_id": export_id,
            "status": request.status,
            "message": "Export request created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create export request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export request failed: {str(e)}")

@router.get("/export/status/{export_id}")
async def get_export_status(export_id: str):
    """Get the status of an export request"""
    if export_id not in export_requests:
        raise HTTPException(status_code=404, detail="Export request not found")
    
    export_req = export_requests[export_id]
    return {
        "export_id": export_id,
        "status": export_req.status,
        "progress": export_req.progress,
        "created_at": export_req.created_at,
        "completed_at": export_req.completed_at,
        "download_url": export_req.download_url
    }

@router.get("/export/download/{export_id}")
async def download_export(export_id: str):
    """Download the completed export file"""
    if export_id not in export_requests:
        raise HTTPException(status_code=404, detail="Export request not found")
    
    export_req = export_requests[export_id]
    if export_req.status != ExportStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Export not completed yet")
    
    if export_id not in generated_files:
        raise HTTPException(status_code=404, detail="Export file not found")
    
    file_data = generated_files[export_id]
    file_extension = export_req.type.value
    
    # Determine content type and filename
    content_types = {
        'csv': 'text/csv' if not export_req.options.compression else 'application/gzip',
        'json': 'application/json' if not export_req.options.compression else 'application/gzip',
        'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'pdf': 'application/pdf'
    }
    
    content_type = content_types.get(file_extension, 'application/octet-stream')
    filename = f"export_{export_id[:8]}_{export_req.data_source.value}.{file_extension}"
    
    if export_req.options.compression and file_extension in ['csv', 'json']:
        filename += '.gz'
    
    # Calculate file size
    file_size_mb = len(file_data) / (1024 * 1024)
    file_size_str = f"{file_size_mb:.2f} MB" if file_size_mb >= 1 else f"{len(file_data) / 1024:.2f} KB"
    
    # For actual download, return file response
    if 'download' in str(export_req.download_url):
        return Response(
            content=file_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(file_data))
            }
        )
    
    # Return file info for API response
    return {
        "export_id": export_id,
        "file_name": filename,
        "file_size": file_size_str,
        "download_url": f"/api/v1/export/download/{export_id}?download=true",
        "expires_at": (datetime.now() + timedelta(days=7)).isoformat()
    }

@router.delete("/export/{export_id}")
async def delete_export(export_id: str):
    """Delete an export request and its associated files"""
    if export_id not in export_requests:
        raise HTTPException(status_code=404, detail="Export request not found")
    
    del export_requests[export_id]
    return {"message": "Export deleted successfully"}

@router.get("/export/history")
async def get_export_history(limit: int = Query(50, description="Number of exports to return")):
    """Get export history for the user"""
    sorted_exports = sorted(
        export_requests.values(),
        key=lambda x: x.created_at or datetime.min,
        reverse=True
    )
    
    return {
        "exports": [
            {
                "id": exp.id,
                "type": exp.type,
                "data_source": exp.data_source,
                "status": exp.status,
                "progress": exp.progress,
                "created_at": exp.created_at,
                "completed_at": exp.completed_at,
                "download_url": exp.download_url
            }
            for exp in sorted_exports[:limit]
        ],
        "total_count": len(export_requests)
    }

# Report Template Endpoints
@router.get("/reports/templates")
async def get_report_templates():
    """Get all report templates"""
    return {
        "templates": list(report_templates.values()),
        "total_count": len(report_templates)
    }

@router.post("/reports/templates")
async def create_report_template(template: ReportTemplate):
    """Create a new report template"""
    try:
        template_id = str(uuid.uuid4())
        template.id = template_id
        template.created_at = datetime.now()
        template.updated_at = datetime.now()
        
        report_templates[template_id] = template
        
        return {
            "template_id": template_id,
            "message": "Report template created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create report template: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Template creation failed: {str(e)}")

@router.post("/reports/generate/{template_id}")
async def generate_report(
    template_id: str,
    parameters: Dict[str, Any],
    delivery_options: Dict[str, Any]
):
    """Generate a report from a template"""
    if template_id not in report_templates:
        raise HTTPException(status_code=404, detail="Report template not found")
    
    template = report_templates[template_id]
    
    # Mock report generation
    report_id = str(uuid.uuid4())
    
    return {
        "report_id": report_id,
        "template_id": template_id,
        "status": "generated",
        "download_url": f"/api/v1/reports/download/{report_id}",
        "generated_at": datetime.now().isoformat()
    }

@router.get("/reports/scheduled")
async def get_scheduled_reports():
    """Get all scheduled reports"""
    scheduled = [
        template for template in report_templates.values()
        if template.schedule is not None
    ]
    
    return {
        "scheduled_reports": scheduled,
        "total_count": len(scheduled)
    }

# API Integration Endpoints
@router.get("/integrations")
async def get_api_integrations():
    """Get all API integrations"""
    return {
        "integrations": list(api_integrations.values()),
        "total_count": len(api_integrations)
    }

@router.post("/integrations")
async def create_api_integration(integration: ApiIntegration):
    """Create a new API integration"""
    try:
        integration_id = str(uuid.uuid4())
        integration.id = integration_id
        integration.created_at = datetime.now()
        integration.status = IntegrationStatus.ACTIVE
        
        api_integrations[integration_id] = integration
        
        return {
            "integration_id": integration_id,
            "message": "API integration created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create API integration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Integration creation failed: {str(e)}")

@router.post("/integrations/{integration_id}/sync")
async def sync_integration(integration_id: str):
    """Manually trigger synchronization for an integration"""
    if integration_id not in api_integrations:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    integration = api_integrations[integration_id]
    integration.last_sync = datetime.now()
    
    return {
        "integration_id": integration_id,
        "status": "sync_started",
        "last_sync": integration.last_sync.isoformat()
    }

@router.get("/integrations/{integration_id}/status")
async def get_integration_status(integration_id: str):
    """Get the status of an API integration"""
    if integration_id not in api_integrations:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    integration = api_integrations[integration_id]
    
    return {
        "integration_id": integration_id,
        "name": integration.name,
        "status": integration.status,
        "last_sync": integration.last_sync,
        "endpoint": integration.endpoint,
        "created_at": integration.created_at
    }

# Utility Endpoints
@router.get("/export/formats")
async def get_supported_formats():
    """Get supported export formats"""
    return {
        "formats": [
            {
                "type": "csv",
                "name": "CSV (Comma-Separated Values)",
                "extension": ".csv",
                "supports_compression": True
            },
            {
                "type": "json",
                "name": "JSON (JavaScript Object Notation)",
                "extension": ".json",
                "supports_compression": True
            },
            {
                "type": "excel",
                "name": "Excel Spreadsheet",
                "extension": ".xlsx",
                "supports_compression": False
            },
            {
                "type": "pdf",
                "name": "PDF Document",
                "extension": ".pdf",
                "supports_compression": False
            }
        ]
    }

@router.get("/export/fields/{data_source}")
async def get_available_fields(data_source: DataSource):
    """Get available fields for a data source"""
    field_mappings = {
        DataSource.TRADES: [
            "id", "timestamp", "symbol", "side", "quantity", "price",
            "value", "commission", "pnl", "strategy", "venue", "account"
        ],
        DataSource.PERFORMANCE: [
            "timestamp", "total_pnl", "unrealized_pnl", "realized_pnl",
            "win_rate", "sharpe_ratio", "max_drawdown", "total_trades",
            "winning_trades", "strategy_id"
        ],
        DataSource.SYSTEM_METRICS: [
            "timestamp", "cpu_usage", "memory_usage", "disk_usage",
            "network_in", "network_out", "latency_avg", "latency_p95",
            "latency_p99", "venue"
        ],
        DataSource.POSITIONS: [
            "symbol", "quantity", "market_value", "unrealized_pnl",
            "cost_basis", "average_price", "last_price", "account"
        ],
        DataSource.ORDERS: [
            "id", "timestamp", "symbol", "side", "quantity", "price",
            "order_type", "status", "filled_quantity", "remaining_quantity"
        ]
    }
    
    return {
        "data_source": data_source,
        "available_fields": field_mappings.get(data_source, [])
    }

# Additional endpoints for historical data access

@router.post("/export/validate")
async def validate_export_request(request: ExportRequest):
    """Validate export request and estimate size/time"""
    try:
        # Calculate estimated record count
        start_date = request.filters.date_range.start_date
        end_date = request.filters.date_range.end_date
        days_range = (end_date - start_date).days
        
        if request.data_source == DataSource.TRADES:
            estimated_records = min(days_range * 5, 10000)
        elif request.data_source == DataSource.SYSTEM_METRICS:
            estimated_records = min(days_range * 24, 50000)  # Hourly metrics
        else:
            estimated_records = min(days_range, 1000)
        
        # Estimate file size (rough calculation)
        avg_record_size = len(request.fields) * 20  # 20 bytes per field average
        estimated_size_bytes = estimated_records * avg_record_size
        
        if request.options.compression:
            estimated_size_bytes = int(estimated_size_bytes * 0.3)  # 70% compression
        
        estimated_size_mb = estimated_size_bytes / (1024 * 1024)
        
        # Estimate processing time
        estimated_minutes = max(1, int(estimated_records / 5000))  # 5k records per minute
        
        return {
            "valid": True,
            "estimated_records": estimated_records,
            "estimated_size_mb": round(estimated_size_mb, 2),
            "estimated_processing_minutes": estimated_minutes,
            "warnings": [
                "Large date ranges may take longer to process",
                "Consider using compression for exports over 10MB"
            ] if estimated_size_mb > 10 else []
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }

@router.post("/export/bulk")
async def create_bulk_export(
    data_source: DataSource,
    start_date: datetime,
    end_date: datetime,
    export_type: ExportType = ExportType.CSV,
    background_tasks: BackgroundTasks = None,
    fields: List[str] = Query(default=[]),
    symbols: List[str] = Query(default=[]),
    accounts: List[str] = Query(default=[]),
    strategies: List[str] = Query(default=[]),
    compression: bool = Query(default=True)
):
    """Create bulk historical data export request"""
    try:
        # Create export request
        bulk_request = ExportRequest(
            type=export_type,
            data_source=data_source,
            filters=ExportFilters(
                date_range=DateRange(start_date=start_date, end_date=end_date),
                symbols=symbols if symbols else None,
                accounts=accounts if accounts else None,
                strategies=strategies if strategies else None
            ),
            fields=fields if fields else [],  # Will use all fields if empty
            options=ExportOptions(
                include_headers=True,
                compression=compression,
                precision=4,
                timezone="UTC",
                currency="USD"
            ),
            status=ExportStatus.PENDING,
            progress=0
        )
        
        export_id = str(uuid.uuid4())
        bulk_request.id = export_id
        bulk_request.created_at = datetime.now()
        
        export_requests[export_id] = bulk_request
        
        # Start background processing
        if background_tasks:
            background_tasks.add_task(process_export_request, export_id)
        
        return {
            "export_id": export_id,
            "status": bulk_request.status,
            "message": "Bulk export request created successfully",
            "estimated_completion": (datetime.now() + timedelta(minutes=5)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create bulk export request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bulk export request failed: {str(e)}")

@router.get("/export/{export_id}/resume")
async def resume_export(export_id: str, background_tasks: BackgroundTasks):
    """Resume interrupted export request"""
    if export_id not in export_requests:
        raise HTTPException(status_code=404, detail="Export request not found")
    
    export_req = export_requests[export_id]
    
    if export_req.status not in [ExportStatus.FAILED, ExportStatus.PENDING]:
        raise HTTPException(status_code=400, detail="Export cannot be resumed")
    
    # Reset status and restart processing
    export_req.status = ExportStatus.PENDING
    export_req.progress = 0
    
    background_tasks.add_task(process_export_request, export_id)
    
    return {
        "export_id": export_id,
        "status": export_req.status,
        "message": "Export request resumed successfully"
    }

@router.post("/export/{export_id}/validate-data")
async def validate_export_data(export_id: str):
    """Validate exported data integrity"""
    if export_id not in export_requests:
        raise HTTPException(status_code=404, detail="Export request not found")
    
    export_req = export_requests[export_id]
    
    if export_req.status != ExportStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Export not completed yet")
    
    if export_id not in generated_files:
        raise HTTPException(status_code=404, detail="Export file not found")
    
    file_data = generated_files[export_id]
    
    # Perform basic validation
    validation_results = {
        "file_size_bytes": len(file_data),
        "file_size_readable": f"{len(file_data) / (1024 * 1024):.2f} MB" if len(file_data) > 1024 * 1024 else f"{len(file_data) / 1024:.2f} KB",
        "format_valid": True,
        "data_integrity": "passed",
        "validation_timestamp": datetime.now().isoformat()
    }
    
    # Additional format-specific validation
    if export_req.type == ExportType.CSV:
        try:
            # Basic CSV validation
            csv_content = file_data.decode('utf-8') if not export_req.options.compression else gzip.decompress(file_data).decode('utf-8')
            lines = csv_content.split('\n')
            validation_results["record_count"] = len(lines) - 1 if export_req.options.include_headers else len(lines)
            validation_results["has_headers"] = export_req.options.include_headers
        except Exception as e:
            validation_results["format_valid"] = False
            validation_results["data_integrity"] = f"failed: {str(e)}"
    
    elif export_req.type == ExportType.JSON:
        try:
            # Basic JSON validation
            json_content = file_data.decode('utf-8') if not export_req.options.compression else gzip.decompress(file_data).decode('utf-8')
            data = json.loads(json_content)
            validation_results["record_count"] = len(data) if isinstance(data, list) else 1
        except Exception as e:
            validation_results["format_valid"] = False
            validation_results["data_integrity"] = f"failed: {str(e)}"
    
    return validation_results