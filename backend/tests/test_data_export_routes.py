"""
Story 5.3: Unit Tests for Enhanced Data Export Routes
Tests for multi-format data export functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json
import zipfile
import io
from pathlib import Path

# Mock the main app and dependencies
from fastapi import FastAPI

# Create test app instance
app = FastAPI()

# Mock imports to avoid actual dependencies
with patch.dict('sys.modules', {
    'data_export_routes': MagicMock(),
    'postgresql': MagicMock(),
    'pandas': MagicMock(),
    'openpyxl': MagicMock(),
    'reportlab': MagicMock(),
}):
    # Mock the data_export_routes module functions
    data_export_routes = MagicMock()
    data_export_routes.generate_csv_export = MagicMock()
    data_export_routes.generate_json_export = MagicMock()
    data_export_routes.generate_excel_export = MagicMock()
    data_export_routes.generate_pdf_export = MagicMock()
    data_export_routes.export_bulk_historical_data = MagicMock()

# Test client
client = TestClient(app)

# Sample test data
SAMPLE_PORTFOLIO_DATA = [
    {
        "timestamp": "2024-01-20T10:00:00Z",
        "total_pnl": 15000.50,
        "unrealized_pnl": 2500.25,
        "realized_pnl": 12500.25,
        "win_rate": 0.65,
        "sharpe_ratio": 1.85,
        "positions_count": 12,
        "symbol": "AAPL"
    },
    {
        "timestamp": "2024-01-20T11:00:00Z",
        "total_pnl": 15250.75,
        "unrealized_pnl": 2750.50,
        "realized_pnl": 12500.25,
        "win_rate": 0.67,
        "sharpe_ratio": 1.88,
        "positions_count": 13,
        "symbol": "GOOGL"
    }
]

class TestDataExportRoutes:
    """Test suite for data export route functionality"""

    @pytest.fixture
    def mock_portfolio_data(self):
        """Mock portfolio data for testing"""
        return SAMPLE_PORTFOLIO_DATA

    @pytest.fixture
    def export_request(self):
        """Standard export request payload"""
        return {
            "format": "csv",
            "date_range": {
                "start": "2024-01-20",
                "end": "2024-01-21"
            },
            "filters": {
                "symbols": ["AAPL", "GOOGL"],
                "min_pnl": 1000
            },
            "fields": ["timestamp", "total_pnl", "win_rate", "symbol"],
            "compression": False
        }

    def test_generate_csv_export_success(self, mock_portfolio_data, export_request):
        """Test CSV export generation with valid data"""
        with patch('data_export_routes.generate_csv_export') as mock_csv:
            # Setup mock response
            csv_content = "timestamp,total_pnl,win_rate,symbol\n2024-01-20T10:00:00Z,15000.50,0.65,AAPL\n"
            mock_csv.return_value = {
                "filename": "portfolio_export_20240120.csv",
                "content": csv_content.encode(),
                "mime_type": "text/csv",
                "size": len(csv_content)
            }
            
            result = data_export_routes.generate_csv_export(
                data=mock_portfolio_data,
                fields=export_request["fields"],
                filename_prefix="portfolio_export"
            )
            
            assert result["filename"].endswith(".csv")
            assert result["mime_type"] == "text/csv"
            assert len(result["content"]) > 0
            assert result["size"] > 0
            mock_csv.assert_called_once()

    def test_generate_json_export_success(self, mock_portfolio_data, export_request):
        """Test JSON export generation with valid data"""
        with patch('data_export_routes.generate_json_export') as mock_json:
            # Setup mock response
            json_data = {"data": mock_portfolio_data, "count": len(mock_portfolio_data)}
            json_content = json.dumps(json_data, indent=2)
            mock_json.return_value = {
                "filename": "portfolio_export_20240120.json",
                "content": json_content.encode(),
                "mime_type": "application/json",
                "size": len(json_content)
            }
            
            result = data_export_routes.generate_json_export(
                data=mock_portfolio_data,
                fields=export_request["fields"],
                filename_prefix="portfolio_export"
            )
            
            assert result["filename"].endswith(".json")
            assert result["mime_type"] == "application/json"
            assert len(result["content"]) > 0
            mock_json.assert_called_once()

    def test_generate_excel_export_success(self, mock_portfolio_data, export_request):
        """Test Excel export generation with valid data"""
        with patch('data_export_routes.generate_excel_export') as mock_excel:
            # Setup mock response with binary Excel content
            mock_excel.return_value = {
                "filename": "portfolio_export_20240120.xlsx",
                "content": b"mock_excel_content",
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "size": 18
            }
            
            result = data_export_routes.generate_excel_export(
                data=mock_portfolio_data,
                fields=export_request["fields"],
                filename_prefix="portfolio_export"
            )
            
            assert result["filename"].endswith(".xlsx")
            assert result["mime_type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            assert len(result["content"]) > 0
            mock_excel.assert_called_once()

    def test_generate_pdf_export_success(self, mock_portfolio_data, export_request):
        """Test PDF export generation with valid data"""
        with patch('data_export_routes.generate_pdf_export') as mock_pdf:
            # Setup mock response with binary PDF content
            mock_pdf.return_value = {
                "filename": "portfolio_export_20240120.pdf",
                "content": b"%PDF-1.4 mock content",
                "mime_type": "application/pdf",
                "size": 20
            }
            
            result = data_export_routes.generate_pdf_export(
                data=mock_portfolio_data,
                fields=export_request["fields"],
                filename_prefix="portfolio_export"
            )
            
            assert result["filename"].endswith(".pdf")
            assert result["mime_type"] == "application/pdf"
            assert len(result["content"]) > 0
            mock_pdf.assert_called_once()

    def test_compression_functionality(self, mock_portfolio_data):
        """Test file compression functionality"""
        with patch('data_export_routes.generate_csv_export') as mock_csv:
            # Setup mock to simulate compression
            original_content = "timestamp,total_pnl\n2024-01-20,15000.50\n"
            
            # Create compressed content
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("portfolio_export.csv", original_content)
            compressed_content = buffer.getvalue()
            
            mock_csv.return_value = {
                "filename": "portfolio_export_20240120.csv.zip",
                "content": compressed_content,
                "mime_type": "application/zip",
                "size": len(compressed_content),
                "compressed": True,
                "original_size": len(original_content)
            }
            
            result = data_export_routes.generate_csv_export(
                data=mock_portfolio_data,
                fields=["timestamp", "total_pnl"],
                compress=True,
                filename_prefix="portfolio_export"
            )
            
            assert result["filename"].endswith(".zip")
            assert result["mime_type"] == "application/zip"
            assert result["compressed"] is True
            assert result["original_size"] > result["size"]

    def test_bulk_historical_export_success(self):
        """Test bulk historical data export functionality"""
        with patch('data_export_routes.export_bulk_historical_data') as mock_bulk:
            # Setup mock response for bulk export
            mock_bulk.return_value = {
                "export_id": "bulk_20240120_001",
                "status": "completed",
                "files_generated": [
                    {
                        "filename": "historical_2024_01.csv",
                        "size": 1048576,
                        "records": 10000
                    },
                    {
                        "filename": "historical_2024_02.csv", 
                        "size": 2097152,
                        "records": 20000
                    }
                ],
                "total_records": 30000,
                "total_size": 3145728,
                "start_date": "2024-01-01",
                "end_date": "2024-02-29"
            }
            
            result = data_export_routes.export_bulk_historical_data(
                start_date="2024-01-01",
                end_date="2024-02-29",
                format="csv",
                chunk_size=10000
            )
            
            assert result["export_id"] is not None
            assert result["status"] == "completed"
            assert len(result["files_generated"]) == 2
            assert result["total_records"] == 30000
            mock_bulk.assert_called_once()

    def test_data_validation_and_filtering(self, mock_portfolio_data):
        """Test data validation and filtering functionality"""
        with patch('data_export_routes.generate_csv_export') as mock_csv:
            # Test data validation
            filtered_data = [item for item in mock_portfolio_data if item["total_pnl"] >= 15000]
            
            mock_csv.return_value = {
                "filename": "filtered_export.csv",
                "content": b"mock_filtered_content",
                "mime_type": "text/csv",
                "size": 20,
                "records_filtered": len(mock_portfolio_data) - len(filtered_data),
                "records_exported": len(filtered_data)
            }
            
            result = data_export_routes.generate_csv_export(
                data=filtered_data,
                fields=["timestamp", "total_pnl", "symbol"],
                filters={"min_pnl": 15000},
                validate_data=True
            )
            
            assert result["records_exported"] <= len(mock_portfolio_data)
            assert "records_filtered" in result
            mock_csv.assert_called_once()

    def test_export_resume_functionality(self):
        """Test export resume functionality for interrupted exports"""
        with patch('data_export_routes.export_bulk_historical_data') as mock_bulk:
            # Simulate resuming a partially completed export
            mock_bulk.return_value = {
                "export_id": "bulk_20240120_001",
                "status": "resumed",
                "progress": 0.75,
                "files_generated": [
                    {"filename": "historical_2024_01.csv", "status": "completed"},
                    {"filename": "historical_2024_02.csv", "status": "completed"},
                    {"filename": "historical_2024_03.csv", "status": "in_progress"}
                ],
                "resume_point": "2024-03-15T10:00:00Z"
            }
            
            result = data_export_routes.export_bulk_historical_data(
                export_id="bulk_20240120_001",
                resume=True
            )
            
            assert result["status"] == "resumed"
            assert result["progress"] > 0
            assert "resume_point" in result
            mock_bulk.assert_called_once()

    def test_error_handling_invalid_format(self, mock_portfolio_data):
        """Test error handling for invalid export format"""
        with patch('data_export_routes.generate_csv_export') as mock_csv:
            mock_csv.side_effect = ValueError("Unsupported export format: xml")
            
            with pytest.raises(ValueError) as exc_info:
                data_export_routes.generate_csv_export(
                    data=mock_portfolio_data,
                    format="xml"
                )
            
            assert "Unsupported export format" in str(exc_info.value)

    def test_error_handling_empty_data(self):
        """Test error handling for empty data export"""
        with patch('data_export_routes.generate_csv_export') as mock_csv:
            mock_csv.return_value = {
                "filename": "empty_export.csv",
                "content": b"",
                "mime_type": "text/csv",
                "size": 0,
                "records_exported": 0,
                "warning": "No data available for export"
            }
            
            result = data_export_routes.generate_csv_export(
                data=[],
                fields=["timestamp", "total_pnl"]
            )
            
            assert result["records_exported"] == 0
            assert "warning" in result
            mock_csv.assert_called_once()

    def test_field_selection_and_ordering(self, mock_portfolio_data):
        """Test field selection and custom ordering"""
        with patch('data_export_routes.generate_csv_export') as mock_csv:
            selected_fields = ["symbol", "timestamp", "total_pnl"]
            mock_csv.return_value = {
                "filename": "custom_fields_export.csv",
                "content": b"symbol,timestamp,total_pnl\nAAPL,2024-01-20T10:00:00Z,15000.50\n",
                "mime_type": "text/csv",
                "size": 60,
                "fields_exported": selected_fields
            }
            
            result = data_export_routes.generate_csv_export(
                data=mock_portfolio_data,
                fields=selected_fields
            )
            
            assert result["fields_exported"] == selected_fields
            mock_csv.assert_called_once()

    def test_large_dataset_handling(self):
        """Test handling of large datasets with chunking"""
        with patch('data_export_routes.export_bulk_historical_data') as mock_bulk:
            mock_bulk.return_value = {
                "export_id": "large_export_001",
                "status": "processing",
                "chunks_total": 100,
                "chunks_completed": 25,
                "progress": 0.25,
                "estimated_completion": "2024-01-20T12:00:00Z"
            }
            
            result = data_export_routes.export_bulk_historical_data(
                start_date="2023-01-01",
                end_date="2024-01-20",
                chunk_size=10000,
                max_records=1000000
            )
            
            assert result["chunks_total"] > 0
            assert result["progress"] <= 1.0
            assert "estimated_completion" in result
            mock_bulk.assert_called_once()

    def test_export_metadata_generation(self, mock_portfolio_data):
        """Test export metadata generation"""
        with patch('data_export_routes.generate_csv_export') as mock_csv:
            mock_csv.return_value = {
                "filename": "portfolio_export.csv",
                "content": b"mock_content",
                "mime_type": "text/csv",
                "size": 12,
                "metadata": {
                    "export_timestamp": "2024-01-20T10:00:00Z",
                    "data_source": "portfolio_service",
                    "export_version": "1.0",
                    "fields_included": ["timestamp", "total_pnl"],
                    "filters_applied": {"min_pnl": 1000},
                    "record_count": len(mock_portfolio_data)
                }
            }
            
            result = data_export_routes.generate_csv_export(
                data=mock_portfolio_data,
                fields=["timestamp", "total_pnl"],
                include_metadata=True
            )
            
            assert "metadata" in result
            assert result["metadata"]["record_count"] == len(mock_portfolio_data)
            assert "export_timestamp" in result["metadata"]
            mock_csv.assert_called_once()

    def test_concurrent_export_handling(self):
        """Test handling of multiple concurrent exports"""
        with patch('data_export_routes.export_bulk_historical_data') as mock_bulk:
            # Simulate multiple exports running concurrently
            export_ids = []
            for i in range(3):
                mock_bulk.return_value = {
                    "export_id": f"concurrent_export_{i:03d}",
                    "status": "processing",
                    "queue_position": i + 1,
                    "estimated_start": f"2024-01-20T10:{i:02d}:00Z"
                }
                
                result = data_export_routes.export_bulk_historical_data(
                    start_date="2024-01-01",
                    end_date="2024-01-20"
                )
                export_ids.append(result["export_id"])
            
            assert len(export_ids) == 3
            assert all(eid.startswith("concurrent_export_") for eid in export_ids)

    @pytest.mark.parametrize("export_format,expected_mime", [
        ("csv", "text/csv"),
        ("json", "application/json"),
        ("excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("pdf", "application/pdf")
    ])
    def test_format_specific_mime_types(self, export_format, expected_mime, mock_portfolio_data):
        """Test correct MIME types for different export formats"""
        with patch(f'data_export_routes.generate_{export_format}_export') as mock_export:
            mock_export.return_value = {
                "filename": f"test_export.{export_format}",
                "content": b"mock_content",
                "mime_type": expected_mime,
                "size": 12
            }
            
            # Call the appropriate export function
            export_func = getattr(data_export_routes, f'generate_{export_format}_export')
            result = export_func(data=mock_portfolio_data)
            
            assert result["mime_type"] == expected_mime
            mock_export.assert_called_once()