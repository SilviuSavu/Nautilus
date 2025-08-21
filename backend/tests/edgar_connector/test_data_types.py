"""
Tests for EDGAR Data Types
==========================
"""

import pytest
from datetime import datetime
from decimal import Decimal

from edgar_connector.data_types import (
    FilingType,
    SECEntity,
    FilingData,
    CompanyFacts,
    SECFiling,
    FilingSubmission,
    EDGARSubscription,
    create_filing_data,
    create_company_facts,
)


class TestFilingType:
    """Tests for FilingType enum."""
    
    def test_filing_type_values(self):
        """Test filing type enum values."""
        assert FilingType.FORM_10K == "10-K"
        assert FilingType.FORM_10Q == "10-Q"
        assert FilingType.FORM_8K == "8-K"
        assert FilingType.FORM_DEF14A == "DEF 14A"
    
    def test_get_major_forms(self):
        """Test getting major filing forms."""
        major_forms = FilingType.get_major_forms()
        assert FilingType.FORM_10K in major_forms
        assert FilingType.FORM_10Q in major_forms
        assert FilingType.FORM_8K in major_forms
        assert FilingType.FORM_DEF14A in major_forms


class TestSECEntity:
    """Tests for SECEntity model."""
    
    def test_sec_entity_creation(self):
        """Test SEC entity creation."""
        entity = SECEntity(
            cik="0000320193",
            name="Apple Inc.",
            ticker="AAPL",
            exchange="NASDAQ"
        )
        
        assert entity.cik == "0000320193"
        assert entity.name == "Apple Inc."
        assert entity.ticker == "AAPL"
        assert entity.exchange == "NASDAQ"
    
    def test_sec_entity_minimal(self):
        """Test SEC entity with minimal fields."""
        entity = SECEntity(
            cik="0000320193",
            name="Apple Inc."
        )
        
        assert entity.cik == "0000320193"
        assert entity.name == "Apple Inc."
        assert entity.ticker is None
        assert entity.exchange is None
    
    def test_sec_entity_validation(self):
        """Test SEC entity validation."""
        # Missing required field should raise error
        with pytest.raises(ValueError):
            SECEntity(cik="123")  # Missing name


class TestFilingData:
    """Tests for FilingData custom data type."""
    
    def test_filing_data_creation(self):
        """Test filing data creation."""
        filing_date = datetime(2023, 12, 1)
        
        filing = FilingData(
            filing_type=FilingType.FORM_10K,
            cik="0000320193",
            company_name="Apple Inc.",
            accession_number="0000320193-23-000123",
            filing_date=filing_date
        )
        
        assert filing.filing_type == FilingType.FORM_10K
        assert filing.cik == "0000320193"
        assert filing.company_name == "Apple Inc."
        assert filing.accession_number == "0000320193-23-000123"
        assert filing.filing_date == filing_date
        
        # Should have generated timestamps
        assert filing.ts_event > 0
        assert filing.ts_init > 0
    
    def test_filing_data_with_financial_data(self):
        """Test filing data with financial information."""
        financial_data = {
            "revenue": 394328000000,
            "net_income": 99803000000
        }
        
        filing = FilingData(
            filing_type=FilingType.FORM_10K,
            cik="0000320193",
            company_name="Apple Inc.",
            accession_number="0000320193-23-000123",
            filing_date=datetime(2023, 12, 1),
            financial_data=financial_data
        )
        
        assert filing.financial_data == financial_data
    
    def test_filing_data_custom_timestamps(self):
        """Test filing data with custom timestamps."""
        filing_date = datetime(2023, 12, 1)
        custom_ts_event = int(filing_date.timestamp() * 1_000_000_000)
        custom_ts_init = int(datetime.utcnow().timestamp() * 1_000_000_000)
        
        filing = FilingData(
            filing_type=FilingType.FORM_10Q,
            cik="0000320193",
            company_name="Apple Inc.",
            accession_number="0000320193-23-000124",
            filing_date=filing_date,
            ts_event=custom_ts_event,
            ts_init=custom_ts_init
        )
        
        assert filing.ts_event == custom_ts_event
        assert filing.ts_init == custom_ts_init


class TestCompanyFacts:
    """Tests for CompanyFacts custom data type."""
    
    def test_company_facts_creation(self):
        """Test company facts creation."""
        facts_data = {
            "us-gaap:Assets": {"value": 352755000000, "unit": "USD"},
            "us-gaap:Revenues": {"value": 394328000000, "unit": "USD"}
        }
        
        facts = CompanyFacts(
            cik="0000320193",
            company_name="Apple Inc.",
            facts=facts_data,
            fiscal_year=2023
        )
        
        assert facts.cik == "0000320193"
        assert facts.company_name == "Apple Inc."
        assert facts.facts == facts_data
        assert facts.fiscal_year == 2023
        assert facts.fiscal_quarter is None
        
        # Should have generated timestamps
        assert facts.ts_event > 0
        assert facts.ts_init > 0
    
    def test_company_facts_quarterly(self):
        """Test quarterly company facts."""
        facts = CompanyFacts(
            cik="0000320193",
            company_name="Apple Inc.",
            facts={},
            fiscal_year=2023,
            fiscal_quarter=1
        )
        
        assert facts.fiscal_year == 2023
        assert facts.fiscal_quarter == 1


class TestSECFiling:
    """Tests for SECFiling model."""
    
    def test_sec_filing_basic(self):
        """Test basic SEC filing creation."""
        filing_date = datetime(2023, 12, 1)
        
        filing = SECFiling(
            cik="0000320193",
            company_name="Apple Inc.",
            form_type=FilingType.FORM_10K,
            accession_number="0000320193-23-000123",
            filing_date=filing_date
        )
        
        assert filing.cik == "0000320193"
        assert filing.company_name == "Apple Inc."
        assert filing.form_type == FilingType.FORM_10K
        assert filing.accession_number == "0000320193-23-000123"
        assert filing.filing_date == filing_date
        
        # Optional fields should be None
        assert filing.revenue is None
        assert filing.net_income is None
        assert filing.total_assets is None
    
    def test_sec_filing_with_financials(self):
        """Test SEC filing with financial data."""
        filing = SECFiling(
            cik="0000320193",
            company_name="Apple Inc.",
            form_type=FilingType.FORM_10K,
            accession_number="0000320193-23-000123",
            filing_date=datetime(2023, 12, 1),
            revenue=Decimal("394328000000"),
            net_income=Decimal("99803000000"),
            total_assets=Decimal("352755000000")
        )
        
        assert filing.revenue == Decimal("394328000000")
        assert filing.net_income == Decimal("99803000000")
        assert filing.total_assets == Decimal("352755000000")
    
    def test_sec_filing_json_encoding(self):
        """Test SEC filing JSON serialization."""
        filing = SECFiling(
            cik="0000320193",
            company_name="Apple Inc.",
            form_type=FilingType.FORM_10K,
            accession_number="0000320193-23-000123",
            filing_date=datetime(2023, 12, 1, 9, 30),
            revenue=Decimal("394328000000")
        )
        
        # Should be able to serialize to dict
        filing_dict = filing.dict()
        assert filing_dict["filing_date"] == "2023-12-01T09:30:00"
        assert filing_dict["revenue"] == "394328000000"


class TestFilingSubmission:
    """Tests for FilingSubmission model."""
    
    def test_filing_submission_creation(self):
        """Test filing submission creation."""
        filings = [
            SECFiling(
                cik="0000320193",
                company_name="Apple Inc.",
                form_type=FilingType.FORM_10K,
                accession_number="0000320193-23-000123",
                filing_date=datetime(2023, 12, 1)
            ),
            SECFiling(
                cik="0000320193", 
                company_name="Apple Inc.",
                form_type=FilingType.FORM_10Q,
                accession_number="0000320193-23-000124",
                filing_date=datetime(2023, 10, 1)
            )
        ]
        
        submission = FilingSubmission(
            cik="0000320193",
            company_name="Apple Inc.",
            filings=filings,
            submission_date=datetime(2023, 12, 15),
            total_count=2
        )
        
        assert submission.cik == "0000320193"
        assert submission.company_name == "Apple Inc."
        assert len(submission.filings) == 2
        assert submission.total_count == 2
    
    def test_filing_submission_latest_10k(self):
        """Test getting latest 10-K filing."""
        filings = [
            SECFiling(
                cik="0000320193",
                company_name="Apple Inc.",
                form_type=FilingType.FORM_10K,
                accession_number="0000320193-22-000123",
                filing_date=datetime(2022, 12, 1)
            ),
            SECFiling(
                cik="0000320193",
                company_name="Apple Inc.",
                form_type=FilingType.FORM_10K,
                accession_number="0000320193-23-000123",
                filing_date=datetime(2023, 12, 1)
            ),
            SECFiling(
                cik="0000320193",
                company_name="Apple Inc.",
                form_type=FilingType.FORM_10Q,
                accession_number="0000320193-23-000124",
                filing_date=datetime(2023, 10, 1)
            )
        ]
        
        submission = FilingSubmission(
            cik="0000320193",
            company_name="Apple Inc.",
            filings=filings,
            submission_date=datetime(2023, 12, 15),
            total_count=3
        )
        
        latest_10k = submission.latest_10k
        assert latest_10k is not None
        assert latest_10k.form_type == FilingType.FORM_10K
        assert latest_10k.filing_date == datetime(2023, 12, 1)
    
    def test_filing_submission_no_10k(self):
        """Test getting latest 10-K when none exists."""
        filings = [
            SECFiling(
                cik="0000320193",
                company_name="Apple Inc.",
                form_type=FilingType.FORM_8K,
                accession_number="0000320193-23-000125",
                filing_date=datetime(2023, 11, 1)
            )
        ]
        
        submission = FilingSubmission(
            cik="0000320193",
            company_name="Apple Inc.",
            filings=filings,
            submission_date=datetime(2023, 12, 15),
            total_count=1
        )
        
        latest_10k = submission.latest_10k
        assert latest_10k is None


class TestEDGARSubscription:
    """Tests for EDGARSubscription model."""
    
    def test_edgar_subscription_cik(self):
        """Test EDGAR subscription by CIK."""
        subscription = EDGARSubscription(
            cik="0000320193",
            filing_types=[FilingType.FORM_10K, FilingType.FORM_10Q]
        )
        
        assert subscription.cik == "0000320193"
        assert subscription.ticker is None
        assert FilingType.FORM_10K in subscription.filing_types
        assert FilingType.FORM_10Q in subscription.filing_types
        assert subscription.include_amendments is False
        assert subscription.max_age_days == 365
    
    def test_edgar_subscription_ticker(self):
        """Test EDGAR subscription by ticker."""
        subscription = EDGARSubscription(
            ticker="AAPL",
            filing_types=[FilingType.FORM_8K],
            include_amendments=True,
            max_age_days=30
        )
        
        assert subscription.cik is None
        assert subscription.ticker == "AAPL"
        assert subscription.filing_types == [FilingType.FORM_8K]
        assert subscription.include_amendments is True
        assert subscription.max_age_days == 30
    
    def test_edgar_subscription_defaults(self):
        """Test EDGAR subscription with defaults."""
        subscription = EDGARSubscription(cik="0000320193")
        
        # Should have default filing types
        assert FilingType.FORM_10K in subscription.filing_types
        assert FilingType.FORM_10Q in subscription.filing_types
        assert len(subscription.filing_types) == 2
    
    def test_edgar_subscription_validation_error(self):
        """Test EDGAR subscription validation."""
        # Should require either CIK or ticker
        with pytest.raises(ValueError):
            subscription = EDGARSubscription()
            subscription.__post_init__()


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_filing_data(self):
        """Test create_filing_data factory function."""
        filing_date = datetime(2023, 12, 1)
        
        filing = create_filing_data(
            filing_type="10-K",
            cik="0000320193",
            company_name="Apple Inc.",
            accession_number="0000320193-23-000123",
            filing_date=filing_date,
            report_url="https://example.com/filing"
        )
        
        assert isinstance(filing, FilingData)
        assert filing.filing_type == FilingType.FORM_10K
        assert filing.cik == "0000320193"
        assert filing.report_url == "https://example.com/filing"
    
    def test_create_company_facts(self):
        """Test create_company_facts factory function."""
        facts_data = {
            "revenue": 394328000000,
            "assets": 352755000000
        }
        
        facts = create_company_facts(
            cik="0000320193",
            company_name="Apple Inc.",
            facts_data=facts_data,
            fiscal_year=2023
        )
        
        assert isinstance(facts, CompanyFacts)
        assert facts.cik == "0000320193"
        assert facts.facts == facts_data
        assert facts.fiscal_year == 2023