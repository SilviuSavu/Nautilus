"""
Database Models for Collateral Management
=========================================

SQLAlchemy models for persisting margin data, alerts, and optimization results.
Integrates with existing Nautilus database schema.
"""

from sqlalchemy import Column, String, Numeric, DateTime, Boolean, Text, Integer, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid

Base = declarative_base()


class Portfolio(Base):
    """Portfolio table for collateral management"""
    __tablename__ = "cm_portfolios"
    
    id = Column(String, primary_key=True)
    name = Column(String(255), nullable=False)
    currency = Column(String(3), default="USD")
    available_cash = Column(Numeric(20, 8), nullable=False)
    leverage_ratio = Column(Numeric(10, 4))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    margin_calculations = relationship("MarginCalculation", back_populates="portfolio", cascade="all, delete-orphan")
    margin_alerts = relationship("MarginAlert", back_populates="portfolio", cascade="all, delete-orphan")


class Position(Base):
    """Position table for margin calculations"""
    __tablename__ = "cm_positions"
    
    id = Column(String, primary_key=True)
    portfolio_id = Column(String, ForeignKey("cm_portfolios.id"), nullable=False)
    symbol = Column(String(50), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    market_value = Column(Numeric(20, 8), nullable=False)
    asset_class = Column(String(20), nullable=False)
    currency = Column(String(3), default="USD")
    sector = Column(String(100))
    country = Column(String(3))
    duration = Column(Numeric(10, 4))
    implied_volatility = Column(Numeric(10, 6))
    delta = Column(Numeric(10, 6))
    gamma = Column(Numeric(10, 6))
    theta = Column(Numeric(10, 6))
    vega = Column(Numeric(10, 6))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")
    
    # Indexes
    __table_args__ = (
        Index('idx_cm_positions_portfolio_symbol', 'portfolio_id', 'symbol'),
        Index('idx_cm_positions_asset_class', 'asset_class'),
    )


class MarginCalculation(Base):
    """Margin calculation results storage"""
    __tablename__ = "cm_margin_calculations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(String, ForeignKey("cm_portfolios.id"), nullable=False)
    gross_initial_margin = Column(Numeric(20, 8), nullable=False)
    cross_margin_offset = Column(Numeric(20, 8), nullable=False)
    net_initial_margin = Column(Numeric(20, 8), nullable=False)
    variation_margin = Column(Numeric(20, 8), nullable=False)
    maintenance_margin = Column(Numeric(20, 8), nullable=False)
    regulatory_margin = Column(Numeric(20, 8), nullable=False)
    total_margin_requirement = Column(Numeric(20, 8), nullable=False)
    margin_utilization = Column(Numeric(10, 6), nullable=False)
    margin_excess = Column(Numeric(20, 8), nullable=False)
    time_to_margin_call_minutes = Column(Integer)
    position_margins = Column(JSONB)  # Store detailed position-level margins
    calculation_metadata = Column(JSONB)  # Store calculation parameters and performance
    calculated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="margin_calculations")
    
    # Indexes
    __table_args__ = (
        Index('idx_cm_margin_calc_portfolio_time', 'portfolio_id', 'calculated_at'),
        Index('idx_cm_margin_calc_utilization', 'margin_utilization'),
    )


class MarginAlert(Base):
    """Margin alerts and notifications"""
    __tablename__ = "cm_margin_alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(String, ForeignKey("cm_portfolios.id"), nullable=False)
    severity = Column(String(20), nullable=False)  # ok, info, warning, critical, emergency
    message = Column(Text, nullable=False)
    margin_utilization = Column(Numeric(10, 6), nullable=False)
    time_to_margin_call_minutes = Column(Integer)
    recommended_action = Column(String(50))
    affected_positions = Column(JSONB)  # List of position IDs
    required_action_amount = Column(Numeric(20, 8))
    currency = Column(String(3), default="USD")
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="margin_alerts")
    
    # Indexes
    __table_args__ = (
        Index('idx_cm_alerts_portfolio_severity', 'portfolio_id', 'severity'),
        Index('idx_cm_alerts_created_at', 'created_at'),
        Index('idx_cm_alerts_acknowledged', 'acknowledged'),
    )


class CrossMarginBenefit(Base):
    """Cross-margining optimization results"""
    __tablename__ = "cm_cross_margin_benefits"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(String, ForeignKey("cm_portfolios.id"), nullable=False)
    asset_class = Column(String(20), nullable=False)
    position_ids = Column(JSONB, nullable=False)  # List of position IDs
    correlation_coefficient = Column(Numeric(10, 6), nullable=False)
    gross_margin = Column(Numeric(20, 8), nullable=False)
    cross_margin_offset = Column(Numeric(20, 8), nullable=False)
    net_margin = Column(Numeric(20, 8), nullable=False)
    offset_percentage = Column(Numeric(10, 6), nullable=False)
    calculation_method = Column(String(100), nullable=False)
    calculated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Indexes
    __table_args__ = (
        Index('idx_cm_cross_margin_portfolio', 'portfolio_id'),
        Index('idx_cm_cross_margin_asset_class', 'asset_class'),
    )


class RegulatoryCapitalRequirement(Base):
    """Regulatory capital calculations"""
    __tablename__ = "cm_regulatory_capital"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(String, ForeignKey("cm_portfolios.id"), nullable=False)
    basel_iii_requirement = Column(Numeric(20, 8), nullable=False)
    dodd_frank_requirement = Column(Numeric(20, 8), nullable=False)
    emir_requirement = Column(Numeric(20, 8), nullable=False)
    local_regulatory_requirement = Column(Numeric(20, 8), nullable=False)
    total_regulatory_capital = Column(Numeric(20, 8), nullable=False)
    capital_adequacy_ratio = Column(Numeric(10, 6), nullable=False)
    regulatory_framework = Column(String(100), nullable=False)
    jurisdiction = Column(String(10), default="US")
    calculated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Indexes
    __table_args__ = (
        Index('idx_cm_reg_capital_portfolio', 'portfolio_id'),
        Index('idx_cm_reg_capital_jurisdiction', 'jurisdiction'),
    )


class MarginStressTest(Base):
    """Margin stress test results"""
    __tablename__ = "cm_margin_stress_tests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(String, ForeignKey("cm_portfolios.id"), nullable=False)
    scenario_name = Column(String(100), nullable=False)
    base_margin = Column(Numeric(20, 8), nullable=False)
    stressed_margin = Column(Numeric(20, 8), nullable=False)
    margin_increase = Column(Numeric(20, 8), nullable=False)
    margin_increase_percent = Column(Numeric(10, 6), nullable=False)
    positions_at_risk = Column(JSONB, nullable=False)  # List of position IDs
    estimated_liquidation_value = Column(Numeric(20, 8), nullable=False)
    time_to_liquidation_minutes = Column(Integer)
    stress_factors = Column(JSONB, nullable=False)  # Dict of stress parameters
    passes_test = Column(Boolean, nullable=False)
    calculated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Indexes
    __table_args__ = (
        Index('idx_cm_stress_test_portfolio', 'portfolio_id'),
        Index('idx_cm_stress_test_scenario', 'scenario_name'),
        Index('idx_cm_stress_test_passes', 'passes_test'),
    )


class CollateralMovement(Base):
    """Collateral movement tracking"""
    __tablename__ = "cm_collateral_movements"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id = Column(String, ForeignKey("cm_portfolios.id"), nullable=False)
    movement_type = Column(String(20), nullable=False)  # deposit, withdrawal, transfer
    amount = Column(Numeric(20, 8), nullable=False)
    currency = Column(String(3), nullable=False)
    counterparty = Column(String(255))
    settlement_date = Column(DateTime(timezone=True))
    status = Column(String(20), default="pending")  # pending, settled, failed
    reference_id = Column(String(100))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    settled_at = Column(DateTime(timezone=True))
    
    # Indexes
    __table_args__ = (
        Index('idx_cm_collateral_portfolio_status', 'portfolio_id', 'status'),
        Index('idx_cm_collateral_settlement_date', 'settlement_date'),
    )


class OptimizationResult(Base):
    """Collateral optimization results"""
    __tablename__ = "cm_optimization_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(String, ForeignKey("cm_portfolios.id"), nullable=False)
    original_margin = Column(Numeric(20, 8), nullable=False)
    optimized_margin = Column(Numeric(20, 8), nullable=False)
    margin_savings = Column(Numeric(20, 8), nullable=False)
    capital_efficiency_improvement = Column(Numeric(10, 6), nullable=False)
    optimization_method = Column(String(100), nullable=False)
    cross_margin_benefits = Column(JSONB, nullable=False)  # Detailed benefits breakdown
    computation_time_ms = Column(Numeric(10, 2))
    calculated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Indexes
    __table_args__ = (
        Index('idx_cm_optimization_portfolio', 'portfolio_id'),
        Index('idx_cm_optimization_efficiency', 'capital_efficiency_improvement'),
    )


# Engine performance tracking
class EngineMetrics(Base):
    """Engine performance and health metrics"""
    __tablename__ = "cm_engine_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_type = Column(String(50), nullable=False)  # calculation, alert, optimization, health
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Numeric(20, 8), nullable=False)
    metric_unit = Column(String(20))  # ms, count, percent, etc.
    additional_data = Column(JSONB)  # Extra context data
    recorded_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Indexes
    __table_args__ = (
        Index('idx_cm_metrics_type_name', 'metric_type', 'metric_name'),
        Index('idx_cm_metrics_recorded_at', 'recorded_at'),
    )


# Create all tables function
def create_all_tables(engine):
    """Create all collateral management tables"""
    Base.metadata.create_all(engine)


# Migration scripts for integration with existing Nautilus schema
def create_migration_script():
    """Generate SQL migration script for integration with existing database"""
    return """
-- Collateral Management Engine Database Schema
-- Integration with existing Nautilus database

-- Add to existing database schema
-- These tables store collateral management data alongside existing trading data

-- Enable required extensions if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Note: Tables will be created by SQLAlchemy using the models above
-- This is a reference script for manual database setup if needed

-- Additional indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_cm_margin_calc_high_utilization 
ON cm_margin_calculations (portfolio_id, calculated_at) 
WHERE margin_utilization > 0.8;

CREATE INDEX IF NOT EXISTS idx_cm_alerts_unacknowledged 
ON cm_margin_alerts (portfolio_id, created_at, severity) 
WHERE acknowledged = false;

-- Partitioning for historical data (optional)
-- Consider partitioning cm_margin_calculations by date for large datasets
-- CREATE TABLE cm_margin_calculations_y2024m01 PARTITION OF cm_margin_calculations
-- FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
"""