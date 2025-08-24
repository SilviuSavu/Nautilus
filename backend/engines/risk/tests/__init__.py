#!/usr/bin/env python3
"""
PyFolio Integration Test Suite
=============================

Comprehensive test suite for PyFolio integration in Nautilus Risk Engine.

Test Structure:
- test_pyfolio_integration.py: Unit tests for PyFolioAnalytics class
- test_risk_engine_pyfolio.py: Integration tests for Risk Engine endpoints

Usage:
    # Run all tests
    python -m pytest backend/engines/risk/tests/
    
    # Run specific test file
    python -m pytest backend/engines/risk/tests/test_pyfolio_integration.py
    
    # Run with coverage
    python -m pytest backend/engines/risk/tests/ --cov=pyfolio_integration --cov-report=html
    
    # Run performance tests only
    python -m pytest backend/engines/risk/tests/ -k "performance"
"""

__version__ = "1.0.0"
__author__ = "Nautilus Risk Engine Team"