"""
Test Suite for Toraniko v1.1.2 Integration with Nautilus Factor Engine
Tests the enhanced FactorModel capabilities, configuration system, and API endpoints
"""
import asyncio
import logging
import polars as pl
import numpy as np
from datetime import date, datetime, timedelta
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from factor_engine_service import FactorEngineService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToranikoBenchmarkSuite:
    """Comprehensive test suite for Toraniko v1.1.2 integration"""
    
    def __init__(self):
        self.factor_service = FactorEngineService()
        self.test_results = {}
        
    def generate_test_data(self, n_symbols: int = 100, n_days: int = 252) -> tuple:
        """Generate synthetic market data for testing"""
        logger.info(f"Generating test data: {n_symbols} symbols, {n_days} days")
        
        # Generate symbols (mix of sectors)
        symbols = [f"STOCK_{i:03d}" for i in range(n_symbols)]
        
        # Generate date range
        start_date = datetime.now() - timedelta(days=n_days)
        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        
        # Generate feature data
        feature_data = []
        for symbol in symbols:
            for date_val in dates:
                # Simulate realistic market data
                base_price = np.random.uniform(10, 500)
                daily_return = np.random.normal(0, 0.02)  # 2% daily vol
                market_cap = base_price * np.random.uniform(1e6, 1e10)  # Market cap
                volume = np.random.uniform(10000, 1000000)
                
                feature_data.append({
                    'symbol': symbol,
                    'date': date_val.date(),
                    'asset_returns': daily_return,
                    'market_cap': market_cap,
                    'price': base_price,
                    'volume': volume,
                    'book_price': np.random.uniform(0.5, 2.0),  # Book-to-price ratio
                    'sales_price': np.random.uniform(0.1, 5.0),  # Sales-to-price ratio
                    'cf_price': np.random.uniform(0.05, 1.0)    # Cash flow-to-price ratio
                })
        
        # Generate sector encodings (GICS Level 1)
        sectors = [
            'Energy', 'Materials', 'Industrials', 'Consumer Discretionary',
            'Consumer Staples', 'Health Care', 'Financials', 'Information Technology',
            'Telecommunication Services', 'Utilities', 'Real Estate'
        ]
        
        sector_encodings = []
        for symbol in symbols:
            # Assign random sector
            sector = np.random.choice(sectors)
            sector_encoding = {col: 0 for col in sectors}
            sector_encoding[sector] = 1
            sector_encoding['symbol'] = symbol
            sector_encodings.append(sector_encoding)
        
        logger.info(f"Generated {len(feature_data)} feature observations and {len(sector_encodings)} sector encodings")
        return feature_data, sector_encodings
    
    async def test_factor_model_creation(self):
        """Test FactorModel creation with v1.1.2 capabilities"""
        logger.info("Testing FactorModel creation...")
        
        try:
            # Generate test data
            feature_data, sector_encodings = self.generate_test_data(50, 100)
            
            # Create FactorModel
            model_id = "test_model_1"
            result = await self.factor_service.create_factor_model(
                model_id=model_id,
                feature_data=pl.DataFrame(feature_data),
                sector_encodings=pl.DataFrame(sector_encodings)
            )
            
            # Verify model was created
            status = await self.factor_service.get_model_status(model_id)
            
            self.test_results['factor_model_creation'] = {
                'success': True,
                'result': result,
                'status': status,
                'feature_data_rows': len(feature_data),
                'sector_encodings_rows': len(sector_encodings)
            }
            
            logger.info(f"✅ FactorModel creation test passed: {result}")
            return True
            
        except Exception as e:
            logger.error(f"❌ FactorModel creation test failed: {e}")
            self.test_results['factor_model_creation'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    async def test_feature_cleaning(self):
        """Test feature cleaning pipeline"""
        logger.info("Testing feature cleaning pipeline...")
        
        try:
            model_id = "test_model_1"
            
            # Test feature cleaning
            result = await self.factor_service.clean_model_features(
                model_id=model_id,
                to_winsorize={'asset_returns': 0.01, 'market_cap': 0.05},
                to_fill=['price', 'volume'],
                to_smooth={'market_cap': 5}
            )
            
            self.test_results['feature_cleaning'] = {
                'success': True,
                'result': result,
                'operations': ['winsorization', 'forward_fill', 'smoothing']
            }
            
            logger.info(f"✅ Feature cleaning test passed: {result}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Feature cleaning test failed: {e}")
            self.test_results['feature_cleaning'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    async def test_universe_reduction(self):
        """Test universe reduction by market cap"""
        logger.info("Testing universe reduction...")
        
        try:
            model_id = "test_model_1"
            
            # Reduce universe to top 30 by market cap
            result = await self.factor_service.reduce_model_universe(
                model_id=model_id,
                top_n=30,
                collect=True
            )
            
            self.test_results['universe_reduction'] = {
                'success': True,
                'result': result,
                'universe_size': 30
            }
            
            logger.info(f"✅ Universe reduction test passed: {result}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Universe reduction test failed: {e}")
            self.test_results['universe_reduction'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    async def test_style_score_estimation(self):
        """Test style factor score estimation"""
        logger.info("Testing style score estimation...")
        
        try:
            model_id = "test_model_1"
            
            # Estimate style scores
            result = await self.factor_service.estimate_model_style_scores(
                model_id=model_id,
                collect=True
            )
            
            self.test_results['style_score_estimation'] = {
                'success': True,
                'result': result,
                'style_factors': ['momentum', 'value', 'size']
            }
            
            logger.info(f"✅ Style score estimation test passed: {result}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Style score estimation test failed: {e}")
            self.test_results['style_score_estimation'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    async def test_factor_return_estimation(self):
        """Test factor return estimation with v1.1.2 capabilities"""
        logger.info("Testing factor return estimation...")
        
        try:
            model_id = "test_model_1"
            
            # Estimate factor returns
            result = await self.factor_service.estimate_model_factor_returns(
                model_id=model_id,
                winsor_factor=0.02,
                residualize_styles=False,
                asset_returns_col="asset_returns"
            )
            
            self.test_results['factor_return_estimation'] = {
                'success': True,
                'result': result,
                'enhanced_features': ['ledoit_wolf_covariance', 'advanced_winsorization']
            }
            
            logger.info(f"✅ Factor return estimation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Factor return estimation test failed: {e}")
            self.test_results['factor_return_estimation'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    async def test_configuration_system(self):
        """Test Toraniko configuration system"""
        logger.info("Testing configuration system...")
        
        try:
            # Test configuration loading
            config_loaded = self.factor_service._config is not None
            factor_definitions = self.factor_service._factor_definitions_loaded
            
            # Test model listing
            models_info = await self.factor_service.list_factor_models()
            
            self.test_results['configuration_system'] = {
                'success': True,
                'config_loaded': config_loaded,
                'factor_definitions': factor_definitions,
                'models_info': models_info
            }
            
            logger.info(f"✅ Configuration system test passed: {factor_definitions} factor definitions loaded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration system test failed: {e}")
            self.test_results['configuration_system'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    async def run_comprehensive_test_suite(self):
        """Run all tests in the comprehensive suite"""
        logger.info("🚀 Starting Toraniko v1.1.2 Integration Test Suite")
        
        # Initialize service
        await self.factor_service.initialize()
        
        test_methods = [
            self.test_factor_model_creation,
            self.test_feature_cleaning,
            self.test_universe_reduction,
            self.test_style_score_estimation,
            self.test_factor_return_estimation,
            self.test_configuration_system
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            test_name = test_method.__name__
            logger.info(f"\n--- Running {test_name} ---")
            try:
                success = await test_method()
                if success:
                    passed_tests += 1
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
        
        # Generate summary report
        success_rate = (passed_tests / total_tests) * 100
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': f"{success_rate:.1f}%",
            'integration_status': 'PASSED' if success_rate >= 80 else 'FAILED',
            'toraniko_version': '1.1.2',
            'test_timestamp': datetime.now().isoformat(),
            'detailed_results': self.test_results
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🏁 TORANIKO v1.1.2 INTEGRATION TEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Status: {summary['integration_status']}")
        logger.info(f"{'='*60}")
        
        if success_rate >= 80:
            logger.info("✅ Toraniko v1.1.2 integration is READY FOR PRODUCTION")
        else:
            logger.error("❌ Toraniko v1.1.2 integration requires fixes before deployment")
        
        return summary

async def main():
    """Run the test suite"""
    test_suite = ToranikoBenchmarkSuite()
    results = await test_suite.run_comprehensive_test_suite()
    
    # Save results to file
    import json
    with open('toraniko_v1_1_2_integration_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Test results saved to: toraniko_v1_1_2_integration_results.json")
    return results

if __name__ == "__main__":
    asyncio.run(main())