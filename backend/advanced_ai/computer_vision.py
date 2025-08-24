"""
Computer Vision Systems for Alternative Data Analysis
Implementation of chart pattern recognition and satellite imagery analysis
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.dates import DateFormatter
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import asyncio
import aiohttp
import io
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from scipy import ndimage
from scipy.signal import find_peaks
import rasterio
import geopandas as gpd
from satellite_data import SentinelHub  # Placeholder for satellite data API
import base64
import json

logger = logging.getLogger(__name__)


@dataclass
class ChartPattern:
    """Represents a detected chart pattern"""
    pattern_type: str
    confidence: float
    coordinates: List[Tuple[int, int]]
    timeframe: Tuple[datetime, datetime]
    price_range: Tuple[float, float]
    breakout_target: Optional[float] = None
    risk_level: str = "medium"
    trading_signal: str = "hold"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SatelliteInsight:
    """Represents insights from satellite imagery analysis"""
    location: Tuple[float, float]  # lat, lon
    timestamp: datetime
    insight_type: str  # e.g., "industrial_activity", "shipping_traffic", "construction"
    confidence: float
    quantitative_measure: Optional[float] = None
    change_from_previous: Optional[float] = None
    economic_impact: str = "neutral"
    related_companies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CVAnalysisResult:
    """Comprehensive computer vision analysis result"""
    timestamp: datetime
    chart_patterns: List[ChartPattern]
    satellite_insights: List[SatelliteInsight]
    alternative_data_signals: Dict[str, Any]
    confidence_score: float
    trading_recommendations: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]


class ChartPatternNet(nn.Module):
    """Neural network for chart pattern recognition"""
    
    def __init__(self, num_patterns: int = 10, input_size: Tuple[int, int] = (224, 224)):
        super(ChartPatternNet, self).__init__()
        
        # Use pre-trained ResNet50 backbone
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove final layer
        
        # Custom classification head for chart patterns
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_patterns)
        )
        
        # Confidence estimation branch
        self.confidence_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Localization head for pattern coordinates
        self.localization_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # x1, y1, x2, y2
        )
        
        self.num_patterns = num_patterns
        self.input_size = input_size
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Pattern classification
        pattern_logits = self.classifier(features)
        pattern_probs = F.softmax(pattern_logits, dim=1)
        
        # Confidence estimation
        confidence = self.confidence_head(features)
        
        # Pattern localization
        bbox = self.localization_head(features)
        
        return {
            'pattern_probs': pattern_probs,
            'confidence': confidence,
            'bbox': bbox,
            'features': features
        }


class ChartPatternDetector:
    """Advanced chart pattern detector using computer vision"""
    
    PATTERN_TYPES = [
        'head_and_shoulders', 'inverse_head_and_shoulders', 'double_top', 'double_bottom',
        'triangle_ascending', 'triangle_descending', 'triangle_symmetrical',
        'flag_bull', 'flag_bear', 'wedge_rising', 'wedge_falling', 'rectangle',
        'cup_and_handle', 'inverse_cup_and_handle'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = self._create_transform()
        
        # Load or initialize model
        self._load_model()
        
    def _create_transform(self):
        """Create image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        """Load or initialize the chart pattern detection model"""
        try:
            model_path = self.config.get('model_path', 'chart_pattern_model.pth')
            self.model = ChartPatternNet(len(self.PATTERN_TYPES))
            
            try:
                # Try to load pre-trained weights
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded pre-trained model from {model_path}")
            except FileNotFoundError:
                logger.warning(f"No pre-trained model found at {model_path}. Using randomly initialized weights.")
                
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    async def detect_patterns(self, price_data: pd.DataFrame, 
                            symbol: str = "UNKNOWN") -> List[ChartPattern]:
        """Detect chart patterns in price data"""
        try:
            # Create chart image
            chart_image = await self._create_chart_image(price_data, symbol)
            
            # Detect patterns using deep learning
            patterns = await self._detect_patterns_dl(chart_image, price_data)
            
            # Enhance with classical technical analysis
            enhanced_patterns = await self._enhance_with_classical_analysis(patterns, price_data)
            
            return enhanced_patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []
    
    async def _create_chart_image(self, data: pd.DataFrame, symbol: str) -> np.ndarray:
        """Create chart image from price data"""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Ensure data has required columns
        if 'Close' not in data.columns and 'close' in data.columns:
            data['Close'] = data['close']
        
        # Plot candlestick chart
        dates = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data.index)
        
        # OHLC data
        opens = data.get('Open', data.get('open', data['Close']))
        highs = data.get('High', data.get('high', data['Close']))
        lows = data.get('Low', data.get('low', data['Close']))
        closes = data['Close']
        
        # Plot candlesticks
        for i, date in enumerate(dates):
            open_price = opens.iloc[i]
            high_price = highs.iloc[i]
            low_price = lows.iloc[i]
            close_price = closes.iloc[i]
            
            color = 'green' if close_price >= open_price else 'red'
            
            # Candlestick body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            rect = patches.Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                                   facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Wicks
            ax.plot([i, i], [low_price, high_price], color=color, alpha=0.8, linewidth=1)
        
        # Add volume bars if available
        if 'Volume' in data.columns or 'volume' in data.columns:
            volume = data.get('Volume', data.get('volume'))
            volume_scaled = volume / volume.max() * (data['Close'].max() - data['Close'].min()) * 0.1
            ax.bar(range(len(volume)), volume_scaled, bottom=data['Close'].min(), 
                  alpha=0.3, color='blue', width=0.8)
        
        # Add moving averages
        if len(data) >= 20:
            sma20 = data['Close'].rolling(20).mean()
            ax.plot(range(len(sma20)), sma20, color='orange', alpha=0.7, linewidth=2, label='SMA 20')
        
        if len(data) >= 50:
            sma50 = data['Close'].rolling(50).mean()
            ax.plot(range(len(sma50)), sma50, color='blue', alpha=0.7, linewidth=2, label='SMA 50')
        
        # Styling
        ax.set_title(f'{symbol} - Price Chart', fontsize=16, color='white')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Convert to numpy array
        fig.canvas.draw()
        chart_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        chart_array = chart_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return chart_array
    
    async def _detect_patterns_dl(self, chart_image: np.ndarray, 
                                price_data: pd.DataFrame) -> List[ChartPattern]:
        """Detect patterns using deep learning model"""
        try:
            # Preprocess image
            image_pil = Image.fromarray(chart_image)
            image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
            
            # Extract results
            pattern_probs = outputs['pattern_probs'].cpu().numpy()[0]
            confidence = outputs['confidence'].cpu().numpy()[0][0]
            bbox = outputs['bbox'].cpu().numpy()[0]
            
            patterns = []
            
            # Process predictions
            for i, prob in enumerate(pattern_probs):
                if prob > 0.1:  # Threshold for pattern detection
                    pattern = ChartPattern(
                        pattern_type=self.PATTERN_TYPES[i],
                        confidence=prob * confidence,
                        coordinates=[(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))],
                        timeframe=(price_data.index[0], price_data.index[-1]),
                        price_range=(price_data['Close'].min(), price_data['Close'].max()),
                        trading_signal=self._get_trading_signal(self.PATTERN_TYPES[i]),
                        metadata={'model_confidence': confidence, 'raw_probability': prob}
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in deep learning pattern detection: {e}")
            return []
    
    async def _enhance_with_classical_analysis(self, patterns: List[ChartPattern],
                                             price_data: pd.DataFrame) -> List[ChartPattern]:
        """Enhance patterns with classical technical analysis"""
        try:
            enhanced_patterns = []
            
            for pattern in patterns:
                # Calculate breakout targets
                price_range = pattern.price_range[1] - pattern.price_range[0]
                
                if 'bull' in pattern.pattern_type or pattern.pattern_type in ['double_bottom', 'cup_and_handle']:
                    pattern.breakout_target = pattern.price_range[1] + price_range * 0.5
                    pattern.trading_signal = 'buy'
                elif 'bear' in pattern.pattern_type or pattern.pattern_type in ['double_top', 'head_and_shoulders']:
                    pattern.breakout_target = pattern.price_range[0] - price_range * 0.5
                    pattern.trading_signal = 'sell'
                
                # Calculate risk level based on volatility
                volatility = price_data['Close'].pct_change().std()
                if volatility > 0.03:
                    pattern.risk_level = 'high'
                elif volatility > 0.015:
                    pattern.risk_level = 'medium'
                else:
                    pattern.risk_level = 'low'
                
                # Add technical indicators context
                pattern.metadata.update({
                    'volatility': volatility,
                    'avg_volume': price_data.get('Volume', price_data.get('volume', pd.Series([0]))).mean(),
                    'rsi': self._calculate_rsi(price_data['Close'].values),
                    'macd': self._calculate_macd(price_data['Close'].values)
                })
                
                enhanced_patterns.append(pattern)
            
            return enhanced_patterns
            
        except Exception as e:
            logger.error(f"Error enhancing patterns: {e}")
            return patterns
    
    def _get_trading_signal(self, pattern_type: str) -> str:
        """Get trading signal for pattern type"""
        bullish_patterns = [
            'inverse_head_and_shoulders', 'double_bottom', 'triangle_ascending',
            'flag_bull', 'wedge_falling', 'cup_and_handle'
        ]
        bearish_patterns = [
            'head_and_shoulders', 'double_top', 'triangle_descending',
            'flag_bear', 'wedge_rising', 'inverse_cup_and_handle'
        ]
        
        if pattern_type in bullish_patterns:
            return 'buy'
        elif pattern_type in bearish_patterns:
            return 'sell'
        else:
            return 'hold'
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[-period:])
        avg_losses = np.mean(losses[-period:])
        
        if avg_losses == 0:
            return 100.0
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate MACD indicator"""
        if len(prices) < 26:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        prices_series = pd.Series(prices)
        ema12 = prices_series.ewm(span=12).mean()
        ema26 = prices_series.ewm(span=26).mean()
        
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        return {
            'macd': macd.iloc[-1],
            'signal': signal.iloc[-1],
            'histogram': histogram.iloc[-1]
        }


class SatelliteImageryAnalyzer:
    """Analyzer for satellite imagery to extract economic insights"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Initialize satellite data clients (placeholder)
        self.sentinel_hub = None  # Would initialize actual satellite data API
        
        # Load or initialize model
        self._load_model()
        
    def _load_model(self):
        """Load satellite imagery analysis model"""
        try:
            # Use EfficientNet for satellite imagery analysis
            self.model = efficientnet_b0(pretrained=True)
            
            # Modify for satellite imagery tasks
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10)  # Number of economic activity classes
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Loaded satellite imagery analysis model")
            
        except Exception as e:
            logger.error(f"Error loading satellite model: {e}")
            raise
    
    async def analyze_economic_activity(self, locations: List[Tuple[float, float]],
                                      time_range: Tuple[datetime, datetime]) -> List[SatelliteInsight]:
        """Analyze economic activity from satellite imagery"""
        insights = []
        
        for lat, lon in locations:
            try:
                # Get satellite imagery (placeholder)
                imagery_data = await self._fetch_satellite_imagery(lat, lon, time_range)
                
                if imagery_data is not None:
                    # Analyze imagery
                    insight = await self._analyze_single_location(lat, lon, imagery_data)
                    if insight:
                        insights.append(insight)
                        
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error analyzing location ({lat}, {lon}): {e}")
                continue
        
        return insights
    
    async def _fetch_satellite_imagery(self, lat: float, lon: float, 
                                     time_range: Tuple[datetime, datetime]) -> Optional[np.ndarray]:
        """Fetch satellite imagery for location and time range"""
        # Placeholder implementation - would integrate with actual satellite APIs
        try:
            # Simulated satellite data - in reality would use APIs like:
            # - Sentinel Hub
            # - Google Earth Engine
            # - Planet Labs
            # - Maxar
            
            # Generate synthetic satellite imagery for demo
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Add some realistic features
            # Industrial areas (bright spots)
            if np.random.random() > 0.7:
                center_x, center_y = np.random.randint(50, 174, 2)
                cv2.circle(image, (center_x, center_y), 20, (255, 255, 255), -1)
            
            # Water bodies (dark blue areas)
            if np.random.random() > 0.5:
                cv2.rectangle(image, (0, 0), (50, 50), (20, 20, 100), -1)
            
            return image
            
        except Exception as e:
            logger.error(f"Error fetching satellite imagery: {e}")
            return None
    
    async def _analyze_single_location(self, lat: float, lon: float, 
                                     imagery: np.ndarray) -> Optional[SatelliteInsight]:
        """Analyze satellite imagery for a single location"""
        try:
            # Preprocess imagery
            image_pil = Image.fromarray(imagery)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image_pil).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # Interpret results
            activity_classes = [
                'industrial', 'residential', 'commercial', 'agricultural',
                'mining', 'ports', 'airports', 'construction', 'energy', 'transportation'
            ]
            
            dominant_activity = activity_classes[np.argmax(probabilities)]
            confidence = np.max(probabilities)
            
            # Calculate economic metrics
            economic_impact = self._assess_economic_impact(dominant_activity, confidence)
            quantitative_measure = self._calculate_activity_intensity(imagery)
            
            # Generate insight
            insight = SatelliteInsight(
                location=(lat, lon),
                timestamp=datetime.now(),
                insight_type=dominant_activity,
                confidence=confidence,
                quantitative_measure=quantitative_measure,
                economic_impact=economic_impact,
                related_companies=self._get_related_companies(dominant_activity, lat, lon),
                metadata={
                    'activity_distribution': dict(zip(activity_classes, probabilities)),
                    'image_quality': self._assess_image_quality(imagery)
                }
            )
            
            return insight
            
        except Exception as e:
            logger.error(f"Error analyzing single location: {e}")
            return None
    
    def _assess_economic_impact(self, activity_type: str, confidence: float) -> str:
        """Assess economic impact of detected activity"""
        high_impact_activities = ['industrial', 'ports', 'airports', 'mining', 'energy']
        medium_impact_activities = ['commercial', 'construction', 'transportation']
        
        if activity_type in high_impact_activities and confidence > 0.8:
            return 'high_positive'
        elif activity_type in high_impact_activities and confidence > 0.6:
            return 'medium_positive'
        elif activity_type in medium_impact_activities and confidence > 0.7:
            return 'medium_positive'
        else:
            return 'neutral'
    
    def _calculate_activity_intensity(self, imagery: np.ndarray) -> float:
        """Calculate activity intensity from imagery"""
        # Simple intensity calculation based on brightness and edge density
        gray = cv2.cvtColor(imagery, cv2.COLOR_RGB2GRAY)
        
        # Edge density (indicates infrastructure)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Brightness variance (indicates development vs natural areas)
        brightness_var = np.var(gray)
        
        # Combine metrics
        intensity = (edge_density * 0.7 + brightness_var / 10000 * 0.3)
        return min(1.0, intensity)
    
    def _assess_image_quality(self, imagery: np.ndarray) -> Dict[str, float]:
        """Assess satellite image quality metrics"""
        gray = cv2.cvtColor(imagery, cv2.COLOR_RGB2GRAY)
        
        # Calculate sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate contrast (standard deviation)
        contrast = np.std(gray)
        
        # Cloud coverage (simple brightness-based estimation)
        bright_pixels = np.sum(gray > 200)
        cloud_coverage = bright_pixels / gray.size
        
        return {
            'sharpness': min(1.0, sharpness / 1000),
            'contrast': min(1.0, contrast / 50),
            'cloud_coverage': cloud_coverage
        }
    
    def _get_related_companies(self, activity_type: str, lat: float, lon: float) -> List[str]:
        """Get companies related to detected economic activity"""
        # Placeholder - would integrate with company location databases
        company_map = {
            'industrial': ['GE', 'CAT', 'MMM', 'HON'],
            'ports': ['FDX', 'UPS', 'CHRW', 'XPO'],
            'airports': ['DAL', 'UAL', 'AAL', 'LUV'],
            'mining': ['FCX', 'NEM', 'VALE', 'RIO'],
            'energy': ['XOM', 'CVX', 'COP', 'EOG'],
            'commercial': ['WMT', 'AMZN', 'HD', 'LOW'],
            'residential': ['DHI', 'PHM', 'LEN', 'NVR']
        }
        
        return company_map.get(activity_type, [])


class CVDataProcessor:
    """Main processor for computer vision-based alternative data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chart_detector = ChartPatternDetector(config.get('chart_config', {}))
        self.satellite_analyzer = SatelliteImageryAnalyzer(config.get('satellite_config', {}))
        
    async def analyze_comprehensive(self, symbols: List[str],
                                  locations: Optional[List[Tuple[float, float]]] = None) -> CVAnalysisResult:
        """Perform comprehensive CV analysis"""
        try:
            chart_patterns = []
            satellite_insights = []
            alternative_signals = {}
            
            # Analyze chart patterns for each symbol
            for symbol in symbols:
                try:
                    # Fetch price data
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1y")
                    
                    if not hist.empty:
                        patterns = await self.chart_detector.detect_patterns(hist, symbol)
                        chart_patterns.extend(patterns)
                        
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    
                # Rate limiting
                await asyncio.sleep(0.5)
            
            # Analyze satellite imagery if locations provided
            if locations:
                time_range = (datetime.now() - timedelta(days=30), datetime.now())
                satellite_insights = await self.satellite_analyzer.analyze_economic_activity(
                    locations, time_range
                )
            
            # Generate alternative data signals
            alternative_signals = await self._generate_alternative_signals(
                chart_patterns, satellite_insights
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(
                chart_patterns, satellite_insights
            )
            
            # Generate trading recommendations
            trading_recommendations = self._generate_trading_recommendations(
                chart_patterns, satellite_insights, alternative_signals
            )
            
            # Assess risks
            risk_assessment = self._assess_risks(chart_patterns, satellite_insights)
            
            return CVAnalysisResult(
                timestamp=datetime.now(),
                chart_patterns=chart_patterns,
                satellite_insights=satellite_insights,
                alternative_data_signals=alternative_signals,
                confidence_score=confidence_score,
                trading_recommendations=trading_recommendations,
                risk_assessment=risk_assessment
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive CV analysis: {e}")
            return CVAnalysisResult(
                timestamp=datetime.now(),
                chart_patterns=[],
                satellite_insights=[],
                alternative_data_signals={},
                confidence_score=0.0,
                trading_recommendations=[],
                risk_assessment={}
            )
    
    async def _generate_alternative_signals(self, chart_patterns: List[ChartPattern],
                                          satellite_insights: List[SatelliteInsight]) -> Dict[str, Any]:
        """Generate alternative data trading signals"""
        signals = {}
        
        # Chart pattern signals
        bullish_patterns = [p for p in chart_patterns if p.trading_signal == 'buy']
        bearish_patterns = [p for p in chart_patterns if p.trading_signal == 'sell']
        
        signals['chart_sentiment'] = {
            'bullish_count': len(bullish_patterns),
            'bearish_count': len(bearish_patterns),
            'avg_confidence': np.mean([p.confidence for p in chart_patterns]) if chart_patterns else 0.0,
            'net_bias': len(bullish_patterns) - len(bearish_patterns)
        }
        
        # Satellite insights signals
        positive_insights = [i for i in satellite_insights if 'positive' in i.economic_impact]
        negative_insights = [i for i in satellite_insights if 'negative' in i.economic_impact]
        
        signals['satellite_economic_activity'] = {
            'positive_indicators': len(positive_insights),
            'negative_indicators': len(negative_insights),
            'avg_activity_intensity': np.mean([i.quantitative_measure for i in satellite_insights 
                                            if i.quantitative_measure]) if satellite_insights else 0.0,
            'high_confidence_insights': len([i for i in satellite_insights if i.confidence > 0.8])
        }
        
        # Combined signal strength
        signals['combined_strength'] = self._calculate_combined_signal_strength(
            signals['chart_sentiment'], signals['satellite_economic_activity']
        )
        
        return signals
    
    def _calculate_overall_confidence(self, chart_patterns: List[ChartPattern],
                                    satellite_insights: List[SatelliteInsight]) -> float:
        """Calculate overall confidence score"""
        chart_confidence = np.mean([p.confidence for p in chart_patterns]) if chart_patterns else 0.0
        satellite_confidence = np.mean([i.confidence for i in satellite_insights]) if satellite_insights else 0.0
        
        # Weighted average
        total_signals = len(chart_patterns) + len(satellite_insights)
        if total_signals == 0:
            return 0.0
        
        chart_weight = len(chart_patterns) / total_signals
        satellite_weight = len(satellite_insights) / total_signals
        
        overall_confidence = chart_confidence * chart_weight + satellite_confidence * satellite_weight
        return overall_confidence
    
    def _generate_trading_recommendations(self, chart_patterns: List[ChartPattern],
                                        satellite_insights: List[SatelliteInsight],
                                        alternative_signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading recommendations based on CV analysis"""
        recommendations = []
        
        # Chart-based recommendations
        high_confidence_patterns = [p for p in chart_patterns if p.confidence > 0.7]
        for pattern in high_confidence_patterns:
            if pattern.trading_signal in ['buy', 'sell']:
                recommendations.append({
                    'type': 'chart_pattern',
                    'signal': pattern.trading_signal,
                    'confidence': pattern.confidence,
                    'pattern_type': pattern.pattern_type,
                    'target_price': pattern.breakout_target,
                    'risk_level': pattern.risk_level,
                    'timeframe': 'short_term'
                })
        
        # Satellite-based recommendations
        strong_economic_signals = [i for i in satellite_insights 
                                 if i.confidence > 0.8 and 'positive' in i.economic_impact]
        
        for insight in strong_economic_signals:
            if insight.related_companies:
                for company in insight.related_companies[:3]:  # Top 3
                    recommendations.append({
                        'type': 'satellite_insight',
                        'signal': 'buy',
                        'confidence': insight.confidence,
                        'symbol': company,
                        'insight_type': insight.insight_type,
                        'location': insight.location,
                        'timeframe': 'medium_term'
                    })
        
        # Combined signal recommendations
        combined_strength = alternative_signals.get('combined_strength', 0.0)
        if combined_strength > 0.6:
            recommendations.append({
                'type': 'combined_signal',
                'signal': 'buy' if combined_strength > 0 else 'sell',
                'confidence': abs(combined_strength),
                'reasoning': 'Strong convergence of chart patterns and economic activity indicators',
                'timeframe': 'medium_term'
            })
        
        return recommendations
    
    def _assess_risks(self, chart_patterns: List[ChartPattern],
                     satellite_insights: List[SatelliteInsight]) -> Dict[str, Any]:
        """Assess risks from CV analysis"""
        risks = {
            'pattern_reliability': 'medium',
            'data_quality': 'good',
            'conflicting_signals': False,
            'risk_factors': []
        }
        
        # Check for conflicting chart signals
        buy_signals = len([p for p in chart_patterns if p.trading_signal == 'buy'])
        sell_signals = len([p for p in chart_patterns if p.trading_signal == 'sell'])
        
        if buy_signals > 0 and sell_signals > 0:
            risks['conflicting_signals'] = True
            risks['risk_factors'].append('Conflicting chart pattern signals detected')
        
        # Assess pattern reliability
        high_risk_patterns = len([p for p in chart_patterns if p.risk_level == 'high'])
        total_patterns = len(chart_patterns)
        
        if total_patterns > 0 and high_risk_patterns / total_patterns > 0.5:
            risks['pattern_reliability'] = 'low'
            risks['risk_factors'].append('High proportion of high-risk patterns')
        
        # Assess satellite data quality
        low_quality_insights = len([i for i in satellite_insights if i.confidence < 0.5])
        total_insights = len(satellite_insights)
        
        if total_insights > 0 and low_quality_insights / total_insights > 0.3:
            risks['data_quality'] = 'poor'
            risks['risk_factors'].append('Low quality satellite imagery data')
        
        return risks
    
    def _calculate_combined_signal_strength(self, chart_signals: Dict[str, Any],
                                          satellite_signals: Dict[str, Any]) -> float:
        """Calculate combined signal strength"""
        # Chart signal component
        chart_bias = chart_signals.get('net_bias', 0)
        chart_confidence = chart_signals.get('avg_confidence', 0)
        chart_component = (chart_bias / 10) * chart_confidence  # Normalize bias
        
        # Satellite signal component
        sat_positive = satellite_signals.get('positive_indicators', 0)
        sat_negative = satellite_signals.get('negative_indicators', 0)
        sat_bias = sat_positive - sat_negative
        sat_intensity = satellite_signals.get('avg_activity_intensity', 0)
        satellite_component = (sat_bias / 5) * sat_intensity  # Normalize bias
        
        # Combine with equal weighting
        combined = (chart_component + satellite_component) / 2
        return np.clip(combined, -1.0, 1.0)


# Example usage and testing
async def demo_computer_vision():
    """Demonstrate computer vision capabilities"""
    logger.info("Starting computer vision demo")
    
    # Configuration
    config = {
        'chart_config': {
            'model_path': 'chart_pattern_model.pth'
        },
        'satellite_config': {
            'api_key': 'your-satellite-api-key'
        }
    }
    
    # Initialize processor
    cv_processor = CVDataProcessor(config)
    
    # Test symbols and locations
    test_symbols = ['AAPL', 'GOOGL', 'MSFT']
    test_locations = [
        (37.7749, -122.4194),  # San Francisco (tech hub)
        (40.7128, -74.0060),   # New York (financial center)
        (29.7604, -95.3698)    # Houston (energy hub)
    ]
    
    # Perform comprehensive analysis
    try:
        results = await cv_processor.analyze_comprehensive(test_symbols, test_locations)
        
        logger.info(f"Analysis Results:")
        logger.info(f"Chart Patterns Found: {len(results.chart_patterns)}")
        logger.info(f"Satellite Insights: {len(results.satellite_insights)}")
        logger.info(f"Overall Confidence: {results.confidence_score:.3f}")
        logger.info(f"Trading Recommendations: {len(results.trading_recommendations)}")
        
        # Display some results
        for pattern in results.chart_patterns[:3]:
            logger.info(f"Pattern: {pattern.pattern_type} (confidence: {pattern.confidence:.3f})")
            
        for insight in results.satellite_insights[:3]:
            logger.info(f"Satellite Insight: {insight.insight_type} at {insight.location} (confidence: {insight.confidence:.3f})")
            
        for recommendation in results.trading_recommendations[:5]:
            logger.info(f"Recommendation: {recommendation.get('signal')} {recommendation.get('symbol', 'N/A')} ({recommendation.get('type')})")
            
    except Exception as e:
        logger.error(f"Error in demo: {e}")
    
    logger.info("Computer vision demo completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_computer_vision())