"""
Large Language Model Integration for Market Analysis
Implementation of LLM-powered sentiment analysis and market intelligence
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import aiohttp
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM, AutoModel,
    pipeline, BertTokenizer, BertModel,
    GPT2Tokenizer, GPT2LMHeadModel
)
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
from anthropic import Anthropic
import spacy
import yfinance as yf
import feedparser
import asyncpg
import redis.asyncio as aioredis
from bs4 import BeautifulSoup
import re
import time

logger = logging.getLogger(__name__)


@dataclass
class MarketNews:
    """Represents a market news item"""
    title: str
    content: str
    source: str
    timestamp: datetime
    url: str
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    impact_assessment: Optional[str] = None


@dataclass
class SentimentAnalysis:
    """Sentiment analysis results"""
    overall_score: float  # -1 to 1
    confidence: float     # 0 to 1
    emotion_scores: Dict[str, float]
    key_phrases: List[str]
    entity_sentiment: Dict[str, float]
    market_impact: str    # "positive", "negative", "neutral"


@dataclass
class MarketIntelligence:
    """Comprehensive market intelligence report"""
    timestamp: datetime
    overall_sentiment: float
    trend_analysis: Dict[str, Any]
    sector_analysis: Dict[str, float]
    risk_assessment: Dict[str, Any]
    trading_signals: List[Dict[str, Any]]
    narrative_summary: str
    confidence_level: float
    data_sources: List[str]


class LLMMarketAnalyzer(ABC):
    """Abstract base class for LLM-based market analysis"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.model = None
        self.tokenizer = None
        
    @abstractmethod
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text and return insights"""
        pass
    
    @abstractmethod
    async def generate_market_summary(self, news_items: List[MarketNews]) -> str:
        """Generate market summary from news items"""
        pass
    
    @abstractmethod
    async def assess_market_impact(self, text: str, context: Dict[str, Any]) -> float:
        """Assess market impact of given text"""
        pass


class BERTSentimentAnalyzer(LLMMarketAnalyzer):
    """BERT-based sentiment analyzer for financial text"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("BERT_Sentiment", config)
        self.model_name = config.get('model_name', 'ProsusAI/finbert')
        self.max_length = config.get('max_length', 512)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load BERT model for financial sentiment analysis"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info(f"Loaded BERT model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            raise
    
    async def analyze_text(self, text: str) -> SentimentAnalysis:
        """Analyze sentiment of financial text"""
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Get sentiment prediction
            result = self.sentiment_pipeline(cleaned_text)
            
            # Extract entities and key phrases
            entities = self._extract_entities(cleaned_text)
            key_phrases = self._extract_key_phrases(cleaned_text)
            
            # Convert to standardized format
            sentiment_score = self._convert_sentiment_score(result[0])
            confidence = result[0]['score']
            
            # Analyze emotions (simplified)
            emotion_scores = self._analyze_emotions(cleaned_text)
            
            # Entity-level sentiment
            entity_sentiment = {}
            for entity in entities:
                entity_context = self._extract_entity_context(cleaned_text, entity)
                if entity_context:
                    entity_result = self.sentiment_pipeline(entity_context)
                    entity_sentiment[entity] = self._convert_sentiment_score(entity_result[0])
            
            # Market impact assessment
            market_impact = self._assess_market_impact(sentiment_score, confidence)
            
            return SentimentAnalysis(
                overall_score=sentiment_score,
                confidence=confidence,
                emotion_scores=emotion_scores,
                key_phrases=key_phrases,
                entity_sentiment=entity_sentiment,
                market_impact=market_impact
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return SentimentAnalysis(
                overall_score=0.0,
                confidence=0.0,
                emotion_scores={},
                key_phrases=[],
                entity_sentiment={},
                market_impact="neutral"
            )
    
    async def generate_market_summary(self, news_items: List[MarketNews]) -> str:
        """Generate market summary from news items"""
        if not news_items:
            return "No news items available for analysis."
        
        # Analyze sentiment for each news item
        sentiments = []
        for news in news_items:
            sentiment = await self.analyze_text(news.content)
            sentiments.append(sentiment.overall_score)
        
        avg_sentiment = np.mean(sentiments)
        sentiment_std = np.std(sentiments)
        
        # Generate summary
        if avg_sentiment > 0.1:
            tone = "positive"
        elif avg_sentiment < -0.1:
            tone = "negative"
        else:
            tone = "neutral"
        
        summary = f"""
        Market Sentiment Analysis Summary:
        
        Overall Sentiment: {tone.capitalize()} (Score: {avg_sentiment:.3f})
        Sentiment Volatility: {sentiment_std:.3f}
        News Items Analyzed: {len(news_items)}
        
        Key Insights:
        - Average sentiment score suggests {tone} market mood
        - Sentiment volatility of {sentiment_std:.3f} indicates {'high' if sentiment_std > 0.3 else 'moderate' if sentiment_std > 0.15 else 'low'} disagreement
        - Most impactful news: {max(news_items, key=lambda x: abs(x.sentiment_score or 0)).title if news_items else 'None'}
        """
        
        return summary.strip()
    
    async def assess_market_impact(self, text: str, context: Dict[str, Any]) -> float:
        """Assess market impact of given text"""
        sentiment = await self.analyze_text(text)
        
        # Base impact from sentiment
        base_impact = abs(sentiment.overall_score) * sentiment.confidence
        
        # Adjust for context factors
        multipliers = {
            'fed_announcement': 2.0,
            'earnings_release': 1.5,
            'geopolitical_event': 1.8,
            'economic_data': 1.3,
            'company_news': 1.0
        }
        
        event_type = context.get('event_type', 'general')
        multiplier = multipliers.get(event_type, 1.0)
        
        # Adjust for market conditions
        market_volatility = context.get('market_volatility', 0.02)
        volatility_factor = min(2.0, 1.0 + market_volatility * 10)
        
        impact_score = base_impact * multiplier * volatility_factor
        return min(1.0, impact_score)
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate if too long
        if len(text) > self.max_length * 4:  # Rough character to token ratio
            text = text[:self.max_length * 4]
        
        return text
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        # Simple entity extraction (can be enhanced with spaCy)
        entities = []
        
        # Look for ticker symbols
        tickers = re.findall(r'\b[A-Z]{1,5}\b', text)
        entities.extend(tickers)
        
        # Look for company names (simplified)
        companies = re.findall(r'\b[A-Z][a-z]+ (?:Inc|Corp|Ltd|Co|Corporation|Company)\b', text)
        entities.extend(companies)
        
        return list(set(entities))
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple keyword extraction
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
            'bullish', 'bearish', 'rally', 'crash', 'volatility', 'risk',
            'buy', 'sell', 'hold', 'upgrade', 'downgrade', 'target price'
        ]
        
        found_phrases = []
        text_lower = text.lower()
        
        for keyword in financial_keywords:
            if keyword in text_lower:
                found_phrases.append(keyword)
        
        return found_phrases
    
    def _convert_sentiment_score(self, result: Dict[str, Any]) -> float:
        """Convert model output to standardized sentiment score"""
        label = result['label'].upper()
        score = result['score']
        
        if 'POSITIVE' in label:
            return score
        elif 'NEGATIVE' in label:
            return -score
        else:
            return 0.0
    
    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyze emotional content (simplified)"""
        emotions = {
            'fear': len(re.findall(r'\b(fear|afraid|worry|panic|anxious)\b', text.lower())) / 100,
            'greed': len(re.findall(r'\b(bull|rally|boom|surge|soar)\b', text.lower())) / 100,
            'confidence': len(re.findall(r'\b(confident|strong|solid|robust)\b', text.lower())) / 100,
            'uncertainty': len(re.findall(r'\b(uncertain|unclear|volatile|risky)\b', text.lower())) / 100
        }
        
        # Normalize
        max_score = max(emotions.values()) if emotions.values() else 1
        if max_score > 0:
            emotions = {k: min(1.0, v / max_score) for k, v in emotions.items()}
        
        return emotions
    
    def _extract_entity_context(self, text: str, entity: str) -> str:
        """Extract context around an entity"""
        # Find sentences containing the entity
        sentences = text.split('.')
        context_sentences = [s for s in sentences if entity.lower() in s.lower()]
        return '. '.join(context_sentences[:2])  # First two relevant sentences
    
    def _assess_market_impact(self, sentiment_score: float, confidence: float) -> str:
        """Assess market impact based on sentiment"""
        impact_magnitude = abs(sentiment_score) * confidence
        
        if impact_magnitude > 0.7:
            return "positive" if sentiment_score > 0 else "negative"
        elif impact_magnitude > 0.3:
            return "positive" if sentiment_score > 0 else "negative"
        else:
            return "neutral"


class GPTMarketAnalyzer(LLMMarketAnalyzer):
    """GPT-based market analyzer for comprehensive analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("GPT_Market", config)
        self.api_key = config.get('openai_api_key')
        self.model_name = config.get('model_name', 'gpt-3.5-turbo')
        self.max_tokens = config.get('max_tokens', 1000)
        
        if self.api_key:
            openai.api_key = self.api_key
        
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text using GPT"""
        try:
            prompt = f"""
            Analyze the following financial text and provide insights:
            
            Text: {text}
            
            Please provide:
            1. Sentiment score (-1 to 1)
            2. Market impact assessment (Low/Medium/High)
            3. Key themes and topics
            4. Potential trading implications
            5. Risk factors mentioned
            
            Format as JSON.
            """
            
            if self.api_key:
                response = await openai.ChatCompletion.acreate(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=0.1
                )
                
                result = response['choices'][0]['message']['content']
                return json.loads(result)
            else:
                # Fallback analysis
                return await self._fallback_analysis(text)
                
        except Exception as e:
            logger.error(f"Error in GPT analysis: {e}")
            return await self._fallback_analysis(text)
    
    async def generate_market_summary(self, news_items: List[MarketNews]) -> str:
        """Generate comprehensive market summary"""
        try:
            # Prepare news summary
            news_summary = "\n".join([
                f"- {news.title}: {news.content[:200]}..."
                for news in news_items[:10]  # Limit to prevent token overflow
            ])
            
            prompt = f"""
            Based on the following recent market news, provide a comprehensive market analysis:
            
            {news_summary}
            
            Please include:
            1. Overall market sentiment
            2. Key themes and trends
            3. Sector-specific insights
            4. Risk factors
            5. Trading opportunities
            6. Market outlook
            
            Keep the analysis concise but insightful.
            """
            
            if self.api_key:
                response = await openai.ChatCompletion.acreate(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=0.3
                )
                
                return response['choices'][0]['message']['content']
            else:
                return await self._generate_fallback_summary(news_items)
                
        except Exception as e:
            logger.error(f"Error generating GPT summary: {e}")
            return await self._generate_fallback_summary(news_items)
    
    async def assess_market_impact(self, text: str, context: Dict[str, Any]) -> float:
        """Assess market impact using GPT"""
        try:
            prompt = f"""
            Assess the market impact of the following information:
            
            Text: {text}
            Context: {json.dumps(context)}
            
            Provide a market impact score from 0.0 (no impact) to 1.0 (maximum impact).
            Consider factors like:
            - Magnitude of the news
            - Market relevance
            - Timing and context
            - Historical precedents
            
            Return only the numerical score.
            """
            
            if self.api_key:
                response = await openai.ChatCompletion.acreate(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.1
                )
                
                result = response['choices'][0]['message']['content'].strip()
                return min(1.0, max(0.0, float(result)))
            else:
                return await self._fallback_impact_assessment(text, context)
                
        except Exception as e:
            logger.error(f"Error in GPT impact assessment: {e}")
            return await self._fallback_impact_assessment(text, context)
    
    async def _fallback_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback analysis when GPT is not available"""
        # Simple keyword-based analysis
        positive_words = ['growth', 'profit', 'bullish', 'rally', 'gain', 'rise']
        negative_words = ['loss', 'bearish', 'crash', 'decline', 'fall', 'risk']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        sentiment = (pos_count - neg_count) / max(total_words, 1)
        
        return {
            'sentiment_score': max(-1, min(1, sentiment)),
            'market_impact': 'Medium' if abs(sentiment) > 0.01 else 'Low',
            'key_themes': positive_words + negative_words,
            'trading_implications': 'Monitor for trading opportunities',
            'risk_factors': 'General market risks apply'
        }
    
    async def _generate_fallback_summary(self, news_items: List[MarketNews]) -> str:
        """Generate fallback summary"""
        if not news_items:
            return "No news items available for analysis."
        
        summary = f"""
        Market Summary (Fallback Analysis):
        
        Total news items analyzed: {len(news_items)}
        Time period: {min(item.timestamp for item in news_items)} to {max(item.timestamp for item in news_items)}
        
        Key sources: {', '.join(set(item.source for item in news_items[:5]))}
        
        Note: This is a simplified analysis. Full GPT analysis requires API key configuration.
        """
        
        return summary.strip()
    
    async def _fallback_impact_assessment(self, text: str, context: Dict[str, Any]) -> float:
        """Fallback impact assessment"""
        # Simple heuristic based on text length and context
        base_impact = min(0.5, len(text) / 1000)
        
        # Adjust for context
        event_type = context.get('event_type', '')
        if 'fed' in event_type.lower():
            base_impact *= 1.5
        elif 'earnings' in event_type.lower():
            base_impact *= 1.2
        
        return min(1.0, base_impact)


class NewsProcessor:
    """Processes news from multiple sources for market analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sources = config.get('news_sources', [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://www.cnbc.com/id/100003114/device/rss/rss.html'
        ])
        self.redis_url = config.get('redis_url', 'redis://localhost:6379')
        self.cache_ttl = config.get('cache_ttl', 3600)  # 1 hour
        
    async def fetch_news(self, limit: int = 100) -> List[MarketNews]:
        """Fetch news from configured sources"""
        news_items = []
        
        async with aiohttp.ClientSession() as session:
            for source in self.sources:
                try:
                    news = await self._fetch_from_source(session, source, limit)
                    news_items.extend(news)
                except Exception as e:
                    logger.error(f"Error fetching from {source}: {e}")
        
        # Sort by timestamp and limit
        news_items.sort(key=lambda x: x.timestamp, reverse=True)
        return news_items[:limit]
    
    async def _fetch_from_source(self, session: aiohttp.ClientSession, 
                                source: str, limit: int) -> List[MarketNews]:
        """Fetch news from a single RSS source"""
        try:
            async with session.get(source) as response:
                content = await response.text()
                feed = feedparser.parse(content)
                
                news_items = []
                for entry in feed.entries[:limit]:
                    news_item = MarketNews(
                        title=entry.title,
                        content=entry.summary if hasattr(entry, 'summary') else entry.title,
                        source=feed.feed.title if hasattr(feed.feed, 'title') else 'Unknown',
                        timestamp=datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now(),
                        url=entry.link if hasattr(entry, 'link') else '',
                    )
                    news_items.append(news_item)
                
                return news_items
                
        except Exception as e:
            logger.error(f"Error parsing RSS feed {source}: {e}")
            return []
    
    async def enrich_news_with_sentiment(self, news_items: List[MarketNews], 
                                       analyzer: LLMMarketAnalyzer) -> List[MarketNews]:
        """Enrich news items with sentiment analysis"""
        for news in news_items:
            try:
                if isinstance(analyzer, BERTSentimentAnalyzer):
                    sentiment = await analyzer.analyze_text(news.content)
                    news.sentiment_score = sentiment.overall_score
                elif isinstance(analyzer, GPTMarketAnalyzer):
                    analysis = await analyzer.analyze_text(news.content)
                    news.sentiment_score = analysis.get('sentiment_score', 0.0)
            except Exception as e:
                logger.error(f"Error analyzing sentiment for {news.title}: {e}")
                news.sentiment_score = 0.0
        
        return news_items


class SentimentAnalyzer:
    """Main sentiment analyzer that combines multiple LLM approaches"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analyzers = {}
        
        # Initialize available analyzers
        if config.get('use_bert', True):
            self.analyzers['bert'] = BERTSentimentAnalyzer(config.get('bert_config', {}))
        
        if config.get('use_gpt', False) and config.get('openai_api_key'):
            self.analyzers['gpt'] = GPTMarketAnalyzer(config.get('gpt_config', {}))
        
        self.news_processor = NewsProcessor(config.get('news_config', {}))
    
    async def analyze_market_sentiment(self, 
                                     symbols: Optional[List[str]] = None) -> MarketIntelligence:
        """Analyze overall market sentiment"""
        try:
            # Fetch recent news
            news_items = await self.news_processor.fetch_news(limit=50)
            
            # Analyze with available analyzers
            sentiment_scores = []
            sector_sentiment = {}
            trading_signals = []
            
            for analyzer_name, analyzer in self.analyzers.items():
                try:
                    # Enrich news with sentiment
                    enriched_news = await self.news_processor.enrich_news_with_sentiment(
                        news_items, analyzer
                    )
                    
                    # Calculate overall sentiment
                    scores = [news.sentiment_score for news in enriched_news if news.sentiment_score is not None]
                    if scores:
                        sentiment_scores.append(np.mean(scores))
                    
                    # Generate summary
                    summary = await analyzer.generate_market_summary(enriched_news)
                    
                except Exception as e:
                    logger.error(f"Error with analyzer {analyzer_name}: {e}")
            
            # Aggregate results
            overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            confidence = min(0.9, len(sentiment_scores) / len(self.analyzers)) if self.analyzers else 0.1
            
            # Generate trading signals based on sentiment
            if overall_sentiment > 0.2 and confidence > 0.7:
                trading_signals.append({
                    'signal': 'bullish',
                    'strength': min(1.0, overall_sentiment * confidence),
                    'timeframe': 'short-term'
                })
            elif overall_sentiment < -0.2 and confidence > 0.7:
                trading_signals.append({
                    'signal': 'bearish',
                    'strength': min(1.0, abs(overall_sentiment) * confidence),
                    'timeframe': 'short-term'
                })
            
            # Create market intelligence report
            intelligence = MarketIntelligence(
                timestamp=datetime.now(),
                overall_sentiment=overall_sentiment,
                trend_analysis={'direction': 'up' if overall_sentiment > 0 else 'down'},
                sector_analysis=sector_sentiment,
                risk_assessment={'level': 'medium', 'factors': ['general market risk']},
                trading_signals=trading_signals,
                narrative_summary=f"Market sentiment is {'positive' if overall_sentiment > 0 else 'negative'} with confidence {confidence:.2f}",
                confidence_level=confidence,
                data_sources=[analyzer.name for analyzer in self.analyzers.values()]
            )
            
            return intelligence
            
        except Exception as e:
            logger.error(f"Error in market sentiment analysis: {e}")
            return MarketIntelligence(
                timestamp=datetime.now(),
                overall_sentiment=0.0,
                trend_analysis={},
                sector_analysis={},
                risk_assessment={},
                trading_signals=[],
                narrative_summary="Error in sentiment analysis",
                confidence_level=0.0,
                data_sources=[]
            )


# Utility functions and examples
async def demo_llm_integration():
    """Demonstration of LLM integration capabilities"""
    logger.info("Starting LLM integration demo")
    
    # Configuration
    config = {
        'bert_config': {
            'model_name': 'ProsusAI/finbert',
            'max_length': 512
        },
        'gpt_config': {
            'model_name': 'gpt-3.5-turbo',
            'max_tokens': 1000,
            # 'openai_api_key': 'your-api-key-here'  # Uncomment to enable GPT
        },
        'news_config': {
            'news_sources': [
                'https://feeds.finance.yahoo.com/rss/2.0/headline'
            ]
        },
        'use_bert': True,
        'use_gpt': False  # Set to True if you have API key
    }
    
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer(config)
    
    # Analyze market sentiment
    intelligence = await sentiment_analyzer.analyze_market_sentiment()
    
    logger.info(f"Market Intelligence Report:")
    logger.info(f"Overall Sentiment: {intelligence.overall_sentiment:.3f}")
    logger.info(f"Confidence: {intelligence.confidence_level:.3f}")
    logger.info(f"Trading Signals: {len(intelligence.trading_signals)}")
    logger.info(f"Narrative: {intelligence.narrative_summary}")
    
    # Test individual components
    bert_analyzer = BERTSentimentAnalyzer(config['bert_config'])
    
    sample_text = """
    Apple Inc. reported strong quarterly earnings today, beating analyst expectations 
    by 10%. The company's iPhone sales exceeded forecasts, driving revenue growth 
    of 15% year-over-year. CEO Tim Cook expressed optimism about future prospects 
    despite ongoing supply chain challenges.
    """
    
    sentiment = await bert_analyzer.analyze_text(sample_text)
    logger.info(f"Sample Text Sentiment: {sentiment.overall_score:.3f} (confidence: {sentiment.confidence:.3f})")
    logger.info(f"Key Phrases: {sentiment.key_phrases}")
    logger.info(f"Market Impact: {sentiment.market_impact}")
    
    logger.info("LLM integration demo completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_llm_integration())