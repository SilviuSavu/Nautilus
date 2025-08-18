"""
Common enums for market data infrastructure
"""

from enum import Enum


class Venue(Enum):
    """Supported trading venues"""
    BINANCE = "BINANCE"
    COINBASE = "COINBASE"
    KRAKEN = "KRAKEN"
    BITSTAMP = "BITSTAMP"
    GEMINI = "GEMINI"
    BYBIT = "BYBIT"
    KUCOIN = "KUCOIN"
    BITFINEX = "BITFINEX"
    HUOBI = "HUOBI"
    GATEIO = "GATEIO"
    OKEX = "OKEX"
    OKX = "OKX"
    BITGET = "BITGET"
    FTX = "FTX"  # Historical
    BITMEX = "BITMEX"
    HYPERLIQUID = "HYPERLIQUID"
    DATABENTO = "DATABENTO"
    INTERACTIVE_BROKERS = "INTERACTIVE_BROKERS"
    DYDX = "DYDX"
    POLYMARKET = "POLYMARKET"
    BETFAIR = "BETFAIR"


class DataType(Enum):
    """Supported data types"""
    TICK = "tick"
    QUOTE = "quote"
    BAR = "bar"
    ORDER_BOOK = "order_book"
    TRADE = "trade"
    INSTRUMENT = "instrument"
    STATUS = "status"


class MessageBusTopics(Enum):
    """MessageBus topic patterns"""
    MARKET_DATA = "data"
    TICK_DATA = "data.tick"
    QUOTE_DATA = "data.quote"
    BAR_DATA = "data.bar"
    ORDER_BOOK_DATA = "data.order_book"
    TRADE_DATA = "data.trade"
    INSTRUMENT_DATA = "data.instrument"
    STATUS_DATA = "data.status"