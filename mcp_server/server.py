"""
Alpha-Lens | mcp_server/server.py
==================================
DATA-LINKED MCP SERVER:
  This server is now linked to 'gold_data_1h_cleaned.csv'. 
  It provides real historical context from 2024-2026 instead of mock data.
"""

import json
import pandas as pd
import os
import random
from datetime import datetime
from typing import Any, Optional
import statistics
import yfinance as yf
from functools import lru_cache

# MCP Python SDK
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

# ═══════════════════════════════════════════════════════════════════════════
# SERVER INIT & DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
app = Server("alpha-lens-mcp")

# Resolve absolute path to the data folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "gold_data_1h_cleaned.csv")

# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST STATE (Global within this process)
# ═══════════════════════════════════════════════════════════════════════════
_backtest_mode = False
_backtest_data = pd.DataFrame()
_backtest_index = 0

@lru_cache(maxsize=10)
def _load_csv_data() -> pd.DataFrame:
    """Reads the local Gold CSV and prepares it for analysis."""
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    return df

def _fetch_market_data(symbol: str, n: int = 200) -> pd.DataFrame:
    """Fetches data for a symbol. Respects backtest mode if active."""
    global _backtest_mode, _backtest_data, _backtest_index
    
    if _backtest_mode and not _backtest_data.empty:
        # Return a window ENDING at _backtest_index
        start = max(0, _backtest_index - n + 1)
        return _backtest_data.iloc[start : _backtest_index + 1]

    sym = symbol.upper()
    if sym in ["GOLD", "XAUUSD"]:
        df = _load_csv_data()
        if not df.empty:
            return df.tail(n)
    
    # Fallback to yfinance for live data
    # Map GOLD to GC=F (Gold Futures) if user really wants live gold
    yf_sym = "GC=F" if sym in ["GOLD", "XAUUSD"] else sym
    try:
        ticker = yf.Ticker(yf_sym)
        # Fetch slightly more to ensure we have 'n' rows after any cleaning
        df = ticker.history(period="1mo", interval="1h")
        if df.empty:
            df = ticker.history(period="1y", interval="1d") # Fallback to daily if hourly fails
        
        if df.empty:
            return pd.DataFrame()
            
        df = df.reset_index()
        df = df.rename(columns={"Date": "DateTime", "Datetime": "DateTime"})
        return df.tail(n)
    except Exception:
        return pd.DataFrame()

def _get_recent_series(symbol: str, n: int = 100) -> list[float]:
    """Fetches the latest 'n' closing prices for the given symbol."""
    df = _fetch_market_data(symbol, n)
    if df.empty:
        return []
    return df['Close'].tolist()

def _compute_rsi_from_series(prices: list[float], period: int = 14) -> float:
    """Wilder's RSI formula for momentum analysis."""
    if len(prices) < period + 1: return 50.0
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains, losses = [max(d, 0) for d in deltas], [abs(min(d, 0)) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0: return 100.0
    return round(100 - (100 / (1 + (avg_gain / avg_loss))), 2)

def _compute_macd_from_series(prices: list[float], fast: int = 12, slow: int = 26, signal_period: int = 9) -> dict:
    """MACD = EMA(fast) - EMA(slow), with a signal line EMA(macd, signal_period)."""
    def ema(data, period):
        k = 2 / (period + 1)
        result = [data[0]]
        for p in data[1:]:
            result.append(p * k + result[-1] * (1 - k))
        return result

    if len(prices) < slow + signal_period:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0, "bias": "insufficient_data"}

    ema_fast   = ema(prices, fast)
    ema_slow   = ema(prices, slow)
    macd_line  = [f - s for f, s in zip(ema_fast[slow-1:], ema_slow[slow-1:])]
    signal_line = ema(macd_line, signal_period)
    histogram   = [m - s for m, s in zip(macd_line[signal_period-1:], signal_line[signal_period-1:])]

    last_macd = round(macd_line[-1], 4)
    last_sig  = round(signal_line[-1], 4)
    last_hist = round(histogram[-1], 4)
    bias = "bullish" if last_hist > 0 else "bearish"
    return {"macd": last_macd, "signal": last_sig, "histogram": last_hist, "bias": bias}

def _compute_bollinger_from_series(prices: list[float], period: int = 20, std_mult: float = 2.0) -> dict:
    """Bollinger Bands: mid = SMA(period), upper/lower = mid ± std_mult*stdev."""
    if len(prices) < period:
        return {"upper": 0, "mid": 0, "lower": 0, "pct_b": 0.5, "squeeze": False}
    window = prices[-period:]
    mid    = round(statistics.mean(window), 4)
    stdev  = round(statistics.stdev(window), 4)
    upper  = round(mid + std_mult * stdev, 4)
    lower  = round(mid - std_mult * stdev, 4)
    last   = prices[-1]
    pct_b  = round((last - lower) / (upper - lower), 4) if upper != lower else 0.5
    bandwidth = round((upper - lower) / mid, 4)
    squeeze   = bandwidth < 0.02  # Very tight band = potential breakout
    return {"upper": upper, "mid": mid, "lower": lower, "pct_b": pct_b,
            "bandwidth": bandwidth, "squeeze": squeeze,
            "zone": "overbought" if pct_b > 1 else "oversold" if pct_b < 0 else "within_bands"}

def _detect_support_resistance(prices: list[float], lookback: int = 50) -> dict:
    """Detect key S/R levels from recent swing highs/lows."""
    window = prices[-lookback:] if len(prices) >= lookback else prices
    if len(window) < 5:
        return {"support": 0, "resistance": 0, "distance_to_support_pct": 0, "distance_to_resistance_pct": 0}

    # Swing high = local max over 5-bar window; swing low = local min
    highs, lows = [], []
    for i in range(2, len(window) - 2):
        if window[i] == max(window[i-2:i+3]):
            highs.append(window[i])
        if window[i] == min(window[i-2:i+3]):
            lows.append(window[i])

    current = prices[-1]
    resistance = round(min([h for h in highs if h > current], default=current * 1.02), 4)
    support    = round(max([l for l in lows  if l < current], default=current * 0.98), 4)
    dist_res = round((resistance - current) / current * 100, 2)
    dist_sup = round((current - support)    / current * 100, 2)
    return {"support": support, "resistance": resistance,
            "distance_to_support_pct": dist_sup, "distance_to_resistance_pct": dist_res}

# ═══════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_market_data",
            description="Fetch current price and volume for ANY symbol (e.g. AAPL, BTC-USD, GOLD).",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol (e.g. NVDA, TSLA, GC=F)"},
                    "lookback": {"type": "integer", "description": "Number of units to look back", "default": 1}
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="compute_rsi",
            description="Compute RSI for a symbol. RSI > 70 = overbought, < 30 = oversold.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol"},
                    "period": {"type": "integer", "default": 14},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="get_order_book",
            description="Simulate Level 2 liquidity for a symbol centered around the current market price.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol"},
                    "depth": {"type": "integer", "default": 5},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="compute_macd",
            description="Compute MACD trend indicator for a symbol.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol"},
                    "fast":   {"type": "integer", "default": 12},
                    "slow":   {"type": "integer", "default": 26},
                    "signal": {"type": "integer", "default": 9},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="compute_bollinger",
            description="Compute Bollinger Bands for a symbol.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol"},
                    "period": {"type": "integer", "default": 20},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="detect_support_resistance",
            description="Detect key support and resistance levels for a symbol.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol":  {"type": "string", "description": "Ticker symbol"},
                    "lookback": {"type": "integer", "default": 50},
                },
                "required": ["symbol"],
            },
        ),
        Tool(
            name="load_backtest_file",
            description="Load a local CSV file for historical backtesting. Disables live mode.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to CSV"},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="set_backtest_index",
            description="Internal: Move the 'current time' cursor in the backtest dataset.",
            inputSchema={
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "Row index to act as 'now'"},
                },
                "required": ["index"],
            },
        ),
    ]

# ═══════════════════════════════════════════════════════════════════════════
# TOOL HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    try:
        symbol = arguments.get("symbol", "GOLD")
        df = _fetch_market_data(symbol)
        
        if df.empty:
            return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": f"No data found for symbol {symbol}"}))], isError=True)

        latest_row = df.iloc[-1]
        
        if name == "get_market_data":
            result = {
                "symbol": symbol.upper(),
                "timestamp": str(latest_row.get('DateTime', 'N/A')),
                "price": {"current": latest_row['Close'], "high": latest_row['High'], "low": latest_row['Low']},
                "volume": int(latest_row['Volume'])
            }
        
        elif name == "compute_rsi":
            prices = _get_recent_series(symbol, n=60)
            rsi = _compute_rsi_from_series(prices, arguments.get("period", 14))
            result = {
                "symbol": symbol.upper(),
                "rsi": rsi,
                "zone": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
            }

        elif name == "get_order_book":
            mid = latest_row['Close']
            spread = round(mid * 0.0002, 2)
            result = {
                "symbol": symbol.upper(),
                "mid_price": mid,
                "bids": [{"price": round(mid - spread * i, 2), "size": random.randint(100, 1000)} for i in range(1, 6)],
                "asks": [{"price": round(mid + spread * i, 2), "size": random.randint(100, 1000)} for i in range(1, 6)]
            }
        elif name == "compute_macd":
            prices = _get_recent_series(symbol, n=100)
            result = {
                "symbol": symbol.upper(),
                **_compute_macd_from_series(
                    prices,
                    fast=arguments.get("fast", 12),
                    slow=arguments.get("slow", 26),
                    signal_period=arguments.get("signal", 9)
                )
            }

        elif name == "compute_bollinger":
            prices = _get_recent_series(symbol, n=100)
            result = {
                "symbol": symbol.upper(),
                **_compute_bollinger_from_series(prices, period=arguments.get("period", 20))
            }

        elif name == "detect_support_resistance":
            prices = _get_recent_series(symbol, n=200)
            result = {
                "symbol": symbol.upper(),
                **_detect_support_resistance(prices, lookback=arguments.get("lookback", 50))
            }
        elif name == "load_backtest_file":
            global _backtest_mode, _backtest_data, _backtest_index
            path = arguments["path"]
            if os.path.exists(path):
                _backtest_data = pd.read_csv(path)
                if "DateTime" in _backtest_data.columns:
                    _backtest_data["DateTime"] = pd.to_datetime(_backtest_data["DateTime"])
                _backtest_mode = True
                _backtest_index = 0
                result = {"status": "success", "rows": len(_backtest_data), "mode": "backtest"}
            else:
                result = {"error": f"File not found: {path}"}
        elif name == "set_backtest_index":
            _backtest_index = int(arguments["index"])
            result = {"status": "success", "new_index": _backtest_index}
        else:
            result = {"error": f"Unknown tool: {name}"}

        return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, indent=2))])

    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": str(e)}))], isError=True)

# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())