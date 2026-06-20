import sys
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server.server import (
    _compute_rsi_from_series,
    _compute_macd_from_series,
    _compute_bollinger_from_series,
    _detect_support_resistance
)

def test_rsi_insufficient_data():
    prices = [10.0, 11.0, 12.0]
    assert _compute_rsi_from_series(prices, period=14) == 50.0

def test_rsi_increasing_trend():
    # 20 periods of increasing prices
    prices = [float(x) for x in range(100, 121)]
    rsi = _compute_rsi_from_series(prices, period=14)
    assert rsi > 70.0  # Should be heavily overbought

def test_rsi_decreasing_trend():
    # 20 periods of decreasing prices
    prices = [float(x) for x in range(120, 99, -1)]
    rsi = _compute_rsi_from_series(prices, period=14)
    assert rsi < 30.0  # Should be heavily oversold

def test_macd_insufficient_data():
    prices = [10.0] * 10
    macd_res = _compute_macd_from_series(prices, fast=12, slow=26, signal_period=9)
    assert macd_res["bias"] == "insufficient_data"

def test_macd_calculation():
    # Construct a series long enough (slow + signal = 35)
    prices = [float(100 + x * 0.5) for x in range(50)]
    macd_res = _compute_macd_from_series(prices)
    assert "macd" in macd_res
    assert "signal" in macd_res
    assert "histogram" in macd_res
    assert macd_res["bias"] in ["bullish", "bearish"]

def test_bollinger_insufficient_data():
    prices = [10.0] * 5
    res = _compute_bollinger_from_series(prices, period=20)
    assert res["upper"] == 0
    assert res["lower"] == 0

def test_bollinger_bands():
    # 30 periods of flat/constant prices
    prices = [100.0] * 30
    # Add minor noise so standard deviation is not exactly zero (avoid divide by zero warnings)
    prices[-1] = 100.1
    res = _compute_bollinger_from_series(prices, period=20)
    assert res["mid"] == pytest.approx(100.005, abs=0.01)
    assert res["upper"] > res["lower"]
    assert res["zone"] == "overbought"

def test_support_resistance():
    # Test support and resistance levels detection
    prices = [100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 99.0, 100.0] * 10
    res = _detect_support_resistance(prices, lookback=50)
    assert "support" in res
    assert "resistance" in res
    assert res["support"] > 0
    assert res["resistance"] > 0
    assert res["support"] <= prices[-1]
    assert res["resistance"] >= prices[-1]
