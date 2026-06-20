# Alpha-Lens 🔬

> **A self-reasoning quantitative trading agent** — it reads market data, computes technical indicators, and delivers a clear Bullish / Bearish / Neutral signal, showing you its full reasoning chain in real time.

---

## 🎯 Purpose

Alpha-Lens was built to answer one question:

> *"Can an LLM reason about financial data the way a quant analyst does — calling tools, weighing evidence, and issuing a conviction signal — without a human in the loop?"*

The answer is **yes**, and this project is the proof-of-concept. It is designed as a local-first, privacy-respecting agent that:

- Reads **real historical Gold (XAUUSD) OHLCV data** from a local CSV — no paid data feeds required.
- Uses a **Gemini LLM** as the reasoning brain (swappable with a fine-tuned local Llama-3 QLoRA model).
- Exposes **6 technical analysis tools** through the **Model Context Protocol (MCP)** so the LLM can call them on demand.
- Streams its **full thought process** to a browser dashboard in real time via Server-Sent Events.

---

## ⚙️ Core Functions
-----------------------------------------------------------------------------------------------------------------------------------------------------
|       Function            |                                          What it does                                                                 |
|---------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **Market Data Retrieval** | Fetches **live market data** for any symbol (AAPL, TSLA, BTC-USD) via `yfinance`, with a local CSV fallback for Gold. |
| **Backtesting Mode**      | Allows "Time-Travel" through historical CSVs to evaluate agent performance over past market cycles.                   |
| **Momentum Analysis**     | Computes RSI (Relative Strength Index) to detect overbought/oversold conditions                                       |
| **Trend Analysis**        | Computes MACD and its histogram to identify trend direction and strength                                              |
| **Volatility Analysis**   | Calculates Bollinger Bands — flags squeeze setups (potential breakouts)                                               |
| **Level Detection**       | Detects key Support & Resistance levels from swing highs/lows                                                         |
| **Order Book Simulation** | Simulates Level 2 bid/ask depth to reveal immediate supply/demand imbalance                                           |
| **Signal Generation**     | Synthesizes all tool outputs into a single **Bullish / Bearish / Neutral** verdict                                    |
| **Streaming Reasoning**   | Broadcasts the agent's live ReAct reasoning chain to the browser token-by-token                                       |
-----------------------------------------------------------------------------------------------------------------------------------------------------
---

## 🔄 Workflow

This is how a single analysis request flows through the system, end to end:

```
┌─────────────────────────────────────────────────────────────┐
│  1. USER INPUT                                              │
│     User enters a symbol (e.g. GOLD) and a query            │
│     (e.g. "detect momentum breakout") in the browser UI.    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP POST /analyze/stream
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  2. FastAPI (api/main.py)                                   │
│     Receives the request and opens a Server-Sent Events     │
│     (SSE) stream back to the browser. Hands the query to    │
│     the AlphaLensAgent.                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  3. ReAct Agent Loop (agent/runner.py)                      │
│                                                             │
│   ┌─ OBSERVE ──────────────────────────────────────────┐    │
│   │  Agent sends query + system prompt to Gemini LLM.  │    │
│   │  LLM reasons about what data it needs.             │    │
│   └──────────────────────┬─────────────────────────────┘    │
│                          │ LLM emits <tool_call> tags       │
│   ┌─ CALL TOOLS ─────────▼─────────────────────────────┐    │
│   │  Runner intercepts tags, dispatches each call to   │    │
│   │  the MCP Server subprocess via the MCP SDK.        │    │
│   │  Results are streamed live to the browser.         │    │
│   └──────────────────────┬─────────────────────────────┘    │
│                          │ tool results fed back to LLM     │
│   ┌─ SYNTHESIZE ─────────▼─────────────────────────────┐    │
│   │  LLM reads all tool results, forms a conclusion,   │    │
│   │  and emits a final SIGNAL: Bullish/Bearish/Neutral │    │
│   └────────────────────────────────────────────────────┘    │
│                                                             │
│   Loop repeats up to 5 times if the LLM needs more data.    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  4. MCP Server (mcp_server/server.py)                       │
│     A persistent subprocess that implements the 6 tools.    │
│     Reads from the local CSV, computes indicators using     │
│     pure Python math, returns structured JSON results.      │
└────────────────────────┬────────────────────────────────────┘
                         │ SSE chunks streamed token-by-token
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Browser Dashboard (ui/index.html)                       │
│     Renders the reasoning chain live: text → tool badges    │
│     → JSON results → final colour-coded SIGNAL. Logs all    │
│     signals in the sidebar with timestamps.                 │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start — Running the Project

### Prerequisites
- **Python 3.10+** (3.13 works fine — see `problem&sol.txt` for version notes)
- A **Google Gemini API key** — get one free at [aistudio.google.com](https://aistudio.google.com)

---

### Step 1 — Clone & Install Dependencies

```bash
git clone https://github.com/your-username/alpha-lens.git
cd alpha-lens

pip install -r requirements.txt
```

> **Windows users:** If you get version-constraint warnings during install, use:
> ```bash
> pip install -r requirements.txt --ignore-requires-python
> ```

---

### Step 2 — Configure Environment

Create your local `.env` file (it is git-ignored and never pushed):

```bash
# Copy the template
copy .env.example .env      # Windows
cp .env.example .env        # Mac/Linux
```

Then open `.env` and fill in your API key:

```env
GOOGLE_API_KEY=AIza...your_key_here...
```

---

### Step 3 — Run the API Server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

### Step 4 — Open the Dashboard

Open your browser and go to:

```
http://localhost:8000
```

Enter a **symbol** (e.g. `GOLD`) and a **query** (e.g. `detect momentum breakout`), then click **ANALYZE**.

---

### Step 5 — (Optional) Debug the Agent Directly

You can run the agent standalone without the web UI:

```bash
python agent/runner.py
```

---

## 🛠️ Available MCP Tools
---------------------------------------------------------------------------------------------------------
|           Tool              |                      Description                                        |
|-----------------------------|-------------------------------------------------------------------------|
| `get_market_data`           | Fetch current price and volume for **ANY** symbol (AAPL, BTC-USD, GOLD) |
| `compute_rsi`               | RSI for any symbol — overbought >70, oversold <30                       |
| `compute_macd`              | MACD trend indicator for any symbol                                     |
| `compute_bollinger`         | Bollinger Bands for any symbol                                          |
| `detect_support_resistance` | Support/Resistance levels for any symbol                                |
| `get_order_book`            | Simulated Level 2 bid/ask imbalance                                     |
---------------------------------------------------------------------------------------------------------
---

## 🧪 High-Fidelity Backtesting

Alpha-Lens includes a dedicated backtesting engine that simulates historical market conditions to test the agent's accuracy.

### How it works:
1.  **Time-Travel**: The MCP server loads a CSV and freezes "Virtual Time" at a specific row.
2.  **Blind Analysis**: The Agent is triggered. It calls tools (RSI, MACD, etc.) which only "see" data up to that virtual timestamp.
3.  **Simulation Loop**: The `backtester.py` script moves the virtual clock forward step-by-step.
4.  **Win-Rate Evaluation**: The engine looks at the price 10 hours *after* each signal to determine if the agent was correct.

### Run a Backtest:
```bash
python agent/backtester.py
```

---

## 🔍 Performance Audit Engine

For a more detailed analysis, use the Audit engine. It provides a "Performance Certificate" for any asset by testing it against historical data using a 40% randomized sample.

### Features:
- **Signal Distribution**: Analyzes bias (how often the agent buys vs. sells).
- **Risk Analysis**: Calculates maximum consecutive losses (drawdown streaks).
- **Transparency**: Shows specific examples of where the agent was right and where it was wrong.
- **Persistence**: Automatically saves detailed results to the `/audits/` folder.

### 📊 Glossary of Metrics:
To prevent confusion when interpreting the Performance Certificate:
- **Audit Win Rate**: `Wins / Total Samples Processed`. (Includes "Neutral" signals as part of the denominator).
- **Trade Accuracy**: `Wins / (Bullish + Bearish Signals)`. (Excludes "Neutral" signals; measures hit rate of active trades).
- **Strategy Expectancy**: `(Win% × Avg Win) - (Loss% × Avg Loss)`. The average dollar value you expect to make per signal over thousands of trades.
- **Avg. Profit / Signal**: `Total PnL / Total Samples`.

*The engine uses a randomized 40% sample (capped at 5 for quick testing) to conserve API quota.*

---

## 🏗️ Architecture
--------------------------------------------------------------------------------------------------------------
| Component      |            File                 |                   Purpose                               |
|----------------|---------------------------------|---------------------------------------------------------|
| 🧠 Agent      | `agent/runner.py`               | ReAct loop — Observe → Hypothesize → Call Tools → Signal |
| 🤝 MCP Server | `mcp_server/server.py`          | Tool provider via MCP Python SDK                         |
| 🌐 API        | `api/main.py`                   | FastAPI + SSE streaming to browser                       |
| 🖥️ UI         | `ui/index.html`                 | Terminal-aesthetic quant dashboard                       |
| 📊 Data       | `data/gold_data_1h_cleaned.csv` | Real hourly Gold OHLCV 2024–2026                         |
--------------------------------------------------------------------------------------------------------------
---

## Why This Architecture?

**MCP over direct function calls** — Tools are decoupled from the agent. Swap mock data for live broker feeds without touching the LLM code. The MCP protocol is also compatible with Claude Desktop and other clients.

**ReAct loop** — The agent can call `compute_rsi`, then decide it also needs `detect_support_resistance` — iterative reasoning, just like a human analyst.

**SSE streaming** — The UI receives token-by-token reasoning in real time, giving full transparency into the agent's thought process.

---

## File Tree

```
alpha-lens/
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD pipeline (lint → test → Docker build)
├── mcp_server/
│   ├── __init__.py
│   └── server.py               # MCP tool server (RSI, MACD, Bollinger, S/R, order book)
├── agent/
│   ├── runner.py               # ReAct agent loop
│   ├── backtester.py           # Time-travel backtester with Sharpe/Sortino/MDD
│   └── audit.py                # Performance Certificate generator
├── api/
│   └── main.py                 # FastAPI + SSE endpoints
├── ui/
│   └── index.html              # Single-file terminal dashboard + Paper Trading
├── tests/
│   ├── test_mcp_tools.py       # Unit tests for quant indicator math
│   └── test_api.py             # Integration tests for API endpoints
├── scripts/
│   └── finetune.py             # QLoRA fine-tuning pipeline (GPU required)
├── data/
│   └── gold_data_1h_cleaned.csv
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Local orchestration
├── .dockerignore
├── deploy_gcp.md               # Google Cloud Run deployment handbook
├── requirements.txt
├── .env.example                # Copy this → .env and add your API key
├── .gitignore
└── README.md
```

---

## 🐳 Docker Deployment

Run Alpha-Lens in a container with a single command:

```bash
# Build the image
docker build -t alpha-lens .

# Run with your API key
docker run -p 8000:8000 --env-file .env alpha-lens

# Or use Docker Compose
docker-compose up --build
```

For **Google Cloud Run** deployment, see [deploy_gcp.md](deploy_gcp.md) for a complete step-by-step guide covering Artifact Registry, Cloud Build, Secret Manager, and Cloud Run.

---

## 🔄 CI/CD Pipeline

Every push to `main` triggers the GitHub Actions workflow (`.github/workflows/deploy.yml`):

1. **Lint** — `flake8` catches syntax errors and undefined names
2. **Test** — `pytest` runs the full unit + integration test suite
3. **Build** — Validates the Docker image compiles successfully

---

## 💹 Paper Trading Simulator

The dashboard includes a built-in paper trading simulator in the right sidebar:

- **Virtual $10,000 balance** — starts fresh, persisted across page reloads via `localStorage`
- **Signal-driven trading** — When the agent emits a Bullish/Bearish signal, a trade execution button appears
- **Position tracking** — Shows entry price, side (LONG/SHORT), and live unrealized PnL
- **Close positions** — Manually close positions and see the realized profit/loss reflected in your balance

---

## 📈 Advanced Quantitative Metrics

The backtesting and audit engine reports professional-grade financial metrics:

| Metric | Description |
|--------|-------------|
| **Sharpe Ratio** | Annualized risk-adjusted return (higher = better) |
| **Sortino Ratio** | Like Sharpe but penalizes only downside risk |
| **Max Drawdown** | Largest peak-to-trough decline (lower = safer) |
| **Strategy Expectancy** | Average $ expected per signal over time |
| **Equity Curve** | Interactive Chart.js visualization in the Audit modal |

---

## 🧪 Testing

```bash
# Run the full test suite
python -m pytest

# With verbose output
python -m pytest -v
```

**11 tests** covering:
- RSI boundary conditions (overbought/oversold/insufficient data)
- MACD trend direction and data requirements
- Bollinger Band squeeze detection
- Support/Resistance swing-high/low logic
- FastAPI endpoint health, tool listing, and UI serving

---

## 🔑 API Quota Notes

The free tier of `gemini-2.5-flash` allows **~20 requests/day**. If you hit a `429 RESOURCE_EXHAUSTED` error:
- Wait for the daily quota to reset (midnight Pacific time)
- Or upgrade to a paid Gemini API tier

See `problem&sol.txt` (git-ignored) for a full log of issues encountered and how they were resolved.

---

## Fine-Tuning (GPU Required)

```bash
# Requires ~12GB VRAM (RTX 3090 / A10G or better)
python scripts/finetune.py
# Saves LoRA adapter to ./checkpoints/alpha-lens-llama3/
```

