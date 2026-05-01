"""
Alpha-Lens | api/main.py
=========================
FastAPI backend — bridges the UI (index.html) and the agent runner.
Uses Server-Sent Events (SSE) for streaming token-by-token output to the browser.
WHY SSE over WebSockets: SSE is unidirectional (server→client), simpler for
streaming text, and works over standard HTTP/2.
"""

import asyncio
import json
from typing import Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from agent.runner import AlphaLensAgent
from agent.audit import perform_full_audit

app = FastAPI(title="Alpha-Lens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single agent instance (stateful conversation history)
agent = AlphaLensAgent(use_local_model=False)


class AnalysisRequest(BaseModel):
    query: str
    symbol: str | None = None


@app.get("/")
async def serve_ui():
    return FileResponse("ui/index.html")


@app.post("/analyze/stream")
async def analyze_stream(request: AnalysisRequest):
    """
    SSE endpoint — streams agent reasoning token by token.
    Frontend connects with EventSource and appends each chunk to the UI.
    """
    query = request.query
    if request.symbol:
        query = f"[Symbol: {request.symbol}] {query}"

    async def event_generator():
        try:
            async for chunk in agent.analyze(query):
                # SSE format: "data: <payload>\n\n"
                data = json.dumps({"type": "token", "content": chunk})
                yield f"data: {data}\n\n"
                await asyncio.sleep(0)  # Yield control for true streaming
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/tools")
async def list_available_tools():
    """Returns the list of MCP tools the agent can call."""
    return {
        "tools": [
            {"name": "get_market_data",            "description": "OHLCV price snapshot from real dataset"},
            {"name": "compute_rsi",               "description": "RSI momentum indicator (>70 overbought, <30 oversold)"},
            {"name": "compute_macd",              "description": "MACD trend + histogram bias"},
            {"name": "compute_bollinger",         "description": "Bollinger Bands, %B, squeeze detection"},
            {"name": "detect_support_resistance", "description": "Key S/R levels with distance %"},
            {"name": "get_order_book",            "description": "Level 2 bid/ask imbalance ratio"},
        ]
    }

class AuditRequest(BaseModel):
    symbol: str
    csv_path: Optional[str] = "data/gold_data_1h_cleaned.csv"

@app.post("/audit")
async def run_audit(request: AuditRequest):
    """Triggers a 40% randomized backtest audit for the given CSV."""
    try:
        # Default to the local gold data if no path provided
        path = request.csv_path or "data/gold_data_1h_cleaned.csv"
        report = await perform_full_audit(path)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model": "local" if agent.use_local else "api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
