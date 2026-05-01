"""
Alpha-Lens | agent/runner.py
==============================
THE NERVOUS SYSTEM:
  This module manages the ReAct loop using the modern Google GenAI SDK.
  It intercepts <tool_call> tags and dispatches them to the local MCP server.
"""

import json
import re
import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator
from google import genai  # Modern 2026 SDK
from dotenv import load_dotenv

# Load .env variables
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Setup Local Model Availability Check
try:
    from unsloth import FastLanguageModel
    import torch
    LOCAL_MODEL_AVAILABLE = True
except ImportError:
    LOCAL_MODEL_AVAILABLE = False

# Constants
CHECKPOINT_DIR = Path("./checkpoints/alpha-lens-llama3")
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

SYSTEM_PROMPT = """You are Alpha-Lens, a specialized quantitative analysis agent. 
You detect HFT patterns and momentum signals.

When you need market data, emit a tool call in this EXACT format:
<tool_call>{"tool": "TOOL_NAME", "params": {"key": "value"}}</tool_call>

Available tools:
- get_market_data(symbol, lookback=1)         → OHLCV price snapshot
- compute_rsi(symbol, period=14)              → RSI momentum (>70 overbought, <30 oversold)
- get_order_book(symbol, depth=5)             → Level 2 bid/ask with imbalance
- compute_macd(symbol, fast=12, slow=26, signal=9) → MACD trend + histogram bias
- compute_bollinger(symbol, period=20)        → Bollinger Bands, %B, squeeze detection
- detect_support_resistance(symbol, lookback=50)   → Key S/R levels & distance %

Reasoning pattern: Observe → Hypothesize → Call Tools → Synthesize → SIGNAL.
End with a clear SIGNAL: (Bullish/Bearish/Neutral) and a brief rationale."""

class MCPToolDispatcher:
    """Dispatches tool calls to the local MCP server using the MCP Python SDK."""
    
    def __init__(self, server_script: str = "mcp_server/server.py"):
        base_path = Path(__file__).parent.parent
        self.server_path = str(base_path / server_script)
        self.client_session = None
        self._exit_stack = None

    async def _ensure_session(self):
        if self.client_session:
            return

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from contextlib import AsyncExitStack

        self._exit_stack = AsyncExitStack()
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_path],
            env=os.environ.copy()
        )
        
        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
        self.read, self.write = stdio_transport
        self.client_session = await self._exit_stack.enter_async_context(ClientSession(self.read, self.write))
        await self.client_session.initialize()

    async def call_tool(self, tool_name: str, params: dict) -> dict:
        try:
            await self._ensure_session()
            result = await self.client_session.call_tool(tool_name, params)
            
            if hasattr(result, 'content') and result.content:
                text_content = result.content[0].text
                return json.loads(text_content)
            return {"error": "No content in result"}
            
        except Exception as e:
            # If the session is closed, clear it so the next call recreates it
            import anyio
            if isinstance(e, anyio.ClosedResourceError) or "ClosedResourceError" in str(e):
                await self.cleanup()
            
            import traceback
            traceback.print_exc()
            return {"error": repr(e), "note": "Ensure mcp_server/server.py is error-free."}

    async def cleanup(self):
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except:
                pass
            self._exit_stack = None
            self.client_session = None

class AlphaLensAgent:
    def __init__(self, use_local_model: bool = False):
        self.dispatcher = MCPToolDispatcher()
        self.use_local = use_local_model and LOCAL_MODEL_AVAILABLE and CHECKPOINT_DIR.exists()
        self.history = []

        if not self.use_local:
            api_key = os.getenv("GOOGLE_API_KEY")
            print(f"DEBUG: API Key Loaded: {bool(api_key)}")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY missing from .env file.")
            self.client = genai.Client(api_key=api_key)
            self.model_id = "gemini-2.5-flash"

    async def analyze(self, query: str) -> AsyncGenerator[str, None]:
        # Add user query to history
        self.history.append({"role": "user", "parts": [{"text": query}]})
        
        # Limit ReAct loop to 5 iterations to prevent infinite tool loops
        for _ in range(5):
            # 1. Generate Response
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=self.history,
                config={'system_instruction': SYSTEM_PROMPT}
            )
            
            response_text = response.text
            tool_matches = list(TOOL_CALL_PATTERN.finditer(response_text))

            if not tool_matches:
                yield response_text
                self.history.append({"role": "model", "parts": [{"text": response_text}]})
                break

            # 2. Process Tools found in the response
            last_end = 0
            tool_results = []

            for match in tool_matches:
                # Yield text leading up to the tool call
                yield response_text[last_end:match.start()]
                
                try:
                    call_data = json.loads(match.group(1))
                    t_name, t_params = call_data["tool"], call_data.get("params", {})
                    
                    yield f"\n[MCP] Running {t_name}...\n"
                    result = await self.dispatcher.call_tool(t_name, t_params)
                    
                    result_json = json.dumps(result, indent=2)
                    yield f"```json\n{result_json}\n```\n"
                    tool_results.append(f"Result of {t_name}: {result_json}")
                except Exception as e:
                    yield f"\n[Error]: {str(e)}\n"

                last_end = match.end()

            yield response_text[last_end:]

            # 3. Update history with results for the next "thought" cycle
            self.history.append({"role": "model", "parts": [{"text": response_text}]})
            self.history.append({
                "role": "user", 
                "parts": [{"text": f"Tool Results:\n" + "\n".join(tool_results)}]
            })

# ═══════════════════════════════════════════════════════════════
# DEBUG ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    async def run_debug():
        print("--- Alpha-Lens Agent Debug Session ---")
        agent = AlphaLensAgent(use_local_model=False)
        user_query = "What is the current Gold price and is the RSI indicating it is overbought?"
        
        try:
            async for chunk in agent.analyze(user_query):
                print(chunk, end="", flush=True)
        finally:
            await agent.dispatcher.cleanup()
            print("\n--- Session End ---")

    asyncio.run(run_debug())