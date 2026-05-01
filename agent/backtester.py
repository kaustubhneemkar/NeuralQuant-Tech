import asyncio
import json
import os
import pandas as pd
import random
from datetime import datetime
from agent.runner import AlphaLensAgent

def get_backtest_cache_status(csv_path):
    """Checks if a valid backtest cache exists for the given file."""
    if not os.path.exists("backtest_results.json"):
        return None
    try:
        with open("backtest_results.json", "r") as f:
            cache = json.load(f)
            # If the dataset path and row count match, we can consider it a valid cache
            if cache.get("metadata", {}).get("dataset_abs_path") == os.path.abspath(csv_path):
                return cache
    except:
        return None
    return None

async def run_backtest(csv_path="data/gold_data_1h_cleaned.csv", sample_pct=0.40, prediction_offset=10, skip_confirmation=False, max_samples=None):
    """
    Runs the Alpha-Lens agent over a random subset of historical data.
    """
    csv_abs_path = os.path.abspath(csv_path)
    
    # 1. Check Cache
    cache = get_backtest_cache_status(csv_abs_path)
    if cache and not skip_confirmation:
        print(f"\n[CACHE] Found existing backtest results for {os.path.basename(csv_path)}.")
        reuse = input("Use cached results? (y/n): ").lower()
        if reuse == 'y':
            print("Skipping backtest, using cache.")
            return

    # 2. Manual Trigger (Skip if requested)
    if not skip_confirmation:
        print(f"\n--- BACKTEST CONFIGURATION ---")
        print(f"Dataset: {csv_path}")
        print(f"Sampling: {sample_pct*100}% of rows (Randomized)")
        if max_samples:
            print(f"Cap: Limited to {max_samples} samples")
        print(f"Quota Conservation: Active")
        
        choice = input("\nEnter 'run' to start backtest or 'skip' to exit: ").lower()
        if choice != 'run':
            print("Backtest aborted.")
            return

    print("\nInitializing Alpha-Lens Backtester...")
    agent = AlphaLensAgent()
    
    # Load the data into MCP
    await agent.dispatcher.call_tool("load_backtest_file", {"path": csv_abs_path})
    df = pd.read_csv(csv_abs_path)
    
    # Determine valid indices (must have room for future check)
    total_rows = len(df)
    valid_indices = list(range(100, total_rows - prediction_offset))
    
    # Randomly sample
    sample_size = int(len(valid_indices) * sample_pct)
    if max_samples:
        sample_size = min(sample_size, max_samples)
        
    sampled_indices = sorted(random.sample(valid_indices, sample_size))
    
    api_calls_saved = len(valid_indices) - sample_size
    results = []
    
    print(f"Starting simulation on {sample_size} random samples...")

    # 3. Simulation Loop
    count = 0
    for i in sampled_indices:
        count += 1
        print(f"\n--- SAMPLE {count} / {sample_size} (Index {i}) ---")
        
        await agent.dispatcher.call_tool("set_backtest_index", {"index": i})
        
        current_price = df.iloc[i]['Close']
        timestamp = df.iloc[i]['DateTime']
        
        print(f"Virtual Time: {timestamp} | Price: ${current_price}")
        
        full_thought = ""
        last_signal = "NEUTRAL"
        
        # --- Retry Logic for Quota ---
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async for chunk in agent.analyze("Analyze current market state and provide a clear SIGNAL: Bullish, Bearish, or Neutral."):
                    full_thought += chunk
                break # Success!
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        print(f"!! Quota Hit !! Waiting 20s to retry (Attempt {attempt+1}/{max_retries})...")
                        await asyncio.sleep(21)
                        continue
                    else:
                        print("!! QUOTA EXHAUSTED !! Stopping early and saving partial results.")
                        break
                else:
                    print(f"Error during agent analysis: {e}")
                    break
        
        if "bullish" in full_thought.lower():
            last_signal = "BULLISH"
        elif "bearish" in full_thought.lower():
            last_signal = "BEARISH"

        print(f"Agent Signal: {last_signal}")
        
        results.append({
            "index": i,
            "timestamp": str(timestamp),
            "signal": last_signal,
            "price_at_signal": float(current_price)
        })

    # 4. Evaluation
    print("\nEvaluating Signals...")
    trades = []
    wins = 0
    total_pnl = 0.0
    
    for res in results:
        idx = res["index"]
        target_idx = idx + prediction_offset
        future_price = float(df.iloc[target_idx]['Close'])
        price_diff = future_price - res["price_at_signal"]
        
        win = False
        pnl = 0.0
        if res["signal"] == "BULLISH":
            if price_diff > 0: win = True
            pnl = price_diff
        elif res["signal"] == "BEARISH":
            if price_diff < 0: win = True
            pnl = -price_diff
        
        if win: wins += 1
        total_pnl += pnl
        trades.append({**res, "future_price": future_price, "pnl": round(pnl, 2), "win": win})

    # 5. Reporting
    total_trades = len(trades)
    win_rate_val = (wins / total_trades) if total_trades > 0 else 0
    loss_rate_val = 1 - win_rate_val
    
    # Calculate Avg Win / Avg Loss for Expectancy
    win_trades = [t["pnl"] for t in trades if t["win"]]
    loss_trades = [abs(t["pnl"]) for t in trades if not t["win"] and t["signal"] != "NEUTRAL"]
    
    avg_win = (sum(win_trades) / len(win_trades)) if win_trades else 0
    avg_loss = (sum(loss_trades) / len(loss_trades)) if loss_trades else 0
    
    avg_profit_per_signal = (total_pnl / total_trades) if total_trades > 0 else 0
    expectancy = (win_rate_val * avg_win) - (loss_rate_val * avg_loss)
    
    summary = {
        "metadata": {
            "dataset_abs_path": csv_abs_path,
            "sample_pct": sample_pct,
            "total_possible_rows": len(valid_indices),
            "samples_processed": total_trades,
            "api_calls_saved": api_calls_saved,
            "timestamp": str(datetime.now())
        },
        "stats": {
            "win_rate_pct": round(win_rate_val * 100, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_profit_per_signal": round(avg_profit_per_signal, 2),
            "expectancy": round(expectancy, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2)
        },
        "trades": trades
    }

    with open("backtest_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBacktesting complete. Processed {total_trades} random samples.")
    print(f"Avg Profit/Signal: {summary['stats']['avg_profit_per_signal']}")
    print(f"Expectancy: {summary['stats']['expectancy']}")
    
    await agent.dispatcher.cleanup()
    return summary

if __name__ == "__main__":
    asyncio.run(run_backtest())
