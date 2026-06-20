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
    cum_pnl = 0.0
    cum_pnl_series = []
    active_returns = []
    
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
        cum_pnl += pnl
        
        trade_data = {
            **res,
            "future_price": future_price,
            "pnl": round(pnl, 2),
            "win": win,
            "cumulative_pnl": round(cum_pnl, 2)
        }
        trades.append(trade_data)
        
        cum_pnl_series.append({
            "timestamp": res["timestamp"],
            "cumulative_pnl": round(cum_pnl, 2)
        })
        
        if res["signal"] in ["BULLISH", "BEARISH"]:
            active_returns.append(pnl / res["price_at_signal"])

    # 5. Sharpe and Sortino Ratios
    import math
    sharpe_ratio = 0.0
    sortino_ratio = 0.0
    
    if len(active_returns) > 1:
        mean_ret = sum(active_returns) / len(active_returns)
        var_ret = sum((r - mean_ret) ** 2 for r in active_returns) / (len(active_returns) - 1)
        std_ret = math.sqrt(var_ret)
        
        if std_ret > 0:
            sharpe_ratio = (mean_ret / std_ret) * math.sqrt(252)
            
        downside_returns = [r for r in active_returns if r < 0]
        if len(downside_returns) > 1:
            downside_var = sum(r ** 2 for r in downside_returns) / len(downside_returns)
            downside_std = math.sqrt(downside_var)
            if downside_std > 0:
                sortino_ratio = (mean_ret / downside_std) * math.sqrt(252)
        elif len(downside_returns) == 1:
            downside_std = abs(downside_returns[0])
            if downside_std > 0:
                sortino_ratio = (mean_ret / downside_std) * math.sqrt(252)
                
    # 6. Max Drawdown
    equity = 10000.0
    equity_curve = [equity]
    for t in trades:
        equity += t["pnl"]
        equity_curve.append(equity)
        
    peak = equity_curve[0]
    max_dd_fraction = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd_fraction:
            max_dd_fraction = dd
    max_dd_pct = max_dd_fraction * 100

    # 7. Reporting
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
            "avg_loss": round(avg_loss, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "max_drawdown_pct": round(max_dd_pct, 2)
        },
        "cumulative_pnl_series": cum_pnl_series,
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
