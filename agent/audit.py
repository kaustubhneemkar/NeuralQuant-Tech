import asyncio
import json
import os
from datetime import datetime
from agent.backtester import run_backtest

def print_certificate(report):
    meta = report["metadata"]
    stats = report["stats"]
    trades = report["trades"]

    # Calculate Signal Distribution
    buys = len([t for t in trades if t["signal"] == "BULLISH"])
    sells = len([t for t in trades if t["signal"] == "BEARISH"])
    neutrals = len([t for t in trades if t["signal"] == "NEUTRAL"])

    # Calculate Max Consecutive Losses (Risk Analysis)
    max_consecutive_losses = 0
    current_streak = 0
    for t in trades:
        if not t["win"] and t["signal"] != "NEUTRAL":
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0

    # Find Examples
    winners = [t for t in trades if t["win"]][:2]
    losers = [t for t in trades if not t["win"] and t["signal"] != "NEUTRAL"][:2]

    print("\n" + "="*60)
    print("           ALPHA-LENS PERFORMANCE CERTIFICATE")
    print("="*60)
    print(f"Dataset      : {os.path.basename(meta['dataset_abs_path'])}")
    print(f"Audit Date   : {meta['timestamp']}")
    print(f"Samples      : {meta['samples_processed']} (Quota saved: {meta['api_calls_saved']} calls)")
    print("-" * 60)
    print(f"WIN RATE     : {stats['win_rate_pct']}%")
    print(f"TOTAL PnL    : {stats['total_pnl']}")
    print(f"RISK LEVEL   : {max_consecutive_losses} Max Consecutive Losses | Max Drawdown: {stats.get('max_drawdown_pct', 0.0)}%")
    print(f"SHARPE RATIO : {stats.get('sharpe_ratio', 0.0)} | SORTINO RATIO: {stats.get('sortino_ratio', 0.0)}")
    print("-" * 60)
    print(f"SIGNAL BIAS  : [BULLISH: {buys}] [BEARISH: {sells}] [NEUTRAL: {neutrals}]")
    print("-" * 60)
    
    if winners:
        print("\nWINNING EXAMPLES:")
        for w in winners:
            print(f"  [{w['timestamp']}] Signal: {w['signal']} @ {w['price_at_signal']} -> Outcome: {w['future_price']} (PnL: {w['pnl']})")
    
    if losers:
        print("\nLOSING EXAMPLES:")
        for l in losers:
            print(f"  [{l['timestamp']}] Signal: {l['signal']} @ {l['price_at_signal']} -> Outcome: {l['future_price']} (PnL: {l['pnl']})")
    
    print("\n" + "="*60)
    print("Persistence: Full audit saved to /audits/ folder.")
    print("="*60 + "\n")

async def perform_full_audit(file_path: str, sample_pct: float = 0.40, max_samples: int = 5):
    """Core logic to run an audit and save results. No terminal input."""
    if not os.path.exists(file_path):
        return {"error": f"File '{file_path}' not found."}

    if not os.path.exists("audits"):
        os.makedirs("audits")

    # Run the backtest (40% sample, capped at max_samples)
    report = await run_backtest(csv_path=file_path, sample_pct=sample_pct, skip_confirmation=True, max_samples=max_samples)
    if not report:
        return {"error": "Audit failed or was cancelled."}

    # Save timestamped audit
    filename = f"audit_{os.path.basename(file_path).split('.')[0]}_{datetime.now().strftime('%Y_%m_%d_%H%M')}.json"
    audit_path = os.path.join("audits", filename)
    
    with open(audit_path, "w") as f:
        json.dump(report, f, indent=2)
    
    report["metadata"]["audit_filename"] = filename
    return report

async def start_audit_cli():
    print("\nWelcome to the Alpha-Lens Performance Auditor")
    print("-----------------------------------------------")
    
    file_path = input("Enter the path to your CSV file (e.g., data/gold_data_1h_cleaned.csv): ").strip()
    
    report = await perform_full_audit(file_path)
    
    if "error" in report:
        print(f"Error: {report['error']}")
        return

    # Print the formatted summary to terminal
    print_certificate(report)

if __name__ == "__main__":
    try:
        asyncio.run(start_audit_cli())
    except KeyboardInterrupt:
        print("\nAudit interrupted.")
