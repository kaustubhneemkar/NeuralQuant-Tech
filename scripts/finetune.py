"""
Alpha-Lens | scripts/finetune.py
================================
WHY THIS ARCHITECTURE:
  Fine-tuning Llama-3 on financial reasoning data gives the model a specialized
  "vocabulary" for HFT patterns (e.g., order-book imbalance, momentum reversals).
  Instead of prompt-engineering a generic LLM, we bake domain knowledge into weights
  via QLoRA (4-bit quantized LoRA). This means:
    1. Lower inference VRAM (~6GB vs ~40GB for full Llama-3-70B)
    2. Faster token generation for latency-sensitive trading signals
    3. The model learns WHEN to call MCP tools (e.g., "I need RSI before concluding")

TOOL-CALLING PATTERN:
  Training examples include <tool_call> tags so the fine-tuned model learns
  to emit structured JSON when it needs live data from the MCP server.
  The agent loop in agent/runner.py intercepts these tags and routes to MCP tools.
"""

import json
import torch
from datasets import Dataset
from transformers import TrainingArguments, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer

# ── Try unsloth for 2x faster training; fall back to standard transformers ──
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
    print("[Alpha-Lens] ✓ Unsloth detected — using 2x speed optimizations")
except ImportError:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    USE_UNSLOTH = False
    print("[Alpha-Lens] ⚠ Unsloth not found — using standard HF + BitsAndBytes")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
MODEL_NAME   = "meta-llama/Meta-Llama-3-8B-Instruct"   # Base model
OUTPUT_DIR   = "./checkpoints/alpha-lens-llama3"
MAX_SEQ_LEN  = 2048
LORA_RANK    = 16          # Higher rank = more capacity, more VRAM
LORA_ALPHA   = 32          # Scale factor; alpha/rank = effective LR multiplier
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]   # Full attention + FFN


# ═══════════════════════════════════════════════════════════════════════════
# MOCK FINANCIAL REASONING DATASET
# WHY: Real training data would be proprietary tick data + analyst notes.
#      These 8 examples teach the model the reasoning PATTERN:
#      Observe → Hypothesize → Call Tool → Synthesize → Signal
# ═══════════════════════════════════════════════════════════════════════════
MOCK_TRAINING_DATA = [
    {
        "instruction": "Analyze AAPL for momentum signals given recent price action.",
        "input": "AAPL closed at $189.50, up 2.1% on 3x average volume. 5-day return: +4.8%.",
        "output": (
            "Observing a high-volume breakout with positive price momentum. "
            "Before concluding, I need current RSI to check for overbought conditions.\n"
            "<tool_call>{\"tool\": \"compute_rsi\", \"params\": {\"symbol\": \"AAPL\", \"period\": 14}}</tool_call>\n"
            "RSI at 68 — approaching overbought (>70) but not yet there. "
            "Combined with the volume surge, this suggests institutional accumulation. "
            "SIGNAL: Bullish momentum, target +2% in 3 sessions. Watch RSI > 72 as exit trigger."
        ),
    },
    {
        "instruction": "Detect order-book imbalance for SPY options flow.",
        "input": "SPY bid/ask: 445.20/445.22. Put/Call ratio: 0.65. IV rank: 23%.",
        "output": (
            "Low P/C ratio (0.65 < 0.8 threshold) signals call-side dominance — bullish positioning. "
            "IV rank at 23% means options are cheap relative to last year; ideal for long strategies.\n"
            "<tool_call>{\"tool\": \"get_market_data\", \"params\": {\"symbol\": \"SPY\", \"fields\": [\"order_flow\", \"dark_pool\"]}}</tool_call>\n"
            "Dark pool prints show net buying. Order-book imbalance: 62% bid-side. "
            "SIGNAL: HFT momentum likely to continue. Low-cost call spreads favored."
        ),
    },
    {
        "instruction": "Identify mean-reversion opportunity in TSLA.",
        "input": "TSLA dropped 8% in 2 sessions. Sector (EV) flat. No fundamental news.",
        "output": (
            "An 8% idiosyncratic drop with no fundamental catalyst suggests forced selling "
            "(margin calls or ETF rebalancing) — a classic mean-reversion setup.\n"
            "<tool_call>{\"tool\": \"compute_rsi\", \"params\": {\"symbol\": \"TSLA\", \"period\": 14}}</tool_call>\n"
            "RSI at 28 — oversold. Confirming no structural break with market data.\n"
            "<tool_call>{\"tool\": \"get_market_data\", \"params\": {\"symbol\": \"TSLA\", \"fields\": [\"volume\", \"vwap\"]}}</tool_call>\n"
            "Price 4% below VWAP — institutional buy zone. "
            "SIGNAL: Mean-reversion long. Entry near current. Stop -3%. Target VWAP recovery."
        ),
    },
    {
        "instruction": "Assess macro risk for tech sector exposure.",
        "input": "10Y yield rose 15bps today. QQQ down 1.2%. Fed meeting in 3 days.",
        "output": (
            "Rising rates compress tech multiples (DCF sensitivity: high-growth stocks "
            "lose ~2% per 10bps rate rise). Fed uncertainty amplifies this.\n"
            "<tool_call>{\"tool\": \"get_market_data\", \"params\": {\"symbol\": \"QQQ\", \"fields\": [\"sector_flows\", \"options_skew\"]}}</tool_call>\n"
            "Options skew shifted bearish (puts bid up). Sector outflows accelerating. "
            "SIGNAL: Reduce tech beta. Hedge with TLT puts or VIX calls into Fed."
        ),
    },
    {
        "instruction": "Detect potential pump-and-dump in small-cap MNKY.",
        "input": "MNKY up 340% in 48 hours. Volume 50x average. No SEC filings.",
        "output": (
            "300%+ move on 50x volume without fundamental catalyst is a textbook P&D flag. "
            "Need to verify with order flow data.\n"
            "<tool_call>{\"tool\": \"get_market_data\", \"params\": {\"symbol\": \"MNKY\", \"fields\": [\"trade_size_dist\", \"market_makers\"]}}</tool_call>\n"
            "Trade size distribution: 95% sub-$500 lots (retail-driven, no institutional). "
            "Only 1 market maker active. SIGNAL: HIGH RISK. Avoid long. "
            "Short entry only with strict hard stop at +10% — volatility extreme."
        ),
    },
    {
        "instruction": "Evaluate cross-asset correlation breakdown.",
        "input": "Gold +2%, USD +1.5% simultaneously. Normally inversely correlated.",
        "output": (
            "Gold-USD positive correlation signals a flight-to-safety event — both assets "
            "bid simultaneously implies systemic fear (geopolitical risk or credit event).\n"
            "<tool_call>{\"tool\": \"get_market_data\", \"params\": {\"symbol\": \"GLD\", \"fields\": [\"correlation_matrix\", \"cds_spreads\"]}}</tool_call>\n"
            "EM CDS spreads widening 40bps. Cross-asset correlations going to 1 (crisis mode). "
            "SIGNAL: Risk-off. Cut equity exposure. Long VIX, USD, and 2Y treasuries."
        ),
    },
    {
        "instruction": "Analyze NVDA earnings reaction for gamma squeeze potential.",
        "input": "NVDA beat EPS by 18%. Stock up 9% after-hours. Options OI: massive at $500 strike.",
        "output": (
            "Large OI at a strike that is now near-the-money creates dealer gamma exposure. "
            "As price approaches $500, dealers must buy stock to hedge — self-reinforcing squeeze.\n"
            "<tool_call>{\"tool\": \"compute_rsi\", \"params\": {\"symbol\": \"NVDA\", \"period\": 9}}</tool_call>\n"
            "<tool_call>{\"tool\": \"get_market_data\", \"params\": {\"symbol\": \"NVDA\", \"fields\": [\"gamma_exposure\", \"options_chain\"]}}</tool_call>\n"
            "Net dealer gamma: -$2.4B at $500 (they are short gamma = must buy on rally). "
            "SIGNAL: Gamma squeeze likely at open. Target $510-$515. Fade above $520."
        ),
    },
    {
        "instruction": "Identify statistical arbitrage between XOM and CVX.",
        "input": "XOM/CVX spread z-score: 2.3 standard deviations above 90-day mean.",
        "output": (
            "Z-score > 2 on a historically cointegrated pair is a high-conviction stat-arb signal. "
            "Verifying cointegration hasn't broken down with fresh data.\n"
            "<tool_call>{\"tool\": \"get_market_data\", \"params\": {\"symbol\": \"XOM,CVX\", \"fields\": [\"correlation_60d\", \"beta_ratio\"]}}</tool_call>\n"
            "60-day correlation: 0.91 (cointegration intact). Beta ratio stable. "
            "SIGNAL: Pairs trade — short XOM, long CVX. Target z-score reversion to 0. "
            "Expected holding period: 5-12 days."
        ),
    },
]


def build_prompt(example: dict) -> str:
    """
    Llama-3 chat format with custom <tool_call> tokens.
    The model learns to emit tool calls as part of natural reasoning chains.
    """
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are Alpha-Lens, a specialized quantitative analysis agent. You reason about 
financial markets, detect HFT patterns, and call tools when you need live data.
When you need market data, emit: <tool_call>{{"tool": "...", "params": {{...}}}}</tool_call>
Available tools: get_market_data, compute_rsi, compute_macd, get_order_book
Always explain your reasoning before and after tool calls.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{example['instruction']}

Context: {example['input']}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{example['output']}<|eot_id|>"""


def load_model_and_tokenizer():
    """Load model with 4-bit quantization (QLoRA). WHY: Fits Llama-3-8B in ~6GB VRAM."""
    if USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,           # Auto-detect bf16/fp16
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=TARGET_MODULES,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",  # 30% less VRAM
            random_state=42,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,    # Extra 0.5-bit savings
            bnb_4bit_quant_type="nf4",         # NormalFloat4 — best for LLM weights
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=TARGET_MODULES,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def main():
    print("=" * 60)
    print("  Alpha-Lens | QLoRA Fine-Tuning Pipeline")
    print("=" * 60)

    # Build dataset
    prompts = [{"text": build_prompt(ex)} for ex in MOCK_TRAINING_DATA]
    dataset = Dataset.from_list(prompts)
    print(f"[+] Dataset: {len(dataset)} training examples")

    # Load model
    print(f"[+] Loading {MODEL_NAME} with 4-bit quantization...")
    model, tokenizer = load_model_and_tokenizer()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[+] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Training args — WHY these values:
    #   - per_device_train_batch_size=2 + gradient_accumulation=8 → effective batch=16
    #   - warmup_ratio=0.03 → prevents early divergence on small dataset
    #   - lr=2e-4 → standard for LoRA; too high = catastrophic forgetting
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.03,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy="epoch",
        optim="adamw_8bit" if not USE_UNSLOTH else "adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=training_args,
    )

    print("[+] Starting training...")
    trainer.train()

    # Save adapter weights only (~50MB vs 16GB for full model)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[✓] Training complete. LoRA adapter saved to: {OUTPUT_DIR}")
    print("[✓] Load with: FastLanguageModel.from_pretrained(OUTPUT_DIR)")


if __name__ == "__main__":
    main()
