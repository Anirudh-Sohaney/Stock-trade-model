# Stage 3 Trading System

Ensemble-derived rule-based trading system for US equities. 360 hardcoded decision rules distilled from gradient-boosted trees — no ML at inference time.

## How It Works

**Training pipeline** (not required at runtime):
1. 88 technical features computed from 15-min OHLCV data (RSI, EMAs, ATR percentiles, volume ratios, volatility, etc.)
2. 4 regressors (RF, ET, XGB, LGBM) trained on 150 tickers, 60 days of data
3. 4,324 unique leaf paths extracted, scored by risk-adjusted return, filtered for positive mean
4. Top 360 rules forming Stage 3 selected as the final set

**Inference** (`main_model.py`):
- `HardcodedTrader.generate_signal(feature_dict)` → `"BUY"` or `"HOLD"`
- Each rule is an `if/elif` chain checking feature thresholds (e.g. `rsi_14 > 40 and atr_percentile < 0.7`)
- If any rule fires, the signal is `"BUY"`; otherwise `"HOLD"`

## File Structure

```
├── main_model.py                      # 360-rule signal engine
├── holding_period_optimization.py     # Feature computation + analysis
├── alpaca_bot/
│   ├── main.py                        # Entry point
│   ├── config.json                    # Credentials + settings
│   ├── tickers.txt                    # Watchlist
│   ├── portfolio_manager.py           # Orchestrator
│   ├── position_manager.py            # Sizing + lifecycle
│   ├── signal_engine.py               # Wrapper around main_model.py
│   ├── alpaca_client.py               # Alpaca REST wrapper
│   ├── market_hours.py                # Schedule checks
│   ├── logger.py                      # Logging setup
│   └── requirements.txt               # Dependencies
└── README.md
```

## Using the Model Directly

```python
from main_model import HardcodedTrader
from holding_period_optimization import compute_features
import yfinance as yf

# Fetch 15-min data
df = yf.download("AAPL", period="5d", interval="15m")
df.columns = [c[0].lower() for c in df.columns]

# Compute features
features = compute_features(df).dropna()

# Get latest feature row
feat_dict = features.iloc[-1].to_dict()
feat_dict["close"] = df.iloc[-1]["close"]

# Generate signal
trader = HardcodedTrader()
signal = trader.generate_signal(feat_dict)  # "BUY" or "HOLD"
```

## Alpaca Bot Setup

### 1. Install dependencies
```bash
pip install -r alpaca_bot/requirements.txt
```

### 2. Configure credentials
Edit `alpaca_bot/config.json`:
```json
{
    "alpaca_api_key": "YOUR_API_KEY",
    "alpaca_secret_key": "YOUR_SECRET_KEY",
    "paper_trading": true,
    "initial_capital": 10000.0,
    "exit_bars": 1,
    "max_positions": 5,
    "sizing_method": "rank_based",
    "capital_utilization": 1.0,
    "ticker_file": "tickers.txt",
    "state_file": "bot_state.pkl",
    "log_file": "bot.log",
    "data_lookback_days": 60
}
```

Key settings:
- **exit_bars**: Holding period in 15-min bars (`1` = 15 min)
- **max_positions**: Max concurrent positions (`5` recommended)
- **sizing_method**: `rank_based`, `equal`, or `score_weighted`
- **capital_utilization**: Fraction of cash deployed (`1.0` = 100%)

### 3. Set watchlist
Edit `alpaca_bot/tickers.txt` — one ticker per line.

### 4. Dry-run (simulation mode)
Run without API keys to verify logic:
```bash
python alpaca_bot/main.py
```

### 5. Live run
Add valid Alpaca paper keys to `config.json`, then:
```bash
python alpaca_bot/main.py          # Single cycle
python alpaca_bot/main.py --loop   # Check every 15 min
```

## Portfolio Settings

Optimized for:
- **1-bar (15 min) hold** — fastest capital recycling
- **5 positions** — best return/diversification tradeoff
- **rank_based sizing** — top signal gets largest allocation
- **100% utilization** — constrained capital beats unlimited

Historical backtest: **$10,000 → $29,028 (+190%)** in ~60 trading days with -3.52% max drawdown.

## Notes

- Runs on 15-min data only (not designed for other timeframes)
- 150 tickers recommended, but any list works
- No model retraining needed — rules are static
- Simulation mode requires no API keys, no internet after data fetch
