# FTMO ML Trading System — Research & Architecture Document

> **Date**: 2026-03-22
> **Objective**: Build an ML-powered automated trading system that passes the FTMO challenge in minimal time
> **Stack**: Python + MetaTrader 5 + LightGBM/XGBoost ensemble + HMM regime detection

---

## 1. FTMO Challenge Rules & Constraints

### Account Options

| Parameter | 2-Step Normal | 2-Step Aggressive | 1-Step |
|---|---|---|---|
| Phase 1 Target | 10% | 20% | 10% (single phase) |
| Phase 2 Target | 5% | 10% | — |
| Max Daily Loss | 5% | 10% | 5% |
| Max Total Loss | 10% | 20% | 10% |
| Time Limit | **None** (removed) | **None** | **None** |
| Min Trading Days | 4 (Phase 1), 4 (Phase 2) | 4 | 2 |
| Profit Split | 80-90% | 80-90% | 80-90% |

**Recommendation**: Use **2-Step Normal at $100K**. It has the most forgiving structure — no time pressure, reasonable targets, and the 5%/10% drawdown limits are manageable with proper risk controls.

### Leverage

| Instrument | Normal Account | Swing Account |
|---|---|---|
| Forex | 1:100 | 1:30 |
| Indices | 1:50 | 1:15 |
| Metals | 1:50 | 1:9 |
| Crypto | 1:3.3 | 1:1 |

### What's Allowed
- Scalping, EAs/bots, algorithmic/ML-based trading
- Overnight holding (standard account during market hours)
- News trading during evaluation phases

### What's Prohibited
- HFT exploiting data feed delays, latency arbitrage, tick scalping
- Gap trading before major events/market closures
- Hedging across multiple FTMO accounts
- Server-flooding EAs

### The Best Day Rule
Single most profitable day must not exceed **50% of total Positive Days' Profit** at payout. A consistent ML system grinding daily naturally satisfies this. Deliberately splitting trades to circumvent is forbidden.

### Daily Loss Calculation
FTMO calculates daily loss from the **higher of**: (a) starting balance of the day, or (b) equity at end of previous day. This includes **unrealized (floating) losses**. The system must track this in real-time.

---

## 2. ML Architecture — Final Design

### Why This Architecture

The research is clear on a hierarchy:
1. **Gradient Boosted Trees** (LightGBM/XGBoost) dominate tabular financial data — 71% directional accuracy vs LSTM's 62-67% in controlled comparisons
2. **Ensembles** of diverse models outperform any single model — equal-weighted ensemble of DNN + GBT + RF produced 0.45%/day raw returns
3. **LSTM/GRU** add value as sequence feature extractors, not as standalone predictors
4. **RL** is unstable and unproven in live trading — use only for position sizing meta-layer
5. **Transformers** (TFT) need rich covariate data to justify complexity; LSTMs beat them on differential sequences

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SIGNAL GENERATION LAYER                      │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ LightGBM │  │ XGBoost  │  │ CatBoost │  │ GRU Sequence     │ │
│  │ (base 1) │  │ (base 2) │  │ (base 3) │  │ Feature Extractor│ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───────┬──────────┘ │
│       │              │              │                │            │
│       └──────────────┴──────┬───────┴────────────────┘            │
│                             │                                     │
│                    ┌────────▼────────┐                            │
│                    │  Meta-Learner   │                            │
│                    │  (Logistic Reg) │                            │
│                    └────────┬────────┘                            │
│                             │                                     │
│                    Direction + Confidence Score                    │
└─────────────────────────────┼─────────────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────────┐
│                     REGIME DETECTION LAYER                        │
│                             │                                     │
│                    ┌────────▼────────┐                            │
│                    │   HMM (2-4      │                            │
│                    │   hidden states) │                            │
│                    └────────┬────────┘                            │
│                             │                                     │
│              Regime: Trend-Up | Trend-Down | Range | High-Vol     │
│                             │                                     │
│              Gate: TRADE (trend regimes) or SKIP (range/high-vol) │
└─────────────────────────────┼─────────────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────────┐
│                     RISK MANAGEMENT LAYER                         │
│                             │                                     │
│         ┌───────────────────▼───────────────────────┐            │
│         │  Position Sizer (Fractional Kelly)         │            │
│         │  • Base: Quarter-Kelly                     │            │
│         │  • Scaled by model confidence              │            │
│         │  • Hard cap: 0.5% risk per trade           │            │
│         │  • ATR-based stop distance                 │            │
│         └───────────────────┬───────────────────────┘            │
│                             │                                     │
│         ┌───────────────────▼───────────────────────┐            │
│         │  FTMO Safety Controls                      │            │
│         │  • Daily loss halt at 3% (buffer vs 5%)    │            │
│         │  • Total drawdown halt at 8% (buffer vs 10%│            │
│         │  • Max open risk: 1%                       │            │
│         │  • Best Day tracker                        │            │
│         └───────────────────┬───────────────────────┘            │
└─────────────────────────────┼─────────────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────────┐
│                     EXECUTION LAYER (MT5)                         │
│                             │                                     │
│         ┌───────────────────▼───────────────────────┐            │
│         │  Order Manager                             │            │
│         │  • Market orders with deviation control    │            │
│         │  • Hard SL/TP on every trade               │            │
│         │  • Position tracking & reconciliation      │            │
│         │  • Heartbeat & reconnection logic          │            │
│         └───────────────────────────────────────────┘            │
└───────────────────────────────────────────────────────────────────┘
```

### Model Details

**Base Learners (Signal Generation)**:
- **LightGBM**: Fast training, handles categorical features natively, best for large feature sets
- **XGBoost**: More regularization options, slightly more robust to noise
- **CatBoost**: Best handling of categorical features, good with smaller datasets
- **GRU**: Processes raw OHLCV sequences (20-60 bars) → outputs latent embedding vector fed as additional features to GBTs

**Meta-Learner**: Logistic Regression on out-of-fold predictions from base learners. Simple, avoids second-level overfitting.

**Regime Detector**: Hidden Markov Model with 3-4 states fitted on rolling returns + volatility. States map to: trending-up, trending-down, range-bound, high-volatility-chaos. Only trade in trending states.

### Target Variable

Binary classification: **Will price move > X pips in the signal direction within N bars?**
- X = 1.5 × ATR (ensures moves are large enough to be tradeable after spread)
- N = 4-8 bars on H1 (4-8 hours lookahead)
- Triple-barrier labeling (de Prado): upper barrier (TP), lower barrier (SL), time barrier

---

## 3. Feature Engineering

### Feature Categories (Prioritized)

**Tier 1 — Price Action Derivatives (highest signal-to-noise)**
| Feature | Calculation | Rationale |
|---|---|---|
| `log_return_1` | `log(close / close.shift(1))` | Raw momentum signal |
| `log_return_5` | `log(close / close.shift(5))` | Medium-term momentum |
| `log_return_20` | `log(close / close.shift(20))` | Longer-term trend |
| `hl_range_norm` | `(high - low) / ATR(14)` | Normalized bar range |
| `body_ratio` | `abs(close - open) / (high - low)` | Candle conviction |
| `upper_wick_ratio` | Upper wick / total range | Rejection signal |
| `gap` | `open / prev_close - 1` | Opening gap |

**Tier 2 — Volatility Features**
| Feature | Calculation | Rationale |
|---|---|---|
| `atr_14` | ATR(14) normalized by close | Absolute volatility state |
| `bb_width` | Bollinger Band width (20,2) | Squeeze/expansion detection |
| `realized_vol_5` | Std(returns, 5) | Short-term vol |
| `realized_vol_20` | Std(returns, 20) | Medium-term vol |
| `vol_ratio` | vol_5 / vol_20 | Volatility regime change |
| `atr_percentile` | ATR rank over 100 bars | Is vol high or low historically? |

**Tier 3 — Momentum/Mean-Reversion**
| Feature | Calculation | Rationale |
|---|---|---|
| `rsi_14` | RSI(14) | Overbought/oversold |
| `rsi_7` | RSI(7) | Faster momentum |
| `macd_hist` | MACD histogram | Momentum acceleration |
| `adx_14` | ADX(14) | Trend strength |
| `stoch_k` | Stochastic %K(14,3) | Mean-reversion signal |
| `dist_from_20ma` | (close - SMA20) / ATR | Z-scored distance from MA |
| `dist_from_50ma` | (close - SMA50) / ATR | Longer-term trend position |

**Tier 4 — Volume**
| Feature | Calculation | Rationale |
|---|---|---|
| `volume_zscore` | (vol - vol_mean_20) / vol_std_20 | Relative volume |
| `obv_slope` | OBV linear regression slope | Accumulation/distribution |
| `vol_price_corr` | Rolling correlation(volume, abs(return)) | Volume confirms moves? |

**Tier 5 — Time/Session**
| Feature | Calculation | Rationale |
|---|---|---|
| `hour_sin` | sin(2π × hour / 24) | Cyclical hour encoding |
| `hour_cos` | cos(2π × hour / 24) | Cyclical hour encoding |
| `dow_sin` | sin(2π × dow / 5) | Cyclical day encoding |
| `is_london` | Flag: 08:00-16:00 UTC | London session |
| `is_ny` | Flag: 13:00-21:00 UTC | NY session |
| `is_overlap` | Flag: 13:00-16:00 UTC | London-NY overlap (highest liquidity) |

**Tier 6 — Cross-Asset (for multi-instrument)**
| Feature | Calculation | Rationale |
|---|---|---|
| `dxy_return` | DXY log return | Dollar strength |
| `vix_level` | VIX value (for indices) | Risk sentiment |
| `gold_return` | XAUUSD return (for forex) | Safe-haven flow |

### Feature Selection Protocol
1. Compute SHAP importance from initial LightGBM run
2. Remove features with SHAP importance < 1% of max
3. Remove features correlated > 0.85 with higher-importance feature
4. Run forward feature selection with purged CV
5. Target: 25-40 final features (more = overfitting risk)

---

## 4. Instruments & Timeframes

### Primary Instruments

| Instrument | Why | Spread (typical) | Session |
|---|---|---|---|
| **EURUSD** | Tightest spreads, deepest history, highest liquidity | 0.5-1 pip | London + NY |
| **XAUUSD** | High volatility, strong trends, popular FTMO instrument | $0.30-0.50 | All sessions |
| **NAS100** | Strong momentum, excellent for trend-following | 1-2 points | US session |

### Timeframe Strategy
- **H1 (1-hour)**: Primary signal generation — best signal-to-noise for ML
- **M15 (15-min)**: Entry refinement — once H1 gives direction, M15 provides precision entry
- **H4**: Regime context — HMM fitted on H4 for broader regime awareness

### Why H1 Primary
- Enough bars for ML training (8,760 per year per instrument)
- Spread costs are <5% of average bar range (vs >20% on M1)
- News impact is smoothed, reducing noise
- Sufficient for 3-5 trades per day across 3 instruments

---

## 5. Validation Methodology (Non-Negotiable)

### The Core Problem
Quantopian's study of 888 strategies: **backtested Sharpe ratios had near-zero correlation with live returns**. The in-sample to out-of-sample Sharpe correlation is <0.05. Validation methodology matters more than model architecture.

### Required Validation Stack

**Level 1: Combinatorial Purged Cross-Validation (CPCV)**
- From Lopez de Prado's "Advances in Financial Machine Learning"
- Generates multiple alternative historical paths through data
- Purges training observations overlapping with test labels
- Adds embargo gap to prevent autocorrelation leakage
- Implementation: `mlfinlab` Python library
- **Accept only if PBO (Probability of Backtest Overfitting) < 25%**

**Level 2: Walk-Forward Optimization (WFO)**
- Train on 6 months, test on 2 months, roll forward by 1 month
- Simulates real-world periodic retraining
- Track performance degradation over time since last retrain

**Level 3: Deflated Sharpe Ratio (DSR)**
- Corrects Sharpe for: number of trials, non-normality, selection bias
- Use DSR (not raw Sharpe) for all model comparison decisions

**Level 4: Monte Carlo Permutation Test**
- Shuffle returns to create 1,000 synthetic series
- Strategy must beat 95th percentile of shuffled results
- Identifies strategies that exploit data-specific artifacts

**Level 5: FTMO-Specific Simulation**
- Simulate exact FTMO drawdown accounting (equity-relative, including floating P&L)
- Track daily loss vs 5% limit with realistic timing
- Track total drawdown vs 10% limit
- Include realistic spreads (add 0.5 pip conservatively)
- Include swap costs for overnight positions
- Run 1,000 Monte Carlo simulations of the challenge with varying start dates

### Data Split Protocol
```
Total data: 3-5 years of H1 bars
├── Development set (60%): Feature engineering + model training
│   └── Internal: CPCV with purging + embargo
├── Validation set (20%): Hyperparameter selection + ensemble weights
│   └── Walk-forward tested, touched sparingly
└── Deep OOS holdout (20%): Final go/no-go test
    └── Touched ONCE after all development is complete
```

---

## 6. Risk Management — FTMO-Specific

### Position Sizing Formula
```
lots = (account_balance × risk_pct) / (stop_distance_pips × pip_value_per_lot)

Where:
  risk_pct = base_risk × confidence_scalar × regime_scalar
  base_risk = 0.25% (Quarter Kelly baseline)
  confidence_scalar = model probability output (0.5 - 1.0 mapped to 0.5 - 1.0)
  regime_scalar = 1.0 (trending) or 0.5 (uncertain) or 0.0 (high-vol chaos)
  stop_distance = 1.5 × ATR(14)
```

### Hard Safety Controls (Automated, No Override)

| Control | Threshold | Action |
|---|---|---|
| Per-trade risk cap | 0.5% of balance | Reject trade if exceeded |
| Max open risk | 1.0% total across all positions | No new trades until risk freed |
| Daily loss halt | -3.0% from day start balance | Close all positions, halt for day |
| Total drawdown halt | -8.0% from initial balance | Close all, halt until manual review |
| Max concurrent trades | 3 | Queue additional signals |
| Max trades per day | 8 | Prevent overtrading |

### Why These Specific Buffers
- Daily halt at 3% leaves 2% buffer before FTMO's 5% limit (accounts for slippage on close)
- Total halt at 8% leaves 2% buffer before FTMO's 10% limit
- At 0.5% risk/trade, need 6 consecutive full losses to hit daily halt — very unlikely with >50% accuracy

### Target Math ($100K Account)
- Phase 1 target: $10,000 (10%)
- With no time limit, target $500/day average → 20 trading days
- At 0.5% risk/trade ($500 risk), 55% win rate, 1.5R avg win:
  - Expected value per trade = 0.55 × 1.5R - 0.45 × 1R = 0.375R = $187.50
  - Need ~2.7 trades per day for $500/day → round to 3 trades/day
  - Across 3 instruments at H1 = very achievable signal frequency

---

## 7. MetaTrader 5 Python Integration

### Key Technical Facts
- Package: `MetaTrader5` on PyPI (Windows-only, communicates via named pipe to MT5 terminal)
- MT5 terminal must be running with "Algo Trading" enabled
- Purely request-response (polling) — no WebSocket/streaming
- NOT thread-safe — serialize all MT5 calls through a single thread
- 21 timeframes available from M1 to MN1

### Data Retrieval
```python
import MetaTrader5 as mt5
mt5.initialize()

# Get H1 bars
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 10000)
# Returns numpy array: time, open, high, low, close, tick_volume, spread, real_volume

# Get tick data
ticks = mt5.copy_ticks_range("EURUSD", date_from, date_to, mt5.COPY_TICKS_ALL)
```

### Trade Execution
```python
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": "EURUSD",
    "volume": 0.1,
    "type": mt5.ORDER_TYPE_BUY,
    "price": mt5.symbol_info_tick("EURUSD").ask,
    "sl": 1.0850,  # ALWAYS set SL
    "tp": 1.0950,  # ALWAYS set TP
    "deviation": 20,
    "magic": 234000,  # Unique ID for this bot
    "comment": "ftmo_ml_v1",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}
result = mt5.order_send(request)
```

### Critical Gotchas
1. Cast numpy types to Python native before order requests (`int()`, `float()`)
2. Call `mt5.symbol_select(symbol, True)` before any data/trade operations
3. Check `result.retcode == mt5.TRADE_RETCODE_DONE` after every order
4. Implement heartbeat: re-call `mt5.initialize()` if connection drops
5. M1 data can be unreliable — M5+ is more stable for historical pulls
6. Data has gaps during holidays — handle in feature calculation

---

## 8. GitHub Resources Assessment

### Worth Using
| Repo | Stars | Use For |
|---|---|---|
| `stefan-jansen/machine-learning-for-trading` | ~6,400 | ML methodology, walk-forward, feature engineering patterns |
| `ranaroussi/quantstats` | ~6,600 | Backtest evaluation (Sharpe, drawdown, tearsheets) |
| `AI4Finance-Foundation/FinRL` | ~10,000 | RL concepts if adding position sizing meta-layer |
| `Quantreo/MetaTrader-5-AUTOMATED-TRADING-using-Python` | ~200 | MT5 API wrapper patterns |

### Not Worth Using
Every "FTMO bot" repo on GitHub is either spam, a paid product advertisement, or basic EA code with ML buzzwords. No production-quality open-source FTMO ML system exists publicly — which makes sense (anyone with a working system keeps it private).

**Bottom line**: Use `quantstats` as a library dependency. Reference `stefan-jansen` for methodology. Build everything else custom.

---

## 9. Implementation Roadmap

### Phase 1: Data Pipeline & Feature Engineering (Week 1)
- [ ] Set up MT5 Python connection and data download scripts
- [ ] Download 3-5 years of H1 data for EURUSD, XAUUSD, NAS100
- [ ] Build feature engineering pipeline (all Tier 1-5 features)
- [ ] Implement triple-barrier labeling
- [ ] Data quality checks and gap handling

### Phase 2: Model Development & Validation (Week 2-3)
- [ ] Train individual base learners (LightGBM, XGBoost, CatBoost)
- [ ] Train GRU sequence feature extractor
- [ ] Build stacking ensemble with meta-learner
- [ ] Implement CPCV validation
- [ ] Compute PBO and DSR metrics
- [ ] Feature selection via SHAP
- [ ] Hyperparameter tuning (Optuna)

### Phase 3: Regime Detection & Risk Management (Week 3)
- [ ] Fit HMM regime detector on H4 data
- [ ] Implement regime-gated trading logic
- [ ] Build FTMO-specific risk management module
- [ ] Position sizing with confidence scaling
- [ ] Daily loss halt and total drawdown circuit breaker

### Phase 4: Backtesting & Robustness (Week 3-4)
- [ ] FTMO-accurate backtesting engine (equity tracking, drawdown accounting)
- [ ] Walk-forward validation across multiple regimes
- [ ] Monte Carlo permutation tests (1,000 shuffles)
- [ ] Monte Carlo FTMO challenge simulation (1,000 runs, varying start dates)
- [ ] Stress test against 2020 COVID crash, 2022 rate hikes, 2023 banking crisis

### Phase 5: Live Paper Trading (Week 4+)
- [ ] Deploy on FTMO demo account
- [ ] Monitor execution quality (slippage, fill rates)
- [ ] Compare live results to backtest expectations
- [ ] Iterate on model if live performance diverges

### Phase 6: FTMO Challenge
- [ ] Deploy on paid FTMO challenge account
- [ ] Monitor daily with automated alerts
- [ ] No manual intervention (trust the system)

---

## 10. Python Dependencies

```
# Core ML
lightgbm>=4.0
xgboost>=2.0
catboost>=1.2
scikit-learn>=1.3
torch>=2.0  # For GRU

# Data & Features
pandas>=2.0
numpy>=1.24
ta-lib  # Technical indicators (or pandas-ta as fallback)
pandas-ta>=0.3  # Alternative to ta-lib

# Validation
mlfinlab>=2.0  # CPCV, triple-barrier, feature importance

# Backtesting & Analysis
vectorbt>=0.26  # Fast vectorized backtesting
quantstats>=0.0.62  # Performance analytics

# Regime Detection
hmmlearn>=0.3  # Hidden Markov Models

# Optimization
optuna>=3.0  # Hyperparameter tuning

# MT5
MetaTrader5>=5.0  # Windows only

# Utilities
joblib  # Model persistence
logging  # Already in stdlib
schedule  # For periodic retraining
```

---

## 11. Key Principles (Do Not Violate)

1. **Validation > Architecture**: A LightGBM with CPCV will outperform a Transformer with train/test split
2. **Never use raw prices as features**: Always transform to returns or normalized values
3. **Hard stops on every trade**: No exceptions, ever
4. **Risk management is not optional**: The daily loss halt saves the challenge when the model has a bad day
5. **Train on returns, not prices**: Financial data is non-stationary; returns approximate stationarity
6. **Retrain periodically**: Markets change; schedule monthly retraining with walk-forward
7. **Don't overtrade**: 3-5 quality trades/day beats 20 noisy ones
8. **Trust the system**: Manual override historically hurts automated system performance
9. **Start conservative**: Better to pass slowly than fail fast
10. **The Best Day Rule is your friend**: A consistent daily grind naturally satisfies it
