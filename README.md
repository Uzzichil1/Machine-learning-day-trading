# FTMO ML Trading System

Autonomous ML-powered trading system built to pass the FTMO prop firm challenge. Uses a stacking ensemble of gradient boosted trees with regime detection, FTMO-compliant risk management, and a real-time Streamlit dashboard.

## Backtest Results (Out-of-Sample)

| Metric | XAUUSD | USTEC | Combined |
|--------|--------|-------|----------|
| Return | 34.93% | 1.95% | 37.13% |
| Sharpe Ratio | 1.18 | 0.68 | 1.26 |
| Max Drawdown | 6.65% | 3.45% | 6.65% |
| Win Rate | 43.6% | 46.3% | 43.8% |
| Profit Factor | 1.21 | 1.12 | 1.21 |
| Total Trades | 3,288 | 227 | 3,515 |
| FTMO Phase 1 | PASS | - | PASS |

**Monte Carlo Validation (1,000 simulations):** 100% Phase 1 pass rate, 0% blowup rate, worst-case MaxDD 6.35%.

## Architecture

```
Stacking Ensemble
├── LightGBM
├── XGBoost
├── CatBoost
└── Meta-Learner: Logistic Regression

Signal Pipeline
  Raw OHLCV (MT5) -> 36 Features -> Ensemble Prediction -> Median-Centered Thresholds -> Signal

Risk Pipeline
  Signal -> Regime Filter (GMM) -> ATR Position Sizing -> FTMO Limit Checks -> Execution
```

### Key Components

- **Feature Engineering** — 36 features across 5 tiers: price action, volatility, momentum, volume, and time/session. Pure numpy/pandas implementations (no external TA libraries).
- **Triple-Barrier Labeling** — Lopez de Prado's method with ATR-based barriers and time expiry.
- **Regime Detection** — Gaussian Mixture Model identifies 3 market states (low-vol trend, moderate, high-vol chaos) and scales position sizes accordingly.
- **FTMO Risk Management** — Enforces daily loss halt (3%), total drawdown halt (8%), max concurrent trades, and Best Day Rule compliance with safety buffers before FTMO hard limits.
- **Median-Centered Signals** — Thresholds calibrated relative to model output median, eliminating directional bias from asymmetric probability distributions.

## Project Structure

```
├── config/
│   └── settings.yaml          # All system parameters
├── src/
│   ├── data/
│   │   └── mt5_connector.py   # MetaTrader 5 data interface
│   ├── features/
│   │   ├── engineer.py        # 36-feature pipeline
│   │   └── labeler.py         # Triple-barrier labeling
│   ├── models/
│   │   └── ensemble.py        # Stacking ensemble
│   ├── regime/
│   │   └── hmm_detector.py    # GMM regime detection
│   ├── risk/
│   │   └── manager.py         # FTMO-compliant risk manager
│   ├── execution/
│   │   └── mt5_executor.py    # MT5 trade execution
│   ├── backtest/
│   │   └── engine.py          # FTMO-accurate backtester
│   ├── visualization/
│   │   └── charts.py          # 13+ Plotly chart functions
│   ├── pipeline.py            # Main ML pipeline orchestrator
│   └── live_trader.py         # Live trading loop
├── scripts/
│   ├── optimize_params.py     # Grid search optimization
│   ├── combined_backtest.py   # Multi-instrument backtest
│   ├── monte_carlo.py         # Monte Carlo stress test
│   └── generate_final_report.py
├── dashboard.py               # Streamlit dashboard
├── RESEARCH.md                # Full research document
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.10+
- MetaTrader 5 terminal installed and running
- An MT5 account (demo or live)

### Installation

```bash
git clone https://github.com/Uzzichil1/Machine-learning-day-trading.git
cd Machine-learning-day-trading
pip install -r requirements.txt
```

### Configuration

Edit `config/settings.yaml` to set your account parameters, instruments, and risk limits. Key settings:

```yaml
risk:
  risk_per_trade_pct: 0.40      # Optimized via grid search
  stop_loss_atr_multiple: 1.5
  take_profit_atr_multiple: 2.50 # 1.67R reward-to-risk
  signal_offset: 0.02           # Median +/- threshold
```

## Usage

### Run Full Pipeline (Download + Train + Backtest)

```bash
python -m src.pipeline
```

This will:
1. Download historical data from MT5 for all enabled instruments
2. Engineer features and create labels
3. Train the stacking ensemble
4. Run FTMO-compliant backtest on out-of-sample data
5. Generate interactive HTML reports in `reports/`

### Run Optimization

```bash
python scripts/optimize_params.py    # Grid search over risk/reward params
python scripts/monte_carlo.py        # Monte Carlo stress test
python scripts/combined_backtest.py  # Multi-instrument combined backtest
```

### Launch Dashboard

```bash
python -m streamlit run dashboard.py
```

Dashboard pages:
- **Dashboard** — Combined metrics, FTMO compliance gauges, equity curves
- **Backtest Results** — Per-instrument equity, drawdown, daily P&L, trade distribution
- **Model Analysis** — Feature importance, probability distributions, hourly performance
- **Configuration** — Live view of all system parameters
- **Live Trading** — MT5 account monitor, open positions, candlestick charts

### Live Trading

```bash
python -m src.live_trader
```

Requires MT5 terminal running with a connected account. The system polls for new H1 bars and executes trades automatically based on model signals.

## Optimized Parameters

Found via grid search across 200 parameter combinations per instrument:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Risk/Trade | 0.40% | Best speed/safety tradeoff |
| Stop Loss | 1.5x ATR | Tight enough for good R:R |
| Take Profit | 2.5x ATR | 1.67R, sweet spot for win rate vs payoff |
| Signal Offset | 0.02 | From median, balances buy/sell signals |
| Max Holding | 8 bars | H1 timeframe, ~1 trading day |

## FTMO Challenge Rules Targeted

| Rule | Limit | System Buffer |
|------|-------|---------------|
| Max Daily Loss | 5% | Halt at 3% |
| Max Total Loss | 10% | Halt at 8% |
| Profit Target (Phase 1) | 10% | Achieved 37% in backtest |
| Profit Target (Phase 2) | 5% | Achieved 37% in backtest |
| Best Day Rule | < 50% | Best day 4.3% of total |
| Min Trading Days | 4 | 3,515 trades across full period |

## Tech Stack

- **ML**: LightGBM, XGBoost, CatBoost, scikit-learn
- **Data**: MetaTrader5 Python API, pandas, numpy
- **Visualization**: Plotly, Streamlit
- **Risk**: Custom FTMO-compliant engine with real-time monitoring

## License

This project is for educational and research purposes. Trading involves risk of financial loss. Use at your own discretion.
