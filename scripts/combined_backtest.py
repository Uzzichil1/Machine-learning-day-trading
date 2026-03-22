"""Combined multi-instrument backtest with shared FTMO equity and drawdown tracking."""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data.mt5_connector import MT5Connector
from src.features.engineer import FeatureEngineer
from src.features.labeler import triple_barrier_labels
from src.models.ensemble import StackingEnsemble
from src.regime.hmm_detector import RegimeDetector
from src.backtest.engine import BacktestTrade, BacktestResult
from src.pipeline import load_config, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

config = load_config()


def prepare_signals(symbol: str) -> pd.DataFrame:
    """Prepare signal DataFrame for a symbol."""
    connector = MT5Connector()
    feature_eng = FeatureEngineer(config.get("features", {}))

    tf = config["timeframes"]["signal"]
    data_dir = str(PROJECT_ROOT / "data" / "raw")
    df = connector.load_data(symbol, tf, data_dir)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = feature_eng.compute_all(df)

    label_cfg = config.get("labeling", {})
    df["target"] = triple_barrier_labels(
        df,
        upper_barrier_atr=label_cfg.get("upper_barrier_atr", 1.5),
        lower_barrier_atr=label_cfg.get("lower_barrier_atr", 1.5),
        max_holding_bars=label_cfg.get("max_holding_bars", 8),
    )

    feature_cols = feature_eng.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])
    df = df[df["target"] != 0]
    df["target"] = (df["target"] == 1).astype(int)

    data_cfg = config.get("data", {})
    n = len(df)
    val_end = int(n * (data_cfg.get("train_pct", 0.6) + data_cfg.get("validation_pct", 0.2)))

    test_df = df.iloc[val_end:]
    X_test = test_df[feature_cols].values

    # Load model
    ensemble = StackingEnsemble(config.get("model", {}))
    model_path = os.path.join(str(PROJECT_ROOT / "models" / "saved"), f"{symbol}_ensemble.joblib")
    ensemble.load(model_path)

    probas = ensemble.predict_proba(X_test)

    # Load saved regime detector (fitted on training data only — no test leakage)
    regime_scalars = np.ones(len(test_df))
    regime_path = os.path.join(str(PROJECT_ROOT / "models" / "saved"), f"{symbol}_regime.joblib")
    h4_data = connector.load_data(symbol, config["timeframes"]["regime"], data_dir)
    if os.path.exists(regime_path) and not h4_data.empty:
        regime = RegimeDetector(n_states=3)
        regime.load(regime_path)
        try:
            h4_scalars = regime.get_size_scalar(h4_data)
            h4_scalars.index = h4_scalars.index.tz_localize(None) if h4_scalars.index.tz else h4_scalars.index
            test_idx = test_df.index.tz_localize(None) if test_df.index.tz else test_df.index
            regime_reindexed = h4_scalars.reindex(test_idx, method="ffill").fillna(1.0)
            regime_scalars = regime_reindexed.values
        except Exception:
            pass

    # Build signals — use validation-calibrated median from model (no test leakage)
    risk_cfg = config.get("risk", {})
    signal_offset = risk_cfg.get("signal_offset", 0.02)
    median_proba = ensemble._median_proba  # Calibrated from validation set
    buy_thresh = median_proba + signal_offset
    sell_thresh = median_proba - signal_offset

    signals = pd.DataFrame(index=test_df.index)
    signals["signal"] = 0
    signals.loc[probas >= buy_thresh, "signal"] = 1
    signals.loc[probas <= sell_thresh, "signal"] = -1

    signals["confidence"] = np.where(
        signals["signal"] == 1,
        (probas - median_proba) / (1 - median_proba),
        np.where(signals["signal"] == -1, (median_proba - probas) / median_proba, 0.5),
    ).clip(0.5, 1.0)

    signals["atr"] = test_df["atr"].values
    signals["regime_scalar"] = regime_scalars
    signals["symbol"] = symbol

    n_signals = (signals["signal"] != 0).sum()
    logger.info(
        f"{symbol}: {len(test_df)} bars, {n_signals} signals "
        f"({(signals['signal']==1).sum()} buy, {(signals['signal']==-1).sum()} sell), "
        f"median_proba={median_proba:.4f}"
    )

    return signals, test_df


def combined_backtest(
    all_signals: list[tuple[pd.DataFrame, pd.DataFrame, str]],
) -> BacktestResult:
    """Run combined backtest across all instruments with shared equity."""
    initial_balance = config.get("account", {}).get("initial_balance", 100_000)
    # Use internal safety buffers (stricter than FTMO hard limits)
    risk_cfg_halts = config.get("risk", {})
    max_daily_loss = risk_cfg_halts.get("daily_loss_halt_pct", 4.0) / 100
    max_total_loss = risk_cfg_halts.get("total_drawdown_halt_pct", 9.0) / 100
    phase1_target = config.get("account", {}).get("phase1_target_pct", 10.0) / 100
    phase2_target = config.get("account", {}).get("phase2_target_pct", 5.0) / 100
    risk_cfg = config.get("risk", {})
    risk_per_trade = risk_cfg.get("risk_per_trade_pct", 0.40) / 100
    sl_atr_mult = risk_cfg.get("stop_loss_atr_multiple", 1.5)
    tp_atr_mult = risk_cfg.get("take_profit_atr_multiple", 2.5)
    max_concurrent = risk_cfg.get("max_concurrent_trades", 3)
    max_trades_per_day = 50  # No FTMO limit on trades per day

    # Merge all signals into a single time-sorted stream
    events = []
    for signals, prices, symbol in all_signals:
        for i in range(len(signals)):
            ts = signals.index[i]
            sig = signals.iloc[i]
            if sig["signal"] == 0:
                continue
            events.append({
                "time": ts,
                "symbol": symbol,
                "signal": sig["signal"],
                "confidence": sig["confidence"],
                "atr": sig["atr"],
                "regime_scalar": sig["regime_scalar"],
                "price_idx": i,
                "prices_ref": id(prices),
            })

    # Keep price references accessible
    prices_map = {id(prices): prices for _, prices, _ in all_signals}

    events.sort(key=lambda x: x["time"])
    logger.info(f"Combined event stream: {len(events)} signals across {len(all_signals)} instruments")

    # Simulate
    balance = initial_balance
    peak_balance = initial_balance
    trades = []
    equity_points = []
    daily_balances = {}
    day_start_balance = balance
    current_date = None
    halted_daily = False
    halted_total = False
    trades_today = 0
    # Track active positions per instrument: {symbol: list of close_timestamps}
    active_by_symbol = {}

    for event in events:
        ts = event["time"]
        today = ts.date() if hasattr(ts, "date") else ts

        # New day
        if today != current_date:
            current_date = today
            day_start_balance = balance
            halted_daily = False
            trades_today = 0

        # Check total halt
        total_dd = (initial_balance - balance) / initial_balance
        if total_dd >= max_total_loss:
            halted_total = True

        # Check daily halt
        daily_dd = (day_start_balance - balance) / day_start_balance if day_start_balance > 0 else 0
        if daily_dd >= max_daily_loss:
            halted_daily = True

        if halted_total or halted_daily:
            continue

        if trades_today >= max_trades_per_day:
            continue

        # Expire closed positions per instrument
        prices = prices_map[event["prices_ref"]]
        price_idx = event["price_idx"]
        sym = event["symbol"]

        # Expire same-symbol positions by bar index (indices are only valid within same instrument)
        if sym in active_by_symbol:
            active_by_symbol[sym] = [
                (cb, pref_id) for cb, pref_id in active_by_symbol[sym] if cb > price_idx
            ]

        # Expire cross-instrument positions by timestamp
        # Each position stores (close_bar, prices_ref) — convert close_bar to timestamp
        for other_sym in list(active_by_symbol.keys()):
            if other_sym == sym:
                continue
            # For other instruments, we stored (close_bar, prices_ref_id)
            remaining = []
            for cb, pref_id in active_by_symbol[other_sym]:
                other_prices = prices_map[pref_id]
                if cb < len(other_prices):
                    close_time = other_prices.index[cb]
                    if close_time > ts:
                        remaining.append((cb, pref_id))
                # If cb >= len, position already expired
            active_by_symbol[other_sym] = remaining

        # Count total active positions across all instruments
        total_active = sum(len(v) for v in active_by_symbol.values())

        # Enforce concurrent position limit
        if total_active >= max_concurrent:
            continue

        # Position sizing with confidence and regime
        effective_risk = risk_per_trade * event["confidence"] * event["regime_scalar"]
        risk_amount = balance * effective_risk
        atr = event["atr"]
        if atr <= 0 or np.isnan(atr):
            continue

        sl_dist = sl_atr_mult * atr
        tp_dist = tp_atr_mult * atr

        # Enter on NEXT bar's open (realistic: can't enter at signal bar close)
        if price_idx + 1 >= len(prices):
            continue
        entry_price = prices["open"].iloc[price_idx + 1]

        # Simulate trade starting from bar after entry, track close bar
        trade_pnl = 0.0
        max_bars = 8
        close_bar = price_idx + 1 + max_bars  # Default: time barrier
        for j in range(1, max_bars + 1):
            idx = price_idx + 1 + j
            if idx >= len(prices):
                break
            bar_high = prices["high"].iloc[idx]
            bar_low = prices["low"].iloc[idx]

            if event["signal"] == 1:  # Long
                if bar_high >= entry_price + tp_dist:
                    trade_pnl = tp_dist
                    close_bar = idx
                    break
                if bar_low <= entry_price - sl_dist:
                    trade_pnl = -sl_dist
                    close_bar = idx
                    break
            else:  # Short
                if bar_low <= entry_price - tp_dist:
                    trade_pnl = tp_dist
                    close_bar = idx
                    break
                if bar_high >= entry_price + sl_dist:
                    trade_pnl = -sl_dist
                    close_bar = idx
                    break
        else:
            final_idx = min(price_idx + 1 + max_bars, len(prices) - 1)
            final_close = prices["close"].iloc[final_idx]
            trade_pnl = (final_close - entry_price) * event["signal"]
            close_bar = final_idx

        # Track this position as active until it closes
        if sym not in active_by_symbol:
            active_by_symbol[sym] = []
        active_by_symbol[sym].append((close_bar, event["prices_ref"]))

        # Spread cost
        spread_cost = atr * 0.01
        trade_pnl -= spread_cost

        # Dollar P&L
        if sl_dist > 0:
            r_multiple = trade_pnl / sl_dist
            dollar_pnl = r_multiple * risk_amount
        else:
            dollar_pnl = 0

        balance += dollar_pnl
        trades_today += 1

        trade = BacktestTrade(
            entry_time=ts,
            symbol=event["symbol"],
            direction=event["signal"],
            entry_price=entry_price,
            pnl=dollar_pnl,
            pnl_pct=dollar_pnl / initial_balance,
            risk_pct=effective_risk,
            confidence=event["confidence"],
        )
        trades.append(trade)

        equity_points.append({"time": ts, "equity": balance})

        if today not in daily_balances:
            daily_balances[today] = 0.0
        daily_balances[today] += dollar_pnl

    # Compile results
    result = BacktestResult()
    result.trades = trades
    result.total_trades = len(trades)

    if equity_points:
        eq_df = pd.DataFrame(equity_points).set_index("time")
        result.equity_curve = eq_df["equity"]
        final_equity = result.equity_curve.iloc[-1]
        result.total_return = (final_equity - initial_balance) / initial_balance

        daily_returns = result.equity_curve.resample("D").last().pct_change(fill_method=None).dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            result.sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

        # FTMO total drawdown: lowest equity relative to initial balance
        # (Did balance ever drop below initial - 10%?)
        min_equity = result.equity_curve.min()
        ftmo_total_dd = max(0, (initial_balance - min_equity) / initial_balance)
        result.max_drawdown = ftmo_total_dd

    if daily_balances:
        daily_pnl = pd.Series(daily_balances)
        result.daily_pnl = daily_pnl
        # FTMO daily DD: worst day loss as % of that day's start balance
        # Track day-start balances to compute correctly
        running_balance = initial_balance
        worst_daily_dd_pct = 0.0
        for d in sorted(daily_balances.keys()):
            day_start = running_balance
            day_pnl = daily_balances[d]
            if day_start > 0 and day_pnl < 0:
                dd_pct = abs(day_pnl) / day_start
                worst_daily_dd_pct = max(worst_daily_dd_pct, dd_pct)
            running_balance += day_pnl
        result.max_daily_drawdown = worst_daily_dd_pct

    if trades:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        result.win_rate = len(wins) / len(trades)
        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    result.ftmo_phase1_passed = result.total_return >= phase1_target
    result.ftmo_phase2_passed = result.total_return >= phase2_target
    result.ftmo_daily_limit_breached = halted_daily
    result.ftmo_total_limit_breached = halted_total

    if daily_balances:
        positive_days = {d: p for d, p in daily_balances.items() if p > 0}
        if positive_days:
            total_pos = sum(positive_days.values())
            best = max(positive_days.values())
            result.best_day_pct = best / total_pos if total_pos > 0 else 0

    if result.equity_curve is not None:
        target_equity = initial_balance * (1 + phase1_target)
        target_reached = result.equity_curve[result.equity_curve >= target_equity]
        if len(target_reached) > 0:
            start = result.equity_curve.index[0]
            end = target_reached.index[0]
            result.days_to_target = (end - start).days

    return result


def main():
    symbols_data = []

    for instr in config.get("instruments", []):
        if not instr.get("enabled"):
            continue
        symbol = instr["symbol"]
        signals, prices = prepare_signals(symbol)
        if not signals.empty:
            symbols_data.append((signals, prices, symbol))

    if not symbols_data:
        logger.error("No instrument data available")
        return

    # Run combined backtest
    result = combined_backtest(symbols_data)

    # Print summary
    print("\n" + "=" * 60)
    print("COMBINED MULTI-INSTRUMENT FTMO BACKTEST")
    print("=" * 60)
    print(f"Instruments:        {', '.join(s[2] for s in symbols_data)}")
    print(f"Total Return:       {result.total_return:>10.2%}")
    print(f"Sharpe Ratio:       {result.sharpe_ratio:>10.2f}")
    print(f"Max Drawdown:       {result.max_drawdown:>10.2%}")
    print(f"Max Daily DD:       {result.max_daily_drawdown:>10.2%}")
    print(f"Win Rate:           {result.win_rate:>10.2%}")
    print(f"Profit Factor:      {result.profit_factor:>10.2f}")
    print(f"Total Trades:       {result.total_trades:>10d}")
    print(f"Best Day %:         {result.best_day_pct:>10.2%}")
    print("-" * 60)
    print(f"Phase 1 Passed:     {'YES' if result.ftmo_phase1_passed else 'NO':>10}")
    print(f"Phase 2 Passed:     {'YES' if result.ftmo_phase2_passed else 'NO':>10}")
    print(f"Daily Limit Hit:    {'YES' if result.ftmo_daily_limit_breached else 'NO':>10}")
    print(f"Total Limit Hit:    {'YES' if result.ftmo_total_limit_breached else 'NO':>10}")
    if result.days_to_target > 0:
        print(f"Days to Target:     {result.days_to_target:>10d}")
    print("=" * 60)

    # Per-instrument breakdown
    print("\nPER-INSTRUMENT BREAKDOWN:")
    print("-" * 60)
    for _, _, symbol in symbols_data:
        sym_trades = [t for t in result.trades if t.symbol == symbol]
        if sym_trades:
            sym_pnl = sum(t.pnl for t in sym_trades)
            sym_wins = len([t for t in sym_trades if t.pnl > 0])
            sym_wr = sym_wins / len(sym_trades) if sym_trades else 0
            print(
                f"  {symbol:>8}: {len(sym_trades):>5} trades | "
                f"P&L: ${sym_pnl:>10,.2f} | "
                f"WR: {sym_wr:.1%} | "
                f"Return: {sym_pnl/100_000:.2%}"
            )

    # FTMO compliance check
    print("\nFTMO COMPLIANCE CHECK:")
    print("-" * 60)
    checks = [
        ("Max Drawdown < 10%", result.max_drawdown < 0.10, f"{result.max_drawdown:.2%}"),
        ("Max Daily DD < 5%", result.max_daily_drawdown < 0.05, f"{result.max_daily_drawdown:.2%}"),
        ("Best Day < 50% of profit", result.best_day_pct < 0.50, f"{result.best_day_pct:.1%}"),
        ("Phase 1 Target (10%)", result.ftmo_phase1_passed, f"{result.total_return:.2%}"),
        ("Phase 2 Target (5%)", result.ftmo_phase2_passed, f"{result.total_return:.2%}"),
    ]
    all_passed = True
    for name, passed, value in checks:
        status = "PASS" if passed else "FAIL"
        mark = "[+]" if passed else "[X]"
        print(f"  {mark} {name:.<40} {value:>10} {status}")
        if not passed:
            all_passed = False

    print("-" * 60)
    if all_passed:
        print("  >>> ALL FTMO CHECKS PASSED <<<")
    else:
        print("  >>> SOME CHECKS FAILED <<<")
    print("=" * 60)


if __name__ == "__main__":
    main()
