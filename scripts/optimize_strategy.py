"""Phase 3: Strategy Optimization — Find optimal parameters for FTMO pass rate.

Uses trade-level data from Phase 2 (reports/trade_analysis.csv) to rapidly
simulate different strategy configurations WITHOUT retraining models.

Optimizes:
  1. Risk per trade (0.5% - 3.0%)
  2. Signal threshold (prob_deviation filter)
  3. Early-exit kill switch (N trades, PnL threshold)
  4. Dynamic sizing (streak-based scaling)
  5. First-passage optimal sizing (maximize P(+10% before -10%))

Each configuration is evaluated by simulating all 14 folds and computing
FTMO pass rate as the sole objective function.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_trades():
    """Load trade-level data from Phase 2."""
    csv_path = PROJECT_ROOT / "reports" / "trade_analysis.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} trades across {df['fold'].nunique()} folds")
    return df


def simulate_fold(fold_trades, config):
    """Simulate a single fold with given configuration.

    Returns dict with fold outcome metrics.

    Config keys:
        risk_pct: base risk per trade (fraction, e.g. 0.0175)
        min_confidence: minimum prob_deviation to take trade
        kill_switch_n: number of trades before kill switch check
        kill_switch_thresh: PnL threshold (fraction) to halt
        streak_scale: if True, scale down after consecutive losses
        streak_loss_count: N consecutive losses to trigger scale-down
        streak_scale_factor: multiply risk by this after streak (e.g. 0.5)
        max_daily_loss: daily halt threshold (fraction)
        max_total_loss: total halt threshold (fraction)
        buy_only: if True, only take buy signals
    """
    risk_pct = config.get("risk_pct", 0.0175)
    min_confidence = config.get("min_confidence", 0.0)
    kill_n = config.get("kill_switch_n", 0)
    kill_thresh = config.get("kill_switch_thresh", -999)
    streak_scale = config.get("streak_scale", False)
    streak_loss_count = config.get("streak_loss_count", 3)
    streak_scale_factor = config.get("streak_scale_factor", 0.5)
    max_daily_loss = config.get("max_daily_loss", 0.035)
    max_total_loss = config.get("max_total_loss", 0.09)
    buy_only = config.get("buy_only", False)
    dynamic_accel = config.get("dynamic_accel", False)
    accel_threshold = config.get("accel_threshold", 0.07)
    accel_factor = config.get("accel_factor", 1.5)

    initial_balance = 100_000
    balance = initial_balance
    trade_count = 0
    consecutive_losses = 0
    killed = False
    halted = False
    daily_pnl = {}
    current_date = None
    day_start_balance = balance

    for _, trade in fold_trades.iterrows():
        # Filter by confidence
        if trade["prob_deviation"] < min_confidence:
            continue

        # Filter buy-only
        if buy_only and trade["signal"] != 1:
            continue

        # Date tracking for daily halt
        trade_date = str(trade["timestamp"])[:10]
        if trade_date != current_date:
            current_date = trade_date
            day_start_balance = balance

        # Check halts
        total_dd = (initial_balance - balance) / initial_balance
        if total_dd >= max_total_loss:
            halted = True
            break

        daily_dd = (day_start_balance - balance) / day_start_balance if day_start_balance > 0 else 0
        if daily_dd >= max_daily_loss:
            continue

        if killed:
            continue

        # Determine effective risk
        current_risk = risk_pct

        # Streak-based scaling
        if streak_scale and consecutive_losses >= streak_loss_count:
            current_risk *= streak_scale_factor

        # Dynamic acceleration near target
        if dynamic_accel:
            current_return = (balance - initial_balance) / initial_balance
            if current_return >= accel_threshold:
                current_risk *= accel_factor

        # Scale by confidence and regime (use original trade data)
        effective_risk = current_risk * trade["confidence"] * trade["regime_scalar"]
        risk_amount = balance * effective_risk

        # Apply trade outcome (r_multiple from original simulation)
        r_multiple = trade["r_multiple"]
        sl_dist = trade["atr"] * 1.5  # SL distance

        if sl_dist > 0:
            dollar_pnl = r_multiple * risk_amount
        else:
            continue

        balance += dollar_pnl
        trade_count += 1

        # Track daily PnL
        if trade_date not in daily_pnl:
            daily_pnl[trade_date] = 0.0
        daily_pnl[trade_date] += dollar_pnl

        # Track consecutive losses
        if dollar_pnl > 0:
            consecutive_losses = 0
        else:
            consecutive_losses += 1

        # Kill switch check
        if kill_n > 0 and trade_count == kill_n:
            cum_pnl_pct = (balance - initial_balance) / initial_balance
            if cum_pnl_pct < kill_thresh:
                killed = True

    # Compute results
    total_return = (balance - initial_balance) / initial_balance
    max_dd = max(0, (initial_balance - balance) / initial_balance) if balance < initial_balance else 0

    # Best day check
    best_day_pct = 0
    if daily_pnl:
        positive_days = {d: p for d, p in daily_pnl.items() if p > 0}
        if positive_days:
            total_pos = sum(positive_days.values())
            best = max(positive_days.values())
            best_day_pct = best / total_pos if total_pos > 0 else 0

    return {
        "total_return": total_return,
        "total_trades": trade_count,
        "phase1_passed": total_return >= 0.10,
        "blown": halted,
        "killed_early": killed,
        "max_dd": max_dd,
        "best_day_pct": best_day_pct,
        "best_day_violated": best_day_pct >= 0.50,
    }


def evaluate_config(trades_df, config):
    """Evaluate a configuration across all folds.

    Returns summary metrics.
    """
    folds = sorted(trades_df["fold"].unique())
    results = []

    for fold in folds:
        fold_trades = trades_df[trades_df["fold"] == fold].copy()
        if len(fold_trades) == 0:
            continue
        result = simulate_fold(fold_trades, config)
        result["fold"] = fold
        results.append(result)

    if not results:
        return None

    rdf = pd.DataFrame(results)

    n_folds = len(rdf)
    n_passed = rdf["phase1_passed"].sum()
    n_blown = rdf["blown"].sum()
    n_killed = rdf["killed_early"].sum()
    pass_rate = n_passed / n_folds
    blow_rate = n_blown / n_folds
    mean_return = rdf["total_return"].mean()
    median_return = rdf["total_return"].median()
    best_day_violations = rdf["best_day_violated"].sum()

    # FTMO-adjusted EV
    pass_returns = rdf[rdf["phase1_passed"]]["total_return"]
    ev_pass = pass_returns.mean() * 100_000 if len(pass_returns) > 0 else 0
    ftmo_ev = pass_rate * ev_pass - 500  # fee

    return {
        "n_folds": n_folds,
        "n_passed": int(n_passed),
        "n_blown": int(n_blown),
        "n_killed": int(n_killed),
        "pass_rate": pass_rate,
        "blow_rate": blow_rate,
        "mean_return": mean_return,
        "median_return": median_return,
        "ftmo_ev": ftmo_ev,
        "best_day_violations": int(best_day_violations),
        "per_fold": results,
    }


def run_baseline(trades_df):
    """Run current baseline configuration."""
    print("\n" + "=" * 70)
    print("BASELINE (Current Configuration)")
    print("=" * 70)

    config = {
        "risk_pct": 0.0175,
        "min_confidence": 0.0,
        "kill_switch_n": 0,
        "kill_switch_thresh": -999,
        "streak_scale": False,
        "max_daily_loss": 0.035,
        "max_total_loss": 0.09,
        "buy_only": False,
    }

    result = evaluate_config(trades_df, config)
    print_result("BASELINE", config, result)
    return result


def print_result(label, config, result):
    """Print evaluation result."""
    print(f"\n  {label}:")
    print(f"    Pass: {result['n_passed']}/{result['n_folds']} ({result['pass_rate']:.0%})")
    print(f"    Blow: {result['n_blown']}/{result['n_folds']} ({result['blow_rate']:.0%})")
    if result['n_killed'] > 0:
        print(f"    Killed early: {result['n_killed']}/{result['n_folds']}")
    print(f"    Mean return: {result['mean_return']:.2%}")
    print(f"    Median return: {result['median_return']:.2%}")
    print(f"    FTMO EV: ${result['ftmo_ev']:+,.0f}")
    if result['best_day_violations'] > 0:
        print(f"    Best Day violations: {result['best_day_violations']}")

    # Per-fold detail
    for r in result["per_fold"]:
        status = "PASS" if r["phase1_passed"] else ("BLOW" if r["blown"] else ("KILL" if r["killed_early"] else "fail"))
        print(f"      Fold {r['fold']:>2}: {status:>4} | Ret={r['total_return']:>+7.2%} | "
              f"Trades={r['total_trades']:>4} | DD={r['max_dd']:.2%}")


def sweep_kill_switch(trades_df):
    """Sweep kill switch parameters."""
    print("\n" + "=" * 70)
    print("SWEEP 1: KILL SWITCH PARAMETERS")
    print("=" * 70)

    best_pass_rate = 0
    best_config = None
    best_result = None

    results_table = []

    for n_trades in [3, 5, 7, 10, 15]:
        for thresh in [-0.002, -0.005, -0.01, -0.015, -0.02, -0.025, -0.03]:
            config = {
                "risk_pct": 0.0175,
                "kill_switch_n": n_trades,
                "kill_switch_thresh": thresh,
                "max_daily_loss": 0.035,
                "max_total_loss": 0.09,
            }
            result = evaluate_config(trades_df, config)

            results_table.append({
                "n_trades": n_trades,
                "threshold": f"{thresh:.1%}",
                "pass_rate": result["pass_rate"],
                "blow_rate": result["blow_rate"],
                "killed": result["n_killed"],
                "mean_return": result["mean_return"],
                "ftmo_ev": result["ftmo_ev"],
            })

            if result["pass_rate"] > best_pass_rate or \
               (result["pass_rate"] == best_pass_rate and result["blow_rate"] < best_result["blow_rate"]):
                best_pass_rate = result["pass_rate"]
                best_config = config.copy()
                best_result = result

    rdf = pd.DataFrame(results_table)
    print(rdf.to_string(index=False))

    print(f"\n  Best kill switch: n={best_config['kill_switch_n']}, "
          f"thresh={best_config['kill_switch_thresh']:.1%}")
    print_result("BEST KILL SWITCH", best_config, best_result)

    return best_config, best_result


def sweep_risk_sizing(trades_df, kill_config):
    """Sweep risk per trade with best kill switch."""
    print("\n" + "=" * 70)
    print("SWEEP 2: RISK PER TRADE (with kill switch)")
    print("=" * 70)

    best_pass_rate = 0
    best_ev = -999999
    best_config = None
    best_result = None

    results_table = []

    for risk_pct in [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03]:
        config = {
            **kill_config,
            "risk_pct": risk_pct,
        }
        result = evaluate_config(trades_df, config)

        results_table.append({
            "risk_pct": f"{risk_pct:.2%}",
            "pass_rate": result["pass_rate"],
            "blow_rate": result["blow_rate"],
            "mean_return": result["mean_return"],
            "median_return": result["median_return"],
            "ftmo_ev": result["ftmo_ev"],
            "best_day_v": result["best_day_violations"],
        })

        # Optimize for pass_rate first, then EV as tiebreaker
        if result["pass_rate"] > best_pass_rate or \
           (result["pass_rate"] == best_pass_rate and result["ftmo_ev"] > best_ev):
            best_pass_rate = result["pass_rate"]
            best_ev = result["ftmo_ev"]
            best_config = config.copy()
            best_result = result

    rdf = pd.DataFrame(results_table)
    print(rdf.to_string(index=False))

    print(f"\n  Best risk: {best_config['risk_pct']:.2%}")
    print_result("BEST RISK", best_config, best_result)

    return best_config, best_result


def sweep_confidence_filter(trades_df, base_config):
    """Sweep minimum confidence (prob_deviation) threshold."""
    print("\n" + "=" * 70)
    print("SWEEP 3: MINIMUM CONFIDENCE FILTER (with kill + risk)")
    print("=" * 70)

    best_pass_rate = 0
    best_ev = -999999
    best_config = None
    best_result = None

    results_table = []

    for min_conf in [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]:
        config = {
            **base_config,
            "min_confidence": min_conf,
        }
        result = evaluate_config(trades_df, config)

        mean_trades = np.mean([r["total_trades"] for r in result["per_fold"]])

        results_table.append({
            "min_conf": f"{min_conf:.3f}",
            "pass_rate": result["pass_rate"],
            "blow_rate": result["blow_rate"],
            "mean_trades": f"{mean_trades:.0f}",
            "mean_return": result["mean_return"],
            "ftmo_ev": result["ftmo_ev"],
        })

        if result["pass_rate"] > best_pass_rate or \
           (result["pass_rate"] == best_pass_rate and result["ftmo_ev"] > best_ev):
            best_pass_rate = result["pass_rate"]
            best_ev = result["ftmo_ev"]
            best_config = config.copy()
            best_result = result

    rdf = pd.DataFrame(results_table)
    print(rdf.to_string(index=False))

    print(f"\n  Best confidence filter: {best_config['min_confidence']:.3f}")
    print_result("BEST CONFIDENCE", best_config, best_result)

    return best_config, best_result


def sweep_streak_scaling(trades_df, base_config):
    """Sweep streak-based position scaling."""
    print("\n" + "=" * 70)
    print("SWEEP 4: STREAK-BASED SCALING (with kill + risk + conf)")
    print("=" * 70)

    best_pass_rate = 0
    best_ev = -999999
    best_config = None
    best_result = None

    results_table = []

    # No streak scaling baseline
    config_no = {**base_config, "streak_scale": False}
    result_no = evaluate_config(trades_df, config_no)
    results_table.append({
        "streak": "OFF",
        "n_losses": "-",
        "scale": "-",
        "pass_rate": result_no["pass_rate"],
        "blow_rate": result_no["blow_rate"],
        "mean_return": result_no["mean_return"],
        "ftmo_ev": result_no["ftmo_ev"],
    })
    best_pass_rate = result_no["pass_rate"]
    best_ev = result_no["ftmo_ev"]
    best_config = config_no.copy()
    best_result = result_no

    for n_losses in [2, 3, 4, 5]:
        for scale_factor in [0.25, 0.5, 0.75]:
            config = {
                **base_config,
                "streak_scale": True,
                "streak_loss_count": n_losses,
                "streak_scale_factor": scale_factor,
            }
            result = evaluate_config(trades_df, config)

            results_table.append({
                "streak": "ON",
                "n_losses": n_losses,
                "scale": f"{scale_factor:.2f}",
                "pass_rate": result["pass_rate"],
                "blow_rate": result["blow_rate"],
                "mean_return": result["mean_return"],
                "ftmo_ev": result["ftmo_ev"],
            })

            if result["pass_rate"] > best_pass_rate or \
               (result["pass_rate"] == best_pass_rate and result["ftmo_ev"] > best_ev):
                best_pass_rate = result["pass_rate"]
                best_ev = result["ftmo_ev"]
                best_config = config.copy()
                best_result = result

    rdf = pd.DataFrame(results_table)
    print(rdf.to_string(index=False))

    print(f"\n  Best streak config: {'ON' if best_config.get('streak_scale') else 'OFF'}")
    if best_config.get("streak_scale"):
        print(f"    N losses: {best_config['streak_loss_count']}, "
              f"Scale: {best_config['streak_scale_factor']:.2f}")
    print_result("BEST STREAK", best_config, best_result)

    return best_config, best_result


def sweep_dynamic_acceleration(trades_df, base_config):
    """Sweep dynamic acceleration near target."""
    print("\n" + "=" * 70)
    print("SWEEP 5: DYNAMIC ACCELERATION NEAR TARGET")
    print("=" * 70)

    best_pass_rate = 0
    best_ev = -999999
    best_config = None
    best_result = None

    results_table = []

    # No acceleration baseline
    config_no = {**base_config, "dynamic_accel": False}
    result_no = evaluate_config(trades_df, config_no)
    results_table.append({
        "accel": "OFF", "threshold": "-", "factor": "-",
        "pass_rate": result_no["pass_rate"], "blow_rate": result_no["blow_rate"],
        "mean_return": result_no["mean_return"], "ftmo_ev": result_no["ftmo_ev"],
    })
    best_pass_rate = result_no["pass_rate"]
    best_ev = result_no["ftmo_ev"]
    best_config = config_no.copy()
    best_result = result_no

    for accel_thresh in [0.03, 0.05, 0.07, 0.08]:
        for accel_factor in [1.25, 1.5, 2.0]:
            config = {
                **base_config,
                "dynamic_accel": True,
                "accel_threshold": accel_thresh,
                "accel_factor": accel_factor,
            }
            result = evaluate_config(trades_df, config)

            results_table.append({
                "accel": "ON",
                "threshold": f"{accel_thresh:.0%}",
                "factor": f"{accel_factor:.2f}",
                "pass_rate": result["pass_rate"],
                "blow_rate": result["blow_rate"],
                "mean_return": result["mean_return"],
                "ftmo_ev": result["ftmo_ev"],
            })

            if result["pass_rate"] > best_pass_rate or \
               (result["pass_rate"] == best_pass_rate and result["ftmo_ev"] > best_ev):
                best_pass_rate = result["pass_rate"]
                best_ev = result["ftmo_ev"]
                best_config = config.copy()
                best_result = result

    rdf = pd.DataFrame(results_table)
    print(rdf.to_string(index=False))

    if best_config.get("dynamic_accel"):
        print(f"\n  Best accel: threshold={best_config['accel_threshold']:.0%}, "
              f"factor={best_config['accel_factor']:.2f}")
    else:
        print(f"\n  Best: No acceleration")
    print_result("BEST ACCEL", best_config, best_result)

    return best_config, best_result


def sweep_buy_only(trades_df, base_config):
    """Test buy-only vs both directions."""
    print("\n" + "=" * 70)
    print("SWEEP 6: BUY-ONLY vs BOTH DIRECTIONS")
    print("=" * 70)

    for buy_only in [False, True]:
        config = {**base_config, "buy_only": buy_only}
        result = evaluate_config(trades_df, config)
        label = "BUY-ONLY" if buy_only else "BOTH DIRECTIONS"
        print_result(label, config, result)

    # Return whichever is better
    config_both = {**base_config, "buy_only": False}
    config_buy = {**base_config, "buy_only": True}
    r_both = evaluate_config(trades_df, config_both)
    r_buy = evaluate_config(trades_df, config_buy)

    if r_buy["pass_rate"] > r_both["pass_rate"]:
        return config_buy, r_buy
    return config_both, r_both


def monte_carlo_validation(trades_df, config, n_sims=10000):
    """Monte Carlo validation of final configuration.

    For each fold, shuffle trade order and simulate 1000 times.
    Gives confidence interval on pass rate.
    """
    print("\n" + "=" * 70)
    print("MONTE CARLO VALIDATION OF OPTIMIZED STRATEGY")
    print(f"({n_sims} simulations per fold)")
    print("=" * 70)

    folds = sorted(trades_df["fold"].unique())
    fold_pass_rates = []

    rng = np.random.default_rng(42)

    for fold in folds:
        fold_trades = trades_df[trades_df["fold"] == fold].copy()
        if len(fold_trades) < 5:
            fold_pass_rates.append(0.0)
            continue

        passes = 0
        for _ in range(n_sims):
            # Shuffle trade order within fold
            shuffled = fold_trades.sample(frac=1.0, random_state=rng.integers(0, 2**31))
            result = simulate_fold(shuffled, config)
            if result["phase1_passed"]:
                passes += 1

        fold_pr = passes / n_sims
        fold_pass_rates.append(fold_pr)
        n_trades = len(fold_trades)
        print(f"  Fold {fold:>2} ({n_trades:>4} trades): MC pass rate = {fold_pr:.1%}")

    # Overall pass rate = mean of fold pass rates
    mean_pr = np.mean(fold_pass_rates)
    # Bootstrap CI on mean
    boot_prs = []
    for _ in range(10000):
        idx = rng.choice(len(fold_pass_rates), size=len(fold_pass_rates), replace=True)
        boot_prs.append(np.mean(np.array(fold_pass_rates)[idx]))

    ci = np.percentile(boot_prs, [2.5, 97.5])

    print(f"\n  Overall MC pass rate: {mean_pr:.1%}")
    print(f"  95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]")

    return mean_pr, ci


def main():
    trades_df = load_trades()

    # =====================================================================
    # BASELINE
    # =====================================================================
    baseline = run_baseline(trades_df)

    # =====================================================================
    # SWEEP 1: Kill Switch
    # =====================================================================
    kill_config, kill_result = sweep_kill_switch(trades_df)

    # =====================================================================
    # SWEEP 2: Risk Per Trade (with best kill switch)
    # =====================================================================
    risk_config, risk_result = sweep_risk_sizing(trades_df, kill_config)

    # =====================================================================
    # SWEEP 3: Confidence Filter
    # =====================================================================
    conf_config, conf_result = sweep_confidence_filter(trades_df, risk_config)

    # =====================================================================
    # SWEEP 4: Streak Scaling
    # =====================================================================
    streak_config, streak_result = sweep_streak_scaling(trades_df, conf_config)

    # =====================================================================
    # SWEEP 5: Dynamic Acceleration
    # =====================================================================
    accel_config, accel_result = sweep_dynamic_acceleration(trades_df, streak_config)

    # =====================================================================
    # SWEEP 6: Buy-Only
    # =====================================================================
    dir_config, dir_result = sweep_buy_only(trades_df, accel_config)

    # =====================================================================
    # FINAL COMPARISON
    # =====================================================================
    final_config = dir_config
    final_result = evaluate_config(trades_df, final_config)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON: BASELINE vs OPTIMIZED")
    print("=" * 70)

    print(f"\n  {'Metric':<25} {'Baseline':>12} {'Optimized':>12} {'Delta':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Pass Rate':<25} {baseline['pass_rate']:>11.0%} {final_result['pass_rate']:>11.0%} "
          f"{final_result['pass_rate'] - baseline['pass_rate']:>+9.0%}")
    print(f"  {'Blow Rate':<25} {baseline['blow_rate']:>11.0%} {final_result['blow_rate']:>11.0%} "
          f"{final_result['blow_rate'] - baseline['blow_rate']:>+9.0%}")
    print(f"  {'Mean Return':<25} {baseline['mean_return']:>11.2%} {final_result['mean_return']:>11.2%} "
          f"{final_result['mean_return'] - baseline['mean_return']:>+9.2%}")
    print(f"  {'FTMO EV':<25} ${baseline['ftmo_ev']:>10,.0f} ${final_result['ftmo_ev']:>10,.0f} "
          f"${final_result['ftmo_ev'] - baseline['ftmo_ev']:>+9,.0f}")

    # Print final config
    print(f"\n  OPTIMIZED CONFIGURATION:")
    for k, v in final_config.items():
        if k.startswith("max_"):
            print(f"    {k}: {v:.1%}")
        elif isinstance(v, float) and v < 1:
            print(f"    {k}: {v}")
        else:
            print(f"    {k}: {v}")

    # Per-fold comparison
    print(f"\n  PER-FOLD COMPARISON:")
    print(f"  {'Fold':>4} {'Baseline':>12} {'Optimized':>12} {'Change':>10}")
    print(f"  {'-'*45}")
    for b, o in zip(baseline["per_fold"], final_result["per_fold"]):
        b_status = "PASS" if b["phase1_passed"] else ("BLOW" if b["blown"] else "fail")
        o_status = "PASS" if o["phase1_passed"] else ("BLOW" if o["blown"] else ("KILL" if o["killed_early"] else "fail"))
        change = ""
        if b_status != o_status:
            change = f"{b_status}->{o_status}"
        print(f"  {b['fold']:>4} {b['total_return']:>+11.2%} {o['total_return']:>+11.2%} {change:>10}")

    # =====================================================================
    # MONTE CARLO VALIDATION
    # =====================================================================
    mc_pr, mc_ci = monte_carlo_validation(trades_df, final_config, n_sims=2000)

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE — OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"  Baseline pass rate:   {baseline['pass_rate']:.0%}")
    print(f"  Optimized pass rate:  {final_result['pass_rate']:.0%}")
    print(f"  MC validated rate:    {mc_pr:.1%} [{mc_ci[0]:.1%}, {mc_ci[1]:.1%}]")
    print(f"  Blow rate reduction:  {baseline['blow_rate']:.0%} -> {final_result['blow_rate']:.0%}")
    print(f"  FTMO EV improvement:  ${baseline['ftmo_ev']:+,.0f} -> ${final_result['ftmo_ev']:+,.0f}")


if __name__ == "__main__":
    main()
