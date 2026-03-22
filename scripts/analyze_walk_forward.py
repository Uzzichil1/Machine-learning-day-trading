"""Analyze walk-forward results to find optimal strategy.

Key hypotheses to test:
1. Signal density predicts fold success
2. USTEC >> XAUUSD in reliability
3. What kill-switch threshold maximizes expected value per FTMO attempt?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.pipeline import PROJECT_ROOT


def load_results():
    path = str(PROJECT_ROOT / "reports" / "walk_forward_results.csv")
    df = pd.read_csv(path)
    # Split by symbol based on fold numbering reset
    # XAUUSD folds come first (17), then USTEC (14)
    xau = df.iloc[:17].copy()
    xau["symbol"] = "XAUUSD"
    ustec = df.iloc[17:].copy()
    ustec["symbol"] = "USTEC"
    return pd.concat([xau, ustec], ignore_index=True)


def analyze_signal_density(df):
    """Test: does signal count predict fold success?"""
    print("\n" + "=" * 80)
    print("SIGNAL DENSITY ANALYSIS")
    print("=" * 80)

    for sym in ["XAUUSD", "USTEC", "ALL"]:
        subset = df if sym == "ALL" else df[df["symbol"] == sym]
        print(f"\n--- {sym} ---")

        winners = subset[subset["total_return"] > 0]
        losers = subset[subset["total_return"] <= 0]
        passed = subset[subset["phase1_passed"]]
        failed = subset[~subset["phase1_passed"]]

        print(f"  Winners avg trades:  {winners['total_trades'].mean():.0f} "
              f"(min={winners['total_trades'].min()}, max={winners['total_trades'].max()})")
        print(f"  Losers avg trades:   {losers['total_trades'].mean():.0f} "
              f"(min={losers['total_trades'].min()}, max={losers['total_trades'].max()})")
        print(f"  Passed P1 avg trades: {passed['total_trades'].mean():.0f}" if len(passed) > 0 else "")
        print(f"  Failed P1 avg trades: {failed['total_trades'].mean():.0f}" if len(failed) > 0 else "")

        print(f"\n  Winners avg signals: {winners['n_signals'].mean():.0f}")
        print(f"  Losers avg signals:  {losers['n_signals'].mean():.0f}")

        # Test different trade count thresholds
        print(f"\n  TRADE COUNT THRESHOLDS (if trades < threshold, would you have stopped?):")
        print(f"  {'Threshold':>10} {'Caught losers':>15} {'Killed winners':>15} {'Net impact':>12}")
        for thresh in [50, 100, 150, 200, 250, 300]:
            below = subset[subset["total_trades"] < thresh]
            caught_losers = below[below["total_return"] <= 0]
            killed_winners = below[below["total_return"] > 0]

            # Net: savings from not losing + lost gains from killed winners
            saved = caught_losers["total_return"].sum() * -1  # positive = money saved
            lost = killed_winners["total_return"].sum()  # positive = money lost
            net = saved - lost

            print(f"  {thresh:>10} {len(caught_losers):>10}/{len(losers):<4} "
                  f"{len(killed_winners):>10}/{len(winners):<4} "
                  f"{net:>+11.2%}")


def compute_expected_value(df, symbol="ALL", trade_threshold=0, ftmo_fee=500):
    """Compute expected $ per FTMO attempt with optional kill switch."""
    subset = df if symbol == "ALL" else df[df["symbol"] == symbol]

    if trade_threshold > 0:
        # Simulate: if fold generates < threshold trades, assume we stop at -2% (probe loss)
        results = []
        for _, row in subset.iterrows():
            if row["total_trades"] < trade_threshold:
                # Stopped early — assume small loss from probe period
                results.append(-0.02)  # ~2% probe loss
            else:
                results.append(row["total_return"])
        returns = np.array(results)
    else:
        returns = subset["total_return"].values

    mean_ret = returns.mean()
    pass_rate = (returns >= 0.10).mean()
    blow_rate = (returns <= -0.09).mean()
    expected_dollar = mean_ret * 100_000 - ftmo_fee

    return {
        "mean_return": mean_ret,
        "pass_rate": pass_rate,
        "blow_rate": blow_rate,
        "expected_dollar": expected_dollar,
        "n_folds": len(returns),
        "std_return": returns.std(),
    }


def find_optimal_strategy(df):
    """Find the optimal combination of instrument + kill switch."""
    print("\n" + "=" * 80)
    print("EXPECTED VALUE PER FTMO ATTEMPT ($100K account, $500 fee)")
    print("=" * 80)

    configs = [
        ("Both instruments, no kill switch", "ALL", 0),
        ("Both instruments, kill@100 trades", "ALL", 100),
        ("Both instruments, kill@200 trades", "ALL", 200),
        ("USTEC only, no kill switch", "USTEC", 0),
        ("USTEC only, kill@50 trades", "USTEC", 50),
        ("USTEC only, kill@100 trades", "USTEC", 100),
        ("USTEC only, kill@150 trades", "USTEC", 150),
        ("USTEC only, kill@200 trades", "USTEC", 200),
        ("XAUUSD only, no kill switch", "XAUUSD", 0),
        ("XAUUSD only, kill@100 trades", "XAUUSD", 100),
        ("XAUUSD only, kill@200 trades", "XAUUSD", 200),
    ]

    print(f"\n  {'Strategy':<45} {'E[Return]':>10} {'Pass%':>8} {'Blow%':>8} "
          f"{'E[$]':>10} {'Std':>10} {'Sharpe*':>8}")
    print(f"  {'-'*95}")

    best_ev = -999999
    best_config = None

    for name, sym, thresh in configs:
        ev = compute_expected_value(df, sym, thresh)
        sharpe = ev["mean_return"] / ev["std_return"] if ev["std_return"] > 0 else 0

        flag = ""
        if ev["expected_dollar"] > best_ev:
            best_ev = ev["expected_dollar"]
            best_config = name
            flag = " <<<"

        print(f"  {name:<45} {ev['mean_return']:>+9.2%} {ev['pass_rate']:>7.0%} "
              f"{ev['blow_rate']:>7.0%} ${ev['expected_dollar']:>+9,.0f} "
              f"{ev['std_return']:>9.2%} {sharpe:>7.2f}{flag}")

    print(f"\n  BEST STRATEGY: {best_config} (E[${best_ev:+,.0f}] per attempt)")
    return best_config


def analyze_losing_folds(df):
    """What distinguishes winners from losers? Can we predict failure?"""
    print("\n" + "=" * 80)
    print("WINNING vs LOSING FOLD CHARACTERISTICS")
    print("=" * 80)

    for sym in ["XAUUSD", "USTEC"]:
        subset = df[df["symbol"] == sym]
        winners = subset[subset["total_return"] > 0.10]  # Passed Phase 1
        losers = subset[subset["total_breached"]]  # Blew up

        print(f"\n--- {sym} ---")
        metrics = ["val_accuracy", "median_proba", "n_signals", "n_buy", "n_sell",
                    "total_trades", "win_rate", "profit_factor"]

        print(f"  {'Metric':<20} {'Winners':>12} {'Losers':>12} {'Diff':>10}")
        print(f"  {'-'*55}")
        for m in metrics:
            w_mean = winners[m].mean() if len(winners) > 0 else 0
            l_mean = losers[m].mean() if len(losers) > 0 else 0
            diff = w_mean - l_mean
            print(f"  {m:<20} {w_mean:>12.4f} {l_mean:>12.4f} {diff:>+10.4f}")

        # Check buy/sell ratio
        if len(winners) > 0:
            w_buy_ratio = (winners["n_buy"] / winners["n_signals"]).mean()
            print(f"\n  Winners buy signal ratio: {w_buy_ratio:.2%}")
        if len(losers) > 0:
            l_buy_ratio = (losers["n_buy"] / losers["n_signals"]).mean()
            print(f"  Losers buy signal ratio:  {l_buy_ratio:.2%}")


def realistic_ftmo_simulation(df):
    """Simulate repeated FTMO attempts with the best strategy."""
    print("\n" + "=" * 80)
    print("FTMO CHALLENGE SIMULATION (1000 attempts)")
    print("=" * 80)

    # Best strategy: USTEC only, with signal density kill switch
    ustec = df[df["symbol"] == "USTEC"]
    fee = 500
    n_attempts = 1000
    np.random.seed(42)

    strategies = {
        "USTEC no filter": (ustec, 0),
        "USTEC kill@100": (ustec, 100),
        "USTEC kill@150": (ustec, 150),
        "Both no filter": (df, 0),
    }

    for name, (subset, thresh) in strategies.items():
        # Build return distribution
        returns = []
        for _, row in subset.iterrows():
            if thresh > 0 and row["total_trades"] < thresh:
                returns.append(-0.02)
            else:
                returns.append(row["total_return"])
        returns = np.array(returns)

        # Simulate 1000 attempts (sampling with replacement from fold returns)
        cumulative_pnl = []
        total = 0
        for _ in range(n_attempts):
            attempt_return = np.random.choice(returns)
            if attempt_return >= 0.10:
                # Passed! Receive funded account (model with payout)
                payout = attempt_return * 100_000 * 0.80  # 80% profit split
                total += payout - fee
            else:
                # Failed — lose fee
                total += -fee
            cumulative_pnl.append(total)

        final = cumulative_pnl[-1]
        pass_count = sum(1 for r in [np.random.choice(returns) for _ in range(n_attempts)] if r >= 0.10)
        pass_rate = (returns >= 0.10).mean()

        print(f"\n  {name}:")
        print(f"    Pass rate: {pass_rate:.0%}")
        print(f"    After {n_attempts} attempts: ${final:+,.0f}")
        print(f"    Per attempt: ${final/n_attempts:+,.0f}")
        print(f"    Break-even attempts needed: {1/pass_rate:.1f}" if pass_rate > 0 else "    Never breaks even")

        # What about the funded account? If pass, you trade with same system
        # Average return when passing * $100K * 80% split
        passed_returns = returns[returns >= 0.10]
        if len(passed_returns) > 0:
            avg_funded_profit = passed_returns.mean() * 100_000 * 0.80
            print(f"    Avg profit when funded: ${avg_funded_profit:,.0f}")
            print(f"    Avg profit per attempt (incl. failures): "
                  f"${pass_rate * avg_funded_profit - fee:+,.0f}")


def main():
    df = load_results()
    print(f"Loaded {len(df)} walk-forward folds")
    print(f"  XAUUSD: {len(df[df['symbol']=='XAUUSD'])} folds")
    print(f"  USTEC:  {len(df[df['symbol']=='USTEC'])} folds")

    analyze_signal_density(df)
    find_optimal_strategy(df)
    analyze_losing_folds(df)
    realistic_ftmo_simulation(df)


if __name__ == "__main__":
    main()
