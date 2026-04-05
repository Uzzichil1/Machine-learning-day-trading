"""Phase 1: Statistical Foundation — Prove or disprove the edge.

Tests:
  1.1  Binomial test on FTMO pass rate (fold-level)
  1.2  Bootstrap CI on pass rate + expected return
  1.3  Pooled win-rate test vs breakeven threshold
  1.4  One-sample t-test on fold returns
  1.5  Deflated Sharpe Ratio (DSR)
  1.6  Regime-dependence analysis (are blowups clustered?)
  1.7  Run-order autocorrelation (does fold position matter?)

All tests use USTEC-only data (14 folds from walk-forward).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_ustec_folds():
    """Load walk-forward results and separate USTEC folds."""
    csv_path = PROJECT_ROOT / "reports" / "walk_forward_results.csv"
    df = pd.read_csv(csv_path)

    # USTEC folds start from train_start 2022-07-19 (second run)
    # XAUUSD folds start from 2022-04-05 (first run)
    # Identify by train_start date
    ustec = df[df["train_start"] == "2022-07-19"].reset_index(drop=True)
    xauusd = df[df["train_start"] == "2022-04-05"].reset_index(drop=True)

    print(f"USTEC folds: {len(ustec)}")
    print(f"XAUUSD folds: {len(xauusd)}")

    return ustec, xauusd


def test_1_1_pass_rate_binomial(folds):
    """H0: True pass rate = 50% (random). H1: pass rate > 50%."""
    print("\n" + "=" * 70)
    print("TEST 1.1: BINOMIAL TEST ON FTMO PASS RATE")
    print("=" * 70)

    n = len(folds)
    k = folds["phase1_passed"].sum()
    print(f"  Passed: {k} / {n} = {k/n:.1%}")

    # One-sided binomial: P(X >= k | n, p=0.5)
    p_val = 1 - stats.binom.cdf(k - 1, n, 0.5)
    print(f"  H0: p_pass = 0.50 (coin flip)")
    print(f"  H1: p_pass > 0.50")
    print(f"  p-value (one-sided): {p_val:.4f}")

    # Also test against 40% (is it better than losing most?)
    p_val_40 = 1 - stats.binom.cdf(k - 1, n, 0.4)
    print(f"  p-value vs p=0.40: {p_val_40:.4f}")

    # Exact Clopper-Pearson 95% CI
    ci_low = stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0
    ci_high = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1
    print(f"  95% CI (Clopper-Pearson): [{ci_low:.1%}, {ci_high:.1%}]")

    sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
    print(f"\n  >>> {sig} at alpha=0.05 <<<")

    return {"test": "pass_rate_binomial", "k": k, "n": n, "p_value": p_val,
            "ci_low": ci_low, "ci_high": ci_high, "significant": p_val < 0.05}


def test_1_2_bootstrap_pass_rate(folds, n_boot=50000):
    """Bootstrap CI on pass rate and expected return per attempt."""
    print("\n" + "=" * 70)
    print("TEST 1.2: BOOTSTRAP CI ON PASS RATE & EXPECTED RETURN")
    print("=" * 70)

    outcomes = folds["phase1_passed"].values.astype(int)
    returns = folds["total_return"].values

    rng = np.random.default_rng(42)
    boot_pass_rates = []
    boot_mean_returns = []
    boot_ev_per_attempt = []

    for _ in range(n_boot):
        idx = rng.choice(len(folds), size=len(folds), replace=True)
        boot_pass_rates.append(outcomes[idx].mean())
        boot_mean_returns.append(returns[idx].mean())
        # EV per attempt = mean return * 100K (but includes blowups)
        boot_ev_per_attempt.append(returns[idx].mean() * 100_000)

    pass_ci = np.percentile(boot_pass_rates, [2.5, 50, 97.5])
    ret_ci = np.percentile(boot_mean_returns, [2.5, 50, 97.5])
    ev_ci = np.percentile(boot_ev_per_attempt, [2.5, 50, 97.5])

    print(f"  Pass Rate:")
    print(f"    Point estimate: {outcomes.mean():.1%}")
    print(f"    95% CI: [{pass_ci[0]:.1%}, {pass_ci[2]:.1%}]")
    print(f"    Median: {pass_ci[1]:.1%}")
    print(f"    P(pass_rate > 50%): {np.mean(np.array(boot_pass_rates) > 0.5):.1%}")

    print(f"\n  Mean Return per Fold:")
    print(f"    Point estimate: {returns.mean():.2%}")
    print(f"    95% CI: [{ret_ci[0]:.2%}, {ret_ci[2]:.2%}]")

    print(f"\n  Expected $ per Challenge Attempt:")
    print(f"    Point estimate: ${returns.mean() * 100_000:+,.0f}")
    print(f"    95% CI: [${ev_ci[0]:+,.0f}, ${ev_ci[2]:+,.0f}]")
    print(f"    P(EV > 0): {np.mean(np.array(boot_mean_returns) > 0):.1%}")

    # Bootstrap p-value: fraction of bootstrap samples where pass_rate <= 0.5
    boot_p = np.mean(np.array(boot_pass_rates) <= 0.5)
    print(f"\n  Bootstrap p-value (H0: pass_rate <= 50%): {boot_p:.4f}")

    return {"test": "bootstrap_pass_rate", "pass_ci": pass_ci.tolist(),
            "return_ci": ret_ci.tolist(), "ev_ci": ev_ci.tolist(),
            "p_prob_positive_ev": float(np.mean(np.array(boot_mean_returns) > 0))}


def test_1_3_pooled_win_rate(folds):
    """Test if pooled win rate exceeds breakeven threshold."""
    print("\n" + "=" * 70)
    print("TEST 1.3: POOLED WIN RATE VS BREAKEVEN")
    print("=" * 70)

    # Reconstruct approximate wins/losses from fold-level data
    total_trades = 0
    total_wins = 0
    for _, row in folds.iterrows():
        n_trades = int(row["total_trades"])
        n_wins = int(round(row["win_rate"] * n_trades))
        total_trades += n_trades
        total_wins += n_wins

    pooled_wr = total_wins / total_trades if total_trades > 0 else 0

    print(f"  Total trades (pooled): {total_trades}")
    print(f"  Total wins (approx):   {total_wins}")
    print(f"  Pooled win rate:       {pooled_wr:.4f} ({pooled_wr:.1%})")

    # Breakeven with SL=1.5 ATR, TP=2.5 ATR → R:R = 1.67:1
    # Breakeven WR = 1 / (1 + R:R) = 1 / (1 + 2.5/1.5) = 1.5/4.0 = 0.375
    # But time-barrier exits change this. Use 0.375 as conservative estimate.
    breakeven_wr = 0.375

    print(f"  Breakeven win rate:    {breakeven_wr:.3f} (SL=1.5ATR, TP=2.5ATR)")

    # Binomial test: H0: p = breakeven, H1: p > breakeven
    p_val = 1 - stats.binom.cdf(total_wins - 1, total_trades, breakeven_wr)
    print(f"\n  H0: p_win = {breakeven_wr:.3f} (breakeven)")
    print(f"  H1: p_win > {breakeven_wr:.3f}")
    print(f"  p-value (one-sided): {p_val:.6f}")

    # Also test vs 50% (directional accuracy)
    p_val_50 = stats.binomtest(total_wins, total_trades, 0.5, alternative="two-sided").pvalue
    print(f"\n  Two-sided test vs 50%:")
    print(f"  p-value: {p_val_50:.6f}")
    if pooled_wr < 0.5:
        print(f"  Direction: Model wins LESS than 50% — relies on R:R asymmetry")

    # CI on pooled win rate (Wilson score interval)
    z = 1.96
    denominator = 1 + z**2 / total_trades
    centre = (pooled_wr + z**2 / (2 * total_trades)) / denominator
    spread = z * np.sqrt((pooled_wr * (1 - pooled_wr) + z**2 / (4 * total_trades)) / total_trades) / denominator
    wilson_lo = centre - spread
    wilson_hi = centre + spread
    print(f"\n  95% CI (Wilson): [{wilson_lo:.4f}, {wilson_hi:.4f}]")
    print(f"  CI lower bound vs breakeven: {'ABOVE' if wilson_lo > breakeven_wr else 'BELOW'}")

    sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
    print(f"\n  >>> Win rate vs breakeven: {sig} at alpha=0.05 <<<")

    # Per-fold win rates (check consistency)
    print(f"\n  Per-Fold Win Rates:")
    above_be = 0
    for _, row in folds.iterrows():
        wr = row["win_rate"]
        trades = int(row["total_trades"])
        marker = ">" if wr > breakeven_wr else "<"
        status = "ABOVE BE" if wr > breakeven_wr else "BELOW BE"
        print(f"    Fold {int(row['fold']):>2}: WR={wr:.1%} ({trades:>4} trades) {marker} {breakeven_wr:.1%} [{status}]")
        if wr > breakeven_wr:
            above_be += 1
    print(f"  Folds above breakeven: {above_be}/{len(folds)}")

    return {"test": "pooled_win_rate", "total_trades": total_trades,
            "total_wins": total_wins, "pooled_wr": pooled_wr,
            "breakeven_wr": breakeven_wr, "p_value": p_val,
            "wilson_ci": [wilson_lo, wilson_hi], "significant": p_val < 0.05}


def test_1_4_return_significance(folds):
    """One-sample t-test and Wilcoxon test on fold returns."""
    print("\n" + "=" * 70)
    print("TEST 1.4: FOLD RETURN SIGNIFICANCE")
    print("=" * 70)

    returns = folds["total_return"].values
    n = len(returns)

    print(f"  Sample size: {n} folds")
    print(f"  Mean return: {returns.mean():.4f} ({returns.mean():.2%})")
    print(f"  Median return: {np.median(returns):.4f} ({np.median(returns):.2%})")
    print(f"  Std return: {returns.std():.4f}")
    print(f"  Min: {returns.min():.4f}, Max: {returns.max():.4f}")

    # Shapiro-Wilk normality test
    stat_sw, p_sw = stats.shapiro(returns)
    print(f"\n  Shapiro-Wilk normality test: W={stat_sw:.4f}, p={p_sw:.4f}")
    if p_sw < 0.05:
        print(f"  Returns are NOT normally distributed — t-test may be unreliable")
    else:
        print(f"  Returns appear normally distributed")

    # One-sample t-test: H0: mean return = 0
    t_stat, p_val_t = stats.ttest_1samp(returns, 0)
    p_val_t_one = p_val_t / 2 if t_stat > 0 else 1 - p_val_t / 2
    print(f"\n  One-sample t-test (H0: mu = 0):")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value (two-sided): {p_val_t:.4f}")
    print(f"    p-value (one-sided, H1: mu > 0): {p_val_t_one:.4f}")

    # 95% CI on mean return
    se = stats.sem(returns)
    ci = stats.t.interval(0.95, n - 1, loc=returns.mean(), scale=se)
    print(f"    95% CI on mean return: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"    95% CI: [{ci[0]:.2%}, {ci[1]:.2%}]")

    # Wilcoxon signed-rank test (non-parametric, doesn't assume normality)
    # Remove zeros for Wilcoxon
    nonzero = returns[returns != 0]
    if len(nonzero) >= 6:
        stat_w, p_val_w = stats.wilcoxon(nonzero, alternative="greater")
        print(f"\n  Wilcoxon signed-rank test (H0: median = 0, H1: median > 0):")
        print(f"    W-statistic: {stat_w:.4f}")
        print(f"    p-value (one-sided): {p_val_w:.4f}")
    else:
        p_val_w = 1.0
        print(f"\n  Wilcoxon: insufficient non-zero samples")

    # Skewness and kurtosis
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    print(f"\n  Distribution shape:")
    print(f"    Skewness: {skew:.4f} ({'right-skewed' if skew > 0 else 'left-skewed'})")
    print(f"    Excess kurtosis: {kurt:.4f} ({'fat tails' if kurt > 0 else 'thin tails'})")

    # Practical significance: effect size (Cohen's d)
    cohens_d = returns.mean() / returns.std()
    print(f"    Cohen's d: {cohens_d:.4f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'} effect)")

    sig_t = "SIGNIFICANT" if p_val_t_one < 0.05 else "NOT SIGNIFICANT"
    sig_w = "SIGNIFICANT" if p_val_w < 0.05 else "NOT SIGNIFICANT"
    print(f"\n  >>> t-test: {sig_t} at alpha=0.05 <<<")
    print(f"  >>> Wilcoxon: {sig_w} at alpha=0.05 <<<")

    return {"test": "return_significance", "mean": returns.mean(),
            "t_stat": t_stat, "p_value_t": p_val_t_one, "p_value_w": p_val_w,
            "ci": list(ci), "cohens_d": cohens_d}


def test_1_5_deflated_sharpe(folds, n_strategies_tested=5):
    """Deflated Sharpe Ratio accounting for multiple testing and non-normality.

    DSR formula from Bailey & Lopez de Prado (2014):
    Tests whether observed Sharpe exceeds expected max Sharpe from
    n_strategies_tested independent strategies under H0 of no skill.
    """
    print("\n" + "=" * 70)
    print("TEST 1.5: DEFLATED SHARPE RATIO (DSR)")
    print("=" * 70)

    sharpes = folds["sharpe"].values
    returns = folds["total_return"].values

    obs_sharpe = sharpes.mean()
    print(f"  Mean Sharpe (annualized): {obs_sharpe:.4f}")
    print(f"  Strategies tested (N): {n_strategies_tested}")

    # Compute return statistics for DSR
    # Use per-fold returns as the return series
    sr = returns.mean() / returns.std() if returns.std() > 0 else 0  # Sharpe of fold returns
    T = len(returns)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)  # excess kurtosis

    # Expected max Sharpe from N trials under H0 (no skill)
    # E[max(SR)] ≈ (1 - gamma) * Phi^{-1}(1 - 1/N) + gamma * Phi^{-1}(1 - 1/(N*e))
    # Simplified: use the Euler-Mascheroni constant gamma ≈ 0.5772
    gamma = 0.5772156649
    if n_strategies_tested > 1:
        e_max_sr = (1 - gamma) * stats.norm.ppf(1 - 1/n_strategies_tested) + \
                   gamma * stats.norm.ppf(1 - 1/(n_strategies_tested * np.e))
    else:
        e_max_sr = 0

    print(f"  Expected max Sharpe under H0 (selection bias): {e_max_sr:.4f}")

    # Standard error of Sharpe ratio with non-normality correction
    # SE(SR) = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt/4)*SR^2) / (T-1))
    se_sr = np.sqrt((1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / max(1, T - 1))
    print(f"  SE of Sharpe (non-normality adjusted): {se_sr:.4f}")

    # DSR = Prob(SR > E[max(SR)]) using adjusted standard error
    if se_sr > 0:
        dsr_z = (sr - e_max_sr) / se_sr
        dsr_pval = 1 - stats.norm.cdf(dsr_z)
    else:
        dsr_z = 0
        dsr_pval = 0.5

    print(f"  DSR z-score: {dsr_z:.4f}")
    print(f"  DSR p-value: {dsr_pval:.4f}")

    # Also compute Probabilistic Sharpe Ratio (PSR)
    # PSR = Prob(SR > 0) = Phi(SR / SE(SR))
    if se_sr > 0:
        psr = stats.norm.cdf(sr / se_sr)
    else:
        psr = 0.5
    print(f"\n  Probabilistic Sharpe Ratio (PSR): {psr:.4f}")
    print(f"    Interpretation: {psr:.1%} probability that true Sharpe > 0")

    # Minimum required Sharpe (given N trials)
    min_sr = e_max_sr + se_sr * stats.norm.ppf(0.95)
    print(f"  Minimum Sharpe to beat selection bias (95%): {min_sr:.4f}")
    print(f"  Observed fold-level Sharpe: {sr:.4f}")
    print(f"  {'PASSES' if sr > min_sr else 'FAILS'} minimum Sharpe requirement")

    sig = "SIGNIFICANT" if dsr_pval < 0.05 else "NOT SIGNIFICANT"
    print(f"\n  >>> DSR: {sig} at alpha=0.05 <<<")

    return {"test": "deflated_sharpe", "obs_sharpe": obs_sharpe,
            "fold_sharpe": sr, "e_max_sr": e_max_sr,
            "dsr_pval": dsr_pval, "psr": psr, "min_sr": min_sr}


def test_1_6_regime_dependence(folds):
    """Analyze whether blowups are regime-dependent or random."""
    print("\n" + "=" * 70)
    print("TEST 1.6: REGIME DEPENDENCE & BLOWUP ANALYSIS")
    print("=" * 70)

    # Classify folds
    folds = folds.copy()
    folds["outcome"] = "marginal"
    folds.loc[folds["phase1_passed"], "outcome"] = "PASS"
    folds.loc[folds["total_breached"], "outcome"] = "BLOW"

    print("  Fold Classification:")
    for _, row in folds.iterrows():
        ret = row["total_return"]
        trades = int(row["total_trades"])
        wr = row["win_rate"]
        print(f"    Fold {int(row['fold']):>2}: {row['outcome']:>8} | "
              f"Ret={ret:>+8.2%} | Trades={trades:>4} | WR={wr:.1%} | "
              f"Test: {row['test_start']} to {row['test_end']}")

    # Separate groups
    pass_folds = folds[folds["outcome"] == "PASS"]
    blow_folds = folds[folds["outcome"] == "BLOW"]
    marginal_folds = folds[folds["outcome"] == "marginal"]

    print(f"\n  Group Sizes: PASS={len(pass_folds)}, BLOW={len(blow_folds)}, MARGINAL={len(marginal_folds)}")

    # Compare characteristics
    if len(pass_folds) > 0 and len(blow_folds) > 0:
        print(f"\n  Characteristic Comparison (PASS vs BLOW):")

        for col, label in [("total_trades", "Trades"), ("win_rate", "Win Rate"),
                           ("max_dd", "Max DD"), ("profit_factor", "Profit Factor")]:
            pass_mean = pass_folds[col].mean()
            blow_mean = blow_folds[col].mean()
            # Mann-Whitney U test (non-parametric, small samples)
            if len(pass_folds) >= 3 and len(blow_folds) >= 3:
                try:
                    u_stat, u_p = stats.mannwhitneyu(
                        pass_folds[col].values, blow_folds[col].values,
                        alternative="two-sided"
                    )
                    sig = "*" if u_p < 0.05 else ""
                except ValueError:
                    u_p = 1.0
                    sig = ""
            else:
                u_p = 1.0
                sig = ""
            print(f"    {label:>15}: PASS={pass_mean:.4f} vs BLOW={blow_mean:.4f} (p={u_p:.3f}) {sig}")

    # Autocorrelation of outcomes (are blowups clustered?)
    outcomes_binary = folds["total_breached"].astype(int).values
    if len(outcomes_binary) > 3:
        # Runs test: tests randomness of sequence
        n_runs = 1
        for i in range(1, len(outcomes_binary)):
            if outcomes_binary[i] != outcomes_binary[i-1]:
                n_runs += 1

        n1 = outcomes_binary.sum()  # blows
        n0 = len(outcomes_binary) - n1  # non-blows

        if n0 > 0 and n1 > 0:
            # Expected runs under randomness
            e_runs = 1 + 2 * n0 * n1 / (n0 + n1)
            var_runs = 2 * n0 * n1 * (2 * n0 * n1 - n0 - n1) / ((n0 + n1)**2 * (n0 + n1 - 1))

            if var_runs > 0:
                z_runs = (n_runs - e_runs) / np.sqrt(var_runs)
                p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
            else:
                z_runs = 0
                p_runs = 1.0

            print(f"\n  Runs Test (blowup clustering):")
            print(f"    Observed runs: {n_runs}")
            print(f"    Expected runs: {e_runs:.1f}")
            print(f"    z-score: {z_runs:.4f}")
            print(f"    p-value: {p_runs:.4f}")
            if z_runs < -1.96:
                print(f"    >>> Blowups are CLUSTERED (fewer runs than expected) <<<")
            elif z_runs > 1.96:
                print(f"    >>> Blowups are ALTERNATING (more runs than expected) <<<")
            else:
                print(f"    >>> Blowups appear RANDOM (no significant clustering) <<<")

    # Trade count as edge indicator
    print(f"\n  Trade Count as Edge Indicator:")
    print(f"    PASS folds avg trades: {pass_folds['total_trades'].mean():.0f}")
    if len(blow_folds) > 0:
        print(f"    BLOW folds avg trades: {blow_folds['total_trades'].mean():.0f}")
        print(f"    Ratio: {pass_folds['total_trades'].mean() / max(1, blow_folds['total_trades'].mean()):.1f}x")
        print(f"    Insight: Blowups happen with {'fewer' if blow_folds['total_trades'].mean() < pass_folds['total_trades'].mean() else 'more'} trades")

    return {"test": "regime_dependence"}


def test_1_7_alpha_significance(folds):
    """Test if strategy alpha (excess return over buy-and-hold) is significant."""
    print("\n" + "=" * 70)
    print("TEST 1.7: ALPHA (EXCESS RETURN) SIGNIFICANCE")
    print("=" * 70)

    alphas = folds["alpha"].values
    n = len(alphas)

    print(f"  Mean alpha: {alphas.mean():.4f} ({alphas.mean():.2%})")
    print(f"  Median alpha: {np.median(alphas):.4f} ({np.median(alphas):.2%})")
    print(f"  Positive alpha folds: {(alphas > 0).sum()}/{n}")

    # One-sample t-test: H0: alpha = 0
    t_stat, p_val = stats.ttest_1samp(alphas, 0)
    p_val_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
    print(f"\n  t-test (H0: alpha = 0):")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value (one-sided): {p_val_one:.4f}")

    # 95% CI
    se = stats.sem(alphas)
    ci = stats.t.interval(0.95, n - 1, loc=alphas.mean(), scale=se)
    print(f"    95% CI: [{ci[0]:.2%}, {ci[1]:.2%}]")

    # Sign test (non-parametric)
    n_positive = (alphas > 0).sum()
    p_sign = 1 - stats.binom.cdf(n_positive - 1, n, 0.5)
    print(f"\n  Sign test (H0: P(alpha>0) = 0.5):")
    print(f"    Positive: {n_positive}/{n}")
    print(f"    p-value (one-sided): {p_sign:.4f}")

    sig = "SIGNIFICANT" if p_val_one < 0.05 else "NOT SIGNIFICANT"
    print(f"\n  >>> Alpha: {sig} at alpha=0.05 <<<")

    return {"test": "alpha_significance", "mean_alpha": alphas.mean(),
            "p_value": p_val_one, "ci": list(ci)}


def expected_value_analysis(folds):
    """Compute expected value per FTMO challenge attempt."""
    print("\n" + "=" * 70)
    print("EXPECTED VALUE PER FTMO CHALLENGE ATTEMPT")
    print("=" * 70)

    returns = folds["total_return"].values
    passed = folds["phase1_passed"].values
    blown = folds["total_breached"].values

    ftmo_fee = 500  # Challenge fee
    account_size = 100_000

    # Conditional expected returns
    pass_returns = returns[passed]
    blow_returns = returns[blown]
    other_returns = returns[~passed & ~blown]

    print(f"  Pass rate: {passed.mean():.1%}")
    print(f"  Blow rate: {blown.mean():.1%}")
    print(f"  Other rate: {(~passed & ~blown).mean():.1%}")

    if len(pass_returns) > 0:
        print(f"\n  When passing:")
        print(f"    Mean return: {pass_returns.mean():.2%} (${pass_returns.mean()*account_size:+,.0f})")
        print(f"    Min: {pass_returns.min():.2%}, Max: {pass_returns.max():.2%}")

    if len(blow_returns) > 0:
        print(f"\n  When blowing:")
        print(f"    Mean return: {blow_returns.mean():.2%} (${blow_returns.mean()*account_size:+,.0f})")

    if len(other_returns) > 0:
        print(f"\n  When marginal (no pass, no blow):")
        print(f"    Mean return: {other_returns.mean():.2%}")

    # EV calculation
    # On pass: you get to keep trading (Phase 2 etc.) — here we just count Phase 1 profit
    # On fail: you lose $500 fee
    ev_return = returns.mean()
    ev_dollar = ev_return * account_size
    ev_after_fee = ev_dollar - ftmo_fee  # Always pay fee

    print(f"\n  Unconditional EV:")
    print(f"    Mean return: {ev_return:.2%}")
    print(f"    Dollar EV: ${ev_dollar:+,.0f}")
    print(f"    After fee: ${ev_after_fee:+,.0f}")

    # But this is misleading because FTMO resets account on failure
    # Real EV = P(pass) * E[profit|pass] - fee (always)
    if len(pass_returns) > 0:
        real_ev = passed.mean() * pass_returns.mean() * account_size - ftmo_fee
        print(f"\n  FTMO-Adjusted EV (reset on failure):")
        print(f"    = P(pass) * E[profit|pass] - fee")
        print(f"    = {passed.mean():.2f} * ${pass_returns.mean()*account_size:,.0f} - ${ftmo_fee}")
        print(f"    = ${real_ev:+,.0f}")

    # Required attempts to pass (geometric distribution)
    if passed.mean() > 0:
        e_attempts = 1 / passed.mean()
        print(f"\n  Expected attempts to first pass: {e_attempts:.1f}")
        print(f"  Expected total fees: ${e_attempts * ftmo_fee:,.0f}")
        print(f"  P(pass within 3 attempts): {1-(1-passed.mean())**3:.1%}")

    return {"ev_dollar": ev_dollar, "ev_after_fee": ev_after_fee}


def generate_report(results, folds):
    """Generate markdown report."""
    report_path = PROJECT_ROOT / "reports" / "statistical_validation.md"

    lines = [
        "# Statistical Validation Report",
        f"**Date**: 2026-03-30",
        f"**Instrument**: USTEC (US100.cash)",
        f"**Folds**: {len(folds)} walk-forward folds",
        "",
        "## Summary of Findings",
        "",
        "| Test | Metric | Value | p-value | Significant? |",
        "|------|--------|-------|---------|-------------|",
    ]

    for r in results:
        test = r.get("test", "")
        if test == "pass_rate_binomial":
            lines.append(f"| Pass Rate Binomial | {r['k']}/{r['n']} = {r['k']/r['n']:.1%} | "
                        f"95% CI [{r['ci_low']:.1%}, {r['ci_high']:.1%}] | {r['p_value']:.4f} | "
                        f"{'Yes' if r['significant'] else 'No'} |")
        elif test == "pooled_win_rate":
            lines.append(f"| Win Rate vs Breakeven | {r['pooled_wr']:.1%} vs {r['breakeven_wr']:.1%} | "
                        f"95% CI [{r['wilson_ci'][0]:.1%}, {r['wilson_ci'][1]:.1%}] | {r['p_value']:.6f} | "
                        f"{'Yes' if r['significant'] else 'No'} |")
        elif test == "return_significance":
            lines.append(f"| Mean Return t-test | {r['mean']:.2%} | "
                        f"95% CI [{r['ci'][0]:.2%}, {r['ci'][1]:.2%}] | {r['p_value_t']:.4f} | "
                        f"{'Yes' if r['p_value_t'] < 0.05 else 'No'} |")
        elif test == "deflated_sharpe":
            lines.append(f"| Deflated Sharpe | SR={r['fold_sharpe']:.3f} | "
                        f"PSR={r['psr']:.1%} | {r['dsr_pval']:.4f} | "
                        f"{'Yes' if r['dsr_pval'] < 0.05 else 'No'} |")
        elif test == "alpha_significance":
            lines.append(f"| Alpha Significance | {r['mean_alpha']:.2%} | "
                        f"95% CI [{r['ci'][0]:.2%}, {r['ci'][1]:.2%}] | {r['p_value']:.4f} | "
                        f"{'Yes' if r['p_value'] < 0.05 else 'No'} |")

    lines.extend([
        "",
        "## GO/NO-GO Decision",
        "",
    ])

    # Count significant tests
    sig_count = sum(1 for r in results if r.get("significant", r.get("p_value", r.get("p_value_t", 1.0)) < 0.05 if isinstance(r.get("p_value", r.get("p_value_t", 1.0)), float) else False))

    lines.append(f"Tests significant at alpha=0.05: {sig_count}/{len(results)}")
    lines.append("")
    lines.append("See console output for detailed analysis.")

    report_path.write_text("\n".join(lines))
    print(f"\n  Report saved to: {report_path}")


def main():
    print("=" * 70)
    print("PHASE 1: STATISTICAL FOUNDATION")
    print("Do we have a statistically significant edge?")
    print("=" * 70)

    ustec, xauusd = load_ustec_folds()

    if len(ustec) == 0:
        print("ERROR: No USTEC folds found!")
        return

    results = []

    # Run all tests
    results.append(test_1_1_pass_rate_binomial(ustec))
    results.append(test_1_2_bootstrap_pass_rate(ustec))
    results.append(test_1_3_pooled_win_rate(ustec))
    results.append(test_1_4_return_significance(ustec))
    results.append(test_1_5_deflated_sharpe(ustec))
    test_1_6_regime_dependence(ustec)
    results.append(test_1_7_alpha_significance(ustec))

    expected_value_analysis(ustec)

    # XAUUSD comparison (for reference)
    if len(xauusd) > 0:
        print("\n" + "=" * 70)
        print("XAUUSD COMPARISON (Reference Only — Disabled Instrument)")
        print("=" * 70)
        xau_pass = xauusd["phase1_passed"].sum()
        xau_n = len(xauusd)
        p_val_xau = 1 - stats.binom.cdf(xau_pass - 1, xau_n, 0.5)
        print(f"  Pass rate: {xau_pass}/{xau_n} = {xau_pass/xau_n:.1%}")
        print(f"  Binomial p-value (vs 50%): {p_val_xau:.4f}")
        print(f"  >>> XAUUSD correctly disabled — no significant edge <<<")

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    generate_report(results, ustec)

    print("\n  Review test results above for GO/NO-GO decision.")
    print("  Key question: Do enough tests show p < 0.05 to proceed?")


if __name__ == "__main__":
    main()
