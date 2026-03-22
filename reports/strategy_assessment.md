# FTMO ML Strategy Assessment — Walk-Forward Validated

## Date: 2026-03-22

## Executive Summary

After extensive walk-forward validation (31 folds across 2 instruments, 3.5 years of data),
the optimal strategy is **USTEC-only at 1.75% risk per trade**. XAUUSD has been disabled
due to poor walk-forward performance (24% pass rate vs USTEC's 64%).

## Walk-Forward Results (14 USTEC Folds)

- **Pass rate**: 64% (9/14 folds pass Phase 1)
- **Blow rate**: 29% (4/14 hit -9% halt)
- **Mean return**: +46.98%
- **Avg days to target (when passing)**: ~5 trading days
- **Expected value per attempt**: +$46,484 ($100K account)
- **Max drawdown range**: 0-9.49%

### Per-Fold Detail

| Fold | Period | Return | Max DD | Trades | Outcome |
|------|--------|--------|--------|--------|---------|
| 1 | Apr-Jul 2023 | +66.12% | 2.48% | 689 | PASS |
| 2 | Jul-Sep 2023 | +58.07% | 1.96% | 597 | PASS |
| 3 | Sep-Nov 2023 | -9.11% | 9.11% | 15 | BLOW |
| 4 | Nov 2023-Feb 2024 | +70.96% | 5.85% | 557 | PASS |
| 5 | Feb-Apr 2024 | +39.31% | 5.94% | 439 | PASS |
| 6 | Apr-Jun 2024 | +73.36% | 7.03% | 469 | PASS |
| 7 | Jun-Sep 2024 | -2.84% | 8.15% | 534 | fail |
| 8 | Sep-Nov 2024 | +231.18% | 5.45% | 585 | PASS |
| 9 | Nov 2024-Feb 2025 | +29.72% | 4.45% | 563 | PASS |
| 10 | Feb-Apr 2025 | -9.49% | 9.49% | 32 | BLOW |
| 11 | Apr-Jun 2025 | -9.35% | 9.35% | 19 | BLOW |
| 12 | Jun-Oct 2025 | +52.08% | 0.00% | 282 | PASS |
| 13 | Oct 2025-Jan 2026 | -9.14% | 9.14% | 174 | BLOW |
| 14 | Jan-Mar 2026 | +76.91% | 0.46% | 462 | PASS |

## Why XAUUSD Was Disabled

Walk-forward on XAUUSD (17 folds):
- **Pass rate**: 24% (4/17)
- **Blow rate**: 65% (11/17)
- **Median return**: -10.17% (the halt limit)
- Adding XAUUSD reduces combined pass rate from 64% to 42%

## Kill Switch Strategies Tested (All Failed)

1. **Signal density gate** (check signal count at bar 200): No effect — blowups complete
   before bar 200. The account halts within 15-32 trades.

2. **Early halt** (5% total DD instead of 9%): Destructive — kills winning folds that
   temporarily dip (fold 4: +71% killed, fold 8: +231% killed).

3. **Ramp-up probe** (15 trades at 0.5% then scale): Reduces pass rate from 64% to 29% —
   stops too many winning folds that start slow.

4. **Combined density + early halt**: Same problems as early halt alone.

## Risk Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Risk per trade | 1.75% | Aggressive but validated: 0.079R expectancy |
| SL distance | 1.5 ATR | |
| TP distance | 2.5 ATR | 1.67 R:R |
| Signal offset | 0.01 | Low threshold maximizes signal frequency |
| Daily halt | 3.5% | 1.5% buffer below FTMO 5% (was 4%, tightened) |
| Total halt | 9.0% | 1% buffer below FTMO 10% |
| Max concurrent | 5 | |

## Monte Carlo (1000 sims, single OOS period)

- Pass rate: 78.7% (higher than WF because favorable OOS period)
- Pass within 14 cal days: 34.9%
- Pass within 30 cal days: 65.4%
- Account blown: 0.0% (misleading — only tests order, not regime)

**Trust walk-forward over Monte Carlo for regime-dependent metrics.**

## Practical Recommendations

1. **Trade USTEC only** during FTMO challenge
2. **Monitor trade frequency** in first 2 days — if generating <15 trades/day, consider
   pausing (low-confidence regime, though no automated kill switch works)
3. **Retrain model monthly** per config (retrain_interval_days: 30)
4. **Expected timeline**: 5-14 days when model has edge
5. **Cost of failure**: $500 FTMO fee (36% of attempts)
6. **Expected profit per attempt**: ~$29,750 (accounting for 36% failure rate)

## Caveats

- 14 walk-forward folds is a small sample (wide confidence intervals)
- Daily DD buffer is tight (0.33% → 1.5% after tightening halt to 3.5%)
- Model cannot predict in advance whether current regime has edge
- Two consecutive runs showed consistent 9/14 pass rate (robust to model randomness)
