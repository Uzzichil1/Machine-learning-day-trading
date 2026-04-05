# Statistical Validation Report
**Date**: 2026-03-30
**Instrument**: USTEC (US100.cash)
**Folds**: 14 walk-forward folds

## Summary of Findings

| Test | Metric | Value | p-value | Significant? |
|------|--------|-------|---------|-------------|
| Pass Rate Binomial | 9/14 = 64.3% | 95% CI [35.1%, 87.2%] | 0.2120 | No |
| Win Rate vs Breakeven | 46.0% vs 37.5% | 95% CI [44.6%, 47.4%] | 0.000000 | Yes |
| Mean Return t-test | 44.69% | 95% CI [6.88%, 82.50%] | 0.0120 | Yes |
| Deflated Sharpe | SR=0.708 | PSR=100.0% | 0.9972 | No |
| Alpha Significance | 39.36% | 95% CI [2.64%, 76.07%] | 0.0188 | Yes |

## GO/NO-GO Decision

Tests significant at alpha=0.05: 3/6

See console output for detailed analysis.