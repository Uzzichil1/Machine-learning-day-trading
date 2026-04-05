# Quantitative Research Plan: Beat FTMO Phase 1 in <14 Days

**Objective**: Systematically find and validate a method to pass FTMO Phase 1 (10% return on $100K) within 14 calendar days (~10 trading days), without breaching 5% daily or 10% total drawdown.

**Start Date**: 2026-03-30
**Target Completion**: 2026-04-12 (14 days)

---

## Current Baseline (Honest Assessment)

| Metric | Value | Problem |
|--------|-------|---------|
| WF Pass Rate | 64% (9/14) | Not statistically validated |
| Blow Rate | 29% (4/14) | Unacceptably high |
| Model Accuracy | ~51% | Barely above random |
| Trade Win Rate | ~43-46% | Relies entirely on R:R asymmetry |
| Expectancy | 0.079R/trade | Razor thin, eaten by variance |
| Sharpe (annualized) | ~2.3 | Likely inflated (i.i.d. assumption) |
| Kill Switch | None work | Cannot detect bad regimes in advance |

**Bottom line**: The system shows *plausible* edge but hasn't *proven* it. A 51% accuracy model with 0.079R expectancy is one bad week away from blowing up. We need to either (a) prove this edge is real and optimize it, or (b) find a better one.

---

## Research Philosophy

1. **Prove before optimize** — Don't tune hyperparameters on an edge that might not exist
2. **Fail fast** — If a hypothesis doesn't hold, kill it immediately
3. **FTMO-first thinking** — We're not maximizing Sharpe, we're maximizing P(hit +10% before -10%)
4. **Statistical discipline** — Every claim needs a p-value or confidence interval
5. **Regime honesty** — If the strategy only works in trends, we need to know that and price it in

---

## Phase 1: Statistical Foundation (Days 1-2)

### Goal
Determine if we have a statistically significant edge or are fooling ourselves.

### Hypothesis 1.1: "The current model predicts direction better than a coin flip"
**Experiment**:
- Binomial test on trade-level win rate across ALL walk-forward folds (pooled)
- H0: p_win = 0.50 (random)
- H1: p_win ≠ 0.50
- Significance level: α = 0.05

**Data**: All trades from 14 WF folds (~5,000+ trades total)

**Decision Gate**:
- p < 0.05 → Edge exists, proceed to optimization
- p ≥ 0.05 → Edge is noise, pivot to alternative approaches (see Phase 1B)

### Hypothesis 1.2: "The 64% pass rate is better than chance"
**Experiment**:
- Binomial test: P(9/14 successes | p=0.5)
- Bootstrap: Resample 14 folds 10,000× with replacement, compute pass rate distribution
- Compute 95% CI on true pass rate

**Decision Gate**:
- 95% CI lower bound > 50% → Statistically significant pass rate
- Otherwise → Insufficient evidence; need more folds or higher pass rate

### Hypothesis 1.3: "The expectancy per trade is significantly positive"
**Experiment**:
- One-sample t-test on per-trade R-multiples (pooled across folds)
- Bootstrap 95% CI on mean R-multiple
- Compute Deflated Sharpe Ratio (DSR) adjusting for:
  - Number of strategy variants tested
  - Non-normality of returns (skew, kurtosis)
  - Sample size

**Deliverables**:
- `scripts/statistical_tests.py` — All hypothesis tests
- `reports/statistical_validation.md` — Results with p-values and CIs
- GO/NO-GO decision on current approach

### Phase 1B (Contingency): If No Statistical Edge
If Phase 1 shows the ML model has no significant edge:
- **Pivot 1**: Pure rules-based volatility breakout (opening range breakout on USTEC US session)
- **Pivot 2**: Momentum regime filter + trend following (no ML, just ATR channel breakouts)
- **Pivot 3**: Mean reversion during low-vol regimes only (trade smaller, more often)
- Each pivot gets 2 days of rapid prototyping and walk-forward testing before commit/kill

---

## Phase 2: Understand the Edge (Days 2-4)

### Goal
Dissect *what drives wins vs losses* at a granular level. We can't fix what we don't understand.

### Experiment 2.1: Fold Autopsy — What Separates Winners from Blowups?
**Method**:
- For each of 14 WF folds, compute:
  - Average ATR during fold (volatility regime)
  - Trend strength (ADX mean, cumulative directional move)
  - Signal density (trades per day)
  - Average model confidence (mean probability deviation from 0.5)
  - Regime distribution (% time in each GMM state)
- Classify folds: PASS (9) vs BLOW (4) vs MARGINAL (1)
- Statistical comparison (Mann-Whitney U or permutation test) between groups

**Hypothesis 2.1**: "Blowup folds occur in ranging/choppy markets (low ADX, low trend strength)"

**Deliverable**: Feature profile of winning vs losing market conditions

### Experiment 2.2: Trade-Level Edge Analysis
**Method**:
- Extract all ~5,000+ trades across all folds
- For each trade, record: model probability, regime state, hour, ATR, ADX, signal type (buy/sell), outcome (R-multiple)
- Segment edge by:
  - **Confidence bucket**: Does higher model confidence → higher win rate?
  - **Regime**: Edge by GMM state (low_vol / moderate / high_vol)
  - **Session**: Edge by hour (pre-market, US open, US afternoon, overnight)
  - **Trend context**: Edge when ADX > 25 vs ADX < 20
  - **Volatility context**: Edge in high-ATR vs low-ATR periods

**Hypothesis 2.2**: "The edge is concentrated in specific conditions (high confidence + trending regime)"

**Decision Gate**: If edge is concentrated:
- Filter trades to only take high-edge conditions → fewer trades but higher win rate
- Rerun walk-forward with filters applied → check if pass rate improves

### Experiment 2.3: Consecutive Loss Analysis
**Method**:
- Compute maximum consecutive losing streaks per fold
- Compare to theoretical maximum under i.i.d. assumption (geometric distribution)
- Test for autocorrelation in trade outcomes (are losses clustered?)
- Compute probability of ruin under current sizing

**Hypothesis 2.3**: "Blowups are caused by clustered losses, not random bad luck"

**Deliverable**:
- `reports/edge_analysis.md` — Complete edge decomposition
- Concrete list of "trade filters" to test in Phase 3

---

## Phase 3: Signal Quality Improvement (Days 4-7)

### Goal
Improve model accuracy from ~51% to 53-55%+ through systematic optimization. Even 2% accuracy improvement can dramatically change FTMO outcomes.

### Experiment 3.1: Hyperparameter Optimization
**Method**:
- Use Optuna (Bayesian optimization) with walk-forward as the objective
- NOT optimizing accuracy — optimizing FTMO pass rate directly
- Search space:
  ```
  LightGBM: depth [3-8], lr [0.01-0.1], n_estimators [200-1000],
            num_leaves [15-63], min_child [5-50], subsample [0.6-1.0]
  XGBoost:  depth [3-8], lr [0.01-0.1], n_estimators [200-1000],
            min_child_weight [1-10], colsample [0.5-1.0]
  CatBoost: depth [4-8], lr [0.01-0.1], iterations [200-1000], l2 [1-10]
  ```
- Inner loop: 5-fold purged CV on training data → model quality
- Outer loop: Walk-forward pass rate → strategy quality
- Budget: 200 Optuna trials

**Key Safeguard**: Use the FIRST 10 WF folds for optimization, RESERVE folds 11-14 as holdout validation. Never touch the holdout until final validation.

**Decision Gate**:
- Optimized params improve WF pass rate on folds 1-10 by ≥5% → adopt
- No improvement → keep defaults, focus on other levers

### Experiment 3.2: Feature Selection via SHAP
**Method**:
- Compute SHAP values for all 36 features across all training folds
- Rank by mean |SHAP|
- Test reduced feature sets: top-10, top-15, top-20, top-25
- For each set, run full walk-forward and compare pass rates
- Check for redundancy: correlation matrix of top features, prune >0.85

**Hypothesis 3.2**: "A smaller, curated feature set reduces noise and improves OOS performance"

### Experiment 3.3: Label Engineering
**Method**: Test alternative triple-barrier configurations:
- **Asymmetric barriers**: TP=2.0 ATR / SL=1.0 ATR (wider asymmetry)
- **Longer horizon**: max_holding_bars=12 (let trends develop)
- **Shorter horizon**: max_holding_bars=5 (tighter, more signal)
- **Volatility-scaled**: Barriers scale with recent vol (not just ATR)
- Each config → full walk-forward → compare pass rates

**Hypothesis 3.3**: "Current symmetric 1.5/1.5 barriers with 8-bar horizon aren't optimal for USTEC"

### Experiment 3.4: Ensemble Diversity
**Method**:
- Compute correlation between base learner predictions
- If correlation > 0.9: models are redundant, not diverse
- Test alternatives:
  - Replace one tree model with a different architecture (Ridge, SVM, KNN)
  - Add a momentum-based rules model as a base learner
  - Weight by recent fold performance (not equal weight)

**Deliverable**:
- Optimized model configuration (if improvement found)
- `reports/optimization_results.md` with all experiment outcomes

---

## Phase 4: FTMO-Specific Risk Optimization (Days 7-9)

### Goal
The strategy isn't "maximize returns" — it's "maximize P(+10% before -10%)." This is a first-passage-time problem, and optimal sizing is DIFFERENT from Kelly.

### Experiment 4.1: Optimal Sizing for First-Passage Problem
**Method**:
- Given trade-level R-multiple distribution from Phase 2:
  - Simulate 100,000 paths of sequential trades
  - For each path, check: does equity hit +10% or -10% first?
  - Sweep risk_per_trade from 0.5% to 3.0% in 0.25% steps
  - For each risk level, compute:
    - P(hit +10% first) = pass rate
    - E[trades to +10% | pass] = speed
    - P(hit -10% first) = blow rate
    - E[trades to -10% | blow] = how fast you fail

**Hypothesis 4.1**: "Current 1.75% risk is NOT optimal. There exists a sizing that maximizes P(+10% before -10%)"

**Key Insight**: For thin-edge strategies, SMALLER sizing can actually INCREASE pass rate by reducing variance, even though it takes more trades. The question is whether you have enough trading days.

**Constraint**: Must hit +10% within ~150-200 trades (10 days × 15-20 trades/day max)

### Experiment 4.2: Signal Threshold Optimization
**Method**:
- Sweep signal_offset from 0.005 to 0.05 in 0.005 steps
- For each threshold:
  - Compute trade count (fewer trades at higher threshold)
  - Compute win rate (should increase with threshold)
  - Compute expectancy per trade
  - Run first-passage simulation → P(pass FTMO)
- Find the threshold that maximizes pass probability given the time constraint

**Hypothesis 4.2**: "A higher signal_offset (more selective) improves pass rate despite fewer trades"

**Trade-off**: Higher threshold → fewer but better trades → higher win rate but might not generate enough trades in 10 days. Need to find the sweet spot.

### Experiment 4.3: Dynamic Sizing Based on Equity Curve
**Method**:
- Test "anti-martingale" approach:
  - Start at reduced size (1.0%)
  - If up +3% → increase to 1.5%
  - If up +6% → increase to 2.0% (pressing the advantage)
  - If down -2% → reduce to 0.75% (preservation mode)
  - If down -4% → reduce to 0.5% (survival mode)
- Compare to flat sizing via simulation
- Also test: "accelerate near target" — if at +8%, increase size to close out quickly

**Hypothesis 4.3**: "Dynamic sizing improves pass rate by pressing gains and preserving capital during drawdowns"

### Experiment 4.4: Drawdown Halt Optimization
**Method**:
- Current halts: daily 3.5%, total 9.0%
- Test tighter halts: does stopping earlier and retrying a new challenge beat grinding through a bad regime?
- Simulate: halt at -5% total and restart vs continue to -9%
- Factor in $500 per restart

**Hypothesis 4.4**: "Halting earlier and restarting is +EV compared to grinding through drawdown"

**Deliverable**:
- Optimal risk_per_trade, signal_offset, dynamic sizing rules
- `reports/risk_optimization.md` with simulation results

---

## Phase 5: Proper Validation (Days 9-11)

### Goal
Validate the optimized system with methods that would satisfy an institutional quant desk.

### Experiment 5.1: Combinatorial Purged Cross-Validation (CPCV)
**Method**:
- Implement CPCV with:
  - N=10 groups, k=2 test groups → C(10,2) = 45 backtest paths
  - Embargo gap: 24 bars (1 day) between train/test
  - Purging: Remove train samples within 8 bars of test boundaries (label horizon)
- Compute Probability of Backtest Overfitting (PBO):
  - PBO = fraction of CPCV paths where OOS Sharpe < 0
  - Target: PBO < 0.25 (less than 25% chance of overfitting)

**Decision Gate**:
- PBO < 0.25 → Strategy is robust, proceed to live
- PBO 0.25-0.50 → Cautious: strategy may be overfit, needs simplification
- PBO > 0.50 → Strategy is likely overfit, do not deploy

### Experiment 5.2: Walk-Forward on Holdout Folds
**Method**:
- Run optimized model on reserved folds 11-14 (never seen during optimization)
- These folds cover Oct 2025 - Mar 2026 (most recent data)
- Compare performance to training-period folds

**Decision Gate**:
- Holdout pass rate ≥ 50% (2/4 folds) → Proceed
- Holdout pass rate < 50% → Overfit to training period, reassess

### Experiment 5.3: Regime-Conditioned Monte Carlo
**Method**:
- Improve Monte Carlo to capture regime switching:
  - Classify each historical period by regime (trend/range/crisis)
  - Compute transition probabilities between regimes
  - Simulate regime sequence → draw trades from regime-specific distributions
  - Run 10,000 paths, compute pass rate with realistic regime dynamics

**Deliverable**:
- PBO score
- Holdout fold results
- Regime-aware Monte Carlo pass rate estimate
- `reports/validation.md` — final validation report

### Experiment 5.4: Stress Testing
**Method**:
- Identify worst-case periods in available data:
  - 2022 Q4 (rate hike volatility)
  - 2023 Q1 (banking crisis)
  - Any period with >5% daily USTEC move
- Run strategy specifically on these windows
- Verify halts trigger properly and loss is contained

---

## Phase 6: Live Preparation & Deployment (Days 11-14)

### Goal
Verify the system works in real-time execution, then deploy on FTMO challenge.

### Step 6.1: Paper Trading Validation (2-3 days)
- Deploy on FTMO free trial or demo account
- Run live for minimum 2-3 trading days
- Verify:
  - MT5 connection stability
  - Order execution quality (slippage measurement)
  - Signal generation matches backtest expectations
  - Position sizing is correct
  - Halts trigger at correct levels
  - No bugs in live loop

### Step 6.2: Execution Quality Assessment
- Measure actual vs expected:
  - Fill price vs signal price (slippage)
  - Spread during trading hours
  - Order rejection rate
  - Latency (signal → order → fill)
- If slippage > 0.5 ATR: adjust backtest spread model and revalidate

### Step 6.3: Pre-Flight Checklist
Before starting the real FTMO challenge:
- [ ] All statistical tests pass (p < 0.05 on edge existence)
- [ ] PBO < 0.25 (CPCV validation)
- [ ] Holdout folds show positive performance
- [ ] Paper trading matches backtest within tolerance
- [ ] Risk parameters locked (no more tuning)
- [ ] Model retrained on ALL available data (final fit)
- [ ] Monitoring dashboard operational
- [ ] Halt logic verified (daily and total)
- [ ] Backup plan defined (manual halt conditions)

### Step 6.4: FTMO Challenge Execution
- Start Phase 1 challenge
- Monitor daily: actual P&L vs expected range
- Decision rules during challenge:
  - If -3% in first 2 days → evaluate if regime is favorable, consider manual stop
  - If +5% by day 5 → continue, possibly increase size to close early
  - If 0% by day 7 → evaluate trade count, model confidence
  - **Never override the system's signals** — that's the whole point of automation

---

## Decision Framework

```
                    Phase 1: Do we have an edge?
                         /              \
                       YES               NO
                       /                  \
               Phase 2: Where?      Phase 1B: Pivot
                    |                  (rules-based)
               Phase 3: Improve            |
                    |                 Same pipeline
               Phase 4: Size it       from Phase 2+
                    |
               Phase 5: Validate
                  /         \
            PBO<0.25     PBO>0.50
               /              \
        Phase 6: Deploy    STOP: Overfit
                              |
                         Simplify model,
                         reduce complexity,
                         retest
```

---

## Timeline Summary

| Days | Phase | Key Deliverable | GO/NO-GO Gate |
|------|-------|----------------|---------------|
| 1-2 | Statistical Foundation | p-values on edge existence | Edge real? |
| 2-4 | Edge Analysis | What conditions drive wins/losses | Filterable? |
| 4-7 | Signal Optimization | Tuned model + features | WF improvement? |
| 7-9 | Risk Optimization | Optimal sizing + thresholds | Pass rate up? |
| 9-11 | Validation | CPCV, holdout, stress test | PBO < 0.25? |
| 11-14 | Deployment | Paper trade → FTMO challenge | Execution OK? |

---

## What Success Looks Like

| Metric | Current | Target | Method to Achieve |
|--------|---------|--------|-------------------|
| WF Pass Rate | 64% | 75%+ | Signal quality + risk optimization |
| Blow Rate | 29% | <15% | Better sizing + regime filtering |
| Model Accuracy | 51% | 53-55% | HPO + feature selection |
| Expectancy | 0.079R | 0.15R+ | Label optimization + threshold tuning |
| PBO | Unknown | <0.25 | CPCV validation |
| Statistical Sig. | Untested | p<0.05 | Binomial/bootstrap tests |

---

## Risk Budget

- **Research time**: 14 days (hard deadline)
- **FTMO challenge fee**: $500 per attempt
- **Expected attempts to pass**: 1-2 (if research succeeds), 3+ (if marginal)
- **Maximum investment**: 3 attempts × $500 = $1,500
- **Expected return if pass**: $10,000+ (Phase 1 profit) + funded account access

---

## Principles for This Sprint

1. **Every optimization must be validated out-of-sample** — no in-sample tuning without holdout confirmation
2. **Simpler is better** — if a 10-feature model matches a 36-feature model, use 10 features
3. **FTMO pass rate is the ONLY objective function** — not Sharpe, not accuracy, not total return
4. **Kill hypotheses quickly** — 1 day max per experiment before GO/NO-GO
5. **Document everything** — future you needs to understand why decisions were made
6. **Don't optimize what you can't measure** — statistical tests before tuning
7. **The regime problem is the key problem** — solving the 29% blow rate is worth more than improving accuracy by 1%
