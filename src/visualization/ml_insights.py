"""ML Insights — deep model diagnostics, walk-forward analysis, and live monitoring charts."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Color Palette (matches existing charts.py) ──────────────────────

PROFIT = "#00d4aa"
LOSS = "#ff4757"
PRIMARY = "#3742fa"
SECONDARY = "#ffa502"
NEUTRAL = "#747d8c"
BLUE = "#58a6ff"
BG_CARD = "#161b22"
BG_DARK = "#0d1117"
GRID = "#30363d"

TEMPLATE = "plotly_dark"
LAYOUT_DEFAULTS = dict(
    template=TEMPLATE,
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_DARK,
    font=dict(color="#c9d1d9"),
    margin=dict(t=50, b=40, l=50, r=30),
)


def _apply_layout(fig, **kwargs):
    merged = {**LAYOUT_DEFAULTS, **kwargs}
    fig.update_layout(**merged)
    return fig


# ═══════════════════════════════════════════════════════════════��═══════
# 1. MODEL CALIBRATION
# ═══════════════════════════════════════════════════════════════════════


def plot_calibration_curve(probas: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    """Predicted probability vs actual win rate (reliability diagram).

    Args:
        probas: Model predicted probabilities (0-1).
        outcomes: Binary outcomes (1=win, 0=loss).
        n_bins: Number of bins for calibration.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_rates = []
    bin_counts = []

    for i in range(n_bins):
        mask = (probas >= bin_edges[i]) & (probas < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_rates.append(outcomes[mask].mean())
            bin_counts.append(int(mask.sum()))

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.7, 0.3],
        shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=["Calibration Curve", "Prediction Distribution"],
    )

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color=NEUTRAL, dash="dash", width=1),
        name="Perfect calibration", showlegend=True,
    ), row=1, col=1)

    # Actual calibration
    fig.add_trace(go.Scatter(
        x=bin_centers, y=bin_rates, mode="lines+markers",
        line=dict(color=PRIMARY, width=2),
        marker=dict(size=10, color=PRIMARY),
        name="Model calibration",
        text=[f"n={c}" for c in bin_counts],
        hovertemplate="Predicted: %{x:.2f}<br>Actual: %{y:.2f}<br>%{text}",
    ), row=1, col=1)

    # Histogram of predictions
    fig.add_trace(go.Histogram(
        x=probas, nbinsx=50, marker_color=BLUE, opacity=0.7,
        name="Prediction dist.", showlegend=True,
    ), row=2, col=1)

    fig.update_xaxes(title_text="Predicted Probability", row=2, col=1)
    fig.update_yaxes(title_text="Actual Win Rate", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    _apply_layout(fig, height=550, title_text="Model Calibration — Is Confidence Meaningful?")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 2. WALK-FORWARD FOLD COMPARISON
# ═══════════════════════════════════════════════════════════════════════


def plot_fold_comparison(wf_df: pd.DataFrame, symbol_filter: str = None):
    """Side-by-side bars for all walk-forward folds.

    Highlights winners vs blowups with color coding.
    """
    df = wf_df.copy()
    if symbol_filter:
        # Infer symbol from fold numbering: folds restart at 1 for each symbol.
        # XAUUSD folds are rows 0-16 (17 folds), USTEC folds are rows 17-30 (14 folds)
        # The CSV has folds 1-17 then 1-14.
        n = len(df)
        if n > 17:
            # First block = XAUUSD, second block = USTEC
            if symbol_filter == "USTEC":
                df = df.iloc[17:].reset_index(drop=True)
            else:
                df = df.iloc[:17].reset_index(drop=True)

    df["fold_label"] = [f"Fold {i+1}" for i in range(len(df))]
    df["outcome"] = df["total_return"].apply(
        lambda r: "PASS" if r >= 0.10 else ("BLOW" if r <= -0.09 else "FAIL")
    )
    colors = df["outcome"].map({"PASS": PROFIT, "BLOW": LOSS, "FAIL": SECONDARY}).tolist()

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=["Total Return per Fold", "Win Rate", "Trade Count"],
        row_heights=[0.4, 0.3, 0.3],
    )

    # Return bars
    fig.add_trace(go.Bar(
        x=df["fold_label"], y=df["total_return"] * 100,
        marker_color=colors, name="Return %",
        text=df["outcome"], textposition="outside",
        hovertemplate="Fold: %{x}<br>Return: %{y:.1f}%<br>%{text}",
    ), row=1, col=1)
    fig.add_hline(y=10, line_dash="dash", line_color=PROFIT, annotation_text="Phase 1 Target", row=1, col=1)
    fig.add_hline(y=-9, line_dash="dash", line_color=LOSS, annotation_text="Blow Threshold", row=1, col=1)

    # Win rate
    fig.add_trace(go.Bar(
        x=df["fold_label"], y=df["win_rate"] * 100,
        marker_color=[PROFIT if w > 0.40 else (LOSS if w < 0.25 else SECONDARY) for w in df["win_rate"]],
        name="Win Rate %",
        hovertemplate="Win Rate: %{y:.1f}%",
    ), row=2, col=1)
    fig.add_hline(y=40, line_dash="dot", line_color=NEUTRAL, row=2, col=1)

    # Trade count
    fig.add_trace(go.Bar(
        x=df["fold_label"], y=df["total_trades"],
        marker_color=[LOSS if t < 50 else BLUE for t in df["total_trades"]],
        name="Trades",
        hovertemplate="Trades: %{y}",
    ), row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color=LOSS, annotation_text="Low signal zone", row=3, col=1)

    fig.update_yaxes(title_text="Return %", row=1, col=1)
    fig.update_yaxes(title_text="Win Rate %", row=2, col=1)
    fig.update_yaxes(title_text="Trades", row=3, col=1)

    _apply_layout(fig, height=700, showlegend=False,
                  title_text="Walk-Forward Fold Anatomy — Winners vs Blowups")
    return fig


def plot_fold_scatter(wf_df: pd.DataFrame):
    """Scatter: win rate vs trade count, colored by outcome, sized by return."""
    df = wf_df.copy()
    df["outcome"] = df["total_return"].apply(
        lambda r: "PASS" if r >= 0.10 else ("BLOW" if r <= -0.09 else "FAIL")
    )
    color_map = {"PASS": PROFIT, "BLOW": LOSS, "FAIL": SECONDARY}

    fig = go.Figure()
    for outcome in ["PASS", "BLOW", "FAIL"]:
        mask = df["outcome"] == outcome
        sub = df[mask]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["total_trades"], y=sub["win_rate"] * 100,
            mode="markers",
            marker=dict(
                size=sub["total_return"].abs() * 80 + 10,
                color=color_map[outcome], opacity=0.8,
                line=dict(width=1, color="white"),
            ),
            name=outcome,
            text=[f"Fold {i+1}<br>Return: {r:.1%}" for i, r in zip(sub.index, sub["total_return"])],
            hovertemplate="%{text}<br>Win Rate: %{y:.1f}%<br>Trades: %{x}",
        ))

    fig.add_vline(x=50, line_dash="dash", line_color=LOSS, annotation_text="< 50 trades = danger")
    fig.add_hline(y=40, line_dash="dash", line_color=NEUTRAL, annotation_text="40% WR threshold")

    fig.update_xaxes(title_text="Total Trades")
    fig.update_yaxes(title_text="Win Rate %")
    _apply_layout(fig, height=450, title_text="Fold Clustering — Trade Count vs Win Rate")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 3. REGIME PERFORMANCE BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════


def plot_regime_performance(trades_df: pd.DataFrame, regime_labels: pd.Series):
    """P&L, win rate, and trade count split by regime state.

    Args:
        trades_df: DataFrame with columns: entry_time, pnl, confidence, direction.
        regime_labels: Series indexed by time with regime state names.
    """
    if trades_df.empty or regime_labels.empty:
        fig = go.Figure()
        fig.add_annotation(text="No regime data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=18, color=NEUTRAL))
        _apply_layout(fig, height=400)
        return fig

    # Map each trade to its regime
    regime_idx = regime_labels.index
    trade_regimes = []
    for _, row in trades_df.iterrows():
        t = row["entry_time"]
        # Find nearest regime label
        idx = regime_idx.get_indexer([t], method="ffill")[0]
        if idx >= 0:
            trade_regimes.append(regime_labels.iloc[idx])
        else:
            trade_regimes.append("unknown")

    trades_df = trades_df.copy()
    trades_df["regime"] = trade_regimes

    regime_colors = {
        "low_vol_trend": PROFIT,
        "moderate_trend": BLUE,
        "high_vol_chaos": LOSS,
        "unknown": NEUTRAL,
    }

    fig = make_subplots(
        rows=1, cols=3, subplot_titles=["Avg P&L by Regime", "Win Rate by Regime", "Trade Count by Regime"],
    )

    regimes_present = [r for r in ["low_vol_trend", "moderate_trend", "high_vol_chaos"]
                       if r in trades_df["regime"].values]

    avg_pnl = []
    win_rates = []
    counts = []
    for r in regimes_present:
        sub = trades_df[trades_df["regime"] == r]
        avg_pnl.append(sub["pnl"].mean())
        win_rates.append((sub["pnl"] > 0).mean() * 100)
        counts.append(len(sub))

    bar_colors = [regime_colors.get(r, NEUTRAL) for r in regimes_present]
    labels = [r.replace("_", " ").title() for r in regimes_present]

    fig.add_trace(go.Bar(x=labels, y=avg_pnl, marker_color=bar_colors, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=win_rates, marker_color=bar_colors, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=counts, marker_color=bar_colors, showlegend=False), row=1, col=3)

    fig.update_yaxes(title_text="Avg P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Win Rate %", row=1, col=2)
    fig.update_yaxes(title_text="Trade Count", row=1, col=3)

    _apply_layout(fig, height=380, title_text="Performance by Market Regime")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 4. SIGNAL CONFIDENCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════


def plot_confidence_analysis(confidences: np.ndarray, outcomes: np.ndarray):
    """Analyze whether model confidence predicts trade quality.

    Args:
        confidences: Trade-level confidence scores (0.5-1.0).
        outcomes: Binary outcomes (1=win, 0=loss).
    """
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=["Confidence Distribution", "Confidence vs Win Rate"],
    )

    # Distribution split by outcome
    wins_conf = confidences[outcomes == 1]
    loss_conf = confidences[outcomes == 0]

    fig.add_trace(go.Histogram(
        x=wins_conf, nbinsx=20, name="Winners",
        marker_color=PROFIT, opacity=0.7,
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        x=loss_conf, nbinsx=20, name="Losers",
        marker_color=LOSS, opacity=0.7,
    ), row=1, col=1)
    fig.update_layout(barmode="overlay")

    # Binned win rate by confidence
    n_bins = 8
    bin_edges = np.linspace(confidences.min(), confidences.max(), n_bins + 1)
    bin_centers = []
    bin_wr = []
    bin_counts = []
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() >= 5:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_wr.append(outcomes[mask].mean() * 100)
            bin_counts.append(int(mask.sum()))

    fig.add_trace(go.Scatter(
        x=bin_centers, y=bin_wr, mode="lines+markers",
        line=dict(color=PRIMARY, width=2),
        marker=dict(size=10, color=PRIMARY),
        name="Win Rate",
        text=[f"n={c}" for c in bin_counts],
        hovertemplate="Confidence: %{x:.2f}<br>Win Rate: %{y:.1f}%<br>%{text}",
    ), row=1, col=2)

    fig.add_hline(y=50, line_dash="dot", line_color=NEUTRAL, row=1, col=2)

    fig.update_xaxes(title_text="Confidence", row=1, col=1)
    fig.update_xaxes(title_text="Confidence Bin", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Win Rate %", row=1, col=2)

    _apply_layout(fig, height=400, title_text="Signal Confidence — Does Higher Confidence Mean Better Trades?")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 5. FEATURE STABILITY ACROSS FOLDS
# ═══════════════════════════════════════════════════════════════════════


def plot_feature_stability(fold_importances: list[dict], top_n: int = 15):
    """Heatmap showing feature importance stability across walk-forward folds.

    Args:
        fold_importances: List of dicts {feature_name: importance} per fold.
        top_n: Number of top features to show.
    """
    if not fold_importances:
        fig = go.Figure()
        fig.add_annotation(text="No feature importance data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=18, color=NEUTRAL))
        _apply_layout(fig, height=400)
        return fig

    # Build DataFrame
    all_features = set()
    for d in fold_importances:
        all_features.update(d.keys())

    rows = []
    for i, d in enumerate(fold_importances):
        for f in all_features:
            rows.append({"fold": f"Fold {i+1}", "feature": f, "importance": d.get(f, 0)})
    df = pd.DataFrame(rows)

    # Get top features by average importance
    avg_imp = df.groupby("feature")["importance"].mean().nlargest(top_n)
    top_features = avg_imp.index.tolist()

    df = df[df["feature"].isin(top_features)]
    pivot = df.pivot(index="feature", columns="fold", values="importance")
    pivot = pivot.loc[top_features]  # Sort by average importance

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, BG_DARK], [0.5, PRIMARY], [1, PROFIT]],
        hovertemplate="Feature: %{y}<br>Fold: %{x}<br>Importance: %{z:.0f}<extra></extra>",
    ))

    _apply_layout(fig, height=max(400, top_n * 28),
                  title_text=f"Feature Stability — Top {top_n} Features Across Folds",
                  xaxis=dict(title="Walk-Forward Fold"),
                  yaxis=dict(title="Feature", autorange="reversed"))
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 6. DRAWDOWN PATH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════


def plot_drawdown_paths(wf_df: pd.DataFrame):
    """Overlay equity paths for all folds, highlighting blowups.

    Uses normalized returns (starting at 0%) to compare fold trajectories.
    """
    fig = go.Figure()

    for i, row in wf_df.iterrows():
        ret = row["total_return"]
        trades = int(row["total_trades"])
        outcome = "PASS" if ret >= 0.10 else ("BLOW" if ret <= -0.09 else "FAIL")

        # Simulate approximate equity path from return and trade count
        # Linear interpolation as proxy (actual equity curve not in CSV)
        if trades > 0:
            if outcome == "BLOW":
                # Blowup: fast decline
                x = np.linspace(0, 1, min(trades, 100))
                y = ret * (x ** 0.5) * 100  # Concave decline
            else:
                x = np.linspace(0, 1, min(trades, 100))
                y = ret * x * 100  # Roughly linear growth

            color = PROFIT if outcome == "PASS" else (LOSS if outcome == "BLOW" else SECONDARY)
            opacity = 0.8 if outcome == "BLOW" else 0.4

            fig.add_trace(go.Scatter(
                x=x * trades, y=y, mode="lines",
                line=dict(color=color, width=2 if outcome == "BLOW" else 1),
                opacity=opacity,
                name=f"Fold {i+1} ({outcome})",
                hovertemplate=f"Fold {i+1}<br>Trade: %{{x:.0f}}<br>Return: %{{y:.1f}}%",
            ))

    fig.add_hline(y=10, line_dash="dash", line_color=PROFIT, annotation_text="+10% target")
    fig.add_hline(y=-9, line_dash="dash", line_color=LOSS, annotation_text="-9% halt")
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)

    fig.update_xaxes(title_text="Trade Number")
    fig.update_yaxes(title_text="Return %")
    _apply_layout(fig, height=450, title_text="Drawdown Paths — How Blowups Develop vs Winners")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 7. LIVE VS BACKTEST COMPARISON
# ═══════════════════════════════════════════════════════════════════════


def plot_live_vs_backtest(live_trades: pd.DataFrame, backtest_stats: dict):
    """Compare live execution metrics to backtest expectations.

    Args:
        live_trades: DataFrame of live trades (from trading.log parsing).
        backtest_stats: Dict with keys: win_rate, avg_pnl, avg_confidence, trades_per_day.
    """
    if live_trades.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No live trades yet — bot needs to execute trades before comparison is available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=SECONDARY),
        )
        _apply_layout(fig, height=300, title_text="Live vs Backtest — Waiting for Live Data")
        return fig

    metrics = ["Win Rate", "Avg P&L ($)", "Avg Confidence", "Trades/Day"]
    bt_vals = [
        backtest_stats.get("win_rate", 0) * 100,
        backtest_stats.get("avg_pnl", 0),
        backtest_stats.get("avg_confidence", 0),
        backtest_stats.get("trades_per_day", 0),
    ]

    live_wr = (live_trades["pnl"] > 0).mean() * 100 if "pnl" in live_trades else 0
    live_avg = live_trades["pnl"].mean() if "pnl" in live_trades else 0
    live_conf = live_trades["confidence"].mean() if "confidence" in live_trades else 0
    n_days = max(1, (live_trades["time"].max() - live_trades["time"].min()).days) if "time" in live_trades else 1
    live_tpd = len(live_trades) / n_days

    live_vals = [live_wr, live_avg, live_conf, live_tpd]

    fig = go.Figure()
    x_pos = np.arange(len(metrics))
    fig.add_trace(go.Bar(x=metrics, y=bt_vals, name="Backtest", marker_color=BLUE, opacity=0.7))
    fig.add_trace(go.Bar(x=metrics, y=live_vals, name="Live", marker_color=PROFIT, opacity=0.9))

    _apply_layout(fig, height=400, barmode="group",
                  title_text="Live vs Backtest Comparison")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 8. WALK-FORWARD SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════


def compute_wf_summary(wf_df: pd.DataFrame) -> dict:
    """Compute aggregate statistics from walk-forward results."""
    n = len(wf_df)
    passed = (wf_df["total_return"] >= 0.10).sum()
    blown = (wf_df["total_return"] <= -0.09).sum()
    failed = n - passed - blown

    passing = wf_df[wf_df["total_return"] >= 0.10]
    blowing = wf_df[wf_df["total_return"] <= -0.09]

    return {
        "total_folds": n,
        "pass_count": int(passed),
        "blow_count": int(blown),
        "fail_count": int(failed),
        "pass_rate": passed / n if n > 0 else 0,
        "blow_rate": blown / n if n > 0 else 0,
        "mean_return": wf_df["total_return"].mean(),
        "median_return": wf_df["total_return"].median(),
        "mean_return_winners": passing["total_return"].mean() if len(passing) > 0 else 0,
        "mean_return_losers": blowing["total_return"].mean() if len(blowing) > 0 else 0,
        "mean_sharpe": wf_df["sharpe"].mean(),
        "mean_win_rate": wf_df["win_rate"].mean(),
        "mean_win_rate_winners": passing["win_rate"].mean() if len(passing) > 0 else 0,
        "mean_win_rate_losers": blowing["win_rate"].mean() if len(blowing) > 0 else 0,
        "mean_trades_winners": passing["total_trades"].mean() if len(passing) > 0 else 0,
        "mean_trades_losers": blowing["total_trades"].mean() if len(blowing) > 0 else 0,
        "mean_max_dd": wf_df["max_dd"].mean(),
        "expected_value": wf_df["total_return"].mean() * 100_000,
        "days_to_target_avg": passing["days_to_target"].mean() if len(passing) > 0 else 0,
    }


def plot_wf_summary_card(summary: dict):
    """Visual summary card of walk-forward statistics."""
    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{"type": "indicator"}] * 4, [{"type": "indicator"}] * 4],
        vertical_spacing=0.15,
    )

    indicators = [
        ("Pass Rate", f"{summary['pass_rate']:.0%}", PROFIT if summary['pass_rate'] >= 0.6 else LOSS),
        ("Blow Rate", f"{summary['blow_rate']:.0%}", LOSS if summary['blow_rate'] >= 0.3 else PROFIT),
        ("E[V] / Attempt", f"${summary['expected_value']:+,.0f}", PROFIT if summary['expected_value'] > 0 else LOSS),
        ("Avg Days to Target", f"{summary['days_to_target_avg']:.0f}", BLUE),
        ("Mean Return", f"{summary['mean_return']:.1%}", PROFIT if summary['mean_return'] > 0 else LOSS),
        ("Winner Avg Return", f"{summary['mean_return_winners']:.1%}", PROFIT),
        ("Winner Avg WR", f"{summary['mean_win_rate_winners']:.1%}", PROFIT),
        ("Winner Avg Trades", f"{summary['mean_trades_winners']:.0f}", BLUE),
    ]

    for idx, (title, value, color) in enumerate(indicators):
        row = idx // 4 + 1
        col = idx % 4 + 1
        fig.add_trace(go.Indicator(
            mode="number",
            value=None,
            title=dict(text=f"<b>{title}</b>", font=dict(size=13)),
            number=dict(font=dict(size=28, color=color), valueformat="", suffix=""),
        ), row=row, col=col)
        # Use annotation for formatted value since Indicator needs numeric
        fig.add_annotation(
            text=f"<b>{value}</b>",
            xref="paper", yref="paper",
            x=(col - 0.5) / 4, y=1 - (row - 0.35) / 2,
            showarrow=False,
            font=dict(size=26, color=color),
        )

    # Remove the default indicator numbers
    for trace in fig.data:
        trace.visible = False

    _apply_layout(fig, height=280, title_text="Walk-Forward Validation Summary")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 9. WINNER VS BLOWUP ANATOMY TABLE
# ═══════════════════════════════════════════════════════════════════════


def winner_vs_blowup_stats(wf_df: pd.DataFrame) -> pd.DataFrame:
    """Build comparison table: average metrics for winners vs blowups."""
    passing = wf_df[wf_df["total_return"] >= 0.10]
    blowing = wf_df[wf_df["total_return"] <= -0.09]

    metrics = {
        "Metric": [
            "Count", "Avg Return", "Avg Win Rate", "Avg Trades",
            "Avg Sharpe", "Avg Max DD", "Avg Val Accuracy",
            "Avg Median Proba", "Avg Signals", "Buy/Sell Ratio",
        ],
        "Winners": [
            len(passing),
            f"{passing['total_return'].mean():.1%}",
            f"{passing['win_rate'].mean():.1%}",
            f"{passing['total_trades'].mean():.0f}",
            f"{passing['sharpe'].mean():.2f}",
            f"{passing['max_dd'].mean():.1%}",
            f"{passing['val_accuracy'].mean():.3f}",
            f"{passing['median_proba'].mean():.4f}",
            f"{passing['n_signals'].mean():.0f}",
            f"{(passing['n_buy'] / passing['n_signals'].clip(lower=1)).mean():.2f}",
        ],
        "Blowups": [
            len(blowing),
            f"{blowing['total_return'].mean():.1%}",
            f"{blowing['win_rate'].mean():.1%}",
            f"{blowing['total_trades'].mean():.0f}",
            f"{blowing['sharpe'].mean():.2f}",
            f"{blowing['max_dd'].mean():.1%}",
            f"{blowing['val_accuracy'].mean():.3f}",
            f"{blowing['median_proba'].mean():.4f}",
            f"{blowing['n_signals'].mean():.0f}",
            f"{(blowing['n_buy'] / blowing['n_signals'].clip(lower=1)).mean():.2f}",
        ] if len(blowing) > 0 else ["0"] + ["N/A"] * 9,
    }
    return pd.DataFrame(metrics)
