"""Rich visualization module — plotly interactive charts for all pipeline outputs."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Dark theme for all charts
pio.templates.default = "plotly_dark"

COLORS = {
    "profit": "#00d4aa",
    "loss": "#ff4757",
    "neutral": "#747d8c",
    "primary": "#3742fa",
    "secondary": "#ffa502",
    "background": "#1a1a2e",
    "grid": "#2d2d44",
    "text": "#e0e0e0",
    "buy": "#00d4aa",
    "sell": "#ff4757",
    "equity": "#3742fa",
    "drawdown": "#ff6b81",
    "regime_trend": "#00d4aa",
    "regime_range": "#ffa502",
    "regime_chaos": "#ff4757",
}


def plot_equity_curve(equity_curve: pd.Series, initial_balance: float = 100_000,
                      phase1_target: float = 0.10, title: str = "FTMO Challenge Equity Curve") -> go.Figure:
    """Interactive equity curve with FTMO target and drawdown limits."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=["Equity Curve", "Drawdown"]
    )

    # Equity line
    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        name="Equity", line=dict(color=COLORS["equity"], width=2),
        fill="tozeroy", fillcolor="rgba(55, 66, 250, 0.1)"
    ), row=1, col=1)

    # Phase 1 target line
    target = initial_balance * (1 + phase1_target)
    fig.add_hline(y=target, line_dash="dash", line_color=COLORS["profit"],
                  annotation_text=f"Phase 1 Target ({phase1_target:.0%})",
                  annotation_font_color=COLORS["profit"], row=1, col=1)

    # Max loss line
    max_loss = initial_balance * 0.90
    fig.add_hline(y=max_loss, line_dash="dash", line_color=COLORS["loss"],
                  annotation_text="Max Loss (10%)",
                  annotation_font_color=COLORS["loss"], row=1, col=1)

    # Starting balance
    fig.add_hline(y=initial_balance, line_dash="dot", line_color=COLORS["neutral"],
                  annotation_text="Starting Balance", row=1, col=1)

    # Drawdown subplot
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak * 100
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        name="Drawdown %", line=dict(color=COLORS["drawdown"], width=1.5),
        fill="tozeroy", fillcolor="rgba(255, 107, 129, 0.2)"
    ), row=2, col=1)

    # Daily loss limit line on drawdown
    fig.add_hline(y=-5, line_dash="dash", line_color=COLORS["loss"],
                  annotation_text="Daily Limit (-5%)", row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        height=700, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    return fig


def plot_monthly_heatmap(daily_pnl: pd.Series, title: str = "Monthly P&L Heatmap") -> go.Figure:
    """Monthly returns heatmap — green for profit, red for loss."""
    if daily_pnl.empty:
        return go.Figure()

    df = daily_pnl.to_frame("pnl")
    df.index = pd.to_datetime(df.index)
    df["year"] = df.index.year
    df["month"] = df.index.month

    monthly = df.groupby(["year", "month"])["pnl"].sum().unstack(fill_value=0)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Ensure all months present
    for m in range(1, 13):
        if m not in monthly.columns:
            monthly[m] = 0
    monthly = monthly[sorted(monthly.columns)]

    fig = go.Figure(data=go.Heatmap(
        z=monthly.values,
        x=month_names[:monthly.shape[1]],
        y=[str(y) for y in monthly.index],
        text=[[f"${v:,.0f}" for v in row] for row in monthly.values],
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        colorscale=[[0, COLORS["loss"]], [0.5, "#1a1a2e"], [1, COLORS["profit"]]],
        zmid=0,
        colorbar=dict(title="P&L ($)")
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=300 + len(monthly) * 50,
        template="plotly_dark",
        yaxis=dict(autorange="reversed")
    )
    return fig


def plot_trade_distribution(trades: list, title: str = "Trade P&L Distribution") -> go.Figure:
    """Distribution of trade P&L with win/loss breakdown."""
    if not trades:
        return go.Figure()

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    fig = go.Figure()

    if wins:
        fig.add_trace(go.Histogram(
            x=wins, name=f"Wins ({len(wins)})",
            marker_color=COLORS["profit"], opacity=0.7,
            nbinsx=30
        ))
    if losses:
        fig.add_trace(go.Histogram(
            x=losses, name=f"Losses ({len(losses)})",
            marker_color=COLORS["loss"], opacity=0.7,
            nbinsx=30
        ))

    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    fig.add_vline(x=avg_win, line_dash="dash", line_color=COLORS["profit"],
                  annotation_text=f"Avg Win: ${avg_win:,.0f}")
    fig.add_vline(x=avg_loss, line_dash="dash", line_color=COLORS["loss"],
                  annotation_text=f"Avg Loss: ${avg_loss:,.0f}")

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="P&L ($)", yaxis_title="Count",
        barmode="overlay", height=400, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig


def plot_win_rate_over_time(trades: list, window: int = 50,
                            title: str = "Rolling Win Rate") -> go.Figure:
    """Rolling win rate to detect strategy degradation."""
    if len(trades) < window:
        return go.Figure()

    times = [t.entry_time for t in trades]
    outcomes = [1 if t.pnl > 0 else 0 for t in trades]
    sr = pd.Series(outcomes, index=times)
    rolling_wr = sr.rolling(window).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_wr.index, y=rolling_wr.values * 100,
        name=f"Win Rate ({window}-trade rolling)",
        line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy", fillcolor="rgba(55, 66, 250, 0.1)"
    ))

    fig.add_hline(y=50, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text="50% breakeven")
    fig.add_hline(y=55, line_dash="dot", line_color=COLORS["profit"],
                  annotation_text="55% target")

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        yaxis_title="Win Rate (%)", height=350, template="plotly_dark"
    )
    return fig


def plot_feature_importance(importance: dict, top_n: int = 20,
                            title: str = "Feature Importance") -> go.Figure:
    """Horizontal bar chart of feature importance."""
    if not importance:
        return go.Figure()

    items = list(importance.items())[:top_n]
    items.reverse()  # Bottom to top for horizontal bar
    features, values = zip(*items)

    max_val = max(values)
    colors = [COLORS["profit"] if v > max_val * 0.5 else
              COLORS["secondary"] if v > max_val * 0.25 else
              COLORS["neutral"] for v in values]

    fig = go.Figure(go.Bar(
        x=list(values), y=list(features),
        orientation="h", marker_color=colors,
        text=[f"{v:.0f}" for v in values],
        textposition="outside"
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=max(400, top_n * 25), template="plotly_dark",
        xaxis_title="Importance Score",
        margin=dict(l=150)
    )
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          labels: list = None,
                          title: str = "Prediction Confusion Matrix") -> go.Figure:
    """Confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    labels = labels or ["Sell (0)", "Buy (1)"]
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm / cm.sum() * 100

    text = [[f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)" for j in range(len(labels))]
            for i in range(len(labels))]

    fig = go.Figure(data=go.Heatmap(
        z=cm, x=[f"Pred: {l}" for l in labels], y=[f"True: {l}" for l in labels],
        text=text, texttemplate="%{text}", textfont=dict(size=14),
        colorscale=[[0, "#1a1a2e"], [1, COLORS["primary"]]],
        showscale=False
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=400, width=450, template="plotly_dark",
        yaxis=dict(autorange="reversed")
    )
    return fig


def plot_regime_overlay(prices: pd.DataFrame, regimes: pd.Series,
                        title: str = "Price with Regime Detection") -> go.Figure:
    """Candlestick chart with colored regime backgrounds."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.8, 0.2])

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=prices.index, open=prices["open"], high=prices["high"],
        low=prices["low"], close=prices["close"],
        increasing_line_color=COLORS["profit"],
        decreasing_line_color=COLORS["loss"],
        name="Price"
    ), row=1, col=1)

    # Regime colored background regions
    regime_colors = {
        "low_vol_trend": "rgba(0, 212, 170, 0.15)",
        "moderate_trend": "rgba(255, 165, 2, 0.15)",
        "high_vol_chaos": "rgba(255, 71, 87, 0.15)",
    }

    # Add regime as colored bar chart
    regime_map = {"low_vol_trend": 1, "moderate_trend": 0.5, "high_vol_chaos": 0}
    regime_numeric = regimes.map(regime_map).fillna(0.5)
    regime_color_list = [
        COLORS["regime_trend"] if r == "low_vol_trend" else
        COLORS["regime_range"] if r == "moderate_trend" else
        COLORS["regime_chaos"]
        for r in regimes.values
    ]

    fig.add_trace(go.Bar(
        x=regimes.index, y=regime_numeric.values,
        marker_color=regime_color_list, name="Regime",
        showlegend=False, opacity=0.7
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=700, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Regime", row=2, col=1, tickvals=[0, 0.5, 1],
                     ticktext=["Chaos", "Range", "Trend"])
    return fig


def plot_daily_pnl_bars(daily_pnl: pd.Series, title: str = "Daily P&L") -> go.Figure:
    """Bar chart of daily P&L — green for profit days, red for loss days."""
    if daily_pnl.empty:
        return go.Figure()

    colors = [COLORS["profit"] if v >= 0 else COLORS["loss"] for v in daily_pnl.values]

    fig = go.Figure(go.Bar(
        x=daily_pnl.index, y=daily_pnl.values,
        marker_color=colors, name="Daily P&L"
    ))

    # Rolling average
    if len(daily_pnl) > 5:
        rolling_avg = daily_pnl.rolling(5).mean()
        fig.add_trace(go.Scatter(
            x=rolling_avg.index, y=rolling_avg.values,
            name="5-day Avg", line=dict(color=COLORS["secondary"], width=2)
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        yaxis_title="P&L ($)", height=400, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig


def plot_cumulative_trades(trades: list, title: str = "Cumulative P&L by Trade") -> go.Figure:
    """Cumulative P&L line colored by win/loss streak."""
    if not trades:
        return go.Figure()

    pnls = [t.pnl for t in trades]
    cum_pnl = np.cumsum(pnls)
    trade_nums = list(range(1, len(trades) + 1))

    colors = [COLORS["profit"] if p >= 0 else COLORS["loss"] for p in pnls]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trade_nums, y=cum_pnl,
        mode="lines+markers",
        marker=dict(color=colors, size=5),
        line=dict(color=COLORS["primary"], width=2),
        name="Cumulative P&L"
    ))

    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["neutral"])

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Trade #", yaxis_title="Cumulative P&L ($)",
        height=400, template="plotly_dark"
    )
    return fig


def plot_risk_metrics_gauge(metrics: dict, title: str = "Risk Dashboard") -> go.Figure:
    """Gauge charts for key risk metrics."""
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=["Daily P&L", "Total Drawdown", "Win Rate"]
    )

    daily_pnl_pct = metrics.get("daily_pnl_pct", 0) * 100
    total_pnl_pct = metrics.get("total_pnl_pct", 0) * 100
    win_rate = metrics.get("win_rate", 0) * 100

    # Daily P&L gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=daily_pnl_pct,
        number=dict(suffix="%", font=dict(size=24)),
        gauge=dict(
            axis=dict(range=[-5, 5]),
            bar=dict(color=COLORS["profit"] if daily_pnl_pct >= 0 else COLORS["loss"]),
            steps=[
                dict(range=[-5, -3], color="rgba(255, 71, 87, 0.3)"),
                dict(range=[-3, 0], color="rgba(255, 165, 2, 0.2)"),
                dict(range=[0, 5], color="rgba(0, 212, 170, 0.2)"),
            ],
            threshold=dict(line=dict(color=COLORS["loss"], width=3), value=-5)
        )
    ), row=1, col=1)

    # Total drawdown gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=total_pnl_pct,
        number=dict(suffix="%", font=dict(size=24)),
        gauge=dict(
            axis=dict(range=[-10, 15]),
            bar=dict(color=COLORS["profit"] if total_pnl_pct >= 0 else COLORS["loss"]),
            steps=[
                dict(range=[-10, -8], color="rgba(255, 71, 87, 0.4)"),
                dict(range=[-8, 0], color="rgba(255, 165, 2, 0.2)"),
                dict(range=[0, 10], color="rgba(0, 212, 170, 0.2)"),
                dict(range=[10, 15], color="rgba(0, 212, 170, 0.4)"),
            ],
            threshold=dict(line=dict(color=COLORS["loss"], width=3), value=-10)
        )
    ), row=1, col=2)

    # Win rate gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=win_rate,
        number=dict(suffix="%", font=dict(size=24)),
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=COLORS["profit"] if win_rate >= 50 else COLORS["loss"]),
            steps=[
                dict(range=[0, 40], color="rgba(255, 71, 87, 0.2)"),
                dict(range=[40, 50], color="rgba(255, 165, 2, 0.2)"),
                dict(range=[50, 100], color="rgba(0, 212, 170, 0.2)"),
            ],
            threshold=dict(line=dict(color=COLORS["profit"], width=3), value=55)
        )
    ), row=1, col=3)

    fig.update_layout(
        height=300, template="plotly_dark",
        title=dict(text=title, font=dict(size=18)),
        margin=dict(t=80, b=20)
    )
    return fig


def plot_hourly_performance(trades: list, title: str = "Performance by Hour of Day") -> go.Figure:
    """Bar chart showing average P&L by hour — reveals best trading sessions."""
    if not trades:
        return go.Figure()

    hours = []
    pnls = []
    for t in trades:
        if hasattr(t.entry_time, "hour"):
            hours.append(t.entry_time.hour)
            pnls.append(t.pnl)

    if not hours:
        return go.Figure()

    df = pd.DataFrame({"hour": hours, "pnl": pnls})
    hourly = df.groupby("hour").agg(
        avg_pnl=("pnl", "mean"),
        count=("pnl", "count"),
        total_pnl=("pnl", "sum")
    ).reindex(range(24), fill_value=0)

    colors = [COLORS["profit"] if v >= 0 else COLORS["loss"] for v in hourly["avg_pnl"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=hourly.index, y=hourly["avg_pnl"],
        name="Avg P&L", marker_color=colors, opacity=0.8
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=hourly.index, y=hourly["count"],
        name="Trade Count", line=dict(color=COLORS["secondary"], width=2),
        mode="lines+markers"
    ), secondary_y=True)

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Hour (UTC)", height=400, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    fig.update_yaxes(title_text="Avg P&L ($)", secondary_y=False)
    fig.update_yaxes(title_text="Trade Count", secondary_y=True)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list,
                             title: str = "Feature Correlation Matrix") -> go.Figure:
    """Correlation heatmap of features — helps identify redundant features."""
    corr = df[feature_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu_r", zmid=0,
        colorbar=dict(title="Correlation"),
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(size=8)
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=800, width=900, template="plotly_dark",
        xaxis=dict(tickangle=45)
    )
    return fig


def plot_ftmo_summary_card(result, initial_balance: float = 100_000) -> go.Figure:
    """Visual summary card showing FTMO challenge pass/fail status."""
    fig = go.Figure()

    # Background
    passed = result.ftmo_phase1_passed and not result.ftmo_total_limit_breached
    bg_color = "rgba(0, 212, 170, 0.1)" if passed else "rgba(255, 71, 87, 0.1)"
    status_text = "CHALLENGE PASSED" if passed else "CHALLENGE FAILED"
    status_color = COLORS["profit"] if passed else COLORS["loss"]

    metrics = [
        ("Total Return", f"{result.total_return:.2%}",
         COLORS["profit"] if result.total_return > 0 else COLORS["loss"]),
        ("Sharpe Ratio", f"{result.sharpe_ratio:.2f}",
         COLORS["profit"] if result.sharpe_ratio > 1 else COLORS["secondary"]),
        ("Max Drawdown", f"{result.max_drawdown:.2%}", COLORS["loss"]),
        ("Win Rate", f"{result.win_rate:.1%}",
         COLORS["profit"] if result.win_rate > 0.5 else COLORS["loss"]),
        ("Profit Factor", f"{result.profit_factor:.2f}",
         COLORS["profit"] if result.profit_factor > 1.2 else COLORS["secondary"]),
        ("Total Trades", f"{result.total_trades}", COLORS["text"]),
        ("Best Day %", f"{result.best_day_pct:.1%}",
         COLORS["profit"] if result.best_day_pct < 0.5 else COLORS["loss"]),
        ("Days to Target", f"{result.days_to_target}" if result.days_to_target > 0 else "N/A",
         COLORS["text"]),
    ]

    # Title
    fig.add_annotation(x=0.5, y=0.95, text=status_text, showarrow=False,
                       font=dict(size=28, color=status_color, family="Arial Black"),
                       xref="paper", yref="paper")

    # Metrics grid
    for i, (label, value, color) in enumerate(metrics):
        row = i // 4
        col = i % 4
        x = 0.125 + col * 0.25
        y = 0.7 - row * 0.35

        fig.add_annotation(x=x, y=y, text=value, showarrow=False,
                           font=dict(size=22, color=color, family="Arial Black"),
                           xref="paper", yref="paper")
        fig.add_annotation(x=x, y=y - 0.08, text=label, showarrow=False,
                           font=dict(size=12, color=COLORS["neutral"]),
                           xref="paper", yref="paper")

    fig.update_layout(
        height=350, template="plotly_dark",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor=bg_color,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig


def generate_full_report(result, trades: list, daily_pnl: pd.Series,
                         feature_importance: dict, output_dir: str = "reports"):
    """Generate all charts and save as interactive HTML files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    charts = {}

    # Summary card
    charts["00_summary"] = plot_ftmo_summary_card(result)

    # Equity curve
    if result.equity_curve is not None:
        charts["01_equity_curve"] = plot_equity_curve(result.equity_curve)

    # Risk gauges
    charts["02_risk_gauges"] = plot_risk_metrics_gauge({
        "daily_pnl_pct": result.max_daily_drawdown if hasattr(result, "max_daily_drawdown") else 0,
        "total_pnl_pct": result.total_return,
        "win_rate": result.win_rate,
    })

    # Trade distribution
    if trades:
        charts["03_trade_distribution"] = plot_trade_distribution(trades)
        charts["04_cumulative_trades"] = plot_cumulative_trades(trades)
        charts["05_win_rate_rolling"] = plot_win_rate_over_time(trades)
        charts["06_hourly_performance"] = plot_hourly_performance(trades)

    # Daily P&L
    if daily_pnl is not None and not daily_pnl.empty:
        charts["07_daily_pnl"] = plot_daily_pnl_bars(daily_pnl)
        charts["08_monthly_heatmap"] = plot_monthly_heatmap(daily_pnl)

    # Feature importance
    if feature_importance:
        charts["09_feature_importance"] = plot_feature_importance(feature_importance)

    # Save all charts
    for name, fig in charts.items():
        path = os.path.join(output_dir, f"{name}.html")
        fig.write_html(path, include_plotlyjs="cdn")

    # Generate combined HTML report
    _generate_combined_report(charts, output_dir)

    return charts


def _generate_combined_report(charts: dict, output_dir: str):
    """Combine all charts into a single HTML report."""
    import os

    html_parts = ["""
<!DOCTYPE html>
<html>
<head>
    <title>FTMO ML Trading System — Backtest Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            background-color: #0f0f23;
            color: #e0e0e0;
            font-family: 'Segoe UI', Tahoma, sans-serif;
            margin: 0; padding: 20px;
        }
        .header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid #3742fa;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #3742fa, #00d4aa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }
        .header p { color: #747d8c; font-size: 1.1em; }
        .chart-container {
            background: #1a1a2e;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }
        @media (max-width: 1200px) { .grid-2 { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>FTMO ML Trading System</h1>
        <p>Backtest Report — Generated with LightGBM + XGBoost + CatBoost Ensemble</p>
    </div>
"""]

    for name, fig in charts.items():
        div_id = f"chart_{name}"
        html_parts.append(f'<div class="chart-container">')
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id))
        html_parts.append("</div>")

    html_parts.append("</body></html>")

    report_path = os.path.join(output_dir, "full_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
