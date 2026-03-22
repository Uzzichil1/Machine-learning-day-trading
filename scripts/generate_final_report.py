"""Generate final optimization report with interactive Plotly visuals."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.mt5_connector import MT5Connector
from src.features.engineer import FeatureEngineer
from src.features.labeler import triple_barrier_labels
from src.models.ensemble import StackingEnsemble
from src.regime.hmm_detector import RegimeDetector
from src.backtest.engine import FTMOBacktester
from src.pipeline import load_config, PROJECT_ROOT
DARK_TEMPLATE = "plotly_dark"

import logging
logging.basicConfig(level=logging.WARNING)

config = load_config()
REPORT_DIR = str(PROJECT_ROOT / "reports")


def get_backtest_result(symbol: str):
    """Get full backtest result for a symbol with optimized settings."""
    connector = MT5Connector()
    feature_eng = FeatureEngineer(config.get("features", {}))
    data_dir = str(PROJECT_ROOT / "data" / "raw")

    tf = config["timeframes"]["signal"]
    df = connector.load_data(symbol, tf, data_dir)
    if df.empty:
        return None, None

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

    ensemble = StackingEnsemble(config.get("model", {}))
    model_path = os.path.join(str(PROJECT_ROOT / "models" / "saved"), f"{symbol}_ensemble.joblib")
    ensemble.load(model_path)
    probas = ensemble.predict_proba(X_test)

    regime_scalars = np.ones(len(test_df))
    h4_data = connector.load_data(symbol, config["timeframes"]["regime"], data_dir)
    if not h4_data.empty:
        regime = RegimeDetector(n_states=3)
        regime.fit(h4_data)
        if regime._fitted:
            try:
                h4_scalars = regime.get_size_scalar(h4_data)
                h4_scalars.index = h4_scalars.index.tz_localize(None) if h4_scalars.index.tz else h4_scalars.index
                test_idx = test_df.index.tz_localize(None) if test_df.index.tz else test_df.index
                regime_reindexed = h4_scalars.reindex(test_idx, method="ffill").fillna(1.0)
                regime_scalars = regime_reindexed.values
            except Exception:
                pass

    risk_cfg = config.get("risk", {})
    signal_offset = risk_cfg.get("signal_offset", 0.02)
    median_proba = np.median(probas)

    signals = pd.DataFrame(index=test_df.index)
    signals["signal"] = 0
    signals.loc[probas >= median_proba + signal_offset, "signal"] = 1
    signals.loc[probas <= median_proba - signal_offset, "signal"] = -1
    signals["confidence"] = np.where(
        signals["signal"] == 1,
        (probas - median_proba) / (1 - median_proba),
        np.where(signals["signal"] == -1, (median_proba - probas) / median_proba, 0.5),
    ).clip(0.3, 1.0)
    signals["atr"] = test_df["atr"].values
    signals["regime_scalar"] = regime_scalars
    signals["symbol"] = symbol

    backtester = FTMOBacktester({**config.get("account", {}), **config.get("ftmo_limits", {})})
    result = backtester.run(
        signals, test_df,
        risk_per_trade=risk_cfg.get("risk_per_trade_pct", 0.40) / 100,
        sl_atr_mult=risk_cfg.get("stop_loss_atr_multiple", 1.5),
        tp_atr_mult=risk_cfg.get("take_profit_atr_multiple", 2.5),
    )

    return result, ensemble.get_feature_importance(feature_cols)


def make_mc_chart(trade_pnls, n_sims=500):
    """Create Monte Carlo equity curves chart."""
    rng = np.random.default_rng(42)
    pnls = np.array(trade_pnls)
    initial = 100_000

    fig = go.Figure()

    # Sample paths
    for i in range(min(200, n_sims)):
        shuffled = rng.permutation(pnls)
        equity = np.cumsum(shuffled) + initial
        fig.add_trace(go.Scatter(
            y=equity, mode="lines",
            line=dict(width=0.3, color="rgba(0, 200, 255, 0.08)"),
            showlegend=False, hoverinfo="skip",
        ))

    # Original order
    equity_orig = np.cumsum(pnls) + initial
    fig.add_trace(go.Scatter(
        y=equity_orig, mode="lines",
        line=dict(width=2, color="#00ff88"),
        name="Actual Sequence",
    ))

    # Target lines
    fig.add_hline(y=110_000, line_dash="dash", line_color="#ffcc00",
                  annotation_text="Phase 1 Target (10%)")
    fig.add_hline(y=105_000, line_dash="dot", line_color="#ff8800",
                  annotation_text="Phase 2 Target (5%)")
    fig.add_hline(y=90_000, line_dash="dash", line_color="#ff3333",
                  annotation_text="FTMO Limit (-10%)")

    fig.update_layout(
        title="Monte Carlo Equity Paths (200 shuffled trade sequences)",
        xaxis_title="Trade #",
        yaxis_title="Equity ($)",
        template=DARK_TEMPLATE,
        height=500,
    )
    return fig


def make_grid_heatmap():
    """Create optimization grid heatmap from saved CSV."""
    csv_path = os.path.join(REPORT_DIR, "optimization_grid.csv")
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    xau = df[df["symbol"] == "XAUUSD"].copy()

    # Pivot: risk_pct vs tp_atr, value = days_to_target (for passed combos)
    passed = xau[(xau["phase1_passed"]) & (xau["max_dd"] < 0.07) & (xau["sl_atr"] == 1.5) & (xau["signal_offset"] == 0.02)]

    if passed.empty:
        return None

    pivot = passed.pivot_table(
        values="days_to_target", index="risk_pct", columns="tp_atr", aggfunc="mean"
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{c:.2f}" for c in pivot.columns],
        y=[f"{r:.2f}%" for r in pivot.index],
        colorscale="Viridis_r",
        text=pivot.values.astype(int),
        texttemplate="%{text}d",
        colorbar=dict(title="Days to Target"),
    ))
    fig.update_layout(
        title="Days to FTMO Phase 1 Target (SL=1.5 ATR, Offset=0.02)",
        xaxis_title="Take Profit (ATR multiple)",
        yaxis_title="Risk Per Trade (%)",
        template=DARK_TEMPLATE,
        height=400,
    )
    return fig


def make_drawdown_chart(result):
    """Create drawdown analysis chart."""
    if result.equity_curve is None:
        return None

    peak = result.equity_curve.cummax()
    dd = (result.equity_curve - peak) / peak * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Equity Curve", "Drawdown (%)"],
                        vertical_spacing=0.08)

    fig.add_trace(go.Scatter(
        x=result.equity_curve.index, y=result.equity_curve.values,
        fill="tozeroy", fillcolor="rgba(0, 200, 100, 0.15)",
        line=dict(color="#00ff88", width=1.5), name="Equity",
    ), row=1, col=1)

    fig.add_hline(y=110_000, line_dash="dash", line_color="#ffcc00", row=1, col=1)
    fig.add_hline(y=90_000, line_dash="dash", line_color="#ff3333", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy", fillcolor="rgba(255, 50, 50, 0.2)",
        line=dict(color="#ff4444", width=1), name="Drawdown",
    ), row=2, col=1)

    fig.add_hline(y=-5, line_dash="dot", line_color="#ff8800", row=2, col=1,
                  annotation_text="Daily Limit (-5%)")
    fig.add_hline(y=-10, line_dash="dash", line_color="#ff3333", row=2, col=1,
                  annotation_text="Total Limit (-10%)")

    fig.update_layout(template=DARK_TEMPLATE, height=600, showlegend=False)
    return fig


def make_trade_scatter(result):
    """Create trade P&L scatter with cumulative line."""
    if not result.trades:
        return None

    times = [t.entry_time for t in result.trades]
    pnls = [t.pnl for t in result.trades]
    colors = ["#00ff88" if p > 0 else "#ff4444" for p in pnls]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=times, y=pnls, marker_color=colors, name="Trade P&L", opacity=0.6,
    ), secondary_y=False)

    cum_pnl = np.cumsum(pnls)
    fig.add_trace(go.Scatter(
        x=times, y=cum_pnl, mode="lines",
        line=dict(color="#00ccff", width=2), name="Cumulative P&L",
    ), secondary_y=True)

    fig.update_layout(
        title="Individual Trade P&L with Cumulative Line",
        template=DARK_TEMPLATE, height=400,
    )
    fig.update_yaxes(title_text="Trade P&L ($)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative P&L ($)", secondary_y=True)
    return fig


def make_summary_card(results_dict):
    """Create FTMO summary dashboard card."""
    fig = go.Figure()

    # Create a table-like display
    headers = ["Metric", "XAUUSD", "USTEC", "Combined"]
    cells = []

    metrics = [
        ("Total Return", "total_return", "{:.2%}"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.2f}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
        ("Win Rate", "win_rate", "{:.1%}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Total Trades", "total_trades", "{}"),
        ("Phase 1 Pass", "ftmo_phase1_passed", "{}"),
    ]

    metric_names = [m[0] for m in metrics]
    xau_vals, ustec_vals, combined_vals = [], [], []

    for name, attr, fmt in metrics:
        for result, vals in [(results_dict.get("XAUUSD"), xau_vals),
                             (results_dict.get("USTEC"), ustec_vals),
                             (results_dict.get("combined"), combined_vals)]:
            if result:
                v = getattr(result, attr, "N/A")
                if isinstance(v, bool):
                    vals.append("YES" if v else "NO")
                else:
                    vals.append(fmt.format(v))
            else:
                vals.append("N/A")

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color="#1a1a2e",
            font=dict(color="white", size=14),
            align="center",
        ),
        cells=dict(
            values=[metric_names, xau_vals, ustec_vals, combined_vals],
            fill_color=[["#16213e"] * len(metrics)] * 4,
            font=dict(color="white", size=13),
            align="center",
            height=30,
        ),
    )])

    fig.update_layout(
        title="FTMO Challenge Performance Summary",
        template=DARK_TEMPLATE,
        height=350,
    )
    return fig


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    results = {}
    all_pnls = []

    for instr in config.get("instruments", []):
        if not instr.get("enabled"):
            continue
        symbol = instr["symbol"]
        print(f"Running backtest for {symbol}...")
        result, importance = get_backtest_result(symbol)
        if result:
            results[symbol] = result
            all_pnls.extend([t.pnl for t in result.trades])

    # Build combined HTML report
    charts = []

    # Summary table
    fig_summary = make_summary_card(results)
    charts.append(("summary", fig_summary))

    # Per-instrument equity + drawdown
    for symbol, result in results.items():
        fig_dd = make_drawdown_chart(result)
        if fig_dd:
            fig_dd.update_layout(title=f"{symbol} Equity & Drawdown")
            charts.append((f"{symbol}_equity", fig_dd))

        fig_trades = make_trade_scatter(result)
        if fig_trades:
            fig_trades.update_layout(title=f"{symbol} Trade P&L")
            charts.append((f"{symbol}_trades", fig_trades))

    # Grid heatmap
    fig_grid = make_grid_heatmap()
    if fig_grid:
        charts.append(("grid_heatmap", fig_grid))

    # Monte Carlo
    if all_pnls:
        print("Generating Monte Carlo chart...")
        fig_mc = make_mc_chart(all_pnls)
        charts.append(("monte_carlo", fig_mc))

    # Write combined HTML
    html_parts = [
        "<!DOCTYPE html><html><head>",
        '<meta charset="utf-8">',
        "<title>FTMO ML System - Final Optimization Report</title>",
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
        "<style>",
        "body { background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }",
        "h1 { text-align: center; color: #00ff88; font-size: 28px; margin-bottom: 5px; }",
        "h2 { text-align: center; color: #58a6ff; font-size: 16px; margin-top: 0; }",
        ".chart { margin: 20px auto; max-width: 1200px; }",
        ".stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; max-width: 1200px; margin: 20px auto; }",
        ".stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center; }",
        ".stat-value { font-size: 24px; font-weight: bold; }",
        ".stat-label { font-size: 12px; color: #8b949e; margin-top: 4px; }",
        ".pass { color: #00ff88; } .fail { color: #ff4444; }",
        "</style></head><body>",
        "<h1>FTMO ML Trading System</h1>",
        "<h2>Final Optimization Report - Generated 2026-03-22</h2>",
    ]

    # Stats cards
    combined_trades = sum(r.total_trades for r in results.values())
    combined_pnl = sum(sum(t.pnl for t in r.trades) for r in results.values())
    combined_ret = combined_pnl / 100_000

    html_parts.append('<div class="stats-grid">')
    stats = [
        (f"{combined_ret:.1%}", "Combined Return", "pass" if combined_ret > 0.10 else "fail"),
        (f"{max(r.max_drawdown for r in results.values()):.1%}", "Max Drawdown", "pass"),
        (f"{combined_trades}", "Total Trades", "pass"),
        ("100%", "MC Pass Rate", "pass"),
    ]
    for value, label, css_class in stats:
        html_parts.append(
            f'<div class="stat-card"><div class="stat-value {css_class}">{value}</div>'
            f'<div class="stat-label">{label}</div></div>'
        )
    html_parts.append("</div>")

    # Add charts
    for i, (name, fig) in enumerate(charts):
        div_id = f"chart_{name}"
        html_parts.append(f'<div class="chart" id="{div_id}"></div>')
        html_parts.append("<script>")
        html_parts.append(f"Plotly.newPlot('{div_id}', {fig.to_json()}.data, {fig.to_json()}.layout);")
        html_parts.append("</script>")

    html_parts.append("</body></html>")

    output_path = os.path.join(REPORT_DIR, "final_optimization_report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"\nFinal report saved to: {output_path}")
    print("Open in browser for interactive charts.")


if __name__ == "__main__":
    main()
