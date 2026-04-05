"""Streamlit dashboard for FTMO ML Trading System."""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mt5_connector import MT5Connector
from src.features.engineer import FeatureEngineer
from src.features.labeler import triple_barrier_labels
from src.models.ensemble import StackingEnsemble
from src.regime.hmm_detector import RegimeDetector
from src.backtest.engine import FTMOBacktester
from src.visualization.ml_insights import (
    plot_calibration_curve, plot_fold_comparison, plot_fold_scatter,
    plot_regime_performance, plot_confidence_analysis, plot_feature_stability,
    plot_drawdown_paths, plot_live_vs_backtest, compute_wf_summary,
    plot_wf_summary_card, winner_vs_blowup_stats,
)

# ── Page Config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="FTMO ML Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Theme CSS ───────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin: 4px;
    }
    .metric-value { font-size: 28px; font-weight: bold; }
    .metric-label { font-size: 12px; color: #8b949e; }
    .pass { color: #00ff88; }
    .fail { color: #ff4444; }
    .warn { color: #ffcc00; }
    .neutral { color: #58a6ff; }
    .big-number { font-size: 42px; font-weight: bold; }
    .section-header {
        font-size: 20px;
        font-weight: bold;
        color: #58a6ff;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_config():
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_model_symbol(symbol: str, config: dict) -> str:
    """Resolve broker symbol to model file symbol (e.g. US100.cash -> USTEC)."""
    for instr in config.get("instruments", []):
        if instr["symbol"] == symbol:
            return instr.get("model_symbol", symbol)
    return symbol


@st.cache_data(ttl=60)
def load_backtest_result(symbol: str, config: dict):
    """Load data and run backtest for a symbol (cached)."""
    model_symbol = _get_model_symbol(symbol, config)
    connector = MT5Connector()
    feature_eng = FeatureEngineer(config.get("features", {}))
    data_dir = str(PROJECT_ROOT / "data" / "raw")

    tf = config["timeframes"]["signal"]
    # Load data using model_symbol (data files are saved under original name)
    df = connector.load_data(model_symbol, tf, data_dir)
    if df.empty:
        return None

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
    model_path = str(PROJECT_ROOT / "models" / "saved" / f"{model_symbol}_ensemble.joblib")
    if not os.path.exists(model_path):
        return None
    ensemble.load(model_path)
    probas = ensemble.predict_proba(X_test)

    # Regime
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

    importance = ensemble.get_feature_importance(feature_cols)

    return {
        "result": result,
        "importance": importance,
        "probas": probas,
        "median_proba": median_proba,
        "test_df": test_df,
        "feature_cols": feature_cols,
    }


def metric_card(label, value, color_class="neutral"):
    """Render a styled metric card."""
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value {color_class}">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def ftmo_gauge(value, limit, label, invert=False):
    """Create a gauge showing proximity to FTMO limit."""
    pct = (value / limit) * 100 if limit > 0 else 0
    if invert:
        color = "#00ff88" if pct >= 100 else "#ffcc00" if pct >= 50 else "#ff4444"
    else:
        color = "#00ff88" if pct < 60 else "#ffcc00" if pct < 85 else "#ff4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": label, "font": {"size": 14, "color": "#c9d1d9"}},
        number={"suffix": "%", "font": {"size": 24, "color": color}},
        gauge={
            "axis": {"range": [0, limit * 100], "tickcolor": "#30363d"},
            "bar": {"color": color},
            "bgcolor": "#161b22",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, limit * 60], "color": "rgba(0, 255, 136, 0.1)"},
                {"range": [limit * 60, limit * 85], "color": "rgba(255, 204, 0, 0.1)"},
                {"range": [limit * 85, limit * 100], "color": "rgba(255, 68, 68, 0.1)"},
            ],
            "threshold": {
                "line": {"color": "#ff4444", "width": 3},
                "thickness": 0.8,
                "value": limit * 100,
            },
        },
    ))
    fig.update_layout(
        height=200, margin=dict(t=40, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#c9d1d9"},
    )
    return fig


# ── SIDEBAR ──────────────────────────────────────────────────────────

config = load_config()

st.sidebar.title("FTMO ML System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Backtest Results", "Model Analysis", "ML Insights", "Configuration", "Live Trading"],
    index=0,
)

enabled_symbols = [
    i["symbol"] for i in config.get("instruments", []) if i.get("enabled")
]

st.sidebar.markdown("---")
st.sidebar.markdown("**Enabled Instruments**")
for sym in enabled_symbols:
    st.sidebar.markdown(f"  - {sym}")

st.sidebar.markdown("---")
st.sidebar.caption(f"Config: {config['account']['challenge_type']}")
st.sidebar.caption(f"Risk/Trade: {config['risk']['risk_per_trade_pct']}%")
st.sidebar.caption(f"SL: {config['risk']['stop_loss_atr_multiple']} ATR")
st.sidebar.caption(f"TP: {config['risk']['take_profit_atr_multiple']} ATR")


# ── PAGE: DASHBOARD ──────────────────────────────────────────────────

if page == "Dashboard":
    st.title("FTMO ML Trading System Dashboard")

    # Load results for all enabled symbols
    all_results = {}
    for sym in enabled_symbols:
        data = load_backtest_result(sym, config)
        if data:
            all_results[sym] = data

    if not all_results:
        st.warning("No backtest results available. Run the pipeline first.")
        st.stop()

    # Combined metrics
    total_trades = sum(d["result"].total_trades for d in all_results.values())
    total_pnl = sum(sum(t.pnl for t in d["result"].trades) for d in all_results.values())
    total_return = total_pnl / config["account"]["initial_balance"]
    max_dd = max(d["result"].max_drawdown for d in all_results.values())
    avg_wr = np.mean([d["result"].win_rate for d in all_results.values()])

    # Top metrics row
    cols = st.columns(5)
    with cols[0]:
        color = "pass" if total_return > 0.10 else "warn" if total_return > 0.05 else "fail"
        metric_card("Combined Return", f"{total_return:.1%}", color)
    with cols[1]:
        color = "pass" if max_dd < 0.07 else "warn" if max_dd < 0.09 else "fail"
        metric_card("Max Drawdown", f"{max_dd:.2%}", color)
    with cols[2]:
        metric_card("Total Trades", f"{total_trades:,}", "neutral")
    with cols[3]:
        color = "pass" if avg_wr > 0.45 else "neutral"
        metric_card("Avg Win Rate", f"{avg_wr:.1%}", color)
    with cols[4]:
        all_passed = all(d["result"].ftmo_phase1_passed for d in all_results.values())
        metric_card("FTMO Phase 1", "PASSED" if all_passed else "IN PROGRESS",
                     "pass" if all_passed else "warn")

    st.markdown("")

    # FTMO Compliance Gauges
    st.markdown('<div class="section-header">FTMO Compliance</div>', unsafe_allow_html=True)
    gauge_cols = st.columns(4)

    with gauge_cols[0]:
        st.plotly_chart(ftmo_gauge(max_dd, 0.10, "Max Drawdown vs 10% Limit"), use_container_width=True)
    with gauge_cols[1]:
        max_daily = max(d["result"].max_daily_drawdown for d in all_results.values())
        st.plotly_chart(ftmo_gauge(max_daily, 0.05, "Daily DD vs 5% Limit"), use_container_width=True)
    with gauge_cols[2]:
        st.plotly_chart(ftmo_gauge(total_return, 0.10, "Return vs 10% Target", invert=True), use_container_width=True)
    with gauge_cols[3]:
        best_day = max(d["result"].best_day_pct for d in all_results.values())
        st.plotly_chart(ftmo_gauge(best_day, 0.50, "Best Day Rule (< 50%)"), use_container_width=True)

    # Equity curves
    st.markdown('<div class="section-header">Equity Curves</div>', unsafe_allow_html=True)

    fig = go.Figure()
    for sym, data in all_results.items():
        result = data["result"]
        if result.equity_curve is not None:
            fig.add_trace(go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                mode="lines",
                name=sym,
                line=dict(width=2),
            ))

    fig.add_hline(y=110_000, line_dash="dash", line_color="#ffcc00",
                  annotation_text="Phase 1 Target (+10%)")
    fig.add_hline(y=105_000, line_dash="dot", line_color="#ff8800",
                  annotation_text="Phase 2 Target (+5%)")
    fig.add_hline(y=90_000, line_dash="dash", line_color="#ff3333",
                  annotation_text="FTMO Limit (-10%)")

    fig.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-instrument summary table
    st.markdown('<div class="section-header">Per-Instrument Breakdown</div>', unsafe_allow_html=True)

    table_data = []
    for sym, data in all_results.items():
        r = data["result"]
        table_data.append({
            "Symbol": sym,
            "Return": f"{r.total_return:.2%}",
            "Sharpe": f"{r.sharpe_ratio:.2f}",
            "Max DD": f"{r.max_drawdown:.2%}",
            "Win Rate": f"{r.win_rate:.1%}",
            "Profit Factor": f"{r.profit_factor:.2f}",
            "Trades": r.total_trades,
            "Days to Target": str(r.days_to_target) if r.days_to_target > 0 else "-",
            "Phase 1": "PASS" if r.ftmo_phase1_passed else "FAIL",
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


# ── PAGE: BACKTEST RESULTS ───────────────────────────────────────────

elif page == "Backtest Results":
    st.title("Backtest Results")

    symbol = st.selectbox("Select Instrument", enabled_symbols)
    data = load_backtest_result(symbol, config)

    if not data:
        st.warning(f"No data for {symbol}")
        st.stop()

    result = data["result"]

    # Summary metrics
    cols = st.columns(6)
    metrics = [
        ("Return", f"{result.total_return:.2%}"),
        ("Sharpe", f"{result.sharpe_ratio:.2f}"),
        ("Max DD", f"{result.max_drawdown:.2%}"),
        ("Win Rate", f"{result.win_rate:.1%}"),
        ("Profit Factor", f"{result.profit_factor:.2f}"),
        ("Trades", f"{result.total_trades}"),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)

    # Equity + Drawdown
    if result.equity_curve is not None:
        peak = result.equity_curve.cummax()
        dd = (result.equity_curve - peak) / peak * 100

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.65, 0.35], vertical_spacing=0.05)

        fig.add_trace(go.Scatter(
            x=result.equity_curve.index, y=result.equity_curve.values,
            fill="tozeroy", fillcolor="rgba(0, 200, 100, 0.1)",
            line=dict(color="#00ff88", width=1.5), name="Equity",
        ), row=1, col=1)
        fig.add_hline(y=110_000, line_dash="dash", line_color="#ffcc00", row=1, col=1)
        fig.add_hline(y=90_000, line_dash="dash", line_color="#ff3333", row=1, col=1)

        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            fill="tozeroy", fillcolor="rgba(255, 50, 50, 0.15)",
            line=dict(color="#ff4444", width=1), name="Drawdown %",
        ), row=2, col=1)
        fig.add_hline(y=-10, line_dash="dash", line_color="#ff3333", row=2, col=1)

        fig.update_layout(template="plotly_dark", height=550, showlegend=False,
                          margin=dict(t=20, b=30))
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # Daily P&L bar chart
    if result.daily_pnl is not None and len(result.daily_pnl) > 0:
        st.markdown('<div class="section-header">Daily P&L</div>', unsafe_allow_html=True)

        daily = result.daily_pnl
        colors = ["#00ff88" if v >= 0 else "#ff4444" for v in daily.values]

        fig = go.Figure(go.Bar(
            x=daily.index, y=daily.values,
            marker_color=colors, opacity=0.8,
        ))
        fig.update_layout(template="plotly_dark", height=300,
                          xaxis_title="Date", yaxis_title="P&L ($)",
                          margin=dict(t=20, b=30))
        st.plotly_chart(fig, use_container_width=True)

    # Trade distribution
    if result.trades:
        st.markdown('<div class="section-header">Trade Analysis</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            pnls = [t.pnl for t in result.trades]
            fig = go.Figure(go.Histogram(
                x=pnls, nbinsx=50,
                marker_color="#58a6ff", opacity=0.7,
            ))
            fig.add_vline(x=0, line_color="#ffcc00", line_dash="dash")
            fig.update_layout(
                title="Trade P&L Distribution",
                template="plotly_dark", height=350,
                xaxis_title="P&L ($)", yaxis_title="Count",
                margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Win rate over time (rolling 50 trades)
            pnl_series = pd.Series(pnls)
            rolling_wr = (pnl_series > 0).rolling(50).mean() * 100
            fig = go.Figure(go.Scatter(
                y=rolling_wr.values, mode="lines",
                line=dict(color="#00ccff", width=1.5),
            ))
            fig.add_hline(y=50, line_dash="dot", line_color="#ffcc00")
            fig.update_layout(
                title="Rolling Win Rate (50 trades)",
                template="plotly_dark", height=350,
                xaxis_title="Trade #", yaxis_title="Win Rate (%)",
                margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)


# ── PAGE: MODEL ANALYSIS ─────────────────────────────────────────────

elif page == "Model Analysis":
    st.title("Model Analysis")

    symbol = st.selectbox("Select Instrument", enabled_symbols)
    data = load_backtest_result(symbol, config)

    if not data:
        st.warning(f"No data for {symbol}")
        st.stop()

    importance = data["importance"]
    probas = data["probas"]
    median_proba = data["median_proba"]

    col1, col2 = st.columns(2)

    with col1:
        # Feature importance
        st.markdown('<div class="section-header">Feature Importance (Top 20)</div>', unsafe_allow_html=True)
        top_n = 20
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names = [x[0] for x in sorted_imp][::-1]
        values = [x[1] for x in sorted_imp][::-1]

        fig = go.Figure(go.Bar(
            x=values, y=names, orientation="h",
            marker_color="#58a6ff", opacity=0.85,
        ))
        fig.update_layout(
            template="plotly_dark", height=max(400, top_n * 25),
            margin=dict(t=20, b=20, l=150),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Probability distribution
        st.markdown('<div class="section-header">Model Probability Distribution</div>', unsafe_allow_html=True)

        signal_offset = config["risk"].get("signal_offset", 0.02)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=probas, nbinsx=80,
            marker_color="#58a6ff", opacity=0.7,
            name="All predictions",
        ))
        fig.add_vline(x=median_proba, line_color="#ffcc00", line_dash="dash",
                      annotation_text=f"Median: {median_proba:.4f}")
        fig.add_vline(x=median_proba + signal_offset, line_color="#00ff88", line_dash="dot",
                      annotation_text="Buy threshold")
        fig.add_vline(x=median_proba - signal_offset, line_color="#ff4444", line_dash="dot",
                      annotation_text="Sell threshold")
        fig.update_layout(
            template="plotly_dark", height=400,
            xaxis_title="Predicted Probability",
            yaxis_title="Count",
            margin=dict(t=20, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Hourly performance analysis
    st.markdown('<div class="section-header">Performance by Hour (UTC)</div>', unsafe_allow_html=True)
    result = data["result"]
    if result.trades:
        trade_hours = {}
        for t in result.trades:
            h = t.entry_time.hour if hasattr(t.entry_time, "hour") else 0
            if h not in trade_hours:
                trade_hours[h] = {"wins": 0, "losses": 0, "pnl": 0}
            if t.pnl > 0:
                trade_hours[h]["wins"] += 1
            else:
                trade_hours[h]["losses"] += 1
            trade_hours[h]["pnl"] += t.pnl

        hours = sorted(trade_hours.keys())
        win_rates = [trade_hours[h]["wins"] / max(1, trade_hours[h]["wins"] + trade_hours[h]["losses"]) * 100 for h in hours]
        pnls = [trade_hours[h]["pnl"] for h in hours]
        colors = ["#00ff88" if p >= 0 else "#ff4444" for p in pnls]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=hours, y=pnls, name="P&L ($)",
            marker_color=colors, opacity=0.6,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=hours, y=win_rates, name="Win Rate %",
            mode="lines+markers", line=dict(color="#ffcc00", width=2),
            marker=dict(size=8),
        ), secondary_y=True)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", secondary_y=True)
        fig.update_layout(template="plotly_dark", height=350, margin=dict(t=20, b=30))
        fig.update_xaxes(title_text="Hour (UTC)", dtick=1)
        fig.update_yaxes(title_text="P&L ($)", secondary_y=False)
        fig.update_yaxes(title_text="Win Rate (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)


# ── PAGE: ML INSIGHTS ────────────────────────────────────────────────

elif page == "ML Insights":
    st.title("ML Insights & Diagnostics")
    st.markdown("Deep analysis of model behavior, walk-forward validation, and regime sensitivity.")

    # ── Load walk-forward results ──
    wf_path = PROJECT_ROOT / "reports" / "walk_forward_results.csv"
    if wf_path.exists():
        wf_all = pd.read_csv(wf_path)

        # Split into XAUUSD (first 17 folds) and USTEC (last 14 folds)
        wf_xauusd = wf_all.iloc[:17].reset_index(drop=True)
        wf_ustec = wf_all.iloc[17:].reset_index(drop=True)

        # ── Summary Card ──
        st.markdown('<div class="section-header">Walk-Forward Summary — USTEC</div>', unsafe_allow_html=True)
        summary = compute_wf_summary(wf_ustec)

        cols = st.columns(4)
        cols[0].metric("Pass Rate", f"{summary['pass_rate']:.0%}",
                       delta=f"{summary['pass_count']}/{summary['total_folds']} folds")
        cols[1].metric("Blow Rate", f"{summary['blow_rate']:.0%}",
                       delta=f"{summary['blow_count']} blowups", delta_color="inverse")
        cols[2].metric("E[V] per Attempt", f"${summary['expected_value']:+,.0f}")
        cols[3].metric("Avg Days to Target", f"{summary['days_to_target_avg']:.0f}",
                       delta="when passing")

        cols2 = st.columns(4)
        cols2[0].metric("Mean Return", f"{summary['mean_return']:.1%}")
        cols2[1].metric("Winner Avg Return", f"{summary['mean_return_winners']:.1%}")
        cols2[2].metric("Winner Avg Win Rate", f"{summary['mean_win_rate_winners']:.1%}")
        cols2[3].metric("Loser Avg Trades", f"{summary['mean_trades_losers']:.0f}",
                        delta="low signal = danger", delta_color="inverse")

        st.markdown("---")

        # ── Winner vs Blowup Anatomy ──
        st.markdown('<div class="section-header">Winner vs Blowup Anatomy</div>', unsafe_allow_html=True)
        comparison_df = winner_vs_blowup_stats(wf_ustec)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Fold Comparison ──
        st.markdown('<div class="section-header">Fold-by-Fold Breakdown</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_fold_comparison(wf_all, symbol_filter="USTEC"), use_container_width=True)

        # ── Fold Clustering ──
        st.plotly_chart(plot_fold_scatter(wf_ustec), use_container_width=True)

        # ── Drawdown Paths ──
        st.markdown('<div class="section-header">Drawdown Path Analysis</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_drawdown_paths(wf_ustec), use_container_width=True)
        st.caption(
            "Blowup folds (red) crash fast with very few trades (<50). "
            "Winning folds (green) grow steadily with 400-700+ trades. "
            "This explains why kill switches fail — blowups happen before there's enough data to detect them."
        )

    else:
        st.warning("Walk-forward results not found. Run `scripts/walk_forward.py` first.")

    st.markdown("---")

    # ── Model Diagnostics (from backtest data) ──
    st.markdown('<div class="section-header">Model Diagnostics</div>', unsafe_allow_html=True)

    # Use USTEC model (or US100.cash mapped to USTEC)
    diag_symbol = enabled_symbols[0] if enabled_symbols else "US100.cash"
    bt_data = load_backtest_result(diag_symbol, config)

    if bt_data:
        probas = bt_data["probas"]
        result = bt_data["result"]
        test_df = bt_data["test_df"]

        # Build outcomes from trades
        if result.trades:
            outcomes = np.array([1 if t.pnl > 0 else 0 for t in result.trades])
            trade_confs = np.array([t.confidence for t in result.trades])

            # ── Calibration ──
            st.markdown("**Model Calibration**")
            # For calibration, use probas vs target labels
            targets = test_df.iloc[:len(probas)].get("target")
            if targets is not None:
                st.plotly_chart(
                    plot_calibration_curve(probas, targets.values),
                    use_container_width=True,
                )
            else:
                st.info("Target labels not available for calibration plot.")

            # ── Confidence Analysis ──
            st.markdown("**Signal Confidence Analysis**")
            st.plotly_chart(
                plot_confidence_analysis(trade_confs, outcomes),
                use_container_width=True,
            )

            # ── Feature Stability ──
            st.markdown("**Feature Stability Across Training**")
            importance = bt_data.get("importance")
            if importance:
                # Single fold importance — show as a reference
                st.plotly_chart(
                    plot_feature_stability([importance]),
                    use_container_width=True,
                )
                st.caption(
                    "With a single training run, this shows current feature importance. "
                    "After walk-forward retraining, multiple folds will appear to show stability."
                )
            else:
                st.info("Feature importance data not available.")

            # ── Regime Performance ──
            st.markdown("**Regime Performance Breakdown**")
            model_sym = _get_model_symbol(diag_symbol, config)
            data_dir = str(PROJECT_ROOT / "data" / "raw")
            connector = MT5Connector()
            h4_data = connector.load_data(model_sym, config["timeframes"]["regime"], data_dir)
            if not h4_data.empty:
                regime = RegimeDetector(n_states=3)
                regime.fit(h4_data)
                if regime._fitted:
                    regime_labels = regime.predict_regime(h4_data)
                    # Build trades DataFrame for regime analysis
                    trades_data = pd.DataFrame([{
                        "entry_time": t.entry_time,
                        "pnl": t.pnl,
                        "confidence": t.confidence,
                        "direction": t.direction,
                    } for t in result.trades])
                    if not trades_data.empty:
                        st.plotly_chart(
                            plot_regime_performance(trades_data, regime_labels),
                            use_container_width=True,
                        )
                    else:
                        st.info("No trades to analyze by regime.")
                else:
                    st.info("Regime detector not fitted.")
            else:
                st.info("H4 data not available for regime analysis.")

        else:
            st.info("No trades in backtest result. Check model and signal configuration.")

        # ── Live vs Backtest ──
        st.markdown("---")
        st.markdown('<div class="section-header">Live vs Backtest Comparison</div>', unsafe_allow_html=True)
        st.plotly_chart(
            plot_live_vs_backtest(pd.DataFrame(), {
                "win_rate": result.win_rate if result else 0,
                "avg_pnl": np.mean([t.pnl for t in result.trades]) if result and result.trades else 0,
                "avg_confidence": np.mean([t.confidence for t in result.trades]) if result and result.trades else 0,
                "trades_per_day": result.total_trades / 30 if result else 0,
            }),
            use_container_width=True,
        )
    else:
        st.warning(f"Cannot load backtest data for {diag_symbol}. Ensure data files exist in data/raw/.")


# ── PAGE: CONFIGURATION ──────────────────────────────────────────────

elif page == "Configuration":
    st.title("System Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-header">Account</div>', unsafe_allow_html=True)
        st.json(config["account"])

        st.markdown('<div class="section-header">FTMO Limits</div>', unsafe_allow_html=True)
        st.json(config["ftmo_limits"])

    with col2:
        st.markdown('<div class="section-header">Risk Management</div>', unsafe_allow_html=True)
        st.json(config["risk"])

        st.markdown('<div class="section-header">Timeframes</div>', unsafe_allow_html=True)
        st.json(config["timeframes"])

    with col3:
        st.markdown('<div class="section-header">Model</div>', unsafe_allow_html=True)
        st.json(config["model"])

        st.markdown('<div class="section-header">Features</div>', unsafe_allow_html=True)
        st.json(config["features"])

    st.markdown('<div class="section-header">Instruments</div>', unsafe_allow_html=True)
    for instr in config["instruments"]:
        status = "Enabled" if instr.get("enabled") else "Disabled"
        color = "green" if instr.get("enabled") else "red"
        st.markdown(f":{color}[**{instr['symbol']}**] - {status} | Session: {instr.get('session_utc', 'N/A')}")

    # Optimization results
    opt_csv = PROJECT_ROOT / "reports" / "optimization_grid.csv"
    if opt_csv.exists():
        st.markdown('<div class="section-header">Optimization Grid Results</div>', unsafe_allow_html=True)
        opt_df = pd.read_csv(opt_csv)
        st.dataframe(
            opt_df.sort_values("total_return", ascending=False).head(20),
            use_container_width=True, hide_index=True,
        )


# ── PAGE: LIVE TRADING ───────────────────────────────────────────────

elif page == "Live Trading":
    st.title("Live Trading Monitor")

    # Try to connect to MT5
    connector = MT5Connector()
    connected = connector.connect()

    if connected:
        account = connector.get_account_info()
        if account:
            st.success(f"Connected to MT5 | Account: {account.get('login', 'N/A')}")

            cols = st.columns(5)
            with cols[0]:
                metric_card("Balance", f"${account['balance']:,.2f}", "neutral")
            with cols[1]:
                metric_card("Equity", f"${account['equity']:,.2f}", "neutral")
            with cols[2]:
                pnl = account['equity'] - account['balance']
                color = "pass" if pnl >= 0 else "fail"
                metric_card("Floating P&L", f"${pnl:,.2f}", color)
            with cols[3]:
                metric_card("Margin Used", f"${account.get('margin', 0):,.2f}", "neutral")
            with cols[4]:
                metric_card("Free Margin", f"${account.get('margin_free', 0):,.2f}", "neutral")

            # Open positions
            st.markdown('<div class="section-header">Open Positions</div>', unsafe_allow_html=True)

            try:
                import MetaTrader5 as mt5
                positions = mt5.positions_get()
                if positions:
                    pos_data = []
                    for p in positions:
                        pos_data.append({
                            "Ticket": p.ticket,
                            "Symbol": p.symbol,
                            "Type": "BUY" if p.type == 0 else "SELL",
                            "Volume": p.volume,
                            "Entry": p.price_open,
                            "Current": p.price_current,
                            "SL": p.sl,
                            "TP": p.tp,
                            "Profit": f"${p.profit:,.2f}",
                            "Comment": p.comment,
                        })
                    st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No open positions")
            except Exception as e:
                st.warning(f"Cannot read positions: {e}")

            # Recent price data
            st.markdown('<div class="section-header">Recent Price Action</div>', unsafe_allow_html=True)
            for sym in enabled_symbols:
                with st.expander(f"{sym} - Latest Bars"):
                    df = connector.get_recent_bars(sym, "H1", 50)
                    if not df.empty:
                        fig = go.Figure(go.Candlestick(
                            x=df.index, open=df["open"], high=df["high"],
                            low=df["low"], close=df["close"],
                            increasing_line_color="#00ff88",
                            decreasing_line_color="#ff4444",
                        ))
                        fig.update_layout(
                            template="plotly_dark", height=350,
                            xaxis_rangeslider_visible=False,
                            margin=dict(t=20, b=30),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No recent data for {sym}")

        connector.disconnect()
    else:
        st.warning("MT5 not connected. Start MetaTrader 5 terminal to enable live monitoring.")
        st.info("The dashboard works without MT5 for backtest analysis. Use the other tabs.")

    # Signal log (always visible, doesn't need MT5)
    st.markdown('<div class="section-header">Signal Log</div>', unsafe_allow_html=True)
    signal_path = PROJECT_ROOT / "logs" / "signals.csv"
    if signal_path.exists():
        sig_df = pd.read_csv(signal_path)
        if not sig_df.empty:
            # Color-code actions
            st.markdown(f"**{len(sig_df)} signals recorded** | "
                        f"Trades: {len(sig_df[sig_df['action'] == 'trade_opened'])} | "
                        f"No signal: {len(sig_df[sig_df['action'] == 'no_signal'])} | "
                        f"Blocked: {len(sig_df[sig_df['action'].isin(['regime_blocked', 'risk_blocked'])])}")

            display_df = sig_df.copy()
            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%m-%d %H:%M")
            display_df["bar_time"] = pd.to_datetime(display_df["bar_time"]).dt.strftime("%m-%d %H:%M")
            display_cols = ["timestamp", "bar_time", "symbol", "proba", "direction", "confidence",
                            "regime_scalar", "lots", "action", "result"]
            st.dataframe(display_df[display_cols].iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.info("No signals logged yet. Waiting for market hours.")
    else:
        st.info("Signal log not found. Start the live trader to begin recording.")


# ── Footer ───────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown("**FTMO ML System v1.0**")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
