"""
visualizer.py
=============
Generates Plotly-compatible JSON chart specifications.
Charts are rendered client-side by Plotly.js — no server-side plotly needed.
All functions return JSON strings (Plotly figure format).
"""

import json
import numpy as np

PALETTE = {
    "S": "#38bdf8",
    "E": "#f59e0b",
    "I": "#ef4444",
    "R": "#22c55e",
    "D": "#a855f7",
}

LAYOUT_BASE = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor":  "rgba(0,0,0,0)",
    "font": {"family": "JetBrains Mono, monospace", "color": "#e2e8f0", "size": 11},
    "legend": {"bgcolor": "rgba(0,0,0,0.3)", "bordercolor": "#334155", "borderwidth": 1},
    "xaxis": {"gridcolor": "#1e293b", "zerolinecolor": "#334155", "color": "#94a3b8"},
    "yaxis": {"gridcolor": "#1e293b", "zerolinecolor": "#334155", "color": "#94a3b8"},
    "margin": {"l": 50, "r": 20, "t": 40, "b": 50},
    "hovermode": "x unified",
}


def _fig(data: list, layout: dict) -> str:
    """Serialize a Plotly figure to JSON string."""
    merged = {**LAYOUT_BASE, **layout}
    return json.dumps({"data": data, "layout": merged})


def compartment_chart(result: dict) -> str:
    t = result["t"]
    traces = []
    defs = [("S", "Susceptible"), ("I", "Infected"), ("R", "Recovered"), ("D", "Deaths")]
    if result.get("model") == "SEIR":
        defs.insert(1, ("E", "Exposed"))

    for key, label in defs:
        trace = {
            "type": "scatter", "mode": "lines",
            "x": t, "y": result[key], "name": label,
            "line": {"color": PALETTE[key], "width": 2},
        }
        if key == "I":
            r, g, b = _hex_rgb(PALETTE[key])
            trace["fill"] = "tozeroy"
            trace["fillcolor"] = f"rgba({r},{g},{b},0.08)"
        traces.append(trace)

    layout = {
        "title": {"text": f"{result['model']} Model  |  R₀ = {result['R0']}",
                  "x": 0.01, "font": {"size": 13, "color": "#e2e8f0"}},
        "xaxis": {**LAYOUT_BASE["xaxis"], "title": "Days"},
        "yaxis": {**LAYOUT_BASE["yaxis"], "title": "Population"},
    }
    return _fig(traces, layout)


def daily_cases_chart(result: dict) -> str:
    t = result["t"]
    daily = result["daily_new_cases"]
    avg7 = _rolling_mean(daily, 7)

    traces = [
        {"type": "bar", "x": t, "y": daily, "name": "Daily New Cases",
         "marker": {"color": PALETTE["I"], "opacity": 0.75}},
        {"type": "scatter", "mode": "lines", "x": t, "y": avg7,
         "name": "7-Day Avg", "line": {"color": "#fbbf24", "width": 2, "dash": "dot"}},
    ]
    layout = {
        "title": {"text": "Daily New Infections", "x": 0.01, "font": {"size": 13, "color": "#e2e8f0"}},
        "xaxis": {**LAYOUT_BASE["xaxis"], "title": "Days"},
        "yaxis": {**LAYOUT_BASE["yaxis"], "title": "New Cases"},
        "barmode": "overlay",
    }
    return _fig(traces, layout)


def death_rate_chart(result: dict) -> str:
    t = result["t"]
    D = result["D"]
    daily_D = [max(0, D[i] - D[i-1]) if i > 0 else 0 for i in range(len(D))]

    traces = [
        {"type": "scatter", "mode": "lines", "x": t, "y": D,
         "name": "Cumulative Deaths", "yaxis": "y",
         "line": {"color": PALETTE["D"], "width": 2},
         "fill": "tozeroy", "fillcolor": "rgba(168,85,247,0.10)"},
        {"type": "bar", "x": t, "y": daily_D, "name": "Daily Deaths",
         "yaxis": "y2", "marker": {"color": "rgba(168,85,247,0.5)"}},
    ]
    layout = {
        "title": {"text": "Death Rate Over Time", "x": 0.01, "font": {"size": 13, "color": "#e2e8f0"}},
        "xaxis": {**LAYOUT_BASE["xaxis"], "title": "Days"},
        "yaxis":  {**LAYOUT_BASE["yaxis"], "title": "Cumulative Deaths"},
        "yaxis2": {"title": "Daily Deaths", "overlaying": "y", "side": "right",
                   "gridcolor": "transparent", "color": "#94a3b8"},
    }
    return _fig(traces, layout)


def recovery_trend_chart(result: dict) -> str:
    t = result["t"]
    R, I = result["R"], result["I"]
    rate = [r / (i + r + 1e-9) * 100 for r, i in zip(R, I)]

    traces = [
        {"type": "scatter", "mode": "lines", "x": t, "y": R,
         "name": "Total Recovered", "yaxis": "y",
         "line": {"color": PALETTE["R"], "width": 2},
         "fill": "tozeroy", "fillcolor": "rgba(34,197,94,0.10)"},
        {"type": "scatter", "mode": "lines", "x": t, "y": rate,
         "name": "Recovery Rate %", "yaxis": "y2",
         "line": {"color": "#34d399", "width": 1.5, "dash": "dash"}},
    ]
    layout = {
        "title": {"text": "Recovery Trends", "x": 0.01, "font": {"size": 13, "color": "#e2e8f0"}},
        "xaxis": {**LAYOUT_BASE["xaxis"], "title": "Days"},
        "yaxis":  {**LAYOUT_BASE["yaxis"], "title": "Recovered Population"},
        "yaxis2": {"title": "Recovery Rate (%)", "overlaying": "y", "side": "right",
                   "gridcolor": "transparent", "color": "#94a3b8"},
    }
    return _fig(traces, layout)


def comparison_chart(scenarios: dict) -> str:
    colors = ["#ef4444", "#f59e0b", "#22c55e", "#38bdf8", "#a855f7"]
    traces = []
    for idx, (name, result) in enumerate(scenarios.items()):
        traces.append({
            "type": "scatter", "mode": "lines",
            "x": result["t"], "y": result["I"],
            "name": name,
            "line": {"color": colors[idx % len(colors)], "width": 2.5},
        })
    layout = {
        "title": {"text": "Scenario Comparison – Active Infections",
                  "x": 0.01, "font": {"size": 13, "color": "#e2e8f0"}},
        "xaxis": {**LAYOUT_BASE["xaxis"], "title": "Days"},
        "yaxis": {**LAYOUT_BASE["yaxis"], "title": "Infected"},
    }
    return _fig(traces, layout)


def heatmap_chart(result: dict) -> str:
    infected = result["I"]
    days = len(infected)
    weeks = max(1, days // 7)
    grid = [infected[w*7:(w+1)*7] for w in range(weeks)]
    # Pad last row if needed
    for row in grid:
        while len(row) < 7:
            row.append(0)

    traces = [{
        "type": "heatmap",
        "z": grid,
        "x": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "y": [f"Wk {i+1}" for i in range(weeks)],
        "colorscale": "Reds",
        "colorbar": {"title": "Infected", "tickfont": {"color": "#94a3b8"},
                     "titlefont": {"color": "#94a3b8"}},
    }]
    layout = {
        "title": {"text": "Weekly Infection Intensity Heatmap",
                  "x": 0.01, "font": {"size": 13, "color": "#e2e8f0"}},
        "xaxis": {**LAYOUT_BASE["xaxis"]},
        "yaxis": {**LAYOUT_BASE["yaxis"], "autorange": "reversed"},
    }
    return _fig(traces, layout)


def prediction_chart(pred_result: dict) -> str:
    fv = pred_result["forecast_values"]
    fd = pred_result["forecast_days"]
    upper = [v * 1.15 for v in fv]
    lower = [max(0, v * 0.85) for v in fv]

    metrics = pred_result.get("metrics", {})
    model_label = pred_result.get("model_type", "ML").replace("_", " ").title()
    subtitle = f"{model_label}  |  MAE: {metrics.get('MAE','N/A')}  |  R²: {metrics.get('R2','N/A')}"

    traces = [
        {"type": "scatter", "mode": "lines",
         "x": pred_result["historical_days"], "y": pred_result["historical_actual"],
         "name": "Actual Infected", "line": {"color": PALETTE["I"], "width": 2}},
        {"type": "scatter", "mode": "lines",
         "x": pred_result["historical_days"], "y": pred_result["historical_predicted"],
         "name": "Model Fit", "line": {"color": "#fbbf24", "width": 1.5, "dash": "dot"}},
        {"type": "scatter", "mode": "lines",
         "x": fd + fd[::-1], "y": upper + lower[::-1],
         "fill": "toself", "fillcolor": "rgba(56,189,248,0.10)",
         "line": {"color": "rgba(0,0,0,0)"}, "name": "85% CI"},
        {"type": "scatter", "mode": "lines+markers",
         "x": fd, "y": fv, "name": "Forecast",
         "line": {"color": "#38bdf8", "width": 2, "dash": "dash"},
         "marker": {"size": 4}},
    ]
    layout = {
        "title": {"text": f"AI Infection Forecast  |  {subtitle}",
                  "x": 0.01, "font": {"size": 12, "color": "#e2e8f0"}},
        "xaxis": {**LAYOUT_BASE["xaxis"], "title": "Days"},
        "yaxis": {**LAYOUT_BASE["yaxis"], "title": "Infected"},
    }
    return _fig(traces, layout)


def _rolling_mean(data: list, window: int) -> list:
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result


def _hex_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
