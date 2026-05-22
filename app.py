"""
app.py
======
Flask application entry point for the Epidemic Disease Spread Modeling System.

Routes:
  GET  /                         – main dashboard
  POST /api/simulate             – run SIR/SEIR simulation
  POST /api/predict              – AI infection forecast
  POST /api/compare              – multi-scenario comparison
  GET  /api/export/csv           – export last simulation as CSV
  GET  /api/export/comparison    – export comparison CSV
  GET  /api/sample-dataset       – download sample dataset CSV
  GET  /api/health               – health check
"""

import json
import io
from flask import Flask, render_template, request, jsonify, send_file, Response
try:
    from flask_cors import CORS
except ImportError:
    def CORS(app, **kw): pass  # flask-cors not installed

from models import SIRModel, SEIRModel, EpidemicPredictor
from utils import (
    compartment_chart, daily_cases_chart, death_rate_chart,
    recovery_trend_chart, comparison_chart, heatmap_chart, prediction_chart,
    simulation_to_csv, comparison_to_csv, generate_sample_dataset,
)

app = Flask(__name__)
CORS(app)

# In-memory store for last simulation results (single-user dev mode)
_state = {
    "last_simulation": None,
    "last_comparison": None,
}


# ──────────────────────────────────────────────────────────────
# HTML
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ──────────────────────────────────────────────────────────────
# Simulation API
# ──────────────────────────────────────────────────────────────

@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    Run SIR or SEIR simulation with provided parameters.

    JSON Body:
      model       : "SIR" | "SEIR"
      N           : int    (population)
      beta        : float  (transmission rate)
      gamma       : float  (recovery rate)
      sigma       : float  (incubation rate, SEIR only)
      nu          : float  (vaccination rate)
      mu          : float  (mortality rate)
      delta       : float  (social distancing [0,1])
      days        : int    (simulation duration)
      I0          : int    (initial infected)
      E0          : int    (initial exposed, SEIR only)
    """
    try:
        data = request.get_json(force=True)
        model_type = data.get("model", "SIR").upper()
        N     = int(data.get("N",     1_000_000))
        beta  = float(data.get("beta",  0.3))
        gamma = float(data.get("gamma", 0.05))
        sigma = float(data.get("sigma", 0.2))
        nu    = float(data.get("nu",    0.0))
        mu    = float(data.get("mu",    0.001))
        delta = float(data.get("delta", 0.0))
        days  = int(data.get("days",   365))
        I0    = int(data.get("I0",     10))
        E0    = int(data.get("E0",     0))

        # Clamp values to reasonable ranges
        beta  = max(0.001, min(beta,  5.0))
        gamma = max(0.001, min(gamma, 1.0))
        sigma = max(0.001, min(sigma, 1.0))
        nu    = max(0.0,   min(nu,    0.1))
        mu    = max(0.0,   min(mu,    0.5))
        delta = max(0.0,   min(delta, 0.99))
        days  = max(30,    min(days,  730))
        I0    = max(1,     min(I0,    N // 100))

        if model_type == "SEIR":
            model = SEIRModel(N=N, beta=beta, sigma=sigma, gamma=gamma,
                              nu=nu, mu=mu, delta=delta)
        else:
            model = SIRModel(N=N, beta=beta, gamma=gamma,
                             nu=nu, mu=mu, delta=delta)

        result = model.simulate(days=days, I0=I0, E0=E0)
        _state["last_simulation"] = result

        # Generate all charts
        charts = {
            "compartment":  compartment_chart(result),
            "daily_cases":  daily_cases_chart(result),
            "death_rate":   death_rate_chart(result),
            "recovery":     recovery_trend_chart(result),
            "heatmap":      heatmap_chart(result),
        }

        return jsonify({
            "success": True,
            "result":  result,
            "charts":  charts,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ──────────────────────────────────────────────────────────────
# AI Prediction API
# ──────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Generate AI forecast for infection counts.

    JSON Body:
      infected      : list[float]  (historical infected counts)
      model_type    : "linear" | "random_forest"
      forecast_days : int
    """
    try:
        data = request.get_json(force=True)
        infected      = data.get("infected", [])
        model_type    = data.get("model_type", "random_forest")
        forecast_days = int(data.get("forecast_days", 30))

        if not infected:
            # Use last simulation if no data provided
            if _state["last_simulation"]:
                infected = _state["last_simulation"]["I"]
            else:
                return jsonify({"success": False, "error": "No data provided"}), 400

        predictor = EpidemicPredictor(model_type=model_type, forecast_days=forecast_days)
        pred_result = predictor.fit_predict(infected)

        if "error" in pred_result:
            return jsonify({"success": False, "error": pred_result["error"]}), 400

        chart = prediction_chart(pred_result)
        return jsonify({
            "success":  True,
            "result":   pred_result,
            "chart":    chart,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ──────────────────────────────────────────────────────────────
# Scenario Comparison API
# ──────────────────────────────────────────────────────────────

@app.route("/api/compare", methods=["POST"])
def compare():
    """
    Run three predefined scenarios and return comparison charts.

    JSON Body:
      N     : int
      beta  : float
      gamma : float
      days  : int
      I0    : int
      model : "SIR" | "SEIR"
    """
    try:
        data = request.get_json(force=True)
        N     = int(data.get("N",     1_000_000))
        beta  = float(data.get("beta",  0.3))
        gamma = float(data.get("gamma", 0.05))
        sigma = float(data.get("sigma", 0.2))
        mu    = float(data.get("mu",    0.001))
        days  = int(data.get("days",   365))
        I0    = int(data.get("I0",     10))
        model_type = data.get("model", "SIR").upper()

        def make_model(nu=0.0, delta=0.0):
            if model_type == "SEIR":
                return SEIRModel(N=N, beta=beta, sigma=sigma, gamma=gamma,
                                 nu=nu, mu=mu, delta=delta)
            return SIRModel(N=N, beta=beta, gamma=gamma, nu=nu, mu=mu, delta=delta)

        scenarios = {
            "No Intervention":  make_model(nu=0.0,  delta=0.0).simulate(days, I0),
            "Lockdown (50%)":   make_model(nu=0.0,  delta=0.50).simulate(days, I0),
            "Lockdown (75%)":   make_model(nu=0.0,  delta=0.75).simulate(days, I0),
            "Vaccination":      make_model(nu=0.003, delta=0.0).simulate(days, I0),
            "Combined":         make_model(nu=0.003, delta=0.50).simulate(days, I0),
        }

        _state["last_comparison"] = scenarios

        comp_chart = comparison_chart(scenarios)

        # Summary table
        summary = []
        for name, res in scenarios.items():
            summary.append({
                "scenario":        name,
                "peak_infected":   res["peak_infected"],
                "peak_day":        res["peak_day"],
                "total_deaths":    res["total_deaths"],
                "total_recovered": res["total_recovered"],
                "R0":              res["R0"],
            })

        return jsonify({
            "success":  True,
            "chart":    comp_chart,
            "summary":  summary,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ──────────────────────────────────────────────────────────────
# Export APIs
# ──────────────────────────────────────────────────────────────

@app.route("/api/export/csv")
def export_csv():
    """Export the last simulation run as a CSV file."""
    if not _state["last_simulation"]:
        return jsonify({"error": "No simulation data. Run a simulation first."}), 400
    csv_data = simulation_to_csv(_state["last_simulation"])
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=simulation_results.csv"},
    )


@app.route("/api/export/comparison")
def export_comparison():
    """Export the last comparison run as a CSV file."""
    if not _state["last_comparison"]:
        return jsonify({"error": "No comparison data. Run a comparison first."}), 400
    csv_data = comparison_to_csv(_state["last_comparison"])
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=scenario_comparison.csv"},
    )


@app.route("/api/sample-dataset")
def sample_dataset():
    """Download the generated sample dataset."""
    csv_data = generate_sample_dataset()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_epidemic_dataset.csv"},
    )


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Epidemic Simulation System  –  http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
