"""
data_exporter.py
================
Handles CSV export of simulation results and scenario comparisons.
"""

import io
import csv
import pandas as pd


def simulation_to_csv(result: dict) -> str:
    """
    Convert a simulation result dict to CSV string.

    Parameters
    ----------
    result : dict – output from SIRModel.simulate() or SEIRModel.simulate()

    Returns
    -------
    str – CSV content as a string
    """
    days = len(result["t"])
    rows = []
    for i in range(days):
        row = {
            "Day":          round(result["t"][i]),
            "Susceptible":  round(result["S"][i]),
            "Infected":     round(result["I"][i]),
            "Recovered":    round(result["R"][i]),
            "Deaths":       round(result["D"][i]),
            "Daily_New":    round(result["daily_new_cases"][i]),
        }
        if result.get("model") == "SEIR":
            row["Exposed"] = round(result["E"][i])
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def comparison_to_csv(scenarios: dict) -> str:
    """
    Convert multi-scenario comparison data to a single CSV.

    Parameters
    ----------
    scenarios : dict – mapping of scenario_name -> simulation result dict

    Returns
    -------
    str – CSV content
    """
    frames = []
    for name, result in scenarios.items():
        days = len(result["t"])
        df = pd.DataFrame({
            "Scenario":  [name] * days,
            "Day":       [round(result["t"][i]) for i in range(days)],
            "Infected":  [round(result["I"][i]) for i in range(days)],
            "Recovered": [round(result["R"][i]) for i in range(days)],
            "Deaths":    [round(result["D"][i]) for i in range(days)],
            "Daily_New": [round(result["daily_new_cases"][i]) for i in range(days)],
        })
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined.to_csv(index=False)


def generate_sample_dataset() -> str:
    """
    Generate a synthetic sample dataset resembling historical epidemic data.
    Follows an approximate SIR curve for demonstration purposes.

    Returns
    -------
    str – CSV content
    """
    import numpy as np

    N = 100000
    days = 180
    t = np.arange(days)

    # Approximate SIR with noise
    peak = days * 0.35
    I_base = N * 0.15 * np.exp(-0.5 * ((t - peak) / 20) ** 2)
    noise = np.random.RandomState(42).normal(0, 500, days)
    I = np.maximum(0, I_base + noise).astype(int)

    cumulative = np.cumsum(I)
    R = np.minimum(cumulative * 0.9, N).astype(int)
    D = (cumulative * 0.02).astype(int)
    S = np.maximum(N - cumulative, 0).astype(int)

    df = pd.DataFrame({
        "Day":          t,
        "Susceptible":  S,
        "Infected":     I,
        "Recovered":    R,
        "Deaths":       D,
        "Daily_New":    I,
        "Region":       ["Sample Region"] * days,
        "Population":   [N] * days,
    })
    return df.to_csv(index=False)
