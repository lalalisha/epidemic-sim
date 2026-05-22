"""
generate_sample_data.py
=======================
Standalone script to generate sample_epidemic_dataset.csv
Run: python generate_sample_data.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 100_000
days = 180
t = np.arange(days)

peak = int(days * 0.35)
I_base = N * 0.15 * np.exp(-0.5 * ((t - peak) / 20) ** 2)
noise = np.random.normal(0, 400, days)
I = np.maximum(0, I_base + noise).astype(int)

cumulative = np.cumsum(I)
R = np.minimum(cumulative * 0.88, N).astype(int)
D = (cumulative * 0.018).astype(int)
S = np.maximum(N - cumulative, 0).astype(int)
E = np.maximum(0, (I * 0.6).astype(int))

df = pd.DataFrame({
    "Day":         t,
    "Date":        pd.date_range("2024-01-01", periods=days).strftime("%Y-%m-%d"),
    "Susceptible": S,
    "Exposed":     E,
    "Infected":    I,
    "Recovered":   R,
    "Deaths":      D,
    "Daily_New":   I,
    "Region":      ["Sample Region"] * days,
    "Population":  [N] * days,
})

df.to_csv("data/sample_epidemic_dataset.csv", index=False)
print(f"✓ Generated data/sample_epidemic_dataset.csv  ({days} rows)")
