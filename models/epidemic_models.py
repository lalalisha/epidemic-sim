"""
epidemic_models.py
==================
Core mathematical models for epidemic disease spread simulation.

SIR Model ODEs:
  dS/dt = -β·S·I/N  - ν·S
  dI/dt =  β·S·I/N  - γ·I - μ·I
  dR/dt =  γ·I      + ν·S
  dD/dt =  μ·I

SEIR Model ODEs:
  dS/dt = -β·S·I/N  - ν·S
  dE/dt =  β·S·I/N  - σ·E
  dI/dt =  σ·E      - γ·I - μ·I
  dR/dt =  γ·I      + ν·S
  dD/dt =  μ·I

Parameters:
  N  – total population
  β  – transmission rate         (contacts × prob_transmission per day)
  γ  – recovery rate             (1/γ = mean infectious period in days)
  σ  – incubation rate  [SEIR]  (1/σ = mean incubation period in days)
  ν  – vaccination rate          (fraction of S vaccinated per day)
  μ  – mortality rate            (fraction of I that die per day)
  δ  – social distancing factor  ([0,1]; β_eff = β·(1-δ))
"""

import numpy as np
from scipy.integrate import odeint


class SIRModel:
    """Susceptible → Infected → Recovered compartmental model."""

    def __init__(self, N: int, beta: float, gamma: float,
                 nu: float = 0.0, mu: float = 0.0, delta: float = 0.0):
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.nu = nu
        self.mu = mu
        self.delta = delta

    @property
    def R0(self) -> float:
        """Basic reproduction number R₀ = β_eff / γ."""
        return self.beta * (1 - self.delta) / self.gamma

    def _deriv(self, y, t):
        S, I, R, D = y
        N = self.N
        be = self.beta * (1 - self.delta)
        dSdt = -be * S * I / N - self.nu * S
        dIdt =  be * S * I / N - self.gamma * I - self.mu * I
        dRdt =  self.gamma * I + self.nu * S
        dDdt =  self.mu * I
        return dSdt, dIdt, dRdt, dDdt

    def simulate(self, days: int, I0: int = 1, E0: int = 0) -> dict:
        S0 = max(self.N - I0, 0)
        y0 = (S0, I0, 0, 0)
        t = np.linspace(0, days, days + 1)
        sol = odeint(self._deriv, y0, t)
        S, I, R, D = sol.T

        vax = self.nu * S
        raw = np.diff(np.concatenate([[S0], S])) * -1
        daily_new = np.maximum(0, np.concatenate([[0], raw[:-0 or None] - vax[:len(raw)]]))
        if len(daily_new) < len(t):
            daily_new = np.concatenate([daily_new, [0]])

        return dict(
            t=t.tolist(), S=S.tolist(), I=I.tolist(), R=R.tolist(),
            D=D.tolist(), E=[0.0]*len(t),
            daily_new_cases=daily_new[:len(t)].tolist(),
            R0=round(self.R0, 4), model="SIR",
            peak_infected=round(float(np.max(I))),
            peak_day=int(np.argmax(I)),
            total_recovered=round(float(R[-1])),
            total_deaths=round(float(D[-1])),
        )


class SEIRModel:
    """Susceptible → Exposed → Infected → Recovered compartmental model."""

    def __init__(self, N: int, beta: float, sigma: float, gamma: float,
                 nu: float = 0.0, mu: float = 0.0, delta: float = 0.0):
        self.N = N
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.nu = nu
        self.mu = mu
        self.delta = delta

    @property
    def R0(self) -> float:
        """R₀ = β_eff / γ  (approximate for SEIR)."""
        return self.beta * (1 - self.delta) / self.gamma

    def _deriv(self, y, t):
        S, E, I, R, D = y
        N = self.N
        be = self.beta * (1 - self.delta)
        dSdt = -be * S * I / N - self.nu * S
        dEdt =  be * S * I / N - self.sigma * E
        dIdt =  self.sigma * E - self.gamma * I - self.mu * I
        dRdt =  self.gamma * I + self.nu * S
        dDdt =  self.mu * I
        return dSdt, dEdt, dIdt, dRdt, dDdt

    def simulate(self, days: int, I0: int = 1, E0: int = 0) -> dict:
        S0 = max(self.N - I0 - E0, 0)
        y0 = (S0, E0, I0, 0, 0)
        t = np.linspace(0, days, days + 1)
        sol = odeint(self._deriv, y0, t)
        S, E, I, R, D = sol.T

        vax = self.nu * S
        raw = np.diff(np.concatenate([[S0], S])) * -1
        daily_new = np.maximum(0, np.concatenate([[0], raw - vax[:len(raw)]]))
        if len(daily_new) < len(t):
            daily_new = np.concatenate([daily_new, [0]])

        return dict(
            t=t.tolist(), S=S.tolist(), E=E.tolist(), I=I.tolist(),
            R=R.tolist(), D=D.tolist(),
            daily_new_cases=daily_new[:len(t)].tolist(),
            R0=round(self.R0, 4), model="SEIR",
            peak_infected=round(float(np.max(I))),
            peak_day=int(np.argmax(I)),
            total_recovered=round(float(R[-1])),
            total_deaths=round(float(D[-1])),
        )
