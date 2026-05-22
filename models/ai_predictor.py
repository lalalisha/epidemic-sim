"""
ai_predictor.py
===============
AI/ML prediction module for forecasting future infection counts.

Uses two approaches:
  1. LinearRegression  – fast baseline trend model
  2. RandomForestRegressor – ensemble model capturing non-linearities

Feature engineering:
  - Day number
  - Rolling mean (7-day)
  - Lag features (t-1, t-3, t-7)
  - Polynomial features (degree 2) for Linear Regression
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


class EpidemicPredictor:
    """
    Trains and predicts infection counts using ML models.

    Parameters
    ----------
    model_type : str – 'linear' or 'random_forest'
    forecast_days : int – number of days to predict beyond training data
    """

    def __init__(self, model_type: str = "random_forest", forecast_days: int = 30):
        self.model_type = model_type
        self.forecast_days = forecast_days
        self.model = None
        self.metrics = {}

    def _build_features(self, infected: list) -> pd.DataFrame:
        """Create feature matrix from infection time series."""
        s = pd.Series(infected, name="infected")
        df = pd.DataFrame({"day": range(len(s)), "infected": s})

        # Rolling statistics
        df["roll_7"]  = s.rolling(7, min_periods=1).mean()
        df["roll_14"] = s.rolling(14, min_periods=1).mean()

        # Lag features
        df["lag_1"] = s.shift(1).fillna(0)
        df["lag_3"] = s.shift(3).fillna(0)
        df["lag_7"] = s.shift(7).fillna(0)

        # Rate of change
        df["delta_1"] = df["infected"].diff(1).fillna(0)
        df["delta_7"] = df["infected"].diff(7).fillna(0)

        return df

    def _build_model(self):
        """Instantiate the chosen sklearn model."""
        if self.model_type == "linear":
            return Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("reg",  LinearRegression()),
            ])
        else:
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )

    def fit_predict(self, infected: list) -> dict:
        """
        Train on historical data and generate future forecast.

        Parameters
        ----------
        infected : list – daily infected counts from simulation

        Returns
        -------
        dict with historical fit, forecast, and model metrics
        """
        if len(infected) < 14:
            return {"error": "Need at least 14 days of data for prediction."}

        df = self._build_features(infected)
        feature_cols = ["day", "roll_7", "roll_14", "lag_1", "lag_3", "lag_7",
                        "delta_1", "delta_7"]
        X = df[feature_cols].values
        y = df["infected"].values

        # Train/test split (80/20)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model = self._build_model()
        self.model.fit(X_train, y_train)

        # In-sample predictions
        y_pred_all = np.maximum(0, self.model.predict(X))

        # Test metrics
        y_pred_test = np.maximum(0, self.model.predict(X_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        r2  = r2_score(y_test, y_pred_test)
        self.metrics = {"MAE": round(mae, 2), "R2": round(r2, 4)}

        # Future forecast
        last_infected = list(infected)
        forecast = []
        last_day = len(infected) - 1

        for step in range(self.forecast_days):
            future_series = last_infected[:]
            feat_df = self._build_features(future_series)
            last_row = feat_df[feature_cols].iloc[-1].values.reshape(1, -1)
            pred = float(np.maximum(0, self.model.predict(last_row)[0]))
            forecast.append(pred)
            last_infected.append(pred)

        forecast_days_range = list(range(last_day + 1, last_day + self.forecast_days + 1))

        return {
            "historical_days":     list(range(len(infected))),
            "historical_actual":   [round(v) for v in infected],
            "historical_predicted":[round(v) for v in y_pred_all.tolist()],
            "forecast_days":       forecast_days_range,
            "forecast_values":     [round(v) for v in forecast],
            "metrics":             self.metrics,
            "model_type":          self.model_type,
            "forecast_horizon":    self.forecast_days,
        }
