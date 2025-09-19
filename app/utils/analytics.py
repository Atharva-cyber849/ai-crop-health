from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


@dataclass
class SoilAssessment:
    moisture_status: str
    temp_status: str
    humidity_status: str
    overall: str


def soil_quality_rules(row: pd.Series) -> SoilAssessment:
    """Simple rule-based soil quality assessment.
    - moisture < 0.20 (or 20 if in %) => Low
    - temp outside 18..35C => Risky
    - humidity < 30% => Dry risk
    """
    # Normalize moisture to 0..1 if it looks like percent in 0..100
    moisture = row.get("moisture", np.nan)
    temp = row.get("temp", np.nan)
    humidity = row.get("humidity", np.nan)

    if np.isnan(moisture) or np.isnan(temp) or np.isnan(humidity):
        return SoilAssessment("Unknown", "Unknown", "Unknown", "Unknown")

    if moisture > 1.0:
        moisture = moisture / 100.0

    moisture_status = "Good" if moisture >= 0.20 else "Low"
    temp_status = "Good" if (18 <= temp <= 35) else "Risky"
    humidity_status = "Good" if humidity >= 30 else "Low"

    # Overall simple aggregation
    risky = any(s in ("Low", "Risky") for s in [moisture_status, temp_status, humidity_status])
    overall = "Good" if not risky else "Risky"

    return SoilAssessment(moisture_status, temp_status, humidity_status, overall)


class PestRiskModel:
    """Tiny DecisionTree-based classifier using features like NDVI mean, humidity, temperature.
    Used for demo with small/fake samples.
    """

    def __init__(self, random_state: int = 42):
        self.model = DecisionTreeClassifier(max_depth=3, random_state=random_state)
        self.is_trained = False

    def make_demo_dataset(self, n: int = 200, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(random_state)
        ndvi_mean = rng.uniform(low=-0.2, high=0.9, size=n)
        humidity = rng.uniform(low=20, high=95, size=n)
        temp = rng.uniform(low=10, high=45, size=n)

        # Heuristic label: high risk when NDVI low + humidity high + temp warm
        risk_score = (
            (ndvi_mean < 0.3).astype(int)
            + (humidity > 70).astype(int)
            + ((temp > 20) & (temp < 35)).astype(int)
        )
        y = (risk_score >= 2).astype(int)  # 1 = High risk, 0 = Low

        X = pd.DataFrame({"ndvi_mean": ndvi_mean, "humidity": humidity, "temp": temp})
        return X, pd.Series(y, name="risk")

    def train(self, X: pd.DataFrame | None = None, y: pd.Series | None = None) -> dict:
        if X is None or y is None:
            X, y = self.make_demo_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        return report

    def predict_risk(self, ndvi_mean: float, humidity: float, temp: float) -> tuple[str, float]:
        if not self.is_trained:
            self.train()
        X = pd.DataFrame({"ndvi_mean": [ndvi_mean], "humidity": [humidity], "temp": [temp]})
        proba = self.model.predict_proba(X)[0][1]  # probability of class 1 (High risk)
        label = "High" if proba >= 0.5 else "Low"
        return label, float(proba)
