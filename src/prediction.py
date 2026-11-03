from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.preprocessing import prepare_features


@dataclass(frozen=True)
class PredictionResult:
    """Resultado de uma predição binária de churn."""

    prediction: int
    probability_no: float
    probability_yes: float

    @property
    def churn(self) -> bool:
        return self.prediction == 1


def predict_customer(
    model: Any,
    customer: dict[str, Any],
) -> PredictionResult:
    """Prepara os dados e executa a predição do pipeline treinado."""
    features = prepare_features(pd.DataFrame([customer]))
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    return PredictionResult(
        prediction=prediction,
        probability_no=float(probabilities[0]),
        probability_yes=float(probabilities[1]),
    )
