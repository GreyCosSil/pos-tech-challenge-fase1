import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def build_experiment_metadata(
    *,
    dataset_name: str,
    best_model_name: str,
    best_params: dict[str, Any],
    test_metrics: dict[str, float],
    feature_columns: list[str],
    random_state: int,
    test_size: float,
    cv_folds: int,
) -> dict[str, Any]:
    """Cria os metadados compatíveis com o experimento existente."""
    return {
        "dataset": dataset_name,
        "target": "Churn",
        "positive_class": "Yes",
        "random_state": random_state,
        "test_size": test_size,
        "cv_folds": cv_folds,
        "search_method": "GridSearchCV",
        "primary_selection_metric": "recall_mean",
        "secondary_selection_metric": "average_precision_mean",
        "selection_rule": "lexicographic_descending",
        "decision_threshold": "estimator_default_predict",
        "best_model": best_model_name,
        "best_params": best_params,
        "test_metrics": test_metrics,
        "feature_columns": feature_columns,
    }


def save_training_artifacts(
    *,
    model: Any,
    comparison_df: pd.DataFrame,
    metadata: dict[str, Any],
    model_path: Path,
    comparison_path: Path,
    metadata_path: Path,
) -> None:
    """Persiste o pipeline, a comparação e os metadados do experimento."""
    for path in (model_path, comparison_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    comparison_df.to_csv(comparison_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
