import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def build_estimators(random_state: int) -> dict[str, any]:
    """Retorna os quatro modelos avaliados."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=5000, random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(random_state=random_state, n_jobs=1),
        "MLPClassifier": MLPClassifier(
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=random_state,
        ),
    }


def build_param_grids() -> dict[str, dict[str, list]]:
    """Retorna os grids utilizados no GridSearchCV."""
    class_weights = [None, "balanced", {0: 1, 1: 1.5}, {0: 1, 1: 2.0}]
    return {
        "LogisticRegression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__class_weight": class_weights,
        },
        "DecisionTree": {
                "model__max_depth": [4, 8, 16, 32, None],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 5, 10],
                "model__class_weight": class_weights,
            },
        "RandomForest": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [4, 8, 16, 32, None],
                "model__min_samples_leaf": [2, 4, 8, 16],
                "model__max_features": ["sqrt", "log2", 0.5],
                "model__class_weight": class_weights,
            },
        "MLPClassifier": {
            "model__hidden_layer_sizes": [
                (64, 32),
                (64, 64),
                (128, 64),
                (64, 64, 32),
                (128, 64, 32),
            ],
            "model__alpha": [0.001, 0.01],
            "model__learning_rate_init": [0.001, 0.005],
            },
    }

def train_and_evaluate_models(
    models: dict,
    param_grids: dict,
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv=None,
    scoring: dict = None,
    random_state: int = 42
) -> tuple[dict, pd.DataFrame]:
    """
    Executa o GridSearchCV para múltiplos modelos usando um pipeline com pré-processamento
    e retorna um dicionário com as buscas ajustadas e um DataFrame consolidado de métricas.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        
    if scoring is None:
        scoring = {
            "recall": "recall",
            "average_precision": "average_precision",
            "precision": "precision",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
        }

    searches = {}
    results = []

    for model_name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[model_name],
            scoring=scoring,
            refit="recall",
            cv=cv,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        
        search.fit(X_train, y_train)
        searches[model_name] = search
        
        best_index = search.best_index_
        result = {"Model": model_name, "BestParams": search.best_params_}
        
        for metric in scoring:
            result[f"{metric}_mean"] = float(
                search.cv_results_[f"mean_test_{metric}"][best_index]
            )
            result[f"{metric}_std"] = float(
                search.cv_results_[f"std_test_{metric}"][best_index]
            )
            
        results.append(result)
        print(
            f"{model_name}: Recall = {result['recall_mean']:.4f}; "
            f"PR-AUC = {result['average_precision_mean']:.4f}; "
            f"parâmetros = {search.best_params_}"
        )

    comparison_df = (
        pd.DataFrame(results)
        .sort_values(
            ["recall_mean", "average_precision_mean"],
            ascending=[False, False],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return searches, comparison_df