from typing import Any

import pandas as pd


def select_champion(
    comparison_df: pd.DataFrame,
    searches: dict[str, Any],
) -> tuple[str, Any]:
    """Seleciona o campeão pela ordenação já calculada no notebook."""
    if comparison_df.empty:
        raise ValueError("A tabela de comparação está vazia.")

    best_model_name = str(comparison_df.iloc[0]["Model"])
    if best_model_name not in searches:
        raise KeyError(
            f"O modelo campeão '{best_model_name}' não possui uma busca."
        )

    return best_model_name, searches[best_model_name].best_estimator_
