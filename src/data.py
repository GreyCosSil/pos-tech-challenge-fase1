from pathlib import Path

import kagglehub
import pandas as pd

from src.config import DATASET_FILENAME, DATASET_PATH


DATASET_HANDLE = "blastchar/telco-customer-churn"


def download_dataset() -> Path:
    """Baixa o dataset do Kaggle e retorna o caminho do CSV baixado."""

    download_directory = Path(
        kagglehub.dataset_download(DATASET_HANDLE)
    )

    downloaded_path = download_directory / DATASET_FILENAME

    if not downloaded_path.is_file():
        raise FileNotFoundError(
            f"O arquivo esperado não foi encontrado no download: "
            f"{downloaded_path}"
        )

    return downloaded_path


def load_dataset(
    path: Path = DATASET_PATH,
    allow_download: bool = False,
) -> pd.DataFrame:
    """Carrega o dataset localmente, com download opcional."""

    if not path.is_file():
        if not allow_download:
            raise FileNotFoundError(
                f"Dataset não encontrado em: {path}. "
                "Baixe o arquivo ou habilite allow_download."
            )

        downloaded_path = download_dataset()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(downloaded_path.read_bytes())

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"O dataset está vazio: {path}")

    return dataframe

def validate_dataset(df: pd.DataFrame) -> None:
    """Valida colunas obrigatórias, target e estrutura mínima."""


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Separa as features da variável Churn e retorna as colunas."""
    feature_columns = [
        column for column in df.columns
        if column not in {"customerID", "Churn"}
    ]

    X = df[feature_columns].copy()

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1,
    })

    return X, y, feature_columns  # Retorna também a lista