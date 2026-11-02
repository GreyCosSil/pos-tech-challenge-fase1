import pandas as pd
import pytest

from src import data

pytestmark = pytest.mark.unit


def test_load_local_dataset_without_network(tmp_path, monkeypatch):
    data_path = tmp_path / data.DATASET_FILENAME
    data_path.write_text("customerID,Churn\nlocal-customer,Yes\n")

    def reject_download(*args, **kwargs):
        pytest.fail("An existing dataset must not trigger a download")

    monkeypatch.setattr(data.kagglehub, "dataset_download", reject_download)

    dataframe = data.load_dataset(data_path)

    assert dataframe.to_dict("records") == [{"customerID": "local-customer", "Churn": "Yes"}]


def test_download_missing_dataset(tmp_path, monkeypatch):
    download_path = tmp_path / "download"
    download_path.mkdir()
    (download_path / data.DATASET_FILENAME).write_text("customerID,Churn\nnew-customer,No\n")
    requested_handles = []

    def download_dataset(handle):
        requested_handles.append(handle)
        return str(download_path)

    monkeypatch.setattr(data.kagglehub, "dataset_download", download_dataset)
    data_path = tmp_path / "data" / data.DATASET_FILENAME

    dataframe = data.load_dataset(data_path, allow_download=True)

    assert requested_handles == [data.DATASET_HANDLE]
    assert data_path.is_file()
    assert dataframe.to_dict("records") == [{"customerID": "new-customer", "Churn": "No"}]


def test_split_features_target():
    dataframe = pd.DataFrame(
        {"customerID": ["customer-1"], "Churn": ["Yes"], "tenure": [2]}
    )

    features, target, feature_columns = data.split_features_target(dataframe)

    assert features.to_dict("records") == [{"tenure": 2}]
    assert target.tolist() == [1]
    assert feature_columns == ["tenure"]
