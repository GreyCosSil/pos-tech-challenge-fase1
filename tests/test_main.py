import pandas as pd
import pytest

from src import dataset, main

pytestmark = pytest.mark.unit


def test_load_local_dataset_without_network(tmp_path, monkeypatch):
    data_path = tmp_path / dataset.DATASET_FILENAME
    data_path.write_text("customerID,Churn\nlocal-customer,Yes\n")

    def reject_download(*args, **kwargs):
        pytest.fail("An existing dataset must not trigger a download")

    monkeypatch.setattr(dataset.kagglehub, "dataset_download", reject_download)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dataset, "DEFAULT_DATA_PATH", data_path)

    dataframe = dataset.load_dataset()

    assert dataframe.to_dict("records") == [{"customerID": "local-customer", "Churn": "Yes"}]


def test_download_missing_dataset(tmp_path, monkeypatch):
    download_path = tmp_path / "download"
    download_path.mkdir()
    (download_path / dataset.DATASET_FILENAME).write_text("customerID,Churn\nnew-customer,No\n")
    requested_handles = []

    def download_dataset(handle):
        requested_handles.append(handle)
        return str(download_path)

    monkeypatch.setattr(dataset.kagglehub, "dataset_download", download_dataset)
    data_path = tmp_path / "data" / dataset.DATASET_FILENAME

    dataframe = dataset.load_dataset(data_path)

    assert requested_handles == [dataset.DATASET_HANDLE]
    assert data_path.is_file()
    assert dataframe.to_dict("records") == [{"customerID": "new-customer", "Churn": "No"}]


def test_main_displays_preview(monkeypatch, capsys):
    monkeypatch.setattr(main, "load_dataset", lambda: pd.DataFrame({"Churn": ["Yes"]}))

    main.main()

    output = capsys.readouterr().out
    assert "First 5 records:" in output
    assert "Yes" in output
