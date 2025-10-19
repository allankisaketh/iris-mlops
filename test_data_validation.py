import pandas as pd

def test_data_not_empty():
    df = pd.read_csv("data/raw/iris.csv")
    assert not df.empty, "Dataset is empty"

def test_no_missing_values():
    df = pd.read_csv("data/raw/iris.csv")
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values"