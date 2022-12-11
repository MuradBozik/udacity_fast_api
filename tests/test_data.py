"""Tests regarding the dataset
"""


def test_missing_data(data):
    """Checks if there are any missing data in dataset
    """
    assert not any(data.isnull().sum())


def test_column_names(data):
    """Checks if columns starts with trailing spaces
    """
    assert not any(data.columns.str.startswith(" "))


def test_categorical_data(data):
    """Checks whether categorical data starts with trailing spaces
    """
    df = data.select_dtypes("object")
    unprocessed_columns = df.columns[df.apply(
        lambda x: any(x.str.startswith(" ")), axis=0)].to_list()
    assert len(
        unprocessed_columns) == 0, f"Following column(s) includes messy data {unprocessed_columns}"
