import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from salary_predictor.ml.data import process_data


def test_load_model(model):
    """Checks the type of model
    """
    assert isinstance(model, LogisticRegression)


def test_encoder(encoder):
    """Checks the type of encoder
    """
    assert isinstance(encoder, OneHotEncoder)


def test_lb(lb):
    """Checks the type of label binarizer
    """
    assert isinstance(lb, LabelBinarizer)


def test_inference(model, encoder, lb):
    """Checks the prediction type
    """

    X_sample_raw = pd.DataFrame.from_dict(
        {
            'age': {2290: 18},
            'workclass': {2290: 'Private'},
            'fnlgt': {2290: 201901},
            'education': {2290: 'HS-grad'},
            'education-num': {2290: 9},
            'marital-status': {2290: 'Never-married'},
            'occupation': {2290: 'Handlers-cleaners'},
            'relationship': {2290: 'Own-child'},
            'race': {2290: 'White'},
            'sex': {2290: 'Female'},
            'capital-gain': {2290: 0},
            'capital-loss': {2290: 1719},
            'hours-per-week': {2290: 15},
            'native-country': {2290: 'United-States'},
            'salary': {2290: '<=50K'}
        }
    )

    X_sample, y_sample, _, _ = process_data(
        X_sample_raw, label="salary", training=False,
        encoder=encoder, lb=lb
    )

    pred = model.predict(X_sample)

    assert isinstance(pred, np.ndarray), "Prediction has invalid type"
