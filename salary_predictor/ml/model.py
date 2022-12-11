from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import pickle
import os
from pathlib import Path


def train_model(X_train, y_train, config):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = LogisticRegression(**config)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model, encoder, lb, name):
    folder = Path(os.path.abspath(__file__)).parent

    with open(f"{folder}/../../model/{name}.pkl", "wb") as fp:
        pickle.dump(model, fp)
    with open(f"{folder}/../../model/{name}_encoder.pkl", "wb") as ep:
        pickle.dump(encoder, ep)
    with open(f"{folder}/../../model/{name}_lb.pkl", "wb") as lp:
        pickle.dump(lb, lp)


def load_model(path):
    with open(path, "rb") as fp:
        model = pickle.load(fp)
    return model
