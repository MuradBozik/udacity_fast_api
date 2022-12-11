import pytest
import pandas as pd
from salary_predictor.ml.model import load_model
from pathlib import Path
import os


root_folder = Path(os.path.abspath(__file__)).parent.parent


def pytest_addoption(parser):
    parser.addoption(
        "--data",
        action="store",
        default=f"{root_folder}/data/census_clean.csv")
    parser.addoption(
        "--model",
        action="store",
        default=f"{root_folder}/model/logreg.pkl")
    parser.addoption(
        "--encoder",
        action="store",
        default=f"{root_folder}/model/logreg_encoder.pkl")
    parser.addoption(
        "--lb",
        action="store",
        default=F"{root_folder}/model/logreg_lb.pkl")


@pytest.fixture(scope='session')
def data(request):
    data_path = request.config.option.data

    if data_path is None:
        pytest.fail("You must provide the --data option on the command line")

    return pd.read_csv(data_path)


@pytest.fixture(scope='session')
def model(request):
    model_path = request.config.option.model

    if model_path is None:
        pytest.fail("You must provide the --model option on the command line")

    return load_model(model_path)


@pytest.fixture(scope='session')
def encoder(request):
    encoder_path = request.config.option.encoder

    if encoder_path is None:
        pytest.fail("You must provide the --encoder option on the command line")

    return load_model(encoder_path)


@pytest.fixture(scope='session')
def lb(request):
    lb_path = request.config.option.lb

    if lb_path is None:
        pytest.fail("You must provide the --lb option on the command line")

    return load_model(lb_path)
