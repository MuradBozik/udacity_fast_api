import json
from fastapi.testclient import TestClient
from main import app
import sys
sys.path.append("..")

client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to salary predictor API!"}


def test_post_1():
    sample_input = {'age': 53,
                    'workclass': 'Local-gov',
                    'fnlgt': 248834,
                    'education': 'Bachelors',
                    'education-num': 13,
                    'marital-status': 'Married-civ-spouse',
                    'occupation': 'Prof-specialty',
                    'relationship': 'Wife',
                    'race': 'White',
                    'sex': 'Female',
                    'capital-gain': 0,
                    'capital-loss': 0,
                    'hours-per-week': 50,
                    'native-country': 'United-States',
                    }
    r = client.post('/predict_salary', data=json.dumps(sample_input))
    assert r.status_code == 200
    assert r.json() == {"prediction": "0"}


def test_post_2():
    sample_input = {'age': 37,
                    'workclass': 'Self-emp-not-inc',
                    'fnlgt': 241306,
                    'education': 'HS-grad',
                    'education-num': 9,
                    'marital-status': 'Divorced',
                    'occupation': 'Craft-repair',
                    'relationship': 'Not-in-family',
                    'race': 'White',
                    'sex': 'Male',
                    'capital-gain': 0,
                    'capital-loss': 0,
                    'hours-per-week': 40,
                    'native-country': 'United-States',
                    }

    r = client.post('/predict_salary', data=json.dumps(sample_input))
    assert r.status_code == 200
    assert r.json() == {"prediction": "0"}


def test_post_type():
    sample_input = {'age': 37,
                    'workclass': 'Self-emp-not-inc',
                    'fnlgt': 241306,
                    'education': 'HS-grad',
                    'education-num': 9,
                    'marital-status': 'Divorced',
                    'occupation': 'Craft-repair',
                    'relationship': 'Not-in-family',
                    # 'race': 'White',
                    # 'sex': 'Male',
                    'capital-gain': 0,
                    'capital-loss': 0,
                    'hours-per-week': 40,
                    'native-country': 'United-States',
                    }

    r = client.post('/predict_salary', data=json.dumps(sample_input))

    assert r.status_code == 422
    assert r.json()['detail'][0]['type'] == 'value_error.missing'
