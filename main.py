"""FastAPI for making inference on Logistic Regression Model

Author  : Murad Bozik
Data    : 12DEC1638 2022
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from salary_predictor.ml.model import inference, load_model
from salary_predictor.ml.data import process_data
import os
import salary_predictor
from pathlib import Path


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    print("Running DVC")
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("Pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Load model, encoder, label_binarizer
model_folder = os.path.join(
    Path(salary_predictor.__file__).parent.parent, "model")
model = load_model(f"{model_folder}/logreg.pkl")
encoder = load_model(f"{model_folder}/logreg_encoder.pkl")
label_binarizer = load_model(f"{model_folder}/logreg_lb.pkl")


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {'age': 36,
                        'workclass': 'Federal-gov',
                        'fnlgt': 192443,
                        'education': 'Some-college',
                        'education-num': 10,
                        'marital-status': 'Never-married',
                        'occupation': 'Exec-managerial',
                        'relationship': 'Not-in-family',
                        'race': 'Black',
                        'sex': 'Male',
                        'capital-gain': 13550,
                        'capital-loss': 0,
                        'hours-per-week': 40,
                        'native-country': 'United-States'}
        }


app = FastAPI(
    title="salary_predictor API",
    description="An API that predicts the salary based on of the inputs.",
    version="0.0.1",
)


@app.get("/")
async def welcome():
    return {"message": "Welcome to salary predictor API!"}


@app.post('/predict_salary')
async def predict_salary(data: Data):
    try:
        data_dict = data.dict()
        X_raw = pd.DataFrame.from_dict({k: [v] for k, v in data_dict.items()})

        X_processed, _, _, _ = process_data(X=X_raw,
                                            training=False,
                                            encoder=encoder,
                                            lb=label_binarizer)

        prediction = inference(model=model, X=X_processed)[0]
        return {'prediction': f"{prediction}"}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="An Error occured during inference") from e
