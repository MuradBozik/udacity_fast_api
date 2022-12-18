import requests
import json

app_url = "https://murad-salary-predictor-app.herokuapp.com/predict_salary"

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

if __name__ == "__main__":
    
    response = requests.post(app_url, data=json.dumps(sample_input))
    
    print(response.status_code)
    print(response.json())