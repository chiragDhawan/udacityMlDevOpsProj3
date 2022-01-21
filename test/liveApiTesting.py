import requests
import json

response = requests.post()

data={
  "age": 39,
  "workclass": "State-gov",
  "fnlgt": 77516,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital-gain": 2174,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}

r = requests.post('https://udacitydevopsmlapp.herokuapp.com/'
                  , auth=('usr', 'pass'), data=json.dumps(data))

assert r.status_code == 200
assert r.json() == {"inference": "[0]"}