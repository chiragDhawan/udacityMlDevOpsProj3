from fastapi.testclient import TestClient

from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


def test_api_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"welcome": "Welcome to the model app!! Happy inferencing"}

#39, State-gov,77516, Bachelors,13, Never-married, Adm-clerical, Not-in-family, White, Male,2174,0,40, United-States, <=50K
def test_inference_1():
    r = client.post("/infer/",
                    json={
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
},)

    assert r.status_code == 200
    assert r.json() == {"inference": "[0]"}


#52, Self-emp-inc,287927, HS-grad,9, Married-civ-spouse, Exec-managerial, Wife, White, Female,15024,0,40, United-States, >50K
def test_inference_2():
    r = client.post("/infer/",
                    json={
  "age": 50,
  "workclass": "Self-emp-not-inc",
  "fnlgt": 287927,
  "education": "HS-grad",
  "education-num": 9,
  "marital-status": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Wife",
  "race": "White",
  "sex": "Female",
  "capital-gain": 15024,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
},)

    assert r.status_code == 200
    assert r.json() == {"inference": "[1]"}

