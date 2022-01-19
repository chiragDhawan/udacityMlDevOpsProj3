from fastapi.testclient import TestClient

from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


def test_api_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"welcome": "Welcome to the model app!! Happy inferencing"}

def test_inference():
    r = client.post("/infer/",
                    json={"39", "State-gov,77516", "Bachelors","13", "Never-married",
                          "Adm-clerical", "Not-in-family", "White", "Male",
                          "2174", "0" , "40", "United-States"},
                    )
    assert r.status_code == 200
