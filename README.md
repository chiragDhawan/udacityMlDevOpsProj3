# udacityMlDevOpsProj3

# Git hub link
https://github.com/chiragDhawan/udacityMlDevOpsProj3

# environment
conda env create -f environment.yml

# Train
If want to train the model then run python ml/train.py from the root folder

# EDA
The jupyter notebook is in the eda folder

# data
data is pushed and pull from aws s3 bucket and is versioned with dvc
use dvc pull to get data

# models
random forest model as well as label binarizer and encoder's pkl file is present in the 
models folder

# test
using pytest the data has been unit tested
It also contains test_api which contains local api testing
The test folder contains slice_check to verify slicing
liveApiTesting tests the live api
For the first 2 points call pytest -vv . from the root folder
For the liveApiTesting call python test/liveApiTesting.py 
for slice call python test/slice_check.py

# Github Actions 
Github actions have been created for the repo
It installs dependencies
Configures aws credentials
Runs dvc pull
Installs dependencies
Runs flake8
Runs pytest

# fast api 
Contains two endpoints
'/' Welcomes to the app -- get method
'/infer' predicts the output of the model -- post method
e.g input to the infer post

{
  "age": 50,
  "workclass": "Self-emp-not-inc",
  "fnlgt": 83311,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital-gain": 0,
  "capital-loss": 0,
  "hours-per-week": 13,
  "native-country": "United-States"
}