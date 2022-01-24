# udacityMlDevOpsProj3

# Git hub link
https://github.com/chiragDhawan/udacityMlDevOpsProj3

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
For the liveApiTesting call python test/test_api.py
for slice call python test/slice_check.py

# Github Actions 
Github actions have been created for the repo
It installs dependencies
Configures aws credentials
Runs dvc pull
Installs dependencies
Runs flake8
Runs pytest