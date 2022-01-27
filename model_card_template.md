# Model Card

## Model Details
Chirag Dhawan created the model. 
It is random forest classifier model using the default hyperparameters in scikit-learn 0.24.2.

## Intended Use
This model should be used to predict the salary whether >50K or less of a person based off a handful of attributes. 
The users could be social/financial analyst.

## Training Data
The data was taken from https://archive.ics.uci.edu/ml/datasets/census+income 

## Evaluation Data
The original data set has 32537 rows, and a 80-20 split was used to break this into a train and test set.  
To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Metrics
precision_train 0.8218670372944175
 recall_train 0.566411238825032
 fbeta_train 0.6706360457423685

precision_test 0.8170377541142304
 recall_test 0.5358730158730158
 fbeta_test 0.6472392638036809


## Ethical Considerations
There is both supervised and unsupervised bias. 
For native-country fpr is 1 Yugoslavia and 0.03 for United-States
For the full list of bias go to ethics_biases folder and check bias_df.csv
For other bias and fairness result check out the ethics.ipynb in the ethics_biases folder

## Caveats and Recommendations
This models fitness may or may not lend it for use in production.
The outcome of the model and the subsequent decision should be taken keeping in mind the biases