# Credit Card Fraud Detection

![MacDown logo](resources/creditcard.png)

## Introduction

This is a demo project to showcase the features of the Cloudera Data Science
Workbench.

The data used in this project is from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud/version/3)

“The datasets contains transactions made by credit cards in September 2013 by european cardholders. 
This dataset presents transactions that occurred in two days, where we have **492 frauds out of 284,807** transactions. 

The project consists of 5 parts:
* Part 1: Importing Data                  `1_create_data.py` 
* Part 2: Data Analysis and Visualization `2_data_analysis.py`
* Part 3: Model Training                  `3_train_modelpy`
* Part 4: Model Deployment                `4_deploy_model.py`
* Part 5: Model Tuning                    `5_check_model.py`

_Note: The data needs to be copied to the cluster first as per the file_ `setup.sh` _in order for this project to 
work properly_ 

## 1: Importing Data
The data is imported from the cluster and saved to a local dataframe.

## 2: Data Analysis and Visualization
Perform some basic data analysis and visualisation techniques to understand the data better.

## 3: Model Training
Train an ML Model to predict additional values.

## 4: Model Deployment
Deploy the trained model to CDSW to integrate with other internal systems.

## 5: Model Tuning
Adjust the model hyper parameter values using an experiment to optimise the model.

## 6: Model Monitoring via Jobs
Checks the current state of model performance via the Jobs interface

## 7: Model Monitoring via Experiments
Checks the current state of model performance via the Experiments interface

## 8: Model Retraining
Called to retrain the model if there the job in 6 finds the model is below threshold

_Note: For the model deployment, use the following JSON as the example input:_

```
{
  "feature": "-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62"
}
```