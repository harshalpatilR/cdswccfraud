from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
import random
import pickle
from datetime import date
import os

cc_data = pd.read_pickle("resources/credit_card_dataframe_final.pkl",compression="gzip")

model = pickle.load(open("cc_model_day_3.pkl","rb"))

predictions = model.predict(cc_data.iloc[:,3:-1])
today = str(date.today())
filename = "resources/default_predictions"+today+".csv"
pd.DataFrame(predictions,columns=["predictions"]).to_csv(filename,index=False)
command = "hdfs dfs -put -f "+filename+" /user/harshal"
os.system(command)