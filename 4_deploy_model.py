import pickle
import numpy as np
import os

#record file time stamp
srctime = os.path.getmtime("cc_model_day_3.pkl")
model = pickle.load(open("cc_model_day_3.pkl","rb"))

def predict(args):
  global model
  
  #check if new pickel file available via Retraining Jobs
  dsttime = os.path.getmtime("cc_model_day_3.pkl")
  if dsttime > srctime:
    model = pickle.load(open("cc_model_day_3.pkl","rb"))
  
  account=np.array(args["feature"].split(",")).reshape(1,-1)
  return {"result" : model.predict(account)[0]}