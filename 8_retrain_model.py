from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd
import random
import pickle

cc_data = pd.read_pickle("resources/credit_card_dataframe_final.pkl",compression="gzip")

# # ML Training Data on new days

new_day = int((os.environ['DAY']))

X = cc_data[cc_data.Day < (new_day+1)].iloc[:,3:len(cc_data.columns)-1]
y = cc_data[cc_data.Day < (new_day+1)].iloc[:,len(cc_data.columns)-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Random Forest

param_numTrees = 20
param_maxDepth = 20 
param_impurity = 'gini' 

randF=RandomForestClassifier(n_jobs=10,
                             n_estimators=param_numTrees, 
                             max_depth=param_maxDepth, 
                             criterion = param_impurity,
                             random_state=0)

randF.fit(X_train, y_train)


predictions_rand=randF.predict(X_test)
pd.crosstab(y_test, predictions_rand, rownames=['Actual'], colnames=['Prediction'])

auroc = roc_auc_score(y_test, predictions_rand)
ap = average_precision_score(y_test, predictions_rand)
print(auroc, ap)


pickle.dump(randF, open("cc_model_day_3.pkl","wb"))


