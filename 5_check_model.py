from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import cdsw


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("cc_fraud_demo").config("spark.kryoserializer.buffer.max.mb", "512").getOrCreate()
import pandas as pd
cc_data_spark = spark.read.parquet("credit_card_dataframe_spark")
cc_data = cc_data_spark.toPandas()

X = cc_data[cc_data.Day < 4].iloc[:,3:len(cc_data.columns)-1]
y = cc_data[cc_data.Day < 4].iloc[:,len(cc_data.columns)-1]

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.3, random_state=42
)

param_numTrees = int(sys.argv[1])
param_maxDepth = int(sys.argv[2])
param_impurity = sys.argv[3]


randF=RandomForestClassifier(
  n_jobs=10,
  n_estimators=param_numTrees, 
  max_depth=param_maxDepth, 
  criterion = param_impurity,
  random_state=0
)

randF.fit(X_train, y_train)

predictions_rand=randF.predict(X_test)
auroc = roc_auc_score(y_test, predictions_rand)
ap = average_precision_score(y_test, predictions_rand)


cdsw.track_metric("auroc", round(auroc,2))
cdsw.track_metric("ap", round(ap,2))

pickle.dump(randF, open("cc_model_check.pkl","wb"))

cdsw.track_file("cc_model_check.pkl")