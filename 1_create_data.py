from pyspark.sql import SparkSession
import pandas as pd
import random
import pickle

spark = SparkSession.builder.appName("cc_fraud_demo").getOrCreate()

# # Create Larger Data Set

def data_randomizer(x):
    if x.name == 'Amount':
        return x + (x*random.uniform(-1,1)*0.1)
    else:
        return x + (x*random.uniform(-1,1)*0.1)

credit_card_data = spark.read.csv(
    "creditcard.csv", header=True, mode="DROPMALFORMED",inferSchema=True
)

credit_card_dataframe_1 = credit_card_data.toPandas()

# Show the Columns
credit_card_dataframe_1.columns

randomizeable_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

credit_card_dataframe_2 = pd.concat(
  [credit_card_dataframe_1[['Time']].apply(lambda x: x + (172800*1)),
  credit_card_dataframe_1[randomizeable_columns].apply(data_randomizer),
  credit_card_dataframe_1[['Class']]],axis=1
)


credit_card_dataframe_3 = pd.concat(
  [credit_card_dataframe_1[['Time']].apply(lambda x: x + (172800*2)),
  credit_card_dataframe_1[randomizeable_columns].apply(data_randomizer),
  credit_card_dataframe_1[['Class']]],axis=1
)


credit_card_dataframe_combined = pd.concat(
  [credit_card_dataframe_1,
  credit_card_dataframe_2,
  credit_card_dataframe_3],ignore_index=True,axis=0
)


credit_card_dataframe_final = pd.concat(
  [credit_card_dataframe_combined.rename(columns={'Time': 'Day'}).Day.apply(lambda x: int(x/3600/24)),
  credit_card_dataframe_combined.rename(columns={'Time': 'Hour'}).Hour.apply(lambda x: int(x/3600%24)),
  credit_card_dataframe_combined],axis=1
)

# Write to pickle file for jobs and analysis
credit_card_dataframe_final.to_pickle("resources/credit_card_dataframe_final.pkl",compression="gzip")

# Write to Spark for Experiments
credit_card_dataframe_spark = spark.createDataFrame(credit_card_dataframe_final)
credit_card_dataframe_spark.write.parquet("credit_card_dataframe_final",mode="overwrite")


