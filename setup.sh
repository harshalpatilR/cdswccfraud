# this will unzip and copy the .csv file into your HDFS home directory
cd resources
unzip creditcard.csv.zip
hdfs dfs -copyFromLocal creditcard.csv creditcard.csv
rm creditcard.csv.zip
rm creditcard.csv
pip3 install scikit-learn --upgrade