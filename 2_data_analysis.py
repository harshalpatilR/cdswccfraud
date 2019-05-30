### Data Analysis
# This file will create a shareable analysis and visualisation of the credit card fraud data.
# Most of this section was taken from the SE Specialization presentation done by my colleague 
# Feng Lu 
# [https://github.com/lufeng76/python/tree/master/credit-card-fraud](https://github.com/lufeng76/python/tree/master/credit-card-fraud)

import numpy as np
import pandas as pd
from pandas import DataFrame

from matplotlib import pyplot as plt
import seaborn as sns

## Load data
data = pd.read_pickle("resources/credit_card_dataframe_final.pkl",compression="gzip")

### Describe the data
# * the name of each column
# * the number of values in each column
# * the number of missing/NaN values in each column; 
# * the contents of the first 5 rows
# * the contents of the last 5 rows
# * the basic stats of each column
data.columns.values
data.info()
data.isnull().sum()
data.head()
data.tail()
data.describe().T

## Frequency of Fraud/Normal
fraud_count = data['Class'].value_counts()
fraud_count
fraud_count.plot.bar(color='r',title='Frequency of Fraud/Normal')

## Percentage of Fraud/Normal
fraud_count_n = data['Class'].value_counts(normalize=True)
fraud_count_n
fraud_count_n.plot.pie(title='Percentage of Fraud/Normal', \
                       labels = ['Normal', 'Fraud'], \
                       colors = ['g', 'r'], \
                       autopct='%.2f')


## Generate two new columns 'Day' and 'Hour'
# Since the column 'Time' is in seconds, we need to  
# add new column 'Hour' which is calculated by Hour = Time / 3600 % 24

# The transaction data is from two days, therefore we need to 
# calculate day and Hour  
cal_hour = lambda x : int(x / 3600 % 24)
cal_day  = lambda x : 'Day 1' if int(x/3600) > 23 else 'Day 2'

data['Day'] = data['Time'].apply(cal_day)
data['Hour'] = data['Time'].apply(cal_hour)

## Plot transaction frequency and amount
# What we can see here is that busiest time is from 9am to 10pm
data['Hour'].value_counts(sort=False)
data['Hour'].value_counts(sort=False).plot.bar()

# We don't know if the Day makes any difference
# Plot transaction frequency by Day and Hour
grouped_count = data.groupby(['Hour','Day',])['Amount'].count().unstack()
grouped_count.head()
grouped_count.plot(title='Transaction Freq',style=['o-','v-'])
grouped_count.plot.bar(title='Transaction Freq',stacked=True)

# Plot transaction amount by Day and Hour
grouped_amount = data.groupby(['Hour','Day',])['Amount'].sum().unstack()
grouped_amount.head()
grouped_amount.plot(title='Transaction Amount',style=['o-','v-'])
grouped_amount.plot.bar(title='Transaction Amount',stacked=True)

# We know want to know fraud transaction frequency and amount grouped 
# by Day and Hour

# Get the fraud only data
fraud = data[data['Class'] == 1]
fraud[['Time','Amount','Class']].head()

# What we found here is that a lot of fraud happed in both 2am and 11am
# This is because people sleep on 2am, and spend lots of money at 11am
# It is difficult for people to find out the fraud at that time since they
# are either dreaming or shopping.
f_grouped_count = fraud.groupby(['Hour','Day',])['Amount'].count().unstack()
f_grouped_count.head()
f_grouped_count.plot(title='Fraud Freq',style=['o-','v-'])
f_grouped_count.plot.bar(title='Fraud Freq',stacked=True)

# We can calculate the fraud dollar amount which is almost the same as frenquency
f_grouped_amount = fraud.groupby(['Hour','Day',])['Amount'].sum().unstack()
f_grouped_amount.head()
f_grouped_amount.plot(title='Fraud Amount',style=['o-','v-'])
f_grouped_amount.plot.bar(title='Fraud Amount',stacked=True)


# Plot Fraud distribution by time
# Fraud with more money tends to happed during the day time
def plot_fraud_dist(data):
  fig, axes = plt.subplots(2,1,sharex=True)
  fraud = data[data["Class"] == 1]
  normal = data[data["Class"] == 0]
  axes[0].scatter(fraud["Hour"], fraud["Amount"])
  axes[0].set_title('Fraud')

  axes[1].scatter(normal["Hour"], normal["Amount"])
  axes[1].set_title('Normal')

  plt.xlabel('Time (in Hours)')
  plt.ylabel('Amount')
  plt.show()

plot_fraud_dist(data)

## Divide amounts into several groups
# What we found here is that most fraud has smaller dollor values
# People usually ignore small amounts money they spend
fraud = data[data['Class'] == 1].copy()
fraud.head()
fraud['Amount'].describe()

fraud['Cut'] = pd.cut(fraud['Amount'], bins=[-0.1,10,50,100,200,300,400,3000])
fraud.groupby('Cut')['Amount'].count().plot.bar()



## Check correlation among variables
# The correlation plot is not really important here 
# as the input data is already gone through a PCA tranformation
fraud = data[data['Class'] == 1]
normal = data[data['Class'] == 1]

corr_fraud = fraud.loc[:, ~fraud.columns.isin(['Class','Day'])].corr()
sns.heatmap(corr_fraud,cmap="BrBG",annot=False)

corr_normal = normal.loc[:, ~fraud.columns.isin(['Class','Day'])].corr()
sns.heatmap(corr_normal,cmap="BrBG",annot=False)

## Distplot for V1 and V28
# What we observe here is that the distribution for varibles 
# V8,V13,V15,V20,V22,V23,V25,V26,V27,V28 have no significant influence 
# between fraud and normal data set.
# We can consider removing these variables during the feature selection phase
features = data.iloc[:,3:29].columns
plt.figure(figsize=(16,28*4))
for i, cn in enumerate(data[features]):
    ax = plt.subplot()
    sns.distplot(data[cn][data["Class"] == 1], bins=50, color='r')
    sns.distplot(data[cn][data["Class"] == 0], bins=100, color='g')
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
    plt.show()