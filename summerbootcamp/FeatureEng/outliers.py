import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.3f' % x)
pd.set_option('display.width',500)


# Big dataset
def load_application_train():
    data = pd.read_csv('datasets/application_train.csv')
    return data


# Small dataset
def load():
    data = pd.read_csv('datasets/titanic.csv')
    return data


bg_dt = load_application_train()
sm_dt = load()


##		sns.boxplot(x=sm_dt["Age"])
##		plt.show()
##		
##		q1 = sm_dt["Age"].quantile(0.25)
##		q3 = sm_dt["Age"].quantile(0.75)
##		iqr = q3 - q1
##		
##		up = q3 + 1.5 * iqr
##		low = q1 - 1.5 * iqr
##		
##		aykirilar = sm_dt[(sm_dt["Age"] < low) |  (sm_dt["Age"] > up)]				#.any(axis=None) true false için
##		aykirilar_olmayanlar = sm_dt[~((sm_dt["Age"] < low) |  (sm_dt["Age"] > up))]


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
    
low, up = outlier_thresholds(sm_dt, "Fare")

outliers = sm_dt[(sm_dt["Fare"] < low) | (sm_dt["Fare"] > up)]

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False
    
# Col variable types reader
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != 'O']
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == 'O']
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(bg_dt)

###
df = sm_dt
dff = bg_dt
###

##  num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]
##  for col in num_cols:
##      print(col, check_outlier(dff, col))

def grap_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
        print("...")
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
        
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index
        
##REMOVING
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


cat_cols, num_cols, cat_but_car = grab_col_names(sm_dt)
num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    new_df = remove_outlier(df, col)
    

##RE-ASSIGNMENT

#low, up = outlier_thresholds(df, "Fare")
#df.loc[(df["Fare"] > up), "Fare"] = up

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit



############################
#   LOCAL OUTLIER FACTOR   #
############################

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()


clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
#print(df_scores[0:5])

# df_scores = -df_scores


print(np.sort(df_scores)[0:5])

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th = np.sort(df_scores)[3]

df[df_scores < th]

df.describe([0.01, 0.05, 0.75, 0.9, 0.99]).T

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)



