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




##############################
# Binary Features: FLag, Bool, True-False
##############################

df = load()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})


from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum()
                                             ,df.loc[df['NEW_CABIN_BOOL'] == 0, "Survived"].sum()],
                                      nobs=[df.loc[df['NEW_CABIN_BOOL'] == 1, "Survived"].shape[0],
                                            df.loc[df['NEW_CABIN_BOOL'] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %4.f' % (test_stat, pvalue))


df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"


df.groupby("NEW_IS_ALONE").agg({"Survived" : "mean"})


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == 1, "Survived"].sum()
                                             ,df.loc[df['NEW_IS_ALONE'] == 0, "Survived"].sum()],
                                      nobs=[df.loc[df['NEW_IS_ALONE'] == 1, "Survived"].shape[0],
                                            df.loc[df['NEW_IS_ALONE'] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %4.f' % (test_stat, pvalue))


##############################
# Text Features
##############################

#Letter count

df["NEW_NAME_COUNT"] = df["Name"].str.len()

#Word count

df["NEW_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

#Special structures

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})




##############################
# Regex Features
##############################

df["NEW_TITLE"] = df.Name.str.extract(' ([A-za-z]+\.)', expand=False) #df.Name = df["Name"]

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})



##############################
# Date Features
##############################

dff = pd.read_csv("/datasets/course_reviews.csv")
dff.head()
dff.info()

dff["Timestamp"] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d") #year-month-day

dff["year"] = dff["Timestamp"].dt.year
dff["month"] = dff["Timestamp"].dt.month

#year diff
dff["year_diff"] = date.today() - dff["Timestamp"].dt.year

#month diff = year diff + month diff
dff["month_diff"] = (date.today() - dff["Timestamp"].dt.year) * 12 + date.today().month - dff["Timestamp"].dt.month

#day name
dff["day_name"] = dff["Timestamp"].dt.day_name()

dff.head()




##############################
# Feature Interactions
##############################

df = load()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df["Sex"] == "male") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngmale"

df.loc[(df["Sex"] == "male") & (df["Age"] > 21) & (df["Age"] <= 50), "NEW_SEX_CAT"] = "maturemale"

df.loc[(df["Sex"] == "male") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniormale"


df.loc[(df["Sex"] == "female") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngfemale"

df.loc[(df["Sex"] == "female") & (df["Age"] > 21) & (df["Age"] <= 50), "NEW_SEX_CAT"] = "maturefemale"

df.loc[(df["Sex"] == "female") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

df.groupby("NEW_SEX_CAT")["Survived"].mean()


