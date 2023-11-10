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


df = sns.load_dataset('titanic')

df.isnull().values.any()
df.isnull().sum()
df.notnull().sum()
df.isnull().sum().sum()
df[df.isnull().any(axis=1)]
df[df.notnull().all(axis=1)]


def missing_values_tables(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    
    if na_name:
        return na_columns

#1) Remove na
df.dropna()

#2) Basic assignment
df["Age"].fillna(df["Age"].mean())

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != 'O' else x, axis=0)
df["Embarked"].fillna(df["Embarked"].mode()[0])

df.apply(lambda x: x.fillna(x.mean()) if (x.dtype != 'O' and len(x.unique() <= 10)) else x, axis=0)

#complex
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

#simple
df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]


#ASSIGNMENT BY GUESS

df = sns.load_dataset('titanic')

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


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

#standardization
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)


from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)

dff = pd.DataFrame(imputer.inverse_transform(dff), columns=dff.columns)


dff["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]

##Analyzing Missing Values
msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

#investing relation between missing values and dependent variable

missing_values_tables(df, True)
na_cols = missing_values_tables(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col[target].mean()),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

    missing_vs_target(df, "Survived", na_cols)
