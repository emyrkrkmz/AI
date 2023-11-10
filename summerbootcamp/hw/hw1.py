import seaborn as sns
import pandas as pd

## 1
df = sns.load_dataset("car_crashes")

a = ['NUM_' + x.upper() if df[x].dtype != 'O' else x.upper() for x in df.columns]

## 2
a = [x.upper() + '_FLAG'  if 'no' not in x else x.upper() for x in df.columns]

## 3
og_list = ["abbrev", "no_previous"]

new_cols = [x for x in df.columns if x not in og_list]

new_df = df[new_cols]

a = new_df.head()

##4
df = sns.load_dataset("titanic")
df.shape

df["sex"].value_counts()

df.nunique()

df["pclass"].unique()

df[["pclass", "parch"]].nunique()

df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")

df[df["embarked"] == 'C'].head()
df[df["embarked"] != 'S'].head()

df[(df["age"] < 30) & (df["sex"] == "female")].head()
df[(df["fare"] > 500) | (df["age"] > 70)].head()


df.isnull().sum()

df.drop("who", axis=1, inplace=True)

df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df["deck"].isnull().sum()

df["age"].fillna(df["age"].median(), inplace=True)
df["age"].isnull().sum()

df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "count", "mean"]})


df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)


df = sns.load_dataset("tips")

df.groupby("time").agg({"total_bill": ["sum", "min","mean", "max"]})
df.groupby(["time", "day"]).agg({"total_bill": ["sum", "min","mean", "max"]})

df[(df["time"] == "lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum", "min", "max", "mean"],
                                                                          "tip": ["sum","min","max","mean"]})

df.loc[(df["size"] < 3 ) & (df["total_bill"] > 10), "total_bill"].mean()

df["total_bill_tip_sum"] = df["totalbill"] + df["tip"]

sorted_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]

