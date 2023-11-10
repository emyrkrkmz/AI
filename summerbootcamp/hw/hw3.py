import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("proje/persona.csv")

df.shape

df["SOURCE"].nunique()
df["SOURCE"].value_counts()

df["PRICE"].nunique()
df["PRICE"].value_counts()

df["COUNTRY"].value_counts()
df.groupby("COUNTRY").agg({"PRICE":"sum"})
df.groupby("SOURCE").agg({"PRICE":"sum"})
df.groupby("COUNTRY").agg({"PRICE":"mean"})
df.groupby("SOURCE").agg({"PRICE":"mean"})
df.groupby(["SOURCE", "COUNTRY"]).agg({"PRICE":"mean"})
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE":"mean"})


agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE":"mean"}).sort_values("PRICE", ascending=False)

agg_df = agg_df.reset_index()


intervals = [0, 18, 23, 30, 40, 70]
labels = ["0_18", "18_23", "24_30", "31_40", "41_70"]

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], intervals, labels=labels)


agg_df["customers_level_based"] = agg_df[["COUNTRY", "SOURCE", "SEX", "AGE_CAT"]].agg(lambda x: '_'.join(x).upper(), axis=1)


agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.groupby("SEGMENT").agg({"PRICE": ["sum", "mean", "max"]})

print(agg_df.groupby("SEGMENT").agg({"PRICE": ["sum", "mean", "max"]}))

new_user = "TUR_ANDROID_FEMALE_31_40"
new_user2 = "FRA_IOS_FEMALE_31_40"

agg_df[agg_df["customers_level_based"] == new_user2]


