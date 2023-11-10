import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")

df["age"].mean()

df.groupby("sex")["age"].mean()


df.groupby("sex").agg({"age": "mean"})
df.groupby("sex").agg({"age": ["mean", "sum"]})


df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"],
                       "survived": "mean"})


df.pivot_table("survived","sex", "embarked")	#means
df.pivot_table("survived","sex", "embarked", aggfunc="std")	#stds

df.pivot_table("survived","sex", ["embarked", "class"])


df["new_age"] = pd.cut(df["age"], [0,10,18,25,40,98])

df.pivot_table("survived", "sex", "new_age")

pd.set_option('display_width', 500)
