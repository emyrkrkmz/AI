import seaborn as sns
import pandas as pd

df = sns.load_dataset("titanic")

df.head()
df.tail()
df.info()
df.shape
df.columns
df.index
df.describe().T #T means transpose
df.isnull().values.any()
df.isnull().sum()
df["sex"].value_counts()

df.drop(0, axis=0).head()

delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10)

#FOR PERMANENT
##	df = df.drop(delete_indexes, axis=0)
##	df.drop(delete_indexes, axis=0, inplace=True)

#df["age"] = df.age


#variable to index
df.index = df["age"]
df.drop("age", axis=1)

#index to variable
df["age"] = df.index
df.reset_index()
