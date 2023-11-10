import seaborn as sns
import pandas as pd

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")

df["age"].head()
type(df["age"].head())	#pandas
type([df["age"].head()])	#dataframe

df[["age", "adult_male", "alive"]]


# df.loc[:, df.columns.str.contains("age")].head() columns which contain age
df.loc[:, ~df.columns.str.contains("age")].head() # columns which not contain age

#iloc: intager vased selection
df.iloc[0:3]
df.iloc[0, 0]

# loc: label based selection

df.loc[0:3]


df.iloc[0:3, 0:3]
df.loc[0:3, "age"]


#Contiton selection
df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count()


df.loc[df["age"] > 50, ["age", "class"]].head()
df.loc[(df["age"] > 50) & (df["sex"] == "male") , ["age", "class"]].head() #multi condition

df.loc[(df["age"] > 50) 
       & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
       ["age", "class", "embark_town"]].head()


