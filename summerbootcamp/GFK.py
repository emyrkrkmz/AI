import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("titanic")



def check_df(dataframe, head=5):
    print("######### Shape #########")
    print(dataframe.shape)
    
    print("######### Types #########")
    print(dataframe.dtypes)
    
    print("######### Head #########")
    print(dataframe.head(head))
      
    print("######### Tail #########")
    print(dataframe.tail(head))
    
    print("######### NA #########")
    print(dataframe.isnull().sum())
    
    print("######### Quantiles #########")
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)
    

cat_cols = [col for col in df.columns if str(df[col.dtypes] in ["category", "object", "bool"])]

num_but_cat = [col for col in df.columns if df[col].nunique < 10 and df[col.dtypes] in ["int", "float"]]

cat_but_car = [col for col in df.columns if df[col].nunique > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#########################")
    
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plot.show(block=True)

cat_summary(df, "sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col)
    else:
    	cat_summary(df, col)
    

