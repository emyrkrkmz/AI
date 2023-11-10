import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("titanic")

## Categoric cols 
cat_cols = [col for col in df.columns if str(df[col.dtypes] in ["category", "object", "bool"])]

num_but_cat = [col for col in df.columns if df[col].nunique < 10 and df[col.dtypes] in ["int", "float"]]

cat_but_car = [col for col in df.columns if df[col].nunique > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]
##


num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]

num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
     
num_summary(df, "age")

for col in num_cols:
    num_summary(df, col, plot=True)
    
    
######
def grab_col_names(dataframe, cat_the=10, car_th=20):
	"""Returns the names of categorical, numeric and categorical but numeric variables in the data set.

	Args:
		dataframe (dataframe): dataframe whose variable names are desired 
		cat_the (int float, optional): threshold value for numeric bus catagoric variables. Defaults to 10.
		car_th (int float, optional): threshold value for categoric but cardinal . Defaults to 20.
	
	Returns:
		cat_cols: list
			categoric var
		num_cols: list
			numeric var
		cat_but_car: list
			categoric but cardinal
 	"""
  
	cat_cols = [col for col in df.columns if str(df[col.dtypes] in ["category", "object", "bool"])]

	num_but_cat = [col for col in df.columns if df[col].nunique < 10 and df[col.dtypes] in ["int", "float"]]

	cat_but_car = [col for col in df.columns if df[col].nunique > 20 and str(df[col].dtypes) in ["category", "object"]]

	cat_cols = cat_cols + num_but_cat

	cat_cols = [col for col in cat_cols if col not in cat_but_car]
 

	num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]

	num_cols = [col for col in num_cols if col not in cat_cols]

	return cat_cols, num_cols, num_but_cat




def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))


for col in cat_cols:
    target_summary_with_cat(df, "survived", col)
    

