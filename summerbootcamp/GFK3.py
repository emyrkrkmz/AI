
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1] # unnecessary values


num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]

#correalation
corr = df[num_cols].corr()

#sns.set(rc={'figure.figsize': (12,12)})
#sns.heatmap(corr, cmap="RdBu")
#plt.show()


##Removing high correlation variables
corr_matrix = df.corr().abs()

upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)) #Remove diogonal

drop_list= [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]

df.drop(drop_list, axis=1)


def high_correlated_cols(dataframe, plot=False, corr_th=0.9):
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)) #Remove diogonal
    drop_list= [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
    
    if plot:
        sns.set(rc={'figure.figsize': (12,12)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
        
    return drop_list