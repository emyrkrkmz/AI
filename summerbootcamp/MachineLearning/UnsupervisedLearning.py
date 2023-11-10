import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


## K-Means

df = pd.read_csv("USArrests.csv", index_col=0)

#print(df.head())
#print(df.isnull().sum())
#print(df.info())
#print(df.describe().T)


sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)


kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
#print(kmeans.get_params())

#print(kmeans.n_clusters)
#print(kmeans.cluster_centers_)
#print(kmeans.labels_)
#print(kmeans.inertia_) # sum of squared distances (SSD)


