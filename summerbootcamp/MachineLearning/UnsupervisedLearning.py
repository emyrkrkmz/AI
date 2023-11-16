import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

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


## Deciding Optimum # of Cluster

#kmeans = KMeans()
#ssd = []
#K = range(1, 30)
#
#for k in K:
#    kmeans = KMeans(n_clusters=k).fit(df)
#    ssd.append(kmeans.inertia_)
#    
#plt.plot(K, ssd, "bx-")
#plt.xlabel("SSE/SSR/SSD values for different K values ")
#plt.title("Elbow method for Optimum # of Cluster ")
#
#plt.show()
    

kmeans = KMeans()

elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
#elbow.show()

#print(elbow.elbow_value_)

## Final Clusters and Model

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)


clusters = kmeans.labels_

df = pd.read_csv("USArrests.csv", index_col=0)

df["cluster"] = clusters

df["cluster"] = df["cluster"] + 1


#print(df.head())
#
#print(df[df["cluster"] == 5])
#
#x = df.groupby("cluster").agg(["count","mean","median"])
#
#print(x)


## Hierarchical Cluster Analysis 

df = pd.read_csv("USArrests.csv", index_col=0)

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

hc_average = linkage(df, "average")


#plt.figure(figsize=(10, 5))
#plt.title("Hierarchical Cluster Dendogram")
#plt.xlabel("Observation Units")
#plt.ylabel("Differencies")
#
#dendrogram(hc_average,
#           truncate_mode="lastp",
#           p=10,
#           show_contracted=True,
#           leaf_font_size=10)
#plt.show()



# Deciding # of cluster

plt.figure(figsize=(10, 5))
plt.title("Hierarchical Cluster Dendogram")
plt.xlabel("Observation Units")
plt.ylabel("Differencies")

dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=0.6, color='r', linestyle='--')
plt.axhline(y=0.5, color='b', linestyle='--')

plt.show()

# Final Model

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average").fit_predict(df)


df = pd.read_csv("USArrests.csv", index_col=0)
df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1


## PCA (Principal Component Analysis)

df = pd.read_csv("hitters.csv")


num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]

df = df[num_cols]
df.dropna(inplace=True)

df = StandardScaler().fit_transform(df)

pca = PCA()
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)


## # of Optimum Compenent 
pca = PCA().fit(df)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("# of Component")
plt.ylabel("# of Cumulative Variance")
plt.show()


## Final PCA

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)


