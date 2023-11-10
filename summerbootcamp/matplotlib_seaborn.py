# Categoric variable --> counptlot bar, column chart
# Numeric variable --> histogram, boxplot

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("titanic")

##	CATEGORIC
#df["sex"].value_counts().plot(kind='bar')

##	NUMERIC
#plt.hist(df["age"])

#plt.boxplot(df["fare"])

######

x = np.array([1, 8])
y = np.array([0, 150])
z = np.array([13, 28, 11, 100])
k = np.array([1, 2, 12, 18])


# plt.plot(x, y)		line
# plt.plot(x, y, 'o')	point

# plt.plot(z, marker='o')
# plt.plot(z, marker='*')
# plt.plot(z, linestyle="dashdot", color='r')


#plt.title("mainlabel")
#plt.xtitle("xlabel")
#plt.ytitle("ylabel")

#plt.grid()

plt.subplot(1, 2, 1) # row, column, first graph
plt.title("1")
plt.plot(x, y)

plt.subplot(1, 2, 2)
plt.title("2")
plt.plot(z, k)

plt.show()

