import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("tips")


#sns.countplot(x=df["sex"], data=df)
#sns.boxplot(x=df["total_bill"])

df["total_bill"].hist() ##PANDAS hist

plt.show()