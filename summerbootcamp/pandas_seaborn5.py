import numpy as np
import pandas as pd

m = np.random.randint(1, 30, size=(5, 3))

df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99


pd.concat([df1, df2], ignore_index=True)


df1 = pd.DataFrame({'employees': ['mark', 'john', 'dennis'],
                    'group': ['accounting', 'engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['mark', 'john', 'dennis'],
                    'start_date': [2010, 2009, 2014]})

pd.merge(df1, df2, on="employees")

