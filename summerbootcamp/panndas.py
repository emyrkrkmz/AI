import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)

print(s)
s.index
s.dtype
s.size
s.ndim
s.values
type(s.values) #numpy

s.head(3)
s.tail(3)

d_csv = pd.read_csv("datasets/advertising.csv")

print(d_csv.head())

