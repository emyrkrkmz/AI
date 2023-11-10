
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_excel("gezinomi/miuul_gezinomi.xlsx")

# df.info()


unq_cities=df["SaleCityName"].nunique()
freq_cities=df["SaleCityName"].value_counts()
unq_concept=df["ConceptName"].nunique()
freq_concept=df["ConceptName"].value_counts()


total_gain_city = df.groupby("SaleCityName").agg({"Price": "sum"})
avr_gain_city = df.groupby("SaleCityName").agg({"Price": "mean"})


total_gain_concept = df.groupby("ConceptName").agg({"Price": "sum"})
avr_gain_concept = df.groupby("ConceptName").agg({"Price": "mean"})

gain_city_concept = df.groupby(["SaleCityName","ConceptName"]).agg({"Price": "mean"})


bins= [-1, 7, 30, 90, df["SaleCheckInDayDiff"].max()]
labels = ["Last Minuters", "Potential Planners", "Planners", "Early Bookers"]

grouping_customers = df["EB_Score"] = pd.cut(df["SaleCheckInDayDiff"], bins, labels=labels)
df.head(50).to_excel("eb_scorew.xlsx", index=False)

grouping_customers_ebscore = df.groupby(["SaleCityName", "ConceptName", "EB_Score"]).agg({"Price": ["mean", "count"]})

agg_df = df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": "mean"}).sort_values("Price",ascending=False)
agg_df.reset_index(inplace=True)

agg_df["sales_level_based"] = agg_df[["SaleCityName", "ConceptName", "Seasons"]].agg(lambda x: '_'.join(x).upper(), axis=1)
agg_df["SEGMENT"] = pd.qcut(agg_df["Price"], 4, labels=["D", "C", "B", "A"])
segmentation = agg_df.groupby("SEGMENT").agg({"Price": ["mean", "max", "sum"]})






