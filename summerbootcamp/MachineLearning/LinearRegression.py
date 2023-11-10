#	y = b + w1*x1 + w2*x2 + w3*x3 ...
#
#	Cost(b, w) = 1/2m (i=1 to m)[∑((b + wi)- yi) ** 2]
#
#


### Sales Prediction with Linear Regression 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score



### Simple Linear Regression with OLS using Scikit-Learn 

df = pd.read_csv("advertising.csv")


X = df[["TV"]]
y = df[["sales"]]

## Model
reg_model = LinearRegression().fit(X, y)

# y_hat(y^) = b + w*x ==> b + w*TV

#constant (b - bias) -> reg_model.intercept_[0]
#print(reg_model.intercept_[0])

#coefficient of TV (w1) -> reg_model.coef_[0][0]
#print(reg_model.coef_[0][0])


## Prediction

# How much sales are expected if 150 units of TV are spent?

y_hat = reg_model.intercept_[0] + reg_model.coef_[0][0] * 150


# How much sales are expected if 500 units of TV are spent?

y_hat = reg_model.intercept_[0] + reg_model.coef_[0][0] * 500


# max TV is 296.4 but we can predict more TV situsations, so 500 worked in previous example --> print(df.describe().T)


## Visualizing The Model

g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")
g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Number of sales")
g.set_xlabel("TV Expenses")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

## Evaluating success of the model

# MSE
y_pred = reg_model.predict(X)
mse = mean_squared_error(y, y_pred) #10

#print(mse)

y.mean() # 14
y.std() # 5

# so interval is 9-19 and mse that is 10 not small 


# RMSE
rmse = np.sqrt(mean_squared_error(y, y_pred))

#print(rmse) #3

# MAE
mae = mean_absolute_error(y, y_pred)

#print(mae) #2.5


#R-Sqrt --> The percent rate at which independent variables explain dependent variables
reg_model.score(X, y) #0.61

