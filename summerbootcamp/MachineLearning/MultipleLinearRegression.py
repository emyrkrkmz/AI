import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score




df = pd.read_csv("advertising.csv")


X = df.drop('sales', axis=1) # independent var
 
y = df[["sales"]] # dependent var


## Model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # test_size=20 --mean-> %20 test %80 train

# X_train -> 160
# X_test -> 40
# y_train -> 160
# y_test -> 40

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
# same as reg_model = LinearRegression().fit(X_train, y_train)

#constant (b - bias)
#reg_model.intercept_ -> 2.90

#coefficients (2 - weights)
#reg_model.coef_ -> 0.04, 0.17, 0.0025


## Prediction

#sales = 2.90 + TV*0.04 + radio * 0.17 + newspaper * 0.002
#y_hat = reg_model.intercept_[0] + reg_model.coef_[0][0]*30 + reg_model.coef_[0][1]*10 + reg_model.coef_[0][2]*40
#print(y_hat)

new_data = [[30], [10], [40]]
new_data = pd.DataFrame(new_data).T

y_hat = reg_model.predict(new_data)


## Evaluation of success


#Train RMSE
y_pred = reg_model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred)) #1.73

#Train Rsqrt
Rsqrt_train = reg_model.score(X_train, y_train) # %89


#Test RMSE
y_pred = reg_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred)) #1.41

#Test Rsqrt
Rsqrt_test = reg_model.score(X_test, y_test) # %89


# 10-Fold Cross_Validation(CV) RMSE
cv_rmse = np.mean(np.sqrt(-cross_val_score(reg_model,
                                           X,
                                           y,
                                           cv=10,
                                           scoring="neg_mean_squared_error")))
#1.69

# 5-Fold Cross_Validation(CV) RMSE
cv_rmse = np.mean(np.sqrt(-cross_val_score(reg_model,
                                           X,
                                           y,
                                           cv=5,
                                           scoring="neg_mean_squared_error")))
#1.71

