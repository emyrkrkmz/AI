## diabet with logistic regression


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay, roc_curve, auc#, plot_roc_curve#
from sklearn.model_selection import train_test_split, cross_validate

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False
        
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#######


df = pd.read_csv("diabetes.csv")


#print(df.head() , "\n" , df.shape)

## target analyze

# print(df["Outcome"].value_counts())
# 
# sns.countplot(x="Outcome", data=df)
# plt.show()

#print(100 * df["Outcome"].value_counts() / len(df))

#print(df.describe().T)

#df["Glucose"].hist(bins=20)
#plt.xlabel("Glucose")
#plt.show()

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)
    

    
cols = [col for col in df.columns if "Outcome" not in col]

#for col in cols:
    #plot_numerical_col(df,col)
    
## target and features

#print(df.groupby("Outcome").agg({"Pregnancies": "mean"}))


def target_summary_with_num(dataframe, target, numerical_cols):
    print(dataframe.groupby(target).agg({numerical_cols:"mean"}),end="\n\n\n")
    
#for col in cols:
#    target_summary_with_num(df,"Outcome", cols)
    

## data preprocess

#print(df.isnull().sum())

#df.describe().T --> min values are 0 so missing_values were filled zero. this will be ignored 

#for col in cols:
#    print(col, check_outlier(df,col))

#!!! function's q1=0.05 and q3=0.95

replace_with_thresholds(df, "Insulin")

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])
#standardization


## model and prediction

y = df["Outcome"]

X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

#print(log_model.intercept_)
#print(log_model.coef_)

y_pred = log_model.predict(X)


## evaluation of success

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title('Accuracy Score {0}'.format(acc), size=10)
    plt.show()
    

#plot_confusion_matrix(y, y_pred)
#print(classification_report(y, y_pred)) ##accuracy=0.78, precision=0.74, recall=0.58, f1=0.65

## ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]

#print(roc_auc_score(y, y_prob))
#0.83939


## Validation (Holdout)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=17)


log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

#print(classification_report(y_test, y_pred)) ##accuracy=0.77, precision=0.79, recall=0.53, f1=0.63


## plot_roc_curve func is not available in sklearn latest version

#fpr, tpr, thresholds = roc_curve(y_test, y_pred)
#roc_auc = auc(fpr, tpr)
#display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
#display.plot()
#plt.show()

#print(roc_auc_score(y_test,y_prob))

## Validation (10-Fold Cross Validation)

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]) #cv -> # of fold


#print(cv_results["test_accuracy"].mean())
#print(cv_results["test_precision"].mean())
#print(cv_results["test_recall"].mean())
#print(cv_results["test_f1"].mean())
#print(cv_results["test_roc_auc"].mean())


####

random_user  = X.sample(1, random_state=45)
print(log_model.predict(random_user))
