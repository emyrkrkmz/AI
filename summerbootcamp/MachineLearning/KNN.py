#K-Nearest Neighbors Regression
#K-Nearest Neighbors Classification


# 1. Exploratory Data Analysis (EDA)
# 2. Data Preprocessing & Feature Eng
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)

###########################################
# 1. Exploratory Data Analysis (EDA)
###########################################

df = pd.read_csv("diabetes.csv")
df.shape
df.describe().T
df["Outcome"].value_counts()


###########################################
# 2. Data Preprocessing & Feature Eng
###########################################

y = df["Outcome"]

X = df.drop(["Outcome"], axis=1)


X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)


###########################################
# 3. Modeling & Prediction
###########################################

knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)

###########################################
# 4. Model Evaluation
###########################################

#for confusion matrix
y_pred = knn_model.predict(X)

#for AUC
y_prob = knn_model.predict_proba(X)[:, 1]


#print(classification_report(y, y_pred))
#
#print(roc_auc_score(y, y_prob)) #AUC


cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])


#print(cv_results["test_accuracy"].mean())
#print(cv_results["test_f1"].mean())
#print(cv_results["test_roc_auc"].mean())


###########################################
# 5. Hyperparameter Optimization
###########################################

knn_model = KNeighborsClassifier()
#print(knn_model.get_params())

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model, knn_params, cv=5, n_jobs=-1, verbose=1).fit(X, y)#n_jobs=-1 -> full performance cpu usage, verbose=1 -> yes for report

#print(knn_gs_best.best_params_) -> 17


###########################################
# 6. Final Model
###########################################


knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy","f1", "roc_auc"])

#print(cv_results["test_accuracy"].mean())
#print(cv_results["test_f1"].mean())
#print(cv_results["test_roc_auc"].mean())


## How To Improve scores?
# increase # of sample
# data preprocess
# featrue eng
# optimization for realted algo

random_user = X.sample(1)

print(knn_final.predict(random_user))