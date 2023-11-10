import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile

pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)



df = pd.read_csv("diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

#y_pred for confusion matrix
y_pred = cart_model.predict(X)

#y_prob for AUC 
y_prob = cart_model.predict_proba(X)[:,1]


##confusion matrix
#print(classification_report(y, y_pred)) -> 1 ???

##AUC
#print(roc_auc_score(y, y_prob)) -> 1 ???



## Holdout
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=17)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
#print(classification_report(y_test, y_pred))
#print(roc_auc_score(y_test ,y_prob))

##CV
cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)# cross validate does not care fit(X, y) because it fits in its function

cv_result = cross_validate(cart_model,
                           X, y,
                           cv=5,
                           scoring=["accuracy", "f1", "roc_auc"])

#print(cv_result["test_accuracy"].mean())
#print(cv_result["test_f1"].mean())
#print(cv_result["test_roc_auc"].mean())


## Hyperparameter optimization 

#print(cart_model.get_params()) # we intereted at max_depth and min_samples_split 

cart_params = {'max_depth': range(1, 11),
               'min_samples_split': range(2, 20)}

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y) # verbose is type of report, n_jobs=-1 for max performance CPU
													  # another param is scoring=. Func makes it for scoring param better


#print(cart_best_grid.best_params_)
#print(cart_best_grid.best_score_)

random = X.sample(1, random_state=45)

cart_best_grid.predict(random)
#! gridserachcv crates a model we can use it

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)

#cart_final = cart_final.set_params(**cart_best_grid.best_params_) -> another way to set params

cv_result = cross_validate(cart_final,
                           X, y,
                           cv=5,
                           scoring=["accuracy", "f1", "roc_auc"])

#print(cv_result["test_accuracy"].mean())
#print(cv_result["test_f1"].mean())
#print(cv_result["test_roc_auc"].mean())


#print(cart_final.feature_importances_)

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize = (10, 10))
    sns.set(font_scale = 1)
    sns.barplot(x="Value", y="Feature", data = feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
        
        
#plot_importance(cart_final, X) # num for # of feature will shown


####
## Analyzing Model Complexity with Learning Curves (BONUS)
####

train_score, test_score = validation_curve(cart_final, X, y,
                                           param_name="max_depth",
                                           param_range=range(1,11),
                                           scoring="roc_auc",
                                           cv=10)

#mean_train_score = np.mean(train_score, axis=1)
#mean_test_score = np.mean(test_score, axis=1)
#
#plt.plot(range(1,11), mean_train_score,
#         label="Training Score", color='b')
#
#plt.plot(range(1,11), mean_test_score,
#         label="Validation Score", color='g')
#
#plt.title("Validation Curve for CART")
#plt.xlabel("Number of max_depth")
#plt.ylabel("AUC")
#plt.tight_layout()
#plt.legend(loc="best")

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(model , X=X, y=y, param_name=param_name,
                                               param_range=param_range, scoring=scoring, cv=cv)
    
    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)
    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show(block=True)

#val_curve_params(cart_final, X, y, "max_depth", range(1,11))

#cart_val_params = [["max_depth", range(1,11)], ["min_samples_split", range(2,20)]]
#
#for i in range(len(cart_val_params)):
#    val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])
    
    
    
    
####
## Visualizing the Decision Tree
####

import graphviz


def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)


tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")

cart_final.get_params()
    
    
####
## Extracting Decision Rules 
####

tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)

####
## Extracting Python Codes of Decision Rules 
####

#sklearn 0.23.1

print(skompile(cart_final.predict).to("python/code"))
print(skompile(cart_final.predict).to("sqlalchemy/sqlite"))
print(skompile(cart_final.predict).to("excel"))


###########################
# SAVING and LOADING MODEL
###########################

joblib.dump(cart_final, "cart_final.pkl")

cart_model_from_disc = joblib.load("cart_final.pkl")

x = [12, 13, 20, 23, 4, 55, 12, 7]


cart_model_from_disc.predict(pd.Dataframe(x).T)
