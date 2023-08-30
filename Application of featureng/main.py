from tools_and_functions import *

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.3f' % x)
pd.set_option('display.width',500)



df = load()

# df.shape
# df.head()

df.columns = [col.upper() for col in df.columns]

###########

#Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
#Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
#Name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
#Name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
#Name title
df["NEW_TITLE"] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
#Familt size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
#Age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
#is alone
df.loc[((df["SIBSP"] + df["PARCH"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SIBSP"] + df["PARCH"]) == 0), "NEW_IS_ALONE"] = "YES"
#Age level
df.loc[(df["AGE"] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df["AGE"] >= 56), 'NEW_AGE_CAT'] = 'senior'
#sex x age
df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df["SEX"] == "male") & (df["AGE"] > 21) & (df["AGE"] <= 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df["SEX"] == "male") & (df["AGE"] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df["SEX"] == "female") & (df["AGE"] > 21) & (df["AGE"] <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df["SEX"] == "female") & (df["AGE"] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'



cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [x for x in num_cols if x != "PASSENGERID"]


#################
#OUTLIERS


#for col in num_cols:
#    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)
    
    
#################
#MISSING VALUES

#missing_values_tables(df)
    
df.drop("CABIN",inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]

df.drop(remove_cols, inplace=True, axis=1)


df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

##Create again for filling missing values caused by age value
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df["AGE"] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df["AGE"] >= 18) & (df["AGE"] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df["AGE"] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df["SEX"] == "male") & (df["AGE"] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df["SEX"] == "male") & (df["AGE"] > 21) & (df["AGE"] <= 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df["SEX"] == "male") & (df["AGE"] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df["SEX"] == "female") & (df["AGE"] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df["SEX"] == "female") & (df["AGE"] > 21) & (df["AGE"] <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df["SEX"] == "female") & (df["AGE"] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == 'O' and len(x.unique()) <= 10) else x, axis=0)


#missing_values_tables(df)

#################
#Label encoding

binary_cols = [col for col in df.columns if df[col].dtype not in [int,float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


#################
#Rare encoding

#rare_analyser(df, "SURVIVED", cat_cols)

df = rare_encoder(df, 0.01)

#print(df["NEW_TITLE"].value_counts())


#################
#One-hot encoding

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#rare_analyser(df, "SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]


# df.drop(useless_cols, axis=1, inplace=True)

#################
#Standard encoding

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#df.head()
#df.shape

#################
#Model


y = df["SURVIVED"]

X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train,x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(x_test)
print(accuracy_score(y_pred, y_test)) #80%

##### What would the score be if no data processing was done?

dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y1 = dff["Survived"]
X1 = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train1, x_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=17)
rf_model1 = RandomForestClassifier(random_state=46).fit(X_train1, y_train1)
y_pred1 = rf_model1.predict(x_test1)
print(accuracy_score(y_pred1, y_test1)) #70%


##### Importance of new variables

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value" : model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importance.png') 
    
plot_importance(rf_model, X_train)   