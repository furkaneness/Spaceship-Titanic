import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder

from helpers import *
import seaborn as sns
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

#######################
# EDA
#######################

check_df(train, 5)
check_df(test, 5)

##################################
# Change column names of train and test datasets
##################################

train.columns = [col.upper() for col in train.columns]
test.columns = [col.upper() for col in test.columns]

##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

cat_cols, num_cols, cat_but_car = grab_col_names(train)
cat_cols = [col for col in cat_cols if col not in "TRANSPORTED"]


##################################
# ANALYSIS OF CATEGORICAL VARIABLES
##################################

for col in cat_cols:
    cat_summary(train, col, True)

for col in cat_cols:
    cat_summary(test, col, True)

##################################
# ANALYSIS OF NUMERICAL VARIABLES
##################################

for col in num_cols:
    num_summary(train, col, True)

for col in num_cols:
    num_summary(test, col, True)

##################################
# ANALYSIS OF CATEGORICAL VARIABLES BY TARGET
##################################

for col in cat_cols:
    target_summary_with_cat(train, "TRANSPORTED", col)

##################################
# ANALYSIS OF NUMERICAL VARIABLES BY TARGET
##################################

for col in num_cols:
    target_summary_with_num(train, "TRANSPORTED", col)

##################################
# ANALYSIS OF OUTLIERS
##################################

for col in num_cols:
    print(col, check_outlier(train, col))

for col in num_cols:
    print(col, check_outlier(test, col))

##################################
# REMOVE OUTLIERS
##################################

for col in num_cols:
    replace_with_thresholds(train, col)

for col in num_cols:
    replace_with_thresholds(test, col)

##################################
# ANALYSIS OF MISSING VALUES
##################################

train.isnull().sum()
test.isnull().sum()

train.dropna(inplace=True)
test.dropna(inplace=True)

train.head()
train.shape

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(train)
cat_cols = [col for col in cat_cols if col not in "TRANSPORTED"]

##################################
# FEATURE ENGINEERING
##################################
# Train
train.loc[(train["AGE"] < 5), "NEW_AGE"] = "BABY"
train.loc[(train["AGE"] >= 5) & (train["AGE"] < 10), "NEW_AGE"] = "KID"
train.loc[(train["AGE"] >= 10) & (train["AGE"] < 20), "NEW_AGE"] = "TEENAGER"
train.loc[(train["AGE"] >= 20) & (train["AGE"] < 40), "NEW_AGE"] = "YOUTH"
train.loc[(train["AGE"] >= 40) & (train["AGE"] < 60), "NEW_AGE"] = "ADULT"
train.loc[(train["AGE"] > 60), "NEW_AGE"] = "OLD"

# Test
test.loc[(test["AGE"] < 5), "NEW_AGE"] = "BABY"
test.loc[(test["AGE"] >= 5) & (test["AGE"] < 10), "NEW_AGE"] = "KID"
test.loc[(test["AGE"] >= 10) & (test["AGE"] < 20), "NEW_AGE"] = "TEENAGER"
test.loc[(test["AGE"] >= 20) & (test["AGE"] < 40), "NEW_AGE"] = "YOUTH"
test.loc[(test["AGE"] >= 40) & (test["AGE"] < 60), "NEW_AGE"] = "ADULT"
test.loc[(test["AGE"] > 60), "NEW_AGE"] = "OLD"

# Train
train["TOTAL_SPENT"] = train["ROOMSERVICE"] + train["FOODCOURT"] + train["SHOPPINGMALL"] + train["SPA"] + train["VRDECK"]

# Test
test["TOTAL_SPENT"] = test["ROOMSERVICE"] + test["FOODCOURT"] + test["SHOPPINGMALL"] + test["SPA"] + test["VRDECK"]

# Train
train.loc[(train["TOTAL_SPENT"] == 0), "SPENDING_HABBIT"] = "NONE"
train.loc[(train["TOTAL_SPENT"] > 0) & (train["TOTAL_SPENT"] < 2000), "SPENDING_HABBIT"] = "LOW"
train.loc[(train["TOTAL_SPENT"] >= 2000) & (train["TOTAL_SPENT"] < 6000), "SPENDING_HABBIT"] = "MIDDLE"
train.loc[(train["TOTAL_SPENT"] >= 6000), "SPENDING_HABBIT"] = "HIGH"

# Test
test.loc[(test["TOTAL_SPENT"] == 0), "SPENDING_HABBIT"] = "NONE"
test.loc[(test["TOTAL_SPENT"] > 0) & (test["TOTAL_SPENT"] < 2000), "SPENDING_HABBIT"] = "LOW"
test.loc[(test["TOTAL_SPENT"] >= 2000) & (test["TOTAL_SPENT"] < 6000), "SPENDING_HABBIT"] = "MIDDLE"
test.loc[(test["TOTAL_SPENT"] >= 6000), "SPENDING_HABBIT"] = "HIGH"

##################################
# LABEL ENCODING
##################################
cat_cols, num_cols, cat_but_car = grab_col_names(train)
cat_cols = [col for col in cat_cols if col not in "TRANSPORTED"]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in train.columns if train[col].dtypes == "O" and train[col].nunique() == 2]
binary_cols

for col in binary_cols:
    train = label_encoder(train, col)

for col in binary_cols:
    test = label_encoder(test, col)

##################################
# One-Hot Encoding
##################################
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in "TRANSPORTED"]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

train = one_hot_encoder(train, cat_cols, drop_first=True)
test = one_hot_encoder(test, cat_cols, drop_first=True)
train.head()


##################################
# MODELLING
##################################

y = train["TRANSPORTED"]
X = train.drop(["TRANSPORTED", "PASSENGERID", "CABIN", "NAME"], axis=1)

classifiers = [('LR', LogisticRegression()),
               ('KNN', KNeighborsClassifier()),
               ("SVC", SVC()),
               ("CART", DecisionTreeClassifier()),
               ("RF", RandomForestClassifier()),
               ('GBM', GradientBoostingClassifier()),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
               ('LightGBM', LGBMClassifier()),
               ('CatBoost', CatBoostClassifier(verbose=False))
               ]

for name, classifier in classifiers:
    cv_results = cross_validate(classifier, X, y, cv=3, scoring=["roc_auc", "accuracy"])
    print("roc_auc : ", f" {round(cv_results['test_roc_auc'].mean(), 4)} ({name})")
    print("acc     : ", f" {round(cv_results['test_accuracy'].mean(), 4)} ({name})")
    print("***********")

##################################
# Hyperparameter Optimization
##################################

lgbm_model = LGBMClassifier()
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.7944
cv_results["test_f1"].mean()  # 0.8017
cv_results["test_roc_auc"].mean()  # 0.8794

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "max_depth": [3, 5, 8]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_best_grid.best_params_
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()  # 0.8004
cv_results["test_f1"].mean()  # 0.8092
cv_results["test_roc_auc"].mean()  # 0.8820


PassengerId = test["PASSENGERID"]
test = test.drop(["PASSENGERID", "CABIN", "NAME"], axis=1)

y_pred = lgbm_final.predict(test)
y_pred = pd.Series(y_pred)

# Create Submission File
submission = pd.DataFrame({"PASSENGERID": PassengerId.values, "TRANSPORTED": y_pred})
submission.head()
submission.tail()

# Save Submission File
submission.to_csv("submission.csv", index=False)
print("My competition submission: \n\n", submission)

