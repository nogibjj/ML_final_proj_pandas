import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, gc, joblib
from pprint import pprint
import lightgbm as lgb
from sklearn import metrics
from functools import reduce
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedGroupKFold,
)
from contextlib import suppress

pathway = ""


def set_table_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    for col in df.columns:
        # Cast Transform DPD (Days past due, P) and Transform Amount (A) as Float64
        if col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
        # Cast Transform date (D) as Date, causes issues with other columns ending in D
        # if col[-1] in ("D"):
        # df = df.with_columns(pl.col(col).cast(pl.Date).alias(col))
        # Cast aggregated columns as Float64, tried combining sum and max, but did not work correctly
        if col[-4:-1] in ("_sum"):
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
        if col[-4:-1] in ("_max"):
            df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
    return df


def convert_strings(df: pl.DataFrame) -> pl.DataFrame:
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))
    return df


# Changed this function to work for Pandas
def missing_values(df, threshold=0.0):
    missing_cols = []
    for col in df.columns:
        decimal = (pd.isnull(df[col]).sum()) / (len(df[col]))
        if decimal > threshold:
            print(f"{col}: {decimal}")
            missing_cols.append(col)
    return missing_cols


# Impute numeric columns with the median and cat with mode
def imputer(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].fillna(df[col].median())
        if df[col].dtype.name in ["category", "object"] and df[col].isnull().any():
            mode_without_nan = df[col].dropna().mode().values[0]
            df[col] = df[col].fillna(mode_without_nan)
    return df


####### THIS SCRIPT ENSURES TRAIN AND TEST DATA IS CONSISTENT SINCE THE COMPETITION KEEPS ADDING NEW DATA ETC AND SOME MINOR DIFFERENCES MIGHT CAUSE PROBLEMS #######

train = pl.read_csv("train_final.csv").pipe(set_table_dtypes).pipe(convert_strings)
train.head()

test = pl.read_csv("test_final.csv").pipe(set_table_dtypes).pipe(convert_strings)
test.head()

common_columns = list(set(train.columns) & set(test.columns))


# len(common_columns)

test = test[common_columns]
# Subset train with only columns seen in test + target
train = train[common_columns + ["target"]]
train.shape

train = train.to_pandas()
test = test.to_pandas()


# save train and test to csv
train.to_csv("train_final_final.csv", index=False)
test.to_csv("test_final_final.csv", index=False)


# y = train.loc[:,'target'].to_frame('target')
# X = train.drop(['target',], axis=1)

# # Do not include case_id, or week_num as numeric
# numeric_cols = test.select_dtypes(include=['number']).columns.tolist()
# numeric_cols.remove('case_id')
# numeric_cols.remove('WEEK_NUM')


# #scale values before passing on to model
# warnings.filterwarnings("ignore")
# scaler = MinMaxScaler(copy=False)
# X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
# test[numeric_cols] = scaler.transform(test[numeric_cols])

# # Drop case_id and week_num from features
# weeks = X["WEEK_NUM"]
# X_feats = X.drop(['case_id', 'WEEK_NUM'], axis=1)

# # Sort columns in alphabetical order for training so columns match test submission
# X_feats = X_feats.reindex(sorted(X_feats.columns), axis=1)


# %%time
# warnings.filterwarnings("ignore")
# cv = StratifiedGroupKFold(n_splits=2, shuffle=True)

# fitted_models = []
# cv_scores = []

# grid_params = {
#     "boosting_type": "gbdt",
#     "objective": "binary",
#     "metric": "auc",
#     "max_depth": 10,
#     "learning_rate": 0.05,
#     "n_estimators": 500,
#     "colsample_bytree": 0.8,
#     "colsample_bynode": 0.8,
#     "random_state": 42,
#     "reg_alpha": 0.1,
#     "reg_lambda": 10,
#     "extra_trees":True,
#     'num_leaves':64,
#     "verbose": -1,
#     "max_bin": 250,
# }

# for idx_train, idx_valid in cv.split(X_feats, y, groups=weeks):
#     X_train, y_train = X_feats.iloc[idx_train], y.iloc[idx_train]
#     X_valid, y_valid = X_feats.iloc[idx_valid], y.iloc[idx_valid]

#     clf = lgb.LGBMClassifier(**grid_params)
#     clf.fit(
#         X_train, y_train,
#         eval_set = [(X_valid, y_valid)],
#         callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)])
#     fitted_models.append(clf)

#     y_pred_valid = clf.predict_proba(X_valid)[:,1]
#     auc_score = roc_auc_score(y_valid, y_pred_valid)
#     cv_scores.append(auc_score)

# print("CV AUC scores: ", cv_scores)
# print("Maximum CV AUC score: ", max(cv_scores))

# warnings.filterwarnings("default")

# class VotingModel(BaseEstimator, RegressorMixin):
#     def __init__(self, estimators):
#         super().__init__()
#         self.estimators = estimators

#     def fit(self, X, y=None):
#         return self

#     def predict(self, X):
#         y_preds = [estimator.predict(X) for estimator in self.estimators]
#         return np.mean(y_preds, axis=0)

#     def predict_proba(self, X):
#         y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
#         return np.mean(y_preds, axis=0)

# model = VotingModel(fitted_models)
# model_plot = fitted_models[np.argmax(cv_scores)]


# model_plot

# base_train = pd.concat([X_train, y_train], axis=1)
# base_train['score'] = model.predict_proba(X_train)[:,1]
# print(f"The AUC score on the train set is: {roc_auc_score(base_train['target'], base_train['score'])}")

# base_valid = pd.concat([X_valid, y_valid], axis=1)
# base_valid['score'] = model.predict_proba(X_valid)[:,1]
# print(f"The AUC score on the valid set is: {roc_auc_score(base_valid['target'], base_valid['score'])}")

# def gini_stability(base, w_fallingrate=88.0, w_resstd=-0.5):
#     gini_in_time = base.loc[:, ["WEEK_NUM", "target", "score"]]\
#         .sort_values("WEEK_NUM")\
#         .groupby("WEEK_NUM")[["target", "score"]]\
#         .apply(lambda x: 2*roc_auc_score(x["target"], x["score"])-1).tolist()

#     x = np.arange(len(gini_in_time))
#     y = gini_in_time
#     a, b = np.polyfit(x, y, 1)
#     y_hat = a*x + b
#     residuals = y - y_hat
#     res_std = np.std(residuals)
#     avg_gini = np.mean(gini_in_time)
#     return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std

# base_train["WEEK_NUM"] = weeks.iloc[idx_train]


# stability_score_valid = gini_stability(base_train)
# print(f'The stability score on the valid set is: {stability_score_valid}')
