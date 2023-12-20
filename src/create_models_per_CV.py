import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier


# n_splits, random_state, X_df, y_dfを引数とする関数
def create_regression_models_per_CV(n_splits, random_state, X_df, y_df, AT_df=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    CV_results = []

    # CVループ
    for train_index, test_index in kf.split(X_df):
        X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
        y_train, y_test = y_df.iloc[train_index], y_df.iloc[test_index]

        # アーキタイプの情報も取得
        AT_train, AT_test = (AT_df.iloc[train_index], AT_df.iloc[test_index]) if AT_df is not None else (None, None)

        # 特徴量とターゲットのスケーリング
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_test_scaled = y_scaler.transform(y_test)

        models = {
            'lr': LinearRegression(),
            'ridge': Ridge(),
            'svr': SVR(),
            'rf': RandomForestRegressor(),
            'lgbm': LGBMRegressor(),
            'xgb': XGBRegressor(),
            'mlp': MLPRegressor(max_iter=1000, alpha=0.0001, hidden_layer_sizes=(60,))
        }

        y_preds_from_test = {}
        y_preds_from_train = {}
        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train_scaled.ravel())

            # テストデータからの予測
            y_pred_scaled_from_test = model.predict(X_test_scaled)
            y_pred_from_test = y_scaler.inverse_transform(y_pred_scaled_from_test.reshape(-1, 1)).ravel()
            y_preds_from_test[model_name] = y_pred_from_test

            # 訓練データからの予測
            y_pred_scaled_from_train = model.predict(X_train_scaled)
            y_pred_from_train = y_scaler.inverse_transform(y_pred_scaled_from_train.reshape(-1, 1)).ravel()
            y_preds_from_train[model_name] = y_pred_from_train

        # アンサンブルモデルの予測
        ensemble_pred_from_test = np.mean(list(y_preds_from_test.values()), axis=0)
        ensemble_pred_from_train = np.mean(list(y_preds_from_train.values()), axis=0)
        y_preds_from_test['ensemble'] = ensemble_pred_from_test
        y_preds_from_train['ensemble'] = ensemble_pred_from_train

        CV_results.append({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'AT_train': AT_train,
            'AT_test': AT_test,
            'y_preds_from_test': y_preds_from_test,
            'y_preds_from_train': y_preds_from_train,
            'X_scaler': X_scaler,
            'y_scaler': y_scaler,
            'models': models
        })

    return CV_results




def create_classification_models_per_CV(n_splits, random_state, X_df, y_df, AT_df=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    CV_results = []

    # CVループ
    for train_index, test_index in kf.split(X_df):
        X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
        y_train, y_test = y_df.iloc[train_index].values.ravel(), y_df.iloc[test_index].values.ravel()

        AT_train, AT_test = (AT_df.iloc[train_index], AT_df.iloc[test_index]) if AT_df is not None else (None, None)

        X_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)

        models = {
            'lg': LogisticRegression(max_iter=1000),
            'svm': SVC(probability=True),
            'rf': RandomForestClassifier(),
            'lgbm': LGBMClassifier(),
            'xgb': XGBClassifier(),
            'mlp': MLPClassifier(max_iter=1000, alpha=0.0001, hidden_layer_sizes=(60,))
        }

        y_preds_from_train = {}
        y_preds_from_test = {}
        y_prob_preds_from_train = {}
        y_prob_preds_from_test = {}

        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_preds_from_train[model_name] = model.predict(X_train_scaled)
            y_preds_from_test[model_name] = model.predict(X_test_scaled)

            if hasattr(model, "predict_proba"):
                y_prob_preds_from_train[model_name] = model.predict_proba(X_train_scaled)[:, 1]
                y_prob_preds_from_test[model_name] = model.predict_proba(X_test_scaled)[:, 1]

        # アンサンブルモデルの予測
        ensemble_prob_pred_from_test = np.mean(list(y_prob_preds_from_test.values()), axis=0)
        ensemble_prob_pred_from_train = np.mean(list(y_prob_preds_from_train.values()), axis=0)
        y_prob_preds_from_test['ensemble'] = ensemble_prob_pred_from_test
        y_prob_preds_from_train['ensemble'] = ensemble_prob_pred_from_train

        # アンサンブル確率がth以上なら1、未満なら0
        th = 0.5
        ensemble_pred_from_test = (ensemble_prob_pred_from_test >= th).astype(int)
        ensemble_pred_from_train = (ensemble_prob_pred_from_train >= th).astype(int)
        y_preds_from_test['ensemble'] = ensemble_pred_from_test
        y_preds_from_train['ensemble'] = ensemble_pred_from_train


        CV_results.append({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'AT_train': AT_train,
            'AT_test': AT_test,
            'y_preds_from_train': y_preds_from_train,
            'y_preds_from_test': y_preds_from_test,
            'y_prob_preds_from_train': y_prob_preds_from_train,
            'y_prob_preds_from_test': y_prob_preds_from_test,
            'X_scaler': X_scaler,
            'models': models
        })

    return CV_results

