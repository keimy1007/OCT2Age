import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy as cp

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, FunctionTransformer

import torch
from torch.utils.data import DataLoader, TensorDataset




def dataframe_to_tensor(df, dtype=torch.float):
    return torch.tensor(df.values, dtype=dtype)


def create_scaled_dataloader_per_CV(X_df, Y_df, scaler=StandardScaler(), n_splits=3, batch_size=64):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    X_train_CV, X_test_CV, y_train_CV, y_test_CV = [], [], [], []
    train_loader_CV, test_loader_CV = [], []
    X_scaler_CV, y_scaler_CV = [], []

    for train_index, test_index in kf.split(X_df):
        # 分割された訓練データとテストデータを取得
        X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
        y_train, y_test = Y_df.iloc[train_index], Y_df.iloc[test_index]

        # 生の値をリストに追加
        X_train_CV.append(X_train)
        X_test_CV.append(X_test)
        y_train_CV.append(y_train)
        y_test_CV.append(y_test)


        # スケーリング処理
        if isinstance(scaler, FunctionTransformer):
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()
            y_train = y_train.to_numpy()
            y_test = y_test.to_numpy()

        X_scaler = cp.deepcopy(scaler)
        y_scaler = cp.deepcopy(scaler)

        X_train_scaled = X_scaler.fit_transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_test_scaled = y_scaler.transform(y_test)

        # スケーリングされた値をリストに追加
        X_scaler_CV.append(X_scaler)
        y_scaler_CV.append(y_scaler)

        # データフレームをPyTorchのテンソルに変換
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

        # データセットを作成
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # データローダーを作成
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # スケーリングされたデータローダーをリストに追加
        train_loader_CV.append(train_loader)
        test_loader_CV.append(test_loader)

    return X_train_CV, X_test_CV, X_scaler_CV, y_train_CV, y_test_CV, y_scaler_CV, train_loader_CV, test_loader_CV





def make_datasets_in_2016(filenames, ROI="cpRNFL"):
    df = pd.DataFrame()
    first_loop = True
    for filename in filenames:
        tmp = pd.read_csv(f"./data_oct/{filename}")
        tmp = tmp.rename(columns={"r1l2": "eye", "RL": "eye"})

        if first_loop: 
            df = tmp
            first_loop = False
        else: 
            df = pd.concat([df, tmp])

    # Variable transformation and missing value deletion
    df = df.rename(columns={"TopQImageQuality": "QS", "kenshinID": "ID"})
    df = df.dropna(subset=["cpRNFL_T"]) if ROI == "cpRNFL" else df.dropna(subset=[f"{ROI}_01_01"])

    # Zero-padding for the central part of mRNFL, mGCLP
    if ROI not in ["retina", "cpRNFL"]:
        for coord in ["05_05", "05_06", "06_05", "06_06"]:
            df[f"{ROI}_{coord}"] = 0

    # Sample size, shuffle rows, and remove duplicate IDs
    df = df.sample(frac=1, random_state=0).drop_duplicates(subset=["ID"])

    # df の最初の3列の列名を取得
    first_three_columns = df.columns[:3]

    # df のOCTデータの列のインデックスを取得
    OCT_index = slice(3, 15) if ROI == "cpRNFL" else slice(3, 103)

    # 各患者の平均厚さを計算し、dfに追加
    df[f"thick_{ROI}"] = df.iloc[:, OCT_index].mean(axis=1)

    # 最初の3列とOCTデータの列のみを含む新しいデータフレームを作成
    X_df = df[first_three_columns].copy()
    X_df = X_df.join(df.iloc[:, OCT_index])
    
    # df のインデックスをリセット
    df = df.reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)

    return df, X_df

