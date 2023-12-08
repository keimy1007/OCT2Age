import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys 

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')



def calc_reg_metrics(y_real_scaled, y_pred_scaled, scaler):
    reg_real_scores = scaler.inverse_transform(y_real_scaled).flatten()
    reg_pred_scores = scaler.inverse_transform(y_pred_scaled).flatten()
    
    return {"metric": {
        "RMSD":mean_squared_error(reg_real_scores, reg_pred_scores)**0.5,
        "MAE":mean_absolute_error(reg_real_scores, reg_pred_scores)
        }, 
        "Real scores": reg_real_scores,
        "Pred scores": reg_pred_scores
    }


def plot_reg_metrics(y_pred_scaled, y_true_scaled, scaler, target_name):
    # 辞書 {metrics, y_real, y_pred}
    res_reg = calc_reg_metrics(y_pred_scaled, y_true_scaled, scaler=scaler)

    offset = 1

    y_real = res_reg["Real scores"]
    y_pred = res_reg["Pred scores"]

    min_ = min(y_real.min(), y_pred.min())
    max_ = max(y_real.max(), y_pred.max())
    rmse = mean_squared_error(y_real, y_pred)**0.5
    mae = mean_absolute_error(y_real, y_pred)

    plt.figure(figsize=(4,4))
    plt.scatter(
        y_pred,
        y_real,
        marker="."
    )

    plt.plot(
        [max_+offset, min_-offset],
        [max_+offset, min_-offset],
        color="orange",
        linestyle="dashed", 
        )
    plt.grid(alpha = 0.3)
    plt.title(f"{target_name}: RMSE={rmse:.3f}, MAE={mae:.3f}")
    plt.xlabel(f"Predicted {target_name}")
    plt.ylabel(f"True {target_name}")


