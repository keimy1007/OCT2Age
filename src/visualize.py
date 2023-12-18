
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.manifold import TSNE
from umap import UMAP



def plot_TSNE(X_df, y_df, var_name, random_state=42):
    reducer = TSNE(n_components=2, random_state=random_state)
    X_2d = reducer.fit_transform(X_df)  # X_df は元の212次元のデータ

    # 年齢に基づいた色のグラデーションを設定
    colors = y_df.values.flatten()  # y_df は年齢のデータ
    cmap = mcolors.LinearSegmentedColormap.from_list("custom1", ["pink", "blue"])

    # 2Dの分布をプロット
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap=cmap)
    plt.colorbar(label=var_name)
    plt.title(f"2D visualization of {var_name} using t-SNE")
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.show()


def plot_UMAP(X_df, y_df, var_name, random_state=42):
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    reducer = UMAP(n_components=2, random_state=random_state)
    X_2d = reducer.fit_transform(X_df)  # X_df は元の212次元のデータ

    # 変数に基づいた色のグラデーションを設定
    colors = y_df.values.flatten()  # y_df は変数のデータ
    cmap = mcolors.LinearSegmentedColormap.from_list("custom2", ["pink", "blue"])

    # 2Dの分布をプロット
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, cmap=cmap)
    plt.colorbar(scatter, label=var_name)
    plt.title(f"2D visualization of {var_name} using UMAP")
    plt.xlabel("UMAP feature 1")
    plt.ylabel("UMAP feature 2")
    plt.show()


def plot_disc_thickness(arr):
    fig, ax = plt.subplots(subplot_kw={'polar': True})

    # cpRNFLの描写
    vmin, vmax = (50, 90)
    angles = np.linspace(0, 360, 13)[:-1]
    normed_values = (arr - vmin) / (vmax - vmin)
    normed_values = np.clip(normed_values, 0, 1)
    normed_values = np.roll(normed_values[::-1], shift=-5)
    colors = plt.cm.RdYlGn(normed_values)
    bars = ax.bar(angles * np.pi / 180, [1]*12, bottom=0, width=np.pi / 6, color=colors)

    ax.set_rorigin(0)  # ここを修正
    ax.set_yticklabels([])  # 半径のラベルを非表示に
    ax.set_xticklabels([])  # 角度のラベルを非表示に

    # S, N, I, Tのラベルを追加
    offset = 0.05
    ax.text(0-offset, 0.5, 'T', ha='center', va='center', transform=ax.transAxes)
    ax.text(1+offset, 0.5, 'N', ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 0-offset, 'I', ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 1+offset, 'S', ha='center', va='center', transform=ax.transAxes)

    # 数字のテキストを図示
    radius, offset = (0.3, 0.5)
    arr = np.roll(arr[::-1], shift = -2)
    for bar, value in zip(bars, arr):
        angle = (bar.get_x() + bar.get_width() / 2) * 180 / np.pi - 90
        x = np.cos(np.deg2rad(angle)) * radius + offset
        y = np.sin(np.deg2rad(angle)) * radius + offset
        plt.text(x, y, str(value), ha='center', va='center', transform=ax.transAxes)

    plt.show()
    # plt.close()


def plot_macula_thickness(arr, ROI="RNFL"):
    fig, ax = plt.subplots()

    # mRNFLやmGCLPの描写
    vmin, vmax = (210, 330) if ROI == "retina" else (10, 110)
    sns.heatmap(arr, annot=True, cmap='RdYlGn', fmt="d", vmin=vmin, vmax=vmax, ax=ax)

    plt.show()
    # plt.close()

