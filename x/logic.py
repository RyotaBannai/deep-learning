from typing import Union

import matplotlib.pylab as plt
import numpy as np
from numpy import floating
from numpy._typing import NDArray

"""
パーセプトロンのAND,NAND,OR ゲートは、{0,1} が入力値として入ってくる時に
真理値表に従って{0,1}を出力するようにパラメータを調整して作る.
i.g. NAND(1,1) は0 を出力.

"""


def AND(x1, x2):
    # どっちも入ってきたら閾値を超えるようにする
    x = np.array([x1, x2])  # 入力
    w = np.array([0.5, 0.5])  # 重み
    b = -0.7  # バイアス （発火のしやすさ）
    if np.sum(x * w) + b <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    # どっちも入ってきたら閾値を超えないようにする
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    if np.sum(x * w) + b <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    # 一つでも入ってきたら閾値を超えるようにする
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    if np.sum(x * w) + b <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    # XOR は非線形による領域を作るから表現できない
    # 非線形な領域(AND,NAND,OR) を組み合わせるとよい
    s1 = OR(x1, x2)
    s2 = NAND(x1, x2)
    ret = AND(s1, s2)
    return ret


def step_function(xs: Union[NDArray[floating], float]):
    # ステップ関数（階段関数）
    return np.array(xs > 0, dtype=np.int64)


def sigmoid(xs: Union[NDArray[floating], float]):
    # シグモイド関数
    return 1 / (1 + np.exp(-x))


def relu(xs: Union[NDArray[floating], float]):
    # ReLU関数
    return np.maximum(0, xs)


def softmax(xs: NDArray[floating]):
    # ソフトマックス関数
    # k層の全ニューロンの出力値の合計でk層のi番目ニューロンの出力値を割る
    c_prime = np.max(xs)
    exp_a = np.exp(xs - c_prime)  # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    # y = step_function(x)  # ブロードキャスト
    # y = sigmoid(x)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 5.0)
    plt.show()
