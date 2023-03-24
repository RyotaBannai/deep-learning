from typing import Union

import matplotlib.pylab as plt
import numpy as np
from numpy import floating
from numpy._typing import NDArray

from libs.dataset.mnist import load_mnist

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
    return 1 / (1 + np.exp(-xs))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(xs: Union[NDArray[floating], float]):
    # ReLU関数
    return np.maximum(0, xs)


def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x >= 0] = 1
    return grad


def softmax(x: NDArray[floating]):
    # ソフトマックス関数
    # k層の全ニューロンの出力値の合計でk層のi番目ニューロンの出力値を割る
    x = x - np.max(x, axis=-1, keepdims=True)  # オーバーフロー対策
    #  If axis is negative it counts from the last to the first axis.
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    # n-D 一度に計算する.
    # 1行が１つのoutput 層だから、１行の和が１になるように計算
    # softmax(np.array([[0, 1], [1, 2], [2,3]]))
    # array([[0.26894142, 0.73105858],
    #       [0.26894142, 0.73105858],
    #       [0.26894142, 0.73105858]])
    #


def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    # np.log(0) のような計算が発生した際に、np.log(0) はマイナス無限大 -inf になって、それ以上計算を進められなくなってしまうためその防止策.
    # sample
    # y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    # t = [0,0,1,0,0,0,0,0,0,0]
    # cross_entropy_error(np.array(y),np.array(t))
    # > 2.302584092994546
    # 2データ以上の場合
    # y2=np.array([[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] * 10]).reshape(10,10)
    # _, t = get_data()
    # cross_entropy_error(np.array(y2),np.array(t))
    # > 6.7938106506644

    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データが「one-hot-vector」の場合、
    # 正解ラベルのインデックスに変換（ラベルデータに変換する）
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def cross_entropy_error_label(y, t):
    # 教師データがone-hot ではなく、ラベルの場合
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    # np.arnage で 0~batch_size のiter を作る. 0,1,2...batch_size-1
    # t がラベルだから、そのindex として使われる.
    # y.shape は[データ数, ラベル数]だから、2d 配列のindexing を行っている.
    # 2d-inding で正解していれば、その数値が大きく、0 ならdelta をつけてinf 防止策を入れて、その合計を計算.
    # sample
    # _, t = batch(batch_size=10,one_hot_label=False)
    # y2=np.array([[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] * 10]).reshape(10,10)
    # cross_entropy_error_label(y2, t)
    # > 7.788242088702825

    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


def get_data(one_hot_label=True):
    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, normalize=False, one_hot_label=one_hot_label
    )
    return (x_train, t_train), (x_test, t_test)


def batch(batch_size: int = 10, one_hot_label=True):
    (x_train, t_train), _ = get_data(one_hot_label)
    data_size = x_train.shape[0]
    mask = np.random.choice(data_size, batch_size)
    x_batch = x_train[mask]
    t_batch = t_train[mask]
    return x_batch, t_batch


if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    # y = step_function(x)  # ブロードキャスト
    # y = sigmoid(x)
    # y = relu(x)
    # plt.plot(x, y)
    # plt.ylim(-0.1, 5.0)
    # plt.show()
