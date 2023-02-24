import numpy as np


# 数値微分
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    # 「中心差分」で求める. 一般的な x+h とx の差分は「前方差分」
    return (f(x + h) - f(x - h)) / (2 * h)


def function_2(x):
    # 引数にNumPy の配列が入力されることを想定.
    return x[0] ** 2 + x[1] ** 2


# 勾配
# 全ての変数の偏微分をベクトルとしてまとめたもの
# numerical_diff は複数変数を扱えるから、それらを一度に計算することと同じで勾配を求めるということになる.
def numerical_gradient(f, x):
    # sample
    # numerical_gradient(function_2, np.array([3.0, 4.0]))
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # 勾配を返すために入力値と同じ形の配列を用意

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        # index がtuple で取得できる.
        # n次元配列でも各マスに対して偏微分できる
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad


# 勾配法
# f: 最適化したい関数
# f の極小値、うまくいけば最小値がもとまる.
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_diff(f, x)
        x -= lr * grad
    return x


if __name__ == "__main__":

    numerical_gradient(function_2, np.array([3.0, 4.0]))
