# %%
import torch
from torch import optim

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

# %%


# 線型回帰だけど、これがいちノードに対する結合を表現していることに注目
def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_p, t_c):
    """
    平均二乗誤差
    t_p predct
    t_c true
    """
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


# fit 学習ループ
def train(
    epoch: int,
    model,
    optimizer_inst,
    params,
    t_u,
    t_c,
):
    optimizer = optimizer_inst([params], lr=lr)
    for i in range(epoch):
        # loop 内で何回もmodel に入れて式を作るから、計算グラフを何回も
        # 追加してしまいそうだけど大丈夫なのか...
        # -> p167
        t_p = model(t_u, *params)  # 最後の出力に対する全結合のNN と等しい.
        loss = loss_fn(t_p=t_p, t_c=t_c)  # loss の計算も定義

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 500 == 0:
            print(f"Epoch {i+1}, Loss {float(loss)}")

    return params


params = torch.tensor([1.0, 0.0], requires_grad=True)  # 値は係数
lr = 1e-1
train(
    epoch=5000,
    model=model,
    optimizer_inst=optim.Adam,
    params=params,
    t_u=t_u,
    t_c=t_c,
)

# %%

from matplotlib import pyplot as plt

t_p = model(t_u, *params)
fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), "o")

# %%
