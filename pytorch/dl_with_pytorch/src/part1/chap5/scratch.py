# %%
import torch

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
t_nu = 0.1 * t_u


# %%


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


def d_loss_fn(t_p, t_c):
    # d L/ d model
    """
    平均二乗誤差の勾配
    t_p predct
    t_c true
    """
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)  # 除算は平均化のため
    return dsq_diffs


def dmodel_dw(t_u, w, b):
    # d model/ dw
    return t_u


def dmodel_db(t_u, w, b):
    # d model/ db
    return 1.0


def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = d_loss_fn(t_p, t_c)
    # print(f"{dloss_dtp=}")
    # 連鎖律
    # loss の微分* model の偏微分
    # 各入力とdloss との内積で計算（最後のsum）
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])


w = torch.ones(())  # tensor(1.)
b = torch.zeros(())  # tensor(0.)
t_p = model(t_u=t_u, w=w, b=b)
print(f"{loss_fn(t_p=t_p, t_c=t_c)=}")
# あとは、このloss をどうやって小さくしていくかの工夫.
# loss のFB からw,b をなんとかうまく調整するアリゴリズムが欲しい.

grad_fn(t_u=t_u, t_c=t_c, t_p=t_p, w=w, b=b)


# %%


# fit 学習ループ
def train(epoch: int, lr: float, params, t_u, t_c, print_params=False):
    for i in range(epoch):
        w, b = params
        # 順伝播
        t_p = model(t_u=t_u, w=w, b=b)
        loss = loss_fn(t_p=t_p, t_c=t_c)

        # 逆伝播
        grad = grad_fn(t_u=t_u, t_c=t_c, t_p=t_p, w=w, b=b)
        params = params - lr * grad

        if print_params:
            print(f"Epoch {i+1}, params {params}")
        print(f"Epoch {i+1}, Loss {float(loss)}")

    return params


params = train(
    epoch=5000, lr=1e-2, params=torch.tensor([1.0, 0.0]), t_u=t_nu, t_c=t_c, print_params=False
)

print(params)
# 摂氏（°C）と華氏（°F）
# 華氏を摂氏に変換するには32を引いてから 5/ 9または0.555をかける。
# 摂氏を華氏に変換するには 9/ 5または1.8をかけてから32を加える


# %%

from matplotlib import pyplot as plt

t_p = model(t_nu, *params)
fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), "o")

# %%
