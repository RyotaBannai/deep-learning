# %%
import torch
from torch import nn, optim

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c).unsqueeze(1)  # <1>
t_u = torch.tensor(t_u).unsqueeze(1)  # <1>

n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)
shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

t_u_train = t_u[train_indices]
t_c_train = t_c[train_indices]

t_u_val = t_u[val_indices]
t_c_val = t_c[val_indices]


# %%


# fit 学習ループ
def train(
    epoch: int,
    model,
    optimizer_inst,
    loss_fn,
    t_u_train,
    t_c_train,
    t_u_val,
    t_c_val,
):
    optimizer = optimizer_inst(model.parameters(), lr=lr)
    for i in range(epoch):
        t_p_train = model(t_u_train)
        loss_train = loss_fn(input=t_p_train, target=t_c_train)

        t_p_val = model(t_u_val)
        loss_val = loss_fn(input=t_p_val, target=t_c_val)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if (i + 1) % 500 == 0:
            print(
                f"Epoch {i+1}\nLoss train {loss_train.item():.4f}\nLoss val {loss_val.item():.4f}"
            )


# コンストラクタの引数を与える
# 引数は入力サイズ、’出力サイズ、バイアスでバイアスはデフォルトでTrue
# linear_model = nn.Linear(1, 1)
seq_model = nn.Sequential(
    nn.Linear(1, 20),
    nn.Tanh(),
    nn.Linear(20, 1),
)
lr = 1e-2
train(
    epoch=5000,
    model=seq_model,
    optimizer_inst=optim.Adam,
    loss_fn=nn.MSELoss(),  # 手書き実装不要.
    t_u_train=t_u_train,
    t_u_val=t_u_val,
    t_c_train=t_c_train,
    t_c_val=t_c_val,
)

# %%

from matplotlib import pyplot as plt

t_p_train = seq_model(t_u_train)
fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
fs = t_u_train.squeeze(1).numpy()
ps = t_p_train.squeeze(1).detach().numpy()
cs = t_c_train.squeeze(1).numpy()
plt.plot(*list(zip(*sorted(list(zip(fs, ps))))))
plt.plot(*list(zip(*sorted(list(zip(fs, cs))))), "o")

# %%
