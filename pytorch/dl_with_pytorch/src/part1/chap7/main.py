# %%
import torch
from torch import nn, optim
from torchvision import datasets, transforms

# https://developer.apple.com/metal/pytorch/
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")

data_path = "../../../data/data-unversioned/p1ch7/"
to_tensor = transforms.ToTensor()
to_normalize = transforms.Normalize(
    (0.4915, 0.4823, 0.4468),
    (0.2470, 0.2435, 0.2616),
)


# どのデータセットを使っても、torch.utils.data.Dataset のサブクラスとして返される
cifar10 = datasets.CIFAR10(
    data_path,
    train=True,
    download=True,
    transform=transforms.Compose([to_tensor, to_normalize]),
)
cifar10_val = datasets.CIFAR10(
    data_path,
    train=False,
    download=True,
    transform=transforms.Compose([to_tensor, to_normalize]),
)

label_map = {0: 0, 2: 1}
class_names = ["airplace", "bird"]
cifar2 = [(img, label_map[lable]) for img, lable in cifar10 if lable in [0, 2]]
cifar2_val = [(img, label_map[lable]) for img, lable in cifar10_val if lable in [0, 2]]


# %%


def train(
    epoch: int,
    model,
    optimizer_inst,
    loss_fn,
    t_u_train,
    t_u_val,
):
    train_loader = torch.utils.data.DataLoader(t_u_train, batch_size=64, shuffle=True)
    optimizer = optimizer_inst(model.parameters(), lr=lr)
    for i in range(epoch):
        for imgs, lables in train_loader:
            batch_size = imgs.shape[0]
            outs = model(imgs.view(batch_size, -1))
            loss_train = loss_fn(input=outs, target=lables)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        print(f"Epoch {i}, Loss train {loss_train.item():.4f}")


seq_model = nn.Sequential(
    nn.Linear(3072, 1024),
    nn.Tanh(),
    nn.Linear(1024, 512),
    nn.Tanh(),
    nn.Linear(512, 128),
    nn.Tanh(),
    nn.Linear(128, 2),
)
lr = 1e-2
train(
    epoch=30,
    model=seq_model,
    optimizer_inst=optim.SGD,
    loss_fn=nn.CrossEntropyLoss(),
    t_u_train=cifar2,
    t_u_val=cifar2_val,
)

# %%

# 正解率チェック
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)
correct = 0
total = 0
with torch.no_grad():
    for imgs, lables in val_loader:
        batch_size = imgs.shape[0]
        outs = seq_model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outs, dim=1)
        total += lables.shape[0]
        correct += int((predicted == lables).sum())

print(f"Accuracy: {correct / total}")
# %%
