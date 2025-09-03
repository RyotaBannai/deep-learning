# %%
import datetime

import torch
from torch import nn, optim
from torch.nn import functional as F
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
    optimizer,
    loss_fn,
    t_u_train,
):
    train_loader = torch.utils.data.DataLoader(t_u_train, batch_size=64, shuffle=True)
    for i in range(1, epoch + 1):
        loss_train = 0.0
        for imgs, lables in train_loader:
            imgs = imgs.to(device=mps_device)
            lables = lables.to(device=mps_device)
            outs = model(imgs)
            loss = loss_fn(input=outs, target=lables)
            l2_lamnda = 0.001
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lamnda * l2_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        if i == 1 or i % 10 == 0:
            print(
                "{} Epoch {}, Training loss {}".format(
                    datetime.datetime.now(),
                    i,
                    loss_train / len(train_loader),
                )
            )


# 正解率チェック
def validate(model, t_u_train, t_u_val):
    train_loader = torch.utils.data.DataLoader(t_u_train, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(t_u_val, batch_size=64, shuffle=True)
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lables in loader:
                imgs = imgs.to(device=mps_device)
                lables = lables.to(device=mps_device)
                outs = model(imgs)
                _, predicted = torch.max(outs, dim=1)
                total += lables.shape[0]
                correct += int((predicted == lables).sum())
            print("Accuracy {}: {:.2f}".format(name, correct / total))


class Net(nn.Module):
    def __init__(self):
        """
        ・nn.Sequencial で連結したnn サブモジュールをインスタンス化して、self で保持
        ・
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=8)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        """
        ・入力に対し順にサブモジュールを適用
        ・ここでsequencial を使っても良い
        ・
        """
        out = F.max_pool2d(torch.tanh(self.conv1_batchnorm(self.conv1(x))), 2)
        out = F.max_pool2d(torch.tanh(self.conv2_batchnorm(self.conv2(out))), 2)
        # バッチ内にいくつのサンプルが存在しているかはわからないため
        # view に-1 を指定
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


dnn = Net()
dnn = nn.DataParallel(dnn)
dnn = dnn.to(device=mps_device)
# numel_list = [p.numel() for p in model.parameters()]
# sum(numel_list), numel_list
dnn.train()
train(
    epoch=100,
    model=dnn,
    optimizer=optim.SGD(dnn.parameters(), lr=1e-2, momentum=0.99),
    loss_fn=nn.CrossEntropyLoss(),
    t_u_train=cifar2,
)

# %%
dnn.eval()
validate(model=dnn, t_u_train=cifar2, t_u_val=cifar2_val)

# %%
