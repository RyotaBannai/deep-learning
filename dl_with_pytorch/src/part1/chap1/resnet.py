"""
ResNet で入力画像がなんであるか、1000 種類ほどのクラスへ分類する
"""

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import models, transforms

torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)

# %%
resnet = models.resnet101(pretrained=True)
# 訓練済みのネットワークを使うときは、前処理関数（transforms）をネットワークの訓練時の前処理と同じ内容にする
preprocess = transforms.Compose(
    [
        # 入力画像を 256x256 へ拡大or縮小
        transforms.Resize(256),
        # 画像中央を中心に224x224 に切り取り
        transforms.CenterCrop(224),
        # テンソルへ変換
        transforms.ToTensor(),
        # RGB 成分を予め定義された平均と標準偏差で正規化（ここでは標準化）
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ??
    ]
)
img = Image.open("../../../data/p1ch2/bobby.jpg")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)  # ??

# %%

# 推論（inference）時、ネットワークをeval モードに切り替える
# 切り替えないと、バッチ正規化やドロップアウトなどの処理が混じってしまう
resnet.eval()
out = resnet(batch_t)

# %%
# 推論結果からラベルの最もクラスの推論確率が高いindex をlist から取り出す.
with open("../../../data/p1ch2/imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# 推論結果
_, index = torch.max(out, 1)
percentage = nn.functional.softmax(out, dim=1)[0] * 100  # 出力を[0,1] に正規化
print(labels[index[0]], percentage[index[0]].item())

# 全てのクラス確率を降順で出力.
_, indices = torch.sort(out, descending=True)
print([(labels[i], percentage[i].item()) for i in indices[0][:5]])

# %%
