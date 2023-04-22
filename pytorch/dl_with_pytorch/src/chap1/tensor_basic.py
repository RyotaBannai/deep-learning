# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import models, transforms

# %%
# weights = torch.tensor([0.1, 0.2, 0.3])
# unsqueezed_weights = weights.unsqueeze(-1).unsqueeze_(-1)
img_t = torch.arange(3 * 5 * 5, dtype=torch.float32).reshape(3, 5, 5)
batch_t = torch.arange(2 * 3 * 5 * 5, dtype=torch.float32).reshape(2, 3, 5, 5)
# img_weights = img_t * unsqueezed_weights
# batch_t = batch_t * unsqueezed_weights
# %%
# それぞれの次元に名前をつけるとわかりやすい
# 特定の形状へbroadcast する時に便利（align_as を参照）
# sum 関数を読んでgrey 化する時もnames を指定することができる
weights_named = torch.tensor([0.1, 0.2, 0.3], names=["channels"])
img_named = img_t.refine_names(..., "channels", "rows", "columns")
batch_named = img_t.refine_names(..., "channels", "rows", "columns")
weights_alighed = weights_named.align_as(img_named)

# %%
points = torch.arange(2 * 3, dtype=torch.float32).reshape(3, 2)
# この時点で、システムRAMではなく、GPU 上のRAM にデータが保存されている.
# GPU のローカル上に保存されたため、テンソルの数値演算を実行する際に、高速化が期待される.
# CPU に戻すときは、to(device="cpu")
points_gpu = points.to(device="mps")
