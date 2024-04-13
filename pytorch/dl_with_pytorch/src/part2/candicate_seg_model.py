import math
import random
import sys
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn as nn

root_dir: str = "../../"
ROOT_DIR = Path(root_dir).resolve()
sys.path.insert(0, str(ROOT_DIR))

from src.utils.logconf import logging  # noqa: E402
from src.utils.unet import UNet

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        # kwargs はコンストラクタに渡される全てのキーワード引数を含む辞書
        super().__init__()

        # BatchNorm2d は入力のチャンネル数を必要とする
        # その情報をキーワード引数から取り出す
        self.input_batchnorm = nn.BatchNorm2d(kwargs["in_channels"])
        # U-Netの取り込み部分はこれだけだが、ほとんどの処理はここで行われる
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

        # 第11章と同じように独自の重み初期化を行う
        self._init_weights()

    def _init_weights(self):
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu", a=0)
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

        # nn.init.constant_(self.unet.last.bias, -4)
        # nn.init.constant_(self.unet.last.bias, 4)

    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.unet(bn_output)
        fn_output = self.final(un_output)
        return fn_output


class SegmentationAugmentation(nn.Module):
    def __init__(self, flip=None, offset=None, scale=None, rotate=None, noise=None):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label_g):
        transform_t = self._build2dTransformMatrix()
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        transform_t = transform_t.to(input_g.device, torch.float32)
        affine_t = F.affine_grid(transform_t[:, :2], input_g.size(), align_corners=False)

        augmented_input_g = F.grid_sample(
            input_g, affine_t, padding_mode="border", align_corners=False
        )
        augmented_label_g = F.grid_sample(
            label_g.to(torch.float32),
            affine_t,
            padding_mode="border",
            align_corners=False,
        )

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise

            augmented_input_g += noise_t

        return augmented_input_g, augmented_label_g > 0.5

    def _build2dTransformMatrix(self):
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i, i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = random.random() * 2 - 1
                transform_t[2, i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = random.random() * 2 - 1
                transform_t[i, i] *= 1.0 + scale_float * random_float

        if self.rotate:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])

            transform_t @= rotation_t

        return transform_t
