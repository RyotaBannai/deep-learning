# %%
import imageio
import numpy as np
import torch

# %%
img_arr = imageio.imread("../../../data/p1ch2/bobby.jpg")
img = torch.from_numpy(img_arr)
out = img.permute(2, 0, 1)

# %%

wine_path = "../../../data/p1ch4/tabular-wine/winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
col_list = next(csv.reader(open(wine_path), delimiter=";"))
# この時点ですべてのテーブルデータの値は浮動小数店のtorch.Tensor になってる
wineq = torch.from_numpy(wineq_numpy)
data = wineq[:, :-1]  # 学習データ
data_normalized = (data - torch.mean(data, dim=0)) / torch.sqrt(torch.var(data, dim=0))
target = wineq[:, -1]  # 教師データ、ラベル、ターゲット
# target = target.long()  # スコアの整数値に変換（もとはfloat）

# %%
# 文字単位のonehot
# 各文字を１つずつを１つのテンソルに格納
# 単語（文字列）などではなく、整数128で表現できる１文字（ASCII）がテンソルとマッチする点に気を付ける
jane_path = "../../../data/p1ch4/jane-austen/1342-0.txt"
with open(jane_path, encoding="utf-8-sig") as f:
    txt = f.read()
lines = txt.split("\n")  # 行でsplit
line = lines[200]
line  # 一行分
letter_t = torch.zeros(len(line), 123)  # ASCII が最大128文字なので、128 を使用
print(letter_t.shape)

for i, letter in enumerate(line.lower().strip()):
    print(ord(letter))
    letter_index = ord(letter) if ord(letter) < 128 else 0  # <1>
    letter_t[i][letter_index] = 1

    # <1>
    # 文章中に　英文字以外の記号（二重引用符 " など）が使われている場合、
    # これはASCII ではないため、ここでは除外する
    # ただし、空白文字やカンマ, などはASCII であるため注意.
    # https://www.asciitable.com/
