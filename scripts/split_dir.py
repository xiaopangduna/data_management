#!/usr/bin/env python3
import os, shutil
from pathlib import Path

src = Path("/home/huangwenhua/project/dataset/tmp/to_guangfeng")
n_groups = 6

# 找所有 jpg 的基名
bases = sorted(p.stem for p in src.glob("*.jpg"))
print(f"配对组数: {len(bases)}")

per = (len(bases) + n_groups - 1) // n_groups
print(f"每组约: {per}")

for i in range(n_groups):
    d = src / f"group_{i:02d}"
    d.mkdir(exist_ok=True)
    for b in bases[i*per:(i+1)*per]:
        for ext in (".jpg", ".json"):
            f = src / (b + ext)
            if f.exists():
                shutil.move(str(f), str(d / f.name))
    print(f"group_{i:02d}: {len(list(d.iterdir()))} 文件")