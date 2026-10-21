# MOSS Image Trainer

## 安装

Linux + NVIDIA GPU：

```bash
git clone https://github.com/waw1w1/moss_image_trainer.git
cd moss_image_trainer
sudo apt-get install -y build-essential curl
bash setup.sh
```

## 启动训练

1. 下载 [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/tree/main) 完整模型到 `models/MOSS-Image-Base/`。
2. 将图片和同名 `.txt` 标注放入 `datasets/train/`，例如 `001.png` 和 `001.txt`。
3. 编辑 `configs/lokr.toml`，设置模型、数据、输出路径及训练参数。
4. 在项目根目录执行：

```bash
.venv/bin/moss-trainer cache --config configs/lokr.toml
.venv/bin/moss-trainer train --config configs/lokr.toml
```

训练 LoRA 时，将上述配置文件替换为 `configs/lora.toml`。

在 `[training]` 中修改 `low_noise_ratio` 可设置低噪声（最终 `sigma < 0.5`）的目标采样比例，例如 `0.60` 表示 60%。示例默认 `0.4992021157051587`（约 49.92%），保持原有默认采样行为；实际有限样本比例会有随机波动。取值必须大于 0 且小于 1，仅支持 `timestep_sampling = "logit_normal"`，填写后覆盖 `logit_mean`，仍使用 `logit_std` 和 `timestep_shift` 换算有效均值。删除该项则沿用旧配置的均值行为；使用 `uniform` 时须删除该项。改变有效采样分布后需新开训练，普通续训和 `--new-stage` 均不允许切换分布；只调整此项不需要重建特征缓存。

训练使用普通执行路径，保留 SITU/RMSNorm 融合、梯度检查点和 CPU 卸载，不再使用 `torch.compile`。旧配置须删除 `compile_blocks`；此前开启编译保存的断点与当前执行设置不兼容，不能直接续训。


数据目录支持递归扫描，通过 `重复次数_名称` 文件夹设置每轮采样次数，例如：

```text
datasets/train/
├── 1_测试/                 # 每张图片每轮 1 次
├── 人物/
│   ├── 5_角色A/            # 每张图片每轮 5 次
│   └── 10_角色B/           # 每张图片每轮 10 次
└── 3_画风/
    ├── 普通子目录/         # 继承 3 次
    └── 2_另一画风/         # 使用 2 次，不与父目录相乘
```

每张图片使用最近的带数字前缀目录的次数，查找范围包含 `image_dir` 本身，不包含其上级目录；无前缀默认 1 次，0 次跳过，负数报错。图片仍须配同名 `.txt` 标注，不使用目录名补全标注。
全局 `num_repeats` 已移除，旧配置需删除此项。可用 `.venv/bin/moss-trainer buckets --config configs/lokr.toml` 查看原始图片数、重复后的每轮样本数与训练步数。
升级后请重新执行 `cache`；目录 repeat 会参与缓存与续训签名，旧版断点不能直接按原进度续训，改变 repeat 后也不能沿用原批次进度。


## Muon + AdamW 混合优化器

默认仍为 `optimizer = "adamw"`，其学习率、权重衰减与 PyTorch 默认 betas 保持原行为。
在 LoRA 或 LoKr 的 `[training]` 中设置 `optimizer = "muon_adamw"` 即可启用混合更新，具体参数和注释见两份示例配置。

混合模式使用 PyTorch 原生 Muon 更新 LoRA up/down 和 LoKr 二维因子（含分解因子），使用 AdamW 更新 Norm 增量、DoRA 缩放和标量；辅助参数不衰减。其他算法暂不支持，未知参数类型会报错而不会静默漏训。纯 LoRA 没有辅助可训练参数时只有 Muon 分支，这是正常行为。
两组使用相同的 `learning_rate`、`min_learning_rate` 和调度曲线；Muon 固定 `match_rms_adamw` 缩放。下限仍只作用于 cosine/cosine_with_restarts，预热规则不变。日志与 TensorBoard 分别记录实际存在的 muon/adamw 分支学习率（Muon 的形状缩放在优化器内部执行）。梯度累积完成后，所有适配器参数一起裁剪，再各更新一次。

默认 Muon momentum=0.95、Nesterov=true、5 次 Newton–Schulz 迭代；混合分支 AdamW betas=(0.9,0.95)、eps=1e-8。它对适配器因子执行更新，不等同于官方底模全参数预训练配方，不能保证更快或更好的 LoRA/LoKr 效果。可从当前学习率开始做固定数据和采样条件的对照。

完整断点保存两个分支的状态、参数名称/顺序和共同调度进度。相同配置可正常 `--resume`；完成的混合训练可用既有 `--new-stage` 机制更换学习率计划，保留两个分支的状态。切换优化器或改变混合超参数不允许通过这两种续训方式绕过检查；当前 CLI 没有仅加载适配器权重另起训练的入口，切换时须从底模新开训练。仅更换优化器不需要重建特征缓存。
未启用混合模式时，新字段不参与续训签名，已有 AdamW 断点保持兼容。
