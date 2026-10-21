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
