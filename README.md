# MOSS Image Trainer

MOSS-Image-Base 的单 GPU LoRA / LyCORIS 训练器，支持 HF 模型目录和 ComfyUI 分组件权重。使用 uv 管理依赖，按 kohya 风格分桶，冻结文本编码器和 VAE 并预缓存特征。

## 安装

需要 Linux x86_64、glibc 2.28 或更新版本、支持 BF16 的 NVIDIA GPU、兼容 CUDA 13.2 的驱动，以及 C/C++ 编译器。Ubuntu/Debian 可通过 `sudo apt-get install build-essential` 安装编译器。

在项目根目录执行：

```bash
bash setup.sh
```

脚本会安装缺失的 uv，使用 uv 管理的 Python 3.12 创建 `.venv`，按 `uv.lock` 安装依赖并执行环境自检。缺少 uv 时需要 curl 或 wget。CUDA 运行库和 Triton 随依赖安装，宿主机驱动须预先可用。安装成功时输出 `"ready": true`，不会启动训练。

单独检查环境：

```bash
.venv/bin/moss-trainer doctor
```

依赖版本由锁文件固定，包括 PyTorch 2.14.1+cu132、Triton 3.8.0、Diffusers 0.41.0、Transformers 5.18.0 和 LyCORIS 4.0.0。注意力使用 PyTorch 自带的 FlashAttention。

## 模型与数据

下载 [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) 到 `models/MOSS-Image-Base/`，保留 transformer、text_encoder、tokenizer、vae、scheduler 和官方 pipeline 文件。模型与数据不包含在本仓库中。

图片与同名 UTF-8 标注放在数据目录，可包含子目录：

```text
datasets/train/
  image001.png
  image001.txt
  image002.jpg
  image002.txt
```

支持 PNG、JPEG、WebP、BMP、TIFF。空标注合法，缺失标注或超长提示词会报错。默认使用中心裁切与 EXIF 方向校正；`bucket_no_upscale=true` 时小图不放大，选桶忽略 min/max_bucket_reso。

编辑 `configs/lokr.toml` 或 `configs/lora.toml`。各参数上方均有中文注释，路径相对于运行命令的工作目录：

| 分组 | 内容 |
| --- | --- |
| `[paths]` | 模型、数据集、缓存与输出路径 |
| `[dataset]` / `[cache]` | 分辨率、分桶、批大小、重复次数与文本长度 |
| `[network]` | 适配器算法、维度、训练目标 |
| `[training]` | epoch、学习率、梯度累积、保存间隔与 TensorBoard |
| `[optimization]` | SITU / RMSNorm 融合、编译与 CPU 检查点卸载 |
| `[sample]` | 预览提示词、CFG、分辨率、调度器、采样器与间隔 |

HF 格式使用完整 `model_dir`。使用 ComfyUI 权重时，在 `[paths]` 中配置：

```toml
# 使用分组件权重。
model_format = "comfyui"
# 官方配置、tokenizer 和 pipeline 所在目录；此模式无需其中的 HF 权重。
model_dir = "models/MOSS-Image-Base"
# 未量化的 DiT 权重。
model = "models/diffusion_models/moss_image_base_bf16.safetensors"
# 匹配的 Qwen3.5-9B 文本编码器。
te = "models/text_encoders/qwen3.5_9b_moss_image_base_bf16.safetensors"
# VAE 权重。
vae = "models/vae/moss_image_vae_fp32.safetensors"
```

支持 FP32/BF16/FP16 Safetensors，加载时检查键名、形状与 dtype。不支持 FP8、INT8、NVFP4、GPTQ 或 GGUF 量化权重。分组件权重仍需要官方配置和 tokenizer。

## 训练

以下以 LoKr 为例，LoRA 改用 `configs/lora.toml`。先填写模型、数据和输出路径，以及总 epoch 与保存间隔。

```bash
# 查看模型结构与适配目标，不加载大权重。
.venv/bin/moss-trainer inspect --config configs/lokr.toml
# 查看分桶和每轮更新次数。
.venv/bin/moss-trainer buckets --config configs/lokr.toml
# 缓存冻结组件的特征。
.venv/bin/moss-trainer cache --config configs/lokr.toml
# 检查缓存完整性。
.venv/bin/moss-trainer check-cache --config configs/lokr.toml
# 启动训练。
.venv/bin/moss-trainer train --config configs/lokr.toml
```

训练使用 BF16 DiT、FP32 适配器和优化器状态，损失为 rectified-flow 速度 MSE。Qwen 与 VAE 只参与缓存，训练过程中的预览解码会临时加载 VAE。

`max_epochs` 控制训练轮数，`save_every_epochs` 控制保存间隔，最后一轮始终保存。1 epoch 遍历含重复次数的全部分桶批次；`warmup_ratio` 按一个 epoch 的比例指定预热：0.3 表示首轮前 30%，乘以每轮优化器更新次数后向上取整，只预热一次。满批有效批大小为 `batch_size × gradient_accumulation_steps`，桶尾与轮尾按实际图片数归一化。

优化器通过 `training.optimizer` 选择 `adamw`（默认）、`adamw_fused`、`adam`、`sgd`（momentum=0.9）或 `rmsprop`。`adamw_fused` 使用原生 PyTorch `AdamW(fused=True)`，保持现有梯度累积、当步全局裁剪与更新顺序，减少优化器的逐参数执行开销；浮点结果不保证与普通 AdamW 逐位一致，实际收益取决于优化器耗时占比。

学习率调度通过 `training.scheduler` 选择 `constant`、`linear`、`cosine`（默认）或 `cosine_with_restarts`；全部先执行一次预热。`cosine_restarts` 是预热后的额外重启次数，仅最后一种调度生效，0 表示不重启。保留旧 `warmup_steps` 配置兼容，未指定调度器的旧文件继续采用预热后恒定学习率；新旧预热字段不能同时填写。

LoKr 示例默认 `full_matrix=true`、`train_norm=true`、`factor=8`，训练完整 Kronecker 因子和 RMSNorm 增量。LoRA 示例默认 `train_text_projection=true`，同时适配 DiT 内部文本投影和两层文本细化；Qwen 保持冻结。

支持 lora/locon、loha、lokr、dylora、glora、ia3、diag-oft、boft 和 full。部分算法需关闭编译；`full` 表示所选层的全参数微调，与 LoKr 的 `full_matrix` 不同。

## 显存与编译

两个示例默认开启双融合、全部梯度检查点、编译与 CPU 检查点卸载。卸载的是激活，不是模型权重；会增加主存占用和数据传输。首次运行或遇到新尺寸时需要编译。

可选 `[optimization].checkpoint_offload_pool=true` 复用已释放检查点的连续输入 pinned CPU 缓冲，要求同时启用 `compile_blocks` 与 `checkpoint_offload`，默认关闭；非连续输入沿用原卸载路径。每个 block 最多保留一个空闲缓冲，只有形状、步幅和 dtype 相同才复用；图释放前不归还，复用前等待该缓冲的复制事件。它减少缓冲申请次数，不减少卸载字节或传输量，也不直接节省显存；PyTorch 本身已有 pinned allocator 缓存，新增事件也有开销，因此吞吐收益需要在目标 GPU 上对照验证。

回归测试与四组 Linux/NVIDIA A/B（压力负载、真实 MOSS 训练及断点重放）见 [benchmarks/README.md](benchmarks/README.md)。

这两个可选项参考 KohakuMossTrainer 报告中的优化方向，使用现有卸载路径和 PyTorch 公共 API 实现。先在新训练中单独比较，再组合启用。普通 AdamW 与 `adamw_fused`、不同 pool 设置不能互相切换后直接续训；`--new-stage` 也保留这些执行设置。未启用新选项时保留旧检查点签名。

`compile_blocks=true` 要求：

- `fused_situ=true`、`fused_rmsnorm=true`、`gradient_checkpointing=true`。
- `checkpoint_policy="full"`、`checkpoint_blocks=-1`。
- 适配器 dropout 与权重分解关闭。
- LoRA/LoCon：`dim=alpha=16`。
- LoKr：`factor=8`，关闭 `decompose_both` / `use_tucker`；低秩模式使用 `dim=alpha=16`，完整因子模式使用 `full_matrix=true` 或足够大的 dim。

冻结文本投影且 batch 为 1 或使用 sequential 模式时，会预缓存文本细化结果。`batch_mode="packed"` 在 batch 大于 1 时使用变长并行注意力；不同计算形状会产生 BF16 舍入差异。增加 batch 或降低检查点数量前，需确认实际训练显存。

## 预览与 TensorBoard

`[sample].every_epochs` 控制预览间隔，0 关闭。支持 moss/linear 调度和 euler/heun 采样，固定 50 个采样区间，噪声种子取 `training.seed`。图片保存到 `output_dir/samples/epoch-XXXXXX.png`。修改正负提示词后重新运行 `cache`，可补齐文本特征并复用图片缓存。

`[training].tensorboard=true` 记录 loss、学习率、梯度范数、epoch、耗时、吞吐和预览图。日志写入在独立进程中进行，失败会停用日志，队列满时丢弃记录，不阻塞训练。

```bash
.venv/bin/tensorboard --logdir outputs --host 127.0.0.1 --port 6006
```

浏览器访问 `http://127.0.0.1:6006`。

## 续训与导出

检查点位于 `outputs/<algo>/epoch-XXXXXX/`，包含适配器、优化器与随机状态、配置和完整性清单。

```bash
.venv/bin/moss-trainer train --config configs/lokr.toml \
  --resume outputs/lokr/epoch-000010
```

恒定学习率续训可增加总 epoch；下降型调度依赖总训练长度，续训须保持原总 epoch/步数。可调整保存、预览和日志设置；模型、数据、适配器结构与执行选项需保持一致。使用最新完整检查点，或指定新的空输出目录。

LoRA/LoCon/LoKr 的 `adapter.safetensors` 使用 ComfyUI 的 `lora_unet_...` 键名，LoKr Norm 增量使用 `.w_norm`。将文件放入 ComfyUI 的 `models/loras/`，用标准 LoRA 加载节点接到匹配的 MOSS MODEL。其他 LyCORIS 算法使用原生导出格式。

参考：[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)、[LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)、[kohya 分桶](https://github.com/kohya-ss/sd-scripts)、[uv](https://docs.astral.sh/uv/)。

VAE image caches use the Flux2 posterior mean, not posterior sampling. Cache format v2 records vae_encoding=posterior_mean; v1 image caches and checkpoints must not be reused for corrected training. Rebuild image caches and start a new run from the base model.

To continue a completed run with its adapter, optimizer moments/step counters and RNG but a fresh learning-rate schedule, use train --resume <completed-checkpoint> --new-stage --config <new-config>. Use a separate empty output directory. New-stage epoch/step counters start at zero; max_epochs is the additional duration. Only LR/scheduler/warmup/duration may differ; model/data/optimizer settings remain strict. Resume a checkpoint produced in the new stage normally, without --new-stage.
