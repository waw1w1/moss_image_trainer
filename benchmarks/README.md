# PR #1：优化验证与 Linux/NVIDIA A/B

审查基线：`waw1w1/moss_image_trainer` PR #1，提交
`4f56a0e28a5873a4c9f806f5e422e4cd33cfa644`。完整审查见
[PR1_REVIEW.md](PR1_REVIEW.md)。默认 optimizer/pool 和原训练入口均保持不变。

## 安装与回归

在仓库已有 Linux/NVIDIA 环境安装测试依赖，不替换项目锁定的 CUDA/PyTorch：

```bash
uv sync --locked
uv pip install -r benchmarks/requirements.txt
uv run --no-sync pytest tests -q --junitxml=regressions.xml
```

CPU 协议/配置/续训测试无需模型权重或 Triton。使用独立 CPU 环境时可安装
`torch pytest safetensors psutil` 后直接 `python -m pytest tests -q`。
CPU 测试通过 AST 选取并执行生产函数，避免导入不相关的 Triton/LyCORIS；没有复制实现。
小模型 checkpoint 测试只替换 adapter 编解码接口，仍执行真实 checkpoint
保存、签名、完整性校验、游标校验和恢复函数。Windows 测试替换 POSIX fsync，
因此其通过结果不代表 Linux 文件持久性已验证。CUDA 测试自动跳过不支持的环境。

真实 CUDA 回归覆盖 eager 和 `compile(checkpoint(...))`、多次反向、
保留计算图、不同 Stream、共享存储别名、非连续视图、多图并存、并发 Python 线程。
生产配置要求 Linux + CUDA PyTorch + 依赖齐全，编译回归需要可用的 Triton/C++ 工具链。

## 四组对照

脚本固定比较普通 AdamW/pool 关、fused AdamW/pool 关、普通 AdamW/pool 开、
fused AdamW/pool 开。所有组均启用检查点卸载和编译，其他选项固定。
每组默认重复三次，随机打乱执行顺序，每次启动独立进程；所有组使用相同训练种子。

先运行不需要 MOSS 权重的压力负载：

```bash
uv run --no-sync python benchmarks/optimization_ab.py \
  --mode synthetic --output benchmark-runs/synthetic-01 \
  --warmup 5 --steps 30 --repeats 3
```

负载：多层 MLP，`torch.compile(checkpoint(...))` 外侧使用生产卸载函数，
BF16 autocast、FP32 可训练参数、梯度累积、样本数归一化、裁剪、optimizer/scheduler 更新。
它用于暴露 DMA/分配/优化器回归，结果不能当作 MOSS 训练收益。
可调整 `--width --depth --batch --tokens --accumulation` 改变工作负载。

用真实 MOSS 训练验证最终收益：

```bash
# 先按原训练器流程准备模型和 feature cache。
uv run --no-sync python benchmarks/optimization_ab.py \
  --mode moss --config configs/lora.toml \
  --output benchmark-runs/moss-lora-01 \
  --warmup 10 --steps 60 --repeats 3
```

源配置必须启用 `compile_blocks` 和 `checkpoint_offload`。从仓库根目录运行，
模型、数据、cache 路径沿用配置。LoKr 用对应配置再跑一次。
脚本只在内存中替换 optimizer、pool、输出目录、总更新数、保存间隔及 seed，
关闭预览/TensorBoard；实际生效配置写入每组的 `effective-config.json`。
它不会覆盖源 TOML 或原有训练输出，输出目录必须尚不存在。
`--accumulation/--batch/--tokens/--width/--depth` 仅影响 synthetic；MOSS 使用源配置。
`--device` 为 CUDA 可见设备序号；与原训练器一致，选择其他物理 GPU 请使用
`CUDA_VISIBLE_DEVICES=<物理序号>` 并保留 `--device 0`。

## 指标与数值验收

- 吞吐：实际图片数 / 测量窗口耗时；step 指一次完整 optimizer update，包含梯度累积。
- 单步耗时：均值、中位数、P95，以及每步原始秒数；CUDA 同步后计时。
- 显存：预热后 `max_memory_allocated`、`max_memory_reserved`，单位 bytes。
- 内存：50ms 采样 worker 与 DataLoader 子进程 RSS、VMS、可用时的 USS 峰值。
  RSS 求和会重复计算共享页；USS 表示私有页。pinned RAM 包含在 RSS 中，未声称独立测得。
- 数值：每步 loss、最终全部参数/adapter 权重对照；只改 pool 的组要求完全一致，
  fused 组默认 `rtol=1e-3, atol=1e-5`，可用参数调整，超限即失败。
- synthetic 保存真实 optimizer/scheduler/model/RNG 文件后恢复并重放后半程，
  验证最终参数、optimizer、scheduler 与 loss；不等同于 MOSS adapter 续训。
- MOSS 每组执行连续训练与从中点重新启动的训练，比较最终 adapter、
  optimizer/scheduler、RNG、step/epoch/batch 游标与后半程 loss。
  同时验证普通 resume 和 `new_stage` 都拒绝 optimizer/pool 切换。

MOSS 借用训练器已有可选 telemetry 接口，在回调处同步并计时，不改原训练循环。
舍弃预热和第一次无法配对的更新，排除前一步保存 checkpoint 的计时间隔。
显存/内存峰值覆盖预热后的连续训练及其保存过程；内存采样可能漏掉短于 50ms 的尖峰。
synthetic 保存中点文件位于计时间隔之外，但进程内存峰值可能包含序列化开销。
预热窗口不会自动判断所有桶是否已编译；多尺寸数据请增加预热并检查日志，
同时报告实际桶分布，避免把冷编译当成稳态差异。

输出包括 `plan.json`、每组环境/源文件哈希/日志、原始 `result.json`、
`final.pt` 和总 `summary.json`。汇总含每组吞吐中位数、最小值、最大值和内存峰值；
原始结果保留耗时分布及每次测量。失败组保留日志且总命令非零退出，数值失败不会被隐藏。
压力负载与 MOSS 的结果必须分开报告。

比较前固定驱动、PyTorch、依赖、数据/cache、适配器、分辨率/文本长度、累积和裁剪。
避免其他 GPU 任务；保留温度/功耗/时钟条件。若收益落在三次重复波动范围内，
结论应是“当前负载无明确收益”。事件开销、pinned allocator 自身缓存和 optimizer
占比都可能让优化无收益或变慢。

## 本次实际执行范围

当前宿主为 Windows，GPU 为 RTX 5090 D，安装的 PyTorch 为 `2.7.1+cpu`；
`torch.cuda.is_available()` 为 false，WSL 访问被环境拒绝。
因此本次只执行 CPU 回归、语法检查、脚本入口和无 CUDA 时的拒绝行为，
没有 Linux/CUDA DMA、编译训练或 MOSS 性能数据。
详见审查报告中的测试结果与限制，不提供假设性的加速百分比。
