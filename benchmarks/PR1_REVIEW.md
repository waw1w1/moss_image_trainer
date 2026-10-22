# PR #1 深入审查

基线：PR `https://github.com/waw1w1/moss_image_trainer/pull/1`，head
`4f56a0e28a5873a4c9f806f5e422e4cd33cfa644`。审查日期：2026-10-22。

## 结论

fused AdamW 的接入没有改变现有梯度累积、实际样本数归一化、全局裁剪、
非有限梯度检查、optimizer/scheduler 顺序。普通 AdamW 与 pool 关闭仍是默认值。
没有发现应当替换现有默认路径的证据，也没有实测吞吐收益。

池化实现的正常路径具有正确的依赖关系：D2H 后记录 ready，H2D 前等待 ready，
每个 H2D 后记录 reader，保存对象释放后归还 host，下一次 D2H 前等待所有依赖。
本次发现并修复异常后的不完整依赖归还，以及 cache 的 check/pop 竞态。
这些结论分别来自静态审查、可重复故障注入和测试；真实 CUDA DMA 仍未验证。

## 已修复问题

### P2：H2D 已提交但 reader Event 记录失败后，host 仍被归还

位置：`compilation.py::_offload_checkpoint_input.unpack` 与 `_PooledCheckpointInput` finalizer。

原实现先 `restored.copy_(host, non_blocking=True)`，再 `stream.record_event()`。
若 copy 或 Event 记录抛异常，saved object 最终释放时仍会把 host 连同缺少 reader
的 events 列表放回手动池。后续复用无法依据这个列表等待已提交的 H2D。
在训练层捕获异常、释放失败图、再次调用同一 block 的场景中存在提前覆盖风险。
CUDA 错误不一定可恢复；本修复也不承诺故障设备可继续训练。

已通过 CPU 协议故障注入在原实现上复现：记录失败后，下次 pack 取得同一块 host。
这证明依赖协议缺口，不是实际 GPU 上已观测到错误梯度。

最小修复：每个 lease 附带独立的可复用标记；H2D copy/Event 记录异常即永久禁止
该 lease 进入手动池。保持异常传播，让 PyTorch pinned allocator 管理其正常退役；
没有自行释放原始内存、全设备同步或吞掉 CUDA 错误。正常成功路径不增加 Event。

### P2：cache 的存在性检查和 pop 之间存在并发窗口

原代码 `if pooled and cache: cache.pop()` 是两个 Python 操作。
共享 block 的其他前向或 saved-object finalizer 可在另一 Python 线程消费/替换槽位，
导致 `IndexError`。GIL 不提供跨两个操作的原子性。普通单线程训练一般不会触发。

修复为直接 `pop()` 并捕获空槽 `IndexError`，回退到原分配路径。
单槽写入仍通过 list slice 完成；不会给两个图借出同一 lease。
CPU 测试模拟检查后槽位被消费；真实 CUDA 并发压力测试已编写但本次未执行。

## CUDA Event / 生命周期 / 重复反向的判断

- finalizer 没有捕获自身，只持有 cache、host、事件列表及独立标记，避免自身引用环。
  图尚存活时 lease 不归还；不能在第一次 unpack 后立即归还，因为 retain_graph
  或同一保存值的多次访问可能再次解包。
- 所有 H2D reader 都必须进入 events；只等待最近一次 ready 不够。
  不同 Stream 的 reader 之间可能无顺序关系，不能只保留“最后一个 Event”。
- 复用前使用 `stream.wait_event`，不需要 CPU 阻塞。连续张量 D2H 在该 Stream
  上排队；pool 对非连续张量禁用，保留原布局路径。
- CUDA input 使用 `record_stream` 保护源 CUDA allocation 的 allocator 生命周期。
  host 由 lease 或空闲槽持有。恢复 tensor 在当前解包 Stream 分配并使用。
- 与普通 CUDA 张量操作一致，调用方仍应保证输入生产 Stream 到前向 Stream 的依赖。
  池化不能修复调用方缺失的 Stream 同步，也没有声称支持 CUDA Graph capture。
- 多次保留图反向会积累多个 reader Event，直到图释放；这不是普通单次反向的
  跨 step 无界增长，但长期 retain_graph 场景会增加内存/事件开销。
- 放弃前向图、前向抛异常或形状变化时，已持有的 host 只在 lease 释放后进入槽位。
  shape/stride/dtype 不匹配则放弃该空闲槽并走原分配；每 block 最多一个空闲 host，
  并不表示所有活跃图总共只占一块 host。
- D2H pack 中途异常尚未创建 lease，不归还手动池。H2D 异常现已明确禁止归还。
  fatal CUDA error 后不应把“异常可捕获”理解为“上下文仍可安全恢复”。

## fused AdamW 与续训

`AdamW(fused=True, foreach=False)` 使用相同超参数与现有 FP32 adapter 参数。
CPU 测试以同梯度比较普通/fused 更新，并覆盖不满批累积后的归一化和裁剪、
磁盘 optimizer/scheduler 恢复。CUDA 数值和运行能力由 GPU 测试与四组基准验证。
没有把浮点融合结果假设为逐位相同。

签名包含 optimizer 以及开启的 pool；未开启 pool 时移除该字段，保留旧默认签名。
`new_stage` 仅允许改变 LR 计划，未放松 optimizer/pool 限制。
测试覆盖四种组合、同设置正常续训、切换拒绝、完成 epoch 后新阶段保留 moments/steps。
这里的 pool 兼容性限制是保守策略：pool 不改变 optimizer state 的数学结构，
但本次遵守既有策略，没有擅自放宽。

## 测试与性能证据

见 `tests/`、`benchmarks/optimization_ab.py` 和相邻 README。
CPU 协议测试模拟 DMA API，不作为 CUDA 硬件正确性的证据。
checkpoint 测试使用小 Linear + 真实 safetensors，仅替换 LyCORIS codec 接口；
完整 MOSS adapter 续训验证属于 `--mode moss`，本次未执行。
Windows smoke 会绕过 POSIX fsync，Linux 测试不会绕过。

本次执行结果：`21 passed, 8 skipped`，跳过项为 6 个真实 CUDA 卸载测试和
2 个 CUDA optimizer 磁盘恢复测试；另完成 `compileall`、`git diff --check`、
基准 CLI 帮助及无 CUDA 拒绝运行检查。原实现的 Event 失败回归先失败，修复后通过。
本次环境为 Windows/Python 3.11/
PyTorch 2.7.1+cpu，低于项目锁定的 Linux/Python 3.12/CUDA PyTorch 版本。
真实 DMA、Inductor/Triton 编译和锁定依赖上的最终验收仍需目标机器执行。

基准会运行 2×2 组合、预热、独立进程及多次重复，记录吞吐、耗时分布、
显存与 RAM 峰值，并对比 loss/最终参数和磁盘续训后的行为。
synthetic 是压力负载，MOSS 模式调用实际训练器，不能混用两者的性能结论。
当前没有 NVIDIA CUDA 实测数字，因此没有“提升 x%”的结论。
