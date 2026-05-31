# OTP 修复报告

## 1. 问题背景

### 1.1 环境信息
- **模型**: DeepSeek-V4-Flash-w8a8-mtp
- **硬件**: Ascend 910B3 × 8 (64GB HBM per card)
- **配置**: DP=8, TP=1, EP=8, max_model_len=133072
- **vLLM 版本**: v0.20.2 (pr-9737 分支)
- **vllm-ascend**: pr-9737 分支

### 1.2 问题现象

**问题 1: KV cache OOM**
```
ValueError: To serve at least one request with the model's max seq len (133072),
(5.11 GiB KV cache is needed, which is larger than the available KV cache memory (4.01 GiB).
```
- 修复前 KV cache: 4.01 GiB
- 所需 KV cache: 5.11 GiB

**问题 2: 推理时 HcclReduceScatter 崩溃**
```
Invalid_Argument(EI0005): The arguments for collective communication are inconsistent between ranks,
ccl_op HcomReduceScatter, group group_name_1, parameter count, local end 8192, remote end 24576.
```

**问题 3: 推理时 HcclAllGather 崩溃**
```
Communication_Error_Get_Socket(EI0006): Getting socket times out.
The possible cause is that the behaviors of different ranks are inconsistent.
RuntimeError: The Inner error is reported as above. The process exits for this inner error,
and the current working operator name is HcclAllGather.
```

## 2. 根因分析

### 2.1 KV cache OOM 根因
- wo_b 权重（60 层 × ~50 MiB ≈ 3 GiB）在 OTP 模式下未被 vLLM 的 `process_weights_after_loading` 遍历到
- wo_b 的 `quant_method` 不是 `AscendLinearMethod`，不在 vLLM 遍历范围内
- 导致 3 GiB 显存未释放

### 2.2 HcclReduceScatter 崩溃根因
- OTP 模式下 `tp_size > 1`（OTP group 跨 8 个 DP rank），但 global TP=1
- 不同 DP rank 在 `_dummy_run` 和推理时 batch size 不同
- `reduce_scatter` 要求所有 rank 贡献相等大小的数据，batch size 不一致导致参数不匹配

### 2.3 HcclAllGather 崩溃根因
- MTP draft model 的 `compute_logits` 调用 `LogitsProcessor._gather_logits`
- `_gather_logits` 调用 `tensor_model_parallel_all_gather`，要求所有 rank tensor shape 一致
- 不同 DP rank batch size 不同导致 all_gather 死锁

## 3. 尝试解决过程

### 3.1 KV cache OOM 修复（成功）
**方案**: 在 vLLM 的 `process_weights_after_loading` 中 monkey-patch，手动遍历 wo_b 做 row-slicing
- 文件: `patch_process_weights_after_loading.py`
- 效果: KV cache 4.01 → 6.84 GiB

### 3.2 HcclReduceScatter 修复
**尝试 1（失败）**: 在 `eager_apply_impl` 中加 `tp_size==1` 捷径用 all_reduce
- 失败原因: `tp_size` 实际为 8（OTP group world_size），不是 1

**尝试 2（部分成功）**: 在 `eager_apply_impl` 和 `_maybe_pad_and_reduce_impl` 中用 padding + all_reduce
- 失败原因: 使用了 `get_tp_group()`（world_size=1），all_reduce 是 no-op

**尝试 3（待验证）**: 使用 `get_otp_group()` 代替 `get_tp_group()`
- commit `737fce5f`: 改用 OTP group 进行 all_reduce

### 3.3 HcclAllGather 修复
**方案**: Monkey-patch `LogitsProcessor._gather_logits` + 跳过 `_maybe_all_gather_and_maybe_unpad_impl`
- 文件: `patch_process_weights_after_loading.py`, `register_custom_ops.py`
- 原理: OTP 模式下每个 rank 已有完整 logits，无需 all_gather

### 3.4 Embedding TP 修复（成功）
**方案**: OTP 模式下跳过 `vocab_parallel_embedding.py` 的 embedding TP 路径
- 文件: `vocab_parallel_embedding.py`
- 原理: fallback 到 `_forward_origin`，不走 all_gather/reduce_scatter

### 3.5 LM Head TP 修复（成功）
**方案**: OTP 模式下跳过 `_get_logits_lmheadtp` 路径
- 文件: `vocab_parallel_embedding.py`
- 原理: 直接计算 logits 而不做 TP gathering

## 4. 当前状态

### 4.1 已成功推送的修改
分支: `https://github.com/li-lizhe/vllm-ascend` → `otp-fix`
Commit 列表:
- `e5e6efe6`: 跳过 lm_head TP 路径 + 修复 vocab embedding
- `1d173457`: padding + all_reduce 修复
- `93a374bd`: 跳过 lm_head TP gather
- `89d77c77`: `_get_logits` 跳过 TP gathering
- `7ef6da31`: LogitsProcessor monkey-patch + maybe_all_gather 跳过
- `737fce5f`: **关键修复** — all_reduce 使用 OTP group

### 4.2 当前阻塞
- **NPU 内存泄漏**: 前几次崩溃导致 NPU 上残留 ~57GB HBM 未释放
- `npu-smi reset` 在 VM/容器中被禁止
- 需要宿主机执行 NPU reset 或重启容器才能继续测试

### 4.3 已验证的效果
| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| KV cache | 4.01 GiB | 6.84 GiB |
| HcclReduceScatter | 崩溃 | 方案已就绪，待 NPU 清理后验证 |
| HcclAllGather | 崩溃 | 方案已就绪，待 NPU 清理后验证 |

## 5. 待验证方案

### 5.1 修复步骤（清理 NPU 后执行）
1. **宿主机执行 NPU reset**: `npu-smi set -t reset -i 0-7 -d 1`
2. **启动服务**: `cd /home/t00446321/script_flash && sh start.sh`
3. **验证启动**: `grep "Available KV cache" log.log` → 应 > 5.11 GiB
4. **验证推理**: `curl -X POST 'http://127.0.0.1:7000/v1/chat/completions' -H 'Content-Type: application/json' -d '{"model":"ds","messages":[{"role":"user","content":"介绍一下人工智能"}],"max_tokens":100}'`

### 5.2 预期结果
- KV cache: 6.84 GiB > 5.11 GiB ✓
- 无 HcclReduceScatter 崩溃
- 无 HcclAllGather 崩溃
- 返回有效 JSON 响应

## 6. 修改文件清单

| 文件 | 修改类型 | 修改内容 |
|------|----------|----------|
| `patch/platform/patch_process_weights_after_loading.py` | 新增 | wo_b row-slicing + LogitsProcessor monkey-patch |
| `patch/platform/__init__.py` | +1 行 | 注册 patch |
| `ops/linear_op.py` | 修改 | OTP eager path 用 padding + all_reduce |
| `ops/register_custom_ops.py` | 修改 | OTP 跳过 maybe_all_gather + all_reduce 用 OTP group |
| `ops/vocab_parallel_embedding.py` | 修改 | OTP 跳过 embedding TP 和 lm_head TP |
