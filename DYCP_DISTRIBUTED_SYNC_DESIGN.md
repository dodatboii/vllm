# Dynamic CP 分布式同步设计

## 1. 背景

### 1.1 目标

在保留 vLLM 原始 per-DP 独立进程架构的前提下，通过轻量级分布式协同协议实现 Dynamic Context Parallel（DYCP）：长序列分布到多个 DP rank 并行处理，短序列路由到单个 DP rank 正常执行。

### 1.2 核心约束

长序列走 CP 时，必须满足三个跨 DP 约束：

| 约束 | 说明 |
|------|------|
| KV cache 原子分配 | 所有 CP rank 都有足够 blocks 才能调度 |
| 同步执行 | CP 请求必须在所有 CP rank 的同一 step 执行 |
| 同步 Preemption | 任一 rank 需要 preempt 时，所有 rank 同步释放 |

---

## 2. 配置参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `dycp_size` | `ParallelConfig` | 1 | 每个 CP group 的 DP rank 数 |
| `long_request_threshold` | `SchedulerConfig` | 1024 | 长/短请求分界阈值（prefill token 数） |
| `num_cp_seqs` | `SchedulerConfig` | 0 | 最大并发 CP 请求数（用于 cudagraph key 预分配） |

`max_num_batched_tokens` 在 `EngineArgs.__post_init__` 中自动乘以 `dycp_size`。

---

## 3. 进程组

`initialize_model_parallel`（`vllm/distributed/parallel_state.py`）中构建 `_DYCP` 进程组：

```python
# all_ranks shape: (ExternalDP, DP, PP, PCP, TP)
group_ranks = all_ranks.transpose(1, 4)  # → (ExternalDP, TP, PP, PCP, DP)
cp_group_ranks = group_ranks.reshape(-1, dycp_size).unbind(0)
```

`transpose(1, 4)` 将 DP 维度移到最后，`reshape(-1, dycp_size)` 将每 `dycp_size` 个连续 DP rank 分为一个 CP group。DYCP 是在 DP 维度上分组，与 TP 无关，TP=1 时同样可用。

---

## 4. 整体架构

```
Frontend (DPLBAsyncMPClient)
  ├── 长请求 → 广播到所有 DP engine
  └── 短请求 → 负载均衡路由到单个 DP engine

 DP_0 EngineCoreProc                   DP_1 EngineCoreProc
  CPAwareScheduler    ←─ all-reduce ─→   CPAwareScheduler
(pending / active CP)                  (pending / active CP)
  MultiprocExecutor                      MultiprocExecutor
 Workers (TP group)     ←─ 待实现 ─→     Workers (TP group)
```

> **注**：Worker 间跨 DP rank 的 attention 通信尚未实现。`_DYCP` 进程组已初始化，`cp_rank` 和 `dycp_local_seq_lens` 已传入 attention backend，但 attention 计算中实际调用 `get_dycp_group()` 进行通信的部分仍待补充。DYCP 采用与 DCP 相同的 interleave 分片语义，可直接复用 DCP 的 attention 流程，详见第 6 节。

---

## 5. 组件详解

### 5.1 CPSyncProtocol（`vllm/v1/core/sched/cp_sync.py`）

每 `sync_interval`（默认 4）步执行一次两阶段提交：

**Phase 1 — Announce**
```python
self._announce_tensor[i] = 1  # 本 rank 已收到第 i 个 pending 请求
all_reduce(MIN, dp_group)      # 结果为 1 → 所有 rank 都已收到
```

**Phase 2 — Vote**
```python
self._vote_tensor[i] = 1 if announced[i] and can_schedule[i] else 0
all_reduce(MIN, dp_group)      # 结果为 1 → 所有 rank 都能分配 blocks
```

**Preemption 同步**
```python
self._preempt_tensor[i] = 1 if needs_preempt[i] else 0
all_reduce(MAX, dp_group)      # 结果为 1 → 至少一个 rank 需要 preempt
```

`can_schedule` 的判断：
```python
num_free = kv_cache_manager.block_pool.get_num_free_blocks()
num_blocks_needed = (local_tokens + block_size - 1) // block_size
can_schedule = num_blocks_needed <= num_free
```

最多同时处理 `MAX_CP_SYNC_SLOTS = 32` 个 pending 请求。

---

### 5.2 CPAwareScheduler（`vllm/v1/core/sched/cp_aware_scheduler.py`）

继承 `Scheduler`，每个 DP rank 独立运行一个实例。

**请求分类**
```python
def _is_long_request(self, request):
    num_prefill_tokens = request.num_tokens - request.num_output_tokens
    return num_prefill_tokens >= self.long_request_threshold
```

**请求路由**
```python
def add_request(self, request):
    if cp_world_size <= 1 or not _is_long_request(request):
        request.cp_ranks = [self.cp_rank]   # 短请求：只在本 rank
        super().add_request(request)
    else:
        request.cp_ranks = list(range(cp_world_size))  # 长请求：所有 rank
        self.pending_cp_requests[request_id] = request
```

**CP rank 计算**
```python
self.cp_rank = data_parallel_rank % dycp_size
```

**本地 token 份额**
```python
def _get_local_cp_tokens(self, total_tokens):
    base = total_tokens // cp_world_size
    remainder = total_tokens % cp_world_size
    return base + (1 if cp_rank < remainder else 0)
```

**SchedulerOutput CP 元数据**（由 `schedule()` 填充）

| 字段 | 类型 | 含义 |
|------|------|------|
| `cp_rank` | `int` | 本 rank 的 CP rank 编号 |
| `num_cp_request` | `int` | 本次调度的 CP 请求数 |
| `cp_rank_to_req_id` | `list[str]` | 本 rank 调度的 CP 请求 ID 列表 |
| `req_id_to_cp_size` | `dict[str, list[int]]` | 每个 CP 请求的 rank 列表 |
| `cp_rank_scheduled_tokens` | `dict[str, int]` | 每个请求的 CP world size（>1 表示 CP 请求） |
| `none_tokens_in_peer_sched` | `bool` | peer rank 是否有 scheduled token |

---

### 5.3 DPEngineCoreProc（`vllm/v1/engine/core.py`）

在 busy loop 中插入 CP sync 调用点：

```python
def run_busy_loop(self):
    while True:
        self._process_input_queue()
        self._maybe_run_cp_sync()      # 新增
        executed = self._process_engine_step()
        self._maybe_publish_request_counts()
        self.engines_running = self._has_global_unfinished_reqs(...)

def _maybe_run_cp_sync(self):
    if (
        hasattr(scheduler, "cp_sync")
        and scheduler.cp_sync is not None
        and scheduler.has_pending_cp_requests()
        and scheduler.cp_sync.should_sync()
    ):
        scheduler.run_cp_sync()
```

`hasattr` 检查保证向后兼容：非 CPAwareScheduler 时为空操作。

---

### 5.4 DPLBAsyncMPClient（`vllm/v1/engine/core_client.py`）

**请求路由**
```python
async def add_request_async(self, request):
    if _is_cp_request(request):          # tokens >= long_request_threshold
        for engine in self.core_engines:
            await _send_input(ADD, request, engine)   # 广播
        self.cp_request_ids.add(request.request_id)
    else:
        chosen = get_core_engine_for_request(request) # 负载均衡
        await _send_input(ADD, request, chosen)
```

广播的是完整的 `EngineCoreRequest` 对象（含完整 `prompt_token_ids`），同一份数据被序列化 `dycp_size` 次通过 ZMQ 发送。对于 100K token 的序列，单份约 400KB，`dycp_size=4` 时总传输约 1.6MB，相对于长序列 prefill 计算时间可忽略。

**Abort 路由**
```python
async def abort_requests_async(self, request_ids):
    for req_id in request_ids:
        if req_id in self.cp_request_ids:
            for eng in self.core_engines:             # CP 请求广播 abort
                await _abort_requests([req_id], eng)
        else:
            await _abort_requests([req_id], engine)   # 短请求定向 abort
```

`cp_request_ids` 在 `process_engine_outputs` 中随请求完成而清理。

---

### 5.5 Worker 层（`vllm/v1/worker/`）

**批次重排**（`gpu_model_runner.py`）

当 `num_cp_request > 0` 时，调用 `reorder_batch_to_split_cp_and_normal`，将 CP 请求移到批次前部：

```
[cp0, cp1, ..., ncp0, ncp1, ...]
```

CP 请求按 `req_id` 字典序排序，确保所有 DYCP rank 的顺序一致（不同 rank 的 DP 请求可能交错，但 CP 请求顺序必须相同）。判断依据：`cp_rank_scheduled_tokens[req_id] > 1`。

**本地序列长度**（`gpu_model_runner.py`）

```python
# CP 请求：计算本 rank 实际存储的 token 数
dycp_local_seq_lens[:num_dycp_reqs] = get_dcp_local_seq_lens(
    seq_lens[:num_dycp_reqs], cp_world_size, cp_rank, interleave_size
)
# 非 CP 请求：直接使用原始 seq_lens
dycp_local_seq_lens[num_dycp_reqs:num_reqs] = seq_lens[num_dycp_reqs:num_reqs]
```

**多 rank 执行**（`gpu_worker.py`）

```python
def execute_model(self, scheduler_outputs):
    if isinstance(scheduler_outputs, list):
        scheduler_output = scheduler_outputs[self.model_runner.cp_rank]
        if scheduler_output.total_num_scheduled_tokens == 0 \
                and not scheduler_output.none_tokens_in_peer_sched:
            self.model_runner._dummy_run(1, uniform_decode=True)
```

当 executor 以列表形式下发多个 SchedulerOutput 时，每个 worker 按自己的 `cp_rank` 取对应的 output；若本 rank 无 token 但 peer 有，执行 dummy run 保持 TP 同步。

---

## 6. Attention 通信：与 DCP 的关系

### 6.1 DCP 的工作方式

DCP（Decode Context Parallel）在 TP 组内复用 GPU，将一个序列的 KV cache 按 interleave 方式分片（token `i` 存在 `rank i % dcp_size`）。Attention 计算流程：

```
all_gather(query, dim=1)          → 每个 rank 拿到完整 query
flash_attn(full_query, local_kv)  → 每个 rank 计算 partial attention output + LSE
cp_lse_ag_out_rs(out, lse, dcp_group)
  ├── all_gather(LSE)             → 聚合所有 rank 的 LSE
  ├── correct_attn_out(Triton)    → 用 log-sum-exp 修正本地 output
  └── reduce_scatter(out, dim=1)  → 沿 H 维度分散最终输出
```

`reduce_scatter(dim=1)` 沿 head 维度分散，是因为 DCP 最终要把 head 分给不同 rank 持有。

### 6.2 DYCP 采用与 DCP 相同的 interleave 分片

DYCP 跨 DP rank，采用与 DCP 完全相同的 interleave 分片语义：token `i` 的 KV 存在 `rank = (i // interleave_size) % dycp_size`。两者的本质区别只在于通信域不同：

| 维度 | DCP | DYCP |
|------|-----|------|
| 分片方式 | interleave（token 级交错） | interleave（token 级交错，相同语义） |
| 通信域 | TP 组内（共享物理 GPU，无跨进程通信） | DP 组间（独立进程，需跨进程通信） |
| 进程组 | `get_dcp_group()` | `get_dycp_group()` |
| Query 处理 | all_gather 到所有 rank | all_gather 到所有 rank（相同） |
| Attention 计算 | `full_query × local_kv` | `full_query × local_kv`（相同） |
| 输出聚合 | `cp_lse_ag_out_rs`（reduce_scatter 沿 H 维度） | `cp_lse_ag_out_rs`（相同，换进程组） |

interleave 分片的好处：decode 阶段每新生成一个 token，其位置 `pos` 按公式自然落到对应 rank，所有 rank 负载完全均衡（每步各写一个 token），无需额外的分配逻辑。

### 6.3 可复用的组件

由于分片语义相同，DCP 的 attention 实现几乎可以整体复用，只需将进程组参数从 `get_dcp_group()` 替换为 `get_dycp_group()`：

| 组件 | 位置 | 能否复用 | 说明 |
|------|------|---------|------|
| `_correct_attn_cp_out_kernel` | `attention/ops/common.py` | **能** | LSE 修正的数学逻辑与进程组无关 |
| `_cp_lse_common` | `attention/ops/common.py` | **能** | all_gather LSE + Triton 修正，接口通用 |
| `cp_lse_ag_out_rs` | `attention/ops/common.py` | **能，直接用** | 传入 `get_dycp_group()` 即可 |
| `get_dcp_local_seq_lens` | `attention/backends/utils.py` | **能，已在用** | `dycp_local_seq_lens` 计算已复用此函数 |
| `_forward_with_dcp` 整体流程 | `attention/backends/flash_attn.py` | **能** | 换 `get_dycp_group()`，加 CP/非 CP 请求分支 |
| `compute_slot_mapping` interleave 逻辑 | `worker/block_table.py` | **能** | `compute_domain_slot_mapping` 应复用此逻辑，换 dycp group 参数 |

### 6.4 DYCP Attention 的实现路径

参照 `_forward_with_dcp`，DYCP attention 的核心逻辑为：

```python
# 与 DCP 相同：all_gather query，用完整 query × 本地 interleave KV
query_across_dycp = get_dycp_group().all_gather(query, dim=1)
local_out, local_lse = flash_attn_varlen_func(
    q=query_across_dycp, k=local_kv, v=local_kv,
    seqused_k=dycp_local_seq_lens,   # 本 rank 存储的 token 数
    causal=False,
    return_softmax_lse=True,
)

# 与 DCP 相同：LSE 聚合 + 修正 + reduce_scatter，只换进程组
final_out = cp_lse_ag_out_rs(local_out, local_lse, get_dycp_group())
```

`compute_domain_slot_mapping`（当前悬空调用，尚未实现）应复用 `compute_slot_mapping` 的 interleave 逻辑，对批次前 `num_cp_reqs` 行使用 `dycp_world_size`/`dycp_rank` 参数，其余行走标准路径。

---

## 7. 请求完整生命周期

```
1. Frontend 收到长请求
   → 广播到所有 DP engine
   → 各 DP: add_request → pending_cp_requests

2. Step 1~3: 请求在 pending 状态等待同步点

3. Step 4（sync_interval）: 触发 CP Sync
   Phase 1: all-reduce MIN（announce）→ 确认所有 rank 已收到
   Phase 2: all-reduce MIN（vote）    → 确认所有 rank 能分配 blocks
   → 批准的请求: pending → active → waiting queue

4. 后续 step: 正常调度执行
   → schedule() 填充 CP 元数据
   → reorder_batch 将 CP 请求移到批次前部
   → compute_domain_slot_mapping 计算本地 slot
   → dycp_local_seq_lens 传入 attention backend
   → Workers 执行 attention（跨 rank 通信待实现，见第 6 节）

5. 完成
   → update_from_output 检测完成 → 清理 active_cp_requests
   → process_engine_outputs 清理 cp_request_ids
```

---

## 8. 性能特性

| 场景 | 开销 |
|------|------|
| 无 CP 请求 | 零（`has_pending_cp_requests()` 为 False，不触发 all-reduce） |
| 有 pending CP 请求 | 每 4 步 2 次 all-reduce（小 tensor，CPU，微秒级） |
| 有 active CP 请求需要 preemption 检查 | 每 4 步额外 1 次 all-reduce |
| CP 请求启动延迟 | 最多 4 步（`sync_interval=4`） |

---

## 9. 文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `vllm/v1/core/sched/cp_sync.py` | CPSyncProtocol：两阶段提交分布式共识协议 |
| `vllm/v1/core/sched/cp_aware_scheduler.py` | CPAwareScheduler：继承 Scheduler，增加 CP 感知 |
| `vllm/distributed/kv_transfer/kv_connector/v1/cross_dp_example_connector.py` | 跨 DP KV 传输示例 connector |

### 修改文件

| 文件 | 改动说明 |
|------|---------|
| `vllm/config/parallel.py` | 新增 `dycp_size` 字段 |
| `vllm/config/scheduler.py` | 新增 `num_cp_seqs`、`long_request_threshold` 字段 |
| `vllm/config/vllm.py` | `executor_supports_async_sched` 支持 `"dmp"` backend |
| `vllm/engine/arg_utils.py` | 新增 CLI 参数；`max_num_batched_tokens *= dycp_size` |
| `vllm/distributed/parallel_state.py` | 初始化 `_DYCP` 进程组 |
| `vllm/forward_context.py` | `BatchDescriptor` 新增 `num_dycp_reqs` 字段 |
| `vllm/v1/kv_cache_interface.py` | `memory_usage` 计算纳入 `dycp_size` |
| `vllm/v1/request.py` | `Request` 新增 `cp_ranks` 字段 |
| `vllm/v1/outputs.py` | `ModelRunnerOutput`/`KVConnectorOutput` 新增 CP 相关字段 |
| `vllm/v1/core/sched/output.py` | `SchedulerOutput` 新增 CP 元数据字段 |
| `vllm/v1/engine/core.py` | `DPEngineCoreProc` busy loop 插入 `_maybe_run_cp_sync()` |
| `vllm/v1/engine/core_client.py` | `DPLBAsyncMPClient` CP 路由、广播、abort 逻辑 |
| `vllm/v1/executor/multiproc_executor.py` | 修复 MessageQueue rank；进程命名加 CP group 信息 |
| `vllm/v1/engine/coordinator.py` | 局部变量命名清理 |
| `vllm/v1/worker/block_table.py` | 新增 `apply_permutation` 方法 |
| `vllm/v1/worker/gpu_input_batch.py` | 新增 `apply_permutation` 方法 |
| `vllm/v1/worker/gpu_model_runner.py` | CP rank 初始化；批次重排；`dycp_local_seq_lens`；cudagraph key 扩展 |
| `vllm/v1/worker/gpu_worker.py` | `execute_model`/`sample_tokens` 支持列表输入 |
| `vllm/v1/worker/worker_base.py` | rank 对 `world_size` 取模 |
| `vllm/v1/attention/backends/utils.py` | 新增 `reorder_batch_to_split_cp_and_normal` |
| `vllm/v1/cudagraph_dispatcher.py` | cudagraph key 增加 `num_dycp_reqs` 维度 |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | 注册 `CrossDPExampleConnector` |
| `vllm/distributed/kv_transfer/kv_connector/utils.py` | 新增 `aggregate_domain()`；原 `aggregate()` 加 debug 日志 |
| `vllm/v1/kv_offload/factory.py` | 注册 `CrossDPExampleConnector` |
| `vllm/benchmarks/datasets.py` | `RandomDataset` 新增 `--use-local-json` 参数 |
