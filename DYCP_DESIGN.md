# Dynamic CP（DYCP）设计文档

## 1. 背景与目标

在保留 vLLM 原始 per-DP 独立进程架构的前提下，实现 Dynamic Context Parallel（DYCP）：长序列分布到多个 DP rank 并行处理，短序列路由到单个 DP rank 正常执行。

长序列走 CP 时，必须满足两个跨 DP 约束：

| 约束 | 说明 |
|------|------|
| 同步执行 | CP 请求必须在所有 CP rank 的同一 step 执行 |
| 同步 Preemption | 任一 rank 需要 preempt 时，所有 rank 同步释放 |

KV cache 分配不需要预先协商——各 rank 独立调度，事后通过后置同步达成共识。

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

`transpose(1, 4)` 将 DP 维度移到最后，`reshape(-1, dycp_size)` 将每 `dycp_size` 个连续 DP rank 分为一个 CP group。DYCP 在 DP 维度上分组，与 TP 无关，TP=1 时同样可用。

---

## 4. 整体架构

```
Frontend (DPLBAsyncMPClient)
  ├── 长请求 → 广播到所有 DP engine（所有 rank 保证收到相同请求）
  └── 短请求 → 负载均衡路由到单个 DP engine

 DP_0 EngineCoreProc                   DP_1 EngineCoreProc
  CPAwareScheduler    ←─ all-reduce ─→   CPAwareScheduler
  (active CP requests)                   (active CP requests)
  MultiprocExecutor                      MultiprocExecutor
 Workers (TP group)     ←─ 待实现 ─→     Workers (TP group)
```

> **注**：Worker 间跨 DP rank 的 attention 通信尚未实现。`_DYCP` 进程组已初始化，`cp_rank` 和 `dycp_local_seq_lens` 已传入 attention backend，但 attention 计算中实际调用 `get_dycp_group()` 进行通信的部分仍待补充。

---

## 5. 组件详解

### 5.1 DPLBAsyncMPClient（`vllm/v1/engine/core_client.py`）

**请求路由**

```python
async def add_request_async(self, request):
    if _is_cp_request(request):          # tokens >= long_request_threshold
        for engine in self.core_engines:
            await _send_input(ADD, request, engine)   # 广播到所有 rank
        self.cp_request_ids.add(request.request_id)
    else:
        chosen = get_core_engine_for_request(request) # 负载均衡
        await _send_input(ADD, request, chosen)
```

广播保证每个 DP rank 收到完全相同的 CP 请求，这是整个协议的基础假设。若某个 rank 未收到 CP 请求，视为 client 层 bug，不在 scheduler 层容错。

**Abort 路由**

CP 请求的 abort 同样广播到所有 engine；短请求定向 abort。`cp_request_ids` 随请求完成而清理。

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
        request.cp_ranks = [self.cp_rank]
        super().add_request(request)          # 短请求：正常路径
    else:
        request.cp_ranks = list(range(cp_world_size))
        self.active_cp_requests[request_id] = request
        self.requests[request_id] = request
        self.waiting.add_request(request)     # 长请求：直接激活，进入 waiting queue
```

client 层广播保证所有 rank 同时收到 CP 请求，因此不需要 pending 状态或 announce 阶段，直接激活。

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

**schedule() 覆盖**

在 `super().schedule()` 前后各有处理：
- **开头**：清空 `_preempted_this_step`
- **结尾**：遍历 `num_scheduled_tokens`，识别 active CP 请求，填充 CP 元数据字段

**SchedulerOutput CP 元数据字段**

| 字段 | 类型 | 含义 |
|------|------|------|
| `cp_rank` | `int` | 本 rank 的 CP rank 编号 |
| `num_cp_request` | `int` | 本次调度的 CP 请求数 |
| `cp_rank_to_req_id` | `list[str]` | 本 rank 调度的 CP 请求 ID 列表 |
| `req_id_to_cp_size` | `dict[str, list[int]]` | 每个 CP 请求的 rank 列表 |
| `cp_rank_scheduled_tokens` | `dict[str, int]` | 每个请求的 CP world size（>1 表示 CP 请求） |
| `cp_req_ids_sorted` | `list[str] \| None` | 后置同步确认的 CP 请求 ID（排序），worker 用来计算 batch indices |

---

### 5.3 CPSyncProtocol（`vllm/v1/core/sched/cp_sync.py`）

每次 `schedule()` 返回后执行一次 all-reduce MIN，对所有 active CP 请求达成共识。

**三态编码**

| 状态码 | 值 | 含义 | 对 `num_computed_tokens` 的影响 |
|--------|---|------|-------------------------------|
| `SCHEDULED` | 2 | 本 rank 成功调度了该请求 | 不变（等待执行后增加） |
| `NOT_SCHEDULED` | 1 | 未调度，留在 waiting（token_budget 不够等） | 不变 |
| `PREEMPTED` | 0 | 被 `schedule()` 内部 preempt（KV blocks 不够） | 已被重置为 0 |

区分 `NOT_SCHEDULED` 和 `PREEMPTED` 是必要的：若 Rank A 调度了请求（`num_computed_tokens = 50K`），Rank B 被 preempt（`num_computed_tokens = 0`），回滚时必须让所有 rank 统一重置为 0，否则下一步各 rank 的 prefill 起点不一致。

**协议**

```
每个 rank 上报三态状态
        ↓
all_reduce(MIN, dp_group)   ← 取所有 rank 中的最差状态
        ↓
min >= SCHEDULED (2)  → confirmed：所有 rank 都调度了，正常执行
min >= NOT_SCHEDULED (1) → soft rollback：有 rank 未调度，回退重试
min == PREEMPTED (0)  → hard rollback：有 rank 被 preempt，所有 rank 统一 preempt
```

`_confirm_tensor` 预分配在 CPU 上（`MAX_CP_SYNC_SLOTS = 32` 个 int32），dp_group 使用 gloo backend，all-reduce 为微秒级开销。

---

### 5.4 post_schedule_cp_sync（`CPAwareScheduler`）

在 `schedule()` 之后、`execute_model()` 之前调用，修正 SchedulerOutput 使所有 rank 对 CP 请求的调度状态一致。

**关键约束**：即使本 rank 没有活跃 CP 请求，也必须调用 `cp_sync.sync_empty()` 参与 all_reduce，否则持有 CP 请求的 peer rank 会在集合通信上永久阻塞。`sync_empty()` 用 `NOT_SCHEDULED(1)` 填充张量——若填 `0(PREEMPTED)`，MIN 操作会将 peer 的 `SCHEDULED(2)` 拉低到 `0`，触发不必要的 hard_rollback。

```python
def post_schedule_cp_sync(self, output: SchedulerOutput) -> SchedulerOutput:
    if cp_sync is None:
        return output
    if not active_cp_requests:
        cp_sync.sync_empty()   # 必须参与，不能跳过
        return output

    active_ids = sorted(self.active_cp_requests.keys())

    # 三态编码
    status = []
    for req_id in active_ids:
        if req_id in output.num_scheduled_tokens:
            status.append(SCHEDULED)
        elif req_id in self._preempted_this_step:
            status.append(PREEMPTED)
        else:
            status.append(NOT_SCHEDULED)

    confirmed, soft_rollback_ids, hard_rollback_ids = (
        self.cp_sync.sync_schedule_confirm(active_ids, status)
    )

    if soft_rollback_ids:
        output = self._soft_rollback(output, soft_rollback_ids)
    if hard_rollback_ids:
        output = self._hard_rollback(output, hard_rollback_ids)

    self._preempted_this_step.clear()
    output.cp_req_ids_sorted = sorted(confirmed) if confirmed else None
    return output
```

**Soft Rollback**（`NOT_SCHEDULED`）

- 从 SchedulerOutput 中移除该请求
- 释放本步新分配的 KV blocks
- `num_computed_tokens` 不变
- 请求放回 waiting queue 头部，下步重试
- 请求保留在 `active_cp_requests`

**Hard Rollback**（`PREEMPTED`）

- 从 SchedulerOutput 中移除该请求
- 若本 rank 未被 `schedule()` preempt，手动执行完整 preempt：释放 KV blocks，`num_computed_tokens = 0`，状态改为 `PREEMPTED`
- 若本 rank 已被 `schedule()` preempt，状态已经正确，无需重复操作
- 请求保留在 `active_cp_requests`，放回 waiting queue 头部，下步重试

---

### 5.5 DPEngineCoreProc（`vllm/v1/engine/core.py`）

**Scheduler 自动选择**：`SchedulerConfig.get_scheduler_cls()` 在 `num_cp_seqs > 0` 时自动返回 `CPAwareScheduler`，无需手动指定 `--scheduler-cls`。

`post_schedule_cp_sync` 在 `step()` 的 `schedule()` 和 `execute_model()` 之间调用。**关键**：`has_requests()=False` 时不能直接返回，必须先调用一次 `post_schedule_cp_sync`（传入空 output），否则 peer rank 死锁：

```python
def step(self):
    if not self.scheduler.has_requests():
        if hasattr(self.scheduler, 'post_schedule_cp_sync'):
            self.scheduler.post_schedule_cp_sync(SchedulerOutput.make_empty())
        return {}, False
    scheduler_output = self.scheduler.schedule()
    if hasattr(self.scheduler, 'post_schedule_cp_sync'):
        scheduler_output = self.scheduler.post_schedule_cp_sync(scheduler_output)
    model_output = self.model_executor.execute_model(scheduler_output)
    ...
```

`step_with_batch_queue()` 的 `has_requests()=False` 分支同样处理。`hasattr` 检查保证向后兼容。

---

### 5.6 Worker 层（`vllm/v1/worker/`）

**本地序列长度**（`gpu_model_runner.py`）

不依赖 batch 中请求的物理顺序，按 index 填充：

```python
# 默认：所有请求用原始 seq_lens
dycp_local_seq_lens[:num_reqs] = seq_lens[:num_reqs]

# CP 请求：按 index 覆盖为本 rank 实际存储的 token 数
if cp_req_indices:
    cp_idx_tensor = torch.tensor(cp_req_indices, dtype=torch.long)
    dycp_local_seq_lens[cp_idx_tensor] = get_dcp_local_seq_lens(
        seq_lens[cp_idx_tensor], cp_world_size, cp_rank, interleave_size
    )
```

`cp_req_indices` 由 worker 根据 `scheduler_output.cp_req_ids_sorted` 和 `input_batch.req_id_to_index` 计算得到，不依赖 batch reorder。

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

## 6. 请求完整生命周期

```
1. Frontend 收到长请求
   → 广播到所有 DP engine
   → 各 DP: add_request → active_cp_requests + waiting queue（直接激活）

2. 每个 step:
   schedule()
   → 各 rank 按自己的 token_budget 和 KV cache 状态独立调度
   → 长请求可能被调度进 running，也可能因 budget 不足留在 waiting，
     或因 KV cache 不足被 preempt

   post_schedule_cp_sync()
   → all-reduce MIN：确认所有 rank 都调度了同一个长请求
   → confirmed：保留在 output，正常执行
   → soft rollback：从 output 移除，放回 waiting，下步重试
   → hard rollback：所有 rank 统一 preempt，num_computed_tokens=0，下步重试

   execute_model(修正后的 output)
   → 保证所有 rank 对长请求的状态一致

3. 完成
   → update_from_output 检测完成 → 清理 active_cp_requests
   → process_engine_outputs 清理 cp_request_ids
```

---

## 7. Attention 通信：与 DCP 的关系

### 7.1 DCP 的工作方式

DCP（Decode Context Parallel）在 TP 组内复用 GPU，将一个序列的 KV cache 按 interleave 方式分片（token `i` 存在 `rank i % dcp_size`）。Attention 计算流程：

```
all_gather(query, dim=1)          → 每个 rank 拿到完整 query
flash_attn(full_query, local_kv)  → 每个 rank 计算 partial attention output + LSE
cp_lse_ag_out_rs(out, lse, dcp_group)
  ├── all_gather(LSE)             → 聚合所有 rank 的 LSE
  ├── correct_attn_out(Triton)    → 用 log-sum-exp 修正本地 output
  └── reduce_scatter(out, dim=1)  → 沿 H 维度分散最终输出
```

### 7.2 DYCP 采用与 DCP 相同的 interleave 分片

DYCP 跨 DP rank，采用与 DCP 完全相同的 interleave 分片语义：token `i` 的 KV 存在 `rank = (i // interleave_size) % dycp_size`。两者的本质区别只在于通信域不同：

| 维度 | DCP | DYCP |
|------|-----|------|
| 分片方式 | interleave（token 级交错） | interleave（相同语义） |
| 通信域 | TP 组内（共享物理 GPU） | DP 组间（独立进程，跨进程通信） |
| 进程组 | `get_dcp_group()` | `get_dycp_group()` |

### 7.3 DYCP Attention 实现路径（待实现）

参照 `_forward_with_dcp`，核心逻辑为：

```python
# 与 DCP 相同：all_gather query，用完整 query × 本地 interleave KV
query_across_dycp = get_dycp_group().all_gather(query, dim=1)
local_out, local_lse = flash_attn_varlen_func(
    q=query_across_dycp, k=local_kv, v=local_kv,
    seqused_k=dycp_local_seq_lens,
    causal=False, return_softmax_lse=True,
)
# 与 DCP 相同：LSE 聚合 + 修正 + reduce_scatter，只换进程组
final_out = cp_lse_ag_out_rs(local_out, local_lse, get_dycp_group())
```

Attention backend 需要根据 `cp_req_indices` 区分 CP 请求和非 CP 请求，分别走 DYCP 路径和普通路径，结果按 token index scatter 回 output tensor。

可复用的 DCP 组件：`cp_lse_ag_out_rs`、`_correct_attn_cp_out_kernel`、`get_dcp_local_seq_lens`，只需将进程组参数替换为 `get_dycp_group()`。

---

## 8. 通信开销

| 场景 | 每步 all_reduce 次数 | 说明 |
|------|---------------------|------|
| 无 CP 请求（`active_cp_requests` 为空） | 1（`sync_empty`） | 必须参与以避免 peer rank 阻塞 |
| 有 active CP 请求 | 1（`sync_schedule_confirm`） | post-schedule 共识 |

all_reduce 在 CPU 上执行（gloo backend），tensor 固定为 `MAX_CP_SYNC_SLOTS = 32` 个 int32，单次开销微秒级。`sync_empty` 和 `sync_schedule_confirm` 使用相同的张量大小，确保集合通信形状匹配。

---

## 9. 文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `vllm/v1/core/sched/cp_sync.py` | CPSyncProtocol：后置共识协议 |
| `vllm/v1/core/sched/cp_aware_scheduler.py` | CPAwareScheduler：继承 Scheduler，增加 CP 感知 |

### 主要修改文件

| 文件 | 改动说明 |
|------|---------|
| `vllm/config/parallel.py` | 新增 `dycp_size` 字段 |
| `vllm/config/scheduler.py` | 新增 `num_cp_seqs`、`long_request_threshold` 字段；`get_scheduler_cls()` 在 `num_cp_seqs > 0` 时自动选择 `CPAwareScheduler` |
| `vllm/engine/arg_utils.py` | 新增 CLI 参数；`max_num_batched_tokens *= dycp_size` |
| `vllm/distributed/parallel_state.py` | 初始化 `_DYCP` 进程组 |
| `vllm/forward_context.py` | `BatchDescriptor` 新增 `num_dycp_reqs` 字段 |
| `vllm/v1/core/sched/output.py` | `SchedulerOutput` 新增 CP 元数据字段（含 `cp_req_ids_sorted`） |
| `vllm/v1/engine/core.py` | `step()` 和 `step_with_batch_queue()` 中在 `schedule()` 后调用 `post_schedule_cp_sync()`；`has_requests()=False` 路径也参与 CP sync |
| `vllm/v1/engine/core_client.py` | `DPLBAsyncMPClient` CP 广播路由、abort 逻辑 |
| `vllm/v1/worker/gpu_model_runner.py` | CP rank 初始化；index-based `dycp_local_seq_lens`；cudagraph key 扩展 |
| `vllm/v1/worker/gpu_worker.py` | `execute_model`/`sample_tokens` 支持列表输入 |
| `vllm/v1/attention/backends/flash_attn.py` | DYCP attention 分支（待实现） |
| `vllm/v1/cudagraph_dispatcher.py` | cudagraph key 增加 `num_dycp_reqs` 维度 |

---

## 10. 待实现项

| 项目 | 文件 | 说明 |
|------|------|------|
| Attention 通信 | `vllm/v1/attention/backends/flash_attn.py` | 参照 `_forward_with_dcp`，换 `get_dycp_group()`，加 CP/非 CP 请求分支 |
| `compute_domain_slot_mapping` | `vllm/v1/worker/block_table.py` | 复用 `compute_slot_mapping` interleave 逻辑，对 CP 请求使用 `dycp_world_size`/`dycp_rank` 参数 |
| executor 分发路径 | `vllm/v1/executor/multiproc_executor.py` | `execute_model` 支持 `list[SchedulerOutput]` 分发 |
| `dp_group` 传入 scheduler | `vllm/v1/engine/core.py` | `EngineCore.__init__` 创建 scheduler 时传入 `dp_group` |
| `get_total_cp_world_size` | `vllm/v1/worker/cp_utils.py` | 纳入 `dycp_size` |
