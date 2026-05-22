# Dynamic CP 后置同步设计（Post-Schedule Sync）

## 1. 问题分析

### 1.1 现有设计的缺陷

当前设计将 CP 请求的跨 rank 同步放在 `schedule()` 调用之前：

```
_maybe_run_cp_sync()   ← 同步点：投票决定长请求是否可调度
    ↓
schedule()             ← token_budget 限制、preemption 逻辑
```

**问题**：即使 `cp_sync` 投票通过（所有 rank 都有足够 KV blocks），请求进入 waiting queue 后，`schedule()` 内部仍可能因为以下原因拒绝调度或 preempt：

1. **token_budget 耗尽**：running queue 中已有请求消耗了大量 budget，长请求无法被调度
2. **max_num_running_reqs 达到上限**：running 请求数已满
3. **KV cache 竞争**：running queue 中的请求在 `schedule()` 期间分配了新 blocks，导致 waiting 中的长请求分配失败
4. **不同 rank 的 running queue 状态不同**：各 rank 独立处理短请求，running queue 内容不同，token_budget 剩余不同

**后果**：Rank A 成功调度了长请求进入 running，Rank B 因 budget 不足未调度，两个 rank 的 running queue 状态不一致，后续 attention 通信会出错。

### 1.2 设计目标

将同步点移到 `schedule()` 之后，基于实际的 SchedulerOutput 进行共识：

- 如果所有 CP rank 都将某个长请求调度进了 running queue → 确认，保留
- 如果有任何 CP rank 未调度某个长请求 → 回滚，所有 rank 都 preempt 该请求

---

## 2. 新架构

```
schedule()                    ← 各 rank 独立调度，长请求可能被调度也可能不被调度
    ↓
_post_schedule_cp_sync()      ← 新同步点：基于 SchedulerOutput 做共识
    ↓                            - 所有 rank 都调度了 → 确认
    ↓                            - 有 rank 未调度 → 所有 rank 回滚（preempt）
    ↓
修正后的 SchedulerOutput       ← 保证所有 rank 对长请求的调度状态一致
```

---

## 3. 核心协议：Post-Schedule Consensus

### 3.1 三态编码

后置同步需要区分"未被调度"的两种原因，因为它们对请求状态的影响不同：

| 状态码 | 含义 | 对 `num_computed_tokens` 的影响 |
|--------|------|-------------------------------|
| `SCHEDULED = 2` | 请求被正常调度 | 不变（等待执行后增加） |
| `NOT_SCHEDULED = 1` | 未调度，留在 waiting（token_budget 不够等） | 不变 |
| `PREEMPTED = 0` | 被 `schedule()` 内部 preempt（KV blocks 不够） | 已被重置为 0 |

**为什么需要区分**：如果 Rank A 正常调度了长请求（`num_computed_tokens = 50K`），Rank B 在 `schedule()` 内部被 preempt（`num_computed_tokens = 0`），回滚时必须让所有 rank 统一 preempt（重置为 0），否则下一步各 rank 的 prefill 起点不一致，chunk 大小和 KV 写入位置对不上。

### 3.2 协议流程

每次 `schedule()` 返回后，立即执行后置同步：

```python
SCHEDULED = 2
NOT_SCHEDULED = 1
PREEMPTED = 0

def _post_schedule_cp_sync(self, scheduler_output: SchedulerOutput) -> SchedulerOutput:
    """
    在 schedule() 之后同步长请求的调度状态。
    
    协议：
    1. 每个 rank 用三态编码标记长请求的调度结果
    2. all-reduce MIN：取最差状态
    3. 根据共识结果执行 soft_rollback 或 hard_rollback
    """
    if not self.active_cp_requests:
        return scheduler_output
    
    active_ids = sorted(self.active_cp_requests.keys())
    
    # Step 1: 三态编码
    self._confirm_tensor.zero_()
    for i, req_id in enumerate(active_ids):
        if req_id in scheduler_output.num_scheduled_tokens:
            self._confirm_tensor[i] = SCHEDULED
        elif req_id in self._preempted_this_step:
            self._confirm_tensor[i] = PREEMPTED
        else:
            self._confirm_tensor[i] = NOT_SCHEDULED
    
    # Step 2: all-reduce MIN — 取所有 rank 中的最差状态
    torch.distributed.all_reduce(
        self._confirm_tensor[:len(active_ids)],
        op=torch.distributed.ReduceOp.MIN,
        group=self.dp_group,
    )
    
    # Step 3: 根据共识结果分类处理
    confirmed_ids = []
    soft_rollback_ids = []
    hard_rollback_ids = []
    for i, req_id in enumerate(active_ids):
        val = self._confirm_tensor[i].item()
        if val == SCHEDULED:
            confirmed_ids.append(req_id)
        elif val == NOT_SCHEDULED:
            soft_rollback_ids.append(req_id)
        else:  # PREEMPTED
            hard_rollback_ids.append(req_id)
    
    # Step 4: 执行回滚
    if soft_rollback_ids:
        scheduler_output = self._soft_rollback(scheduler_output, soft_rollback_ids)
    if hard_rollback_ids:
        scheduler_output = self._hard_rollback(scheduler_output, hard_rollback_ids)
    
    return scheduler_output
```

`_preempted_this_step` 是一个 set，在 `schedule()` 执行过程中收集被 preempt 的 CP 请求 ID。需要在 `schedule()` 的 preempt 路径中记录：

```python
# 在 scheduler.py 的 preempt 路径中（CPAwareScheduler 覆盖）
def _preempt_request(self, request, ...):
    super()._preempt_request(request, ...)
    if request.request_id in self.active_cp_requests:
        self._preempted_this_step.add(request.request_id)
```

### 3.3 Soft Rollback（token_budget 不足导致的不一致）

**触发条件**：共识结果为 `NOT_SCHEDULED`（有 rank 未调度，但没有 rank 被 preempt）

**语义**：请求状态不变，只是本步不执行，下步重试。

```python
def _soft_rollback(self, output: SchedulerOutput, rollback_ids: list[str]) -> SchedulerOutput:
    """
    轻量回滚：从 SchedulerOutput 中移除，但不重置 num_computed_tokens。
    请求留在 running/waiting queue 中，下一步重试。
    """
    for req_id in rollback_ids:
        if req_id in output.num_scheduled_tokens:
            # 本 rank 调度了但共识未通过 → 从 output 中移除
            num_tokens = output.num_scheduled_tokens.pop(req_id)
            output.total_num_scheduled_tokens -= num_tokens
            self._remove_from_output(output, req_id)
            
            # 释放本次 schedule() 中新分配的 KV blocks（如果有）
            # 注意：不释放之前已分配的 blocks，不重置 num_computed_tokens
            self.kv_cache_manager.free_last_allocation(request)
        
        # 不在 output 中的 rank：什么都不用做，请求本来就在 waiting 中
    
    return output
```

**关键**：soft rollback 后所有 rank 的 `num_computed_tokens` 保持一致（都是之前的值），下一步 `schedule()` 时所有 rank 从相同起点调度。

### 3.4 Hard Rollback（KV blocks 不足触发 preempt 导致的不一致）

**触发条件**：共识结果为 `PREEMPTED`（至少有一个 rank 在 `schedule()` 内部 preempt 了该请求）

**语义**：所有 rank 统一执行完整 preempt，重置 `num_computed_tokens = 0`。

```python
def _hard_rollback(self, output: SchedulerOutput, rollback_ids: list[str]) -> SchedulerOutput:
    """
    完整回滚：所有 rank 统一 preempt，重置 num_computed_tokens。
    """
    for req_id in rollback_ids:
        if req_id in output.num_scheduled_tokens:
            # 本 rank 调度了 → 从 output 中移除
            num_tokens = output.num_scheduled_tokens.pop(req_id)
            output.total_num_scheduled_tokens -= num_tokens
            self._remove_from_output(output, req_id)
        
        request = self.active_cp_requests[req_id]
        
        if req_id not in self._preempted_this_step:
            # 本 rank 没有被 schedule() preempt，需要手动执行完整 preempt
            self.kv_cache_manager.free(request)
            request.num_computed_tokens = 0
            request.status = RequestStatus.PREEMPTED
        # else: schedule() 已经 preempt 过了，num_computed_tokens 已经是 0
        
        # 统一放回 waiting queue 头部
        self.waiting.prepend_request(request)
        # 从 active 移回 pending（等待下次 announce 重新激活）
        del self.active_cp_requests[req_id]
        self.pending_cp_requests[req_id] = request
    
    return output
```

**关键**：hard rollback 后所有 rank 的 `num_computed_tokens` 都是 0，状态完全一致。代价是丢失了已计算的 prefix cache，但这是保证正确性的必要代价。

### 3.5 辅助方法

```python
def _remove_from_output(self, output: SchedulerOutput, req_id: str):
    """从 SchedulerOutput 的各个字段中清理指定请求。"""
    output.scheduled_new_reqs = [
        r for r in output.scheduled_new_reqs if r.req_id != req_id
    ]
    if hasattr(output.scheduled_cached_reqs, 'req_ids'):
        self._remove_from_cached_reqs(output, req_id)
    if output.cp_rank_scheduled_tokens and req_id in output.cp_rank_scheduled_tokens:
        del output.cp_rank_scheduled_tokens[req_id]
    if output.cp_rank_to_req_id and req_id in output.cp_rank_to_req_id:
        output.cp_rank_to_req_id.remove(req_id)
    if output.req_id_to_cp_size and req_id in output.req_id_to_cp_size:
        del output.req_id_to_cp_size[req_id]
    output.num_cp_request = max(0, output.num_cp_request - 1)
```

### 3.6 Preemption 同步（已整合到三态协议中）

在新设计中，preemption 同步不再是独立的阶段，而是自然包含在三态编码中：

- 旧设计：需要单独的 `_sync_cp_preemption()` 用 all-reduce MAX 检测
- 新设计：`PREEMPTED = 0` 状态通过 all-reduce MIN 自然传播（任何 rank 为 0，结果就是 0）

唯一需要额外处理的场景是：所有 rank 都调度了长请求（共识通过），但某个 rank 的 KV cache 已经非常紧张，预计下一步会 preempt。这种情况可以通过在 `confirmed_ids` 上做一次额外的 preemption 预检来处理：

```python
# 可选：对已确认的请求做前瞻性 preemption 检查
if confirmed_ids and self._should_preemptive_check():
    self._preemptive_eviction_check(scheduler_output, confirmed_ids)
```

但这是优化项，不是正确性必需的——即使不做，下一步 `schedule()` 中自然会触发 preempt，然后通过三态协议同步。

---

## 4. 请求生命周期（新设计）

### 4.1 状态机

```
                    Frontend 广播
                        ↓
              pending_cp_requests
                        ↓
            [前置同步：announce only]
            确认所有 rank 都收到了请求
                        ↓
              active_cp_requests + waiting queue
                        ↓
                   schedule()
                  /           \
          被调度进 running    未被调度（budget不足等）
                  \           /
                        ↓
            _post_schedule_cp_sync()
                  /           \
        所有 rank 都调度了    有 rank 未调度
              ↓                     ↓
          确认，继续执行        回滚：所有 rank preempt
              ↓                     ↓
          正常 decode         回到 waiting queue，下次重试
```

### 4.2 完整流程

```
1. Frontend 收到长请求
   → 广播到所有 DP engine
   → 各 DP: add_request → pending_cp_requests

2. 前置同步（简化版，仅 announce）
   → all-reduce MIN：确认所有 rank 都收到了请求
   → 收到确认的请求：pending → active → waiting queue
   （不再做 can_schedule 投票，因为这个判断不可靠）

3. schedule() 独立执行
   → 各 rank 按自己的 token_budget 和 KV cache 状态独立调度
   → 长请求可能被调度进 running，也可能因 budget 不足留在 waiting

4. _post_schedule_cp_sync()（新同步点）
   → all-reduce MIN：确认所有 rank 都调度了同一个长请求
   → 通过：保留在 running
   → 未通过：回滚，preempt 到 waiting queue

5. 修正后的 SchedulerOutput 下发给 Worker
   → 保证所有 rank 对长请求的状态一致

6. 正常 decode 执行
   → 如果某个 rank 需要 preempt（KV cache 紧张）
   → _post_schedule_cp_sync 中的 preemption 同步会处理
```

---

## 5. 与现有设计的对比

| 维度 | 现有设计（前置同步） | 新设计（后置同步） |
|------|---------------------|-------------------|
| 同步时机 | `schedule()` 之前 | `schedule()` 之后 |
| 同步依据 | KV blocks 是否足够 | SchedulerOutput 中是否实际被调度 |
| 一致性保证 | 弱（投票通过不代表能被调度） | 强（基于实际调度结果做共识） |
| token_budget 问题 | 存在：投票通过后仍可能被 budget 拒绝 | 不存在：直接看调度结果 |
| preemption 一致性 | 需要额外同步 | 自然包含在后置同步中 |
| 回滚开销 | 无（不存在回滚） | 有：需要从 SchedulerOutput 中移除 + 释放已分配的 blocks |
| 延迟 | 低（投票通过即可调度） | 可能略高（如果频繁回滚） |
| 实现复杂度 | 中 | 中（回滚逻辑需要仔细处理） |

---

## 6. 关键设计决策

### 6.1 前置同步是否保留？

**保留 announce 阶段，去掉 vote 阶段。**

理由：
- announce 仍然有用：确保所有 rank 都收到了请求后再放入 waiting queue，避免某个 rank 还没收到请求就开始调度
- vote（can_schedule 检查）去掉：因为即使投票通过，`schedule()` 内部仍可能因 budget 拒绝，不如直接让 `schedule()` 自己决定

```python
def run_cp_sync(self) -> None:
    """简化版：仅做 announce，不做 vote。"""
    pending_ids = sorted(self.pending_cp_requests.keys())
    
    # 仅 Phase 1：确认所有 rank 都收到了
    announced_ids = self.cp_sync.sync_announce(pending_ids)
    
    # 被确认的请求直接激活
    for req_id in announced_ids:
        self._activate_cp_request(req_id)
```

### 6.2 回滚时 KV cache 的处理

当回滚一个已被 `schedule()` 调度的长请求时：

```python
def _rollback_and_preempt(self, req_id: str):
    """回滚已调度的长请求。"""
    request = self.active_cp_requests[req_id]
    
    # 释放本次 schedule() 中分配的 KV blocks
    self.kv_cache_manager.free(request)
    
    # 重置状态
    request.num_computed_tokens = 0
    request.status = RequestStatus.PREEMPTED
    
    # 放回 waiting queue 头部，下次优先调度
    self.waiting.prepend_request(request)
```

### 6.3 避免活锁

**问题**：如果某个 rank 的短请求持续占满 token_budget，长请求永远无法被调度，导致反复回滚。

**解决方案**：引入退避机制

```python
class CPRequestState:
    rollback_count: int = 0
    next_retry_step: int = 0  # 在此 step 之前不尝试调度

def _should_attempt_schedule(self, req_id: str) -> bool:
    """检查是否应该尝试调度该长请求。"""
    state = self.cp_request_states[req_id]
    return self.current_step >= state.next_retry_step
```

或者更简单的方案：**预留 token_budget**

```python
def schedule(self):
    # 如果有 active CP 请求在 waiting 中，为其预留一部分 budget
    reserved_budget = 0
    if self._has_waiting_cp_requests():
        reserved_budget = self._estimate_cp_budget_needed()
    
    effective_budget = self.max_num_scheduled_tokens - reserved_budget
    # running queue 调度使用 effective_budget
    # waiting queue 中的 CP 请求使用 reserved_budget
```

### 6.4 每次 schedule 都同步 vs 按间隔同步

**建议：每次 schedule 后都同步（如果有 active CP 请求在 running 中或刚被调度）。**

理由：
- 后置同步的开销是一次 all-reduce（小 tensor，微秒级）
- 如果不每次同步，可能出现某个 rank 在 step N 调度了长请求，另一个 rank 在 step N+1 才调度，中间有一个 step 状态不一致
- 前置 announce 仍然可以按间隔执行（因为请求到达有延迟，不需要每步检查）

```python
def _process_engine_step(self):
    scheduler_output = self.scheduler.schedule()
    
    # 后置同步：每次都执行（如果有 active CP 请求）
    if self.scheduler.has_active_cp_requests():
        scheduler_output = self.scheduler.post_schedule_cp_sync(scheduler_output)
    
    # 下发给 worker
    ...
```

---

## 7. 通信开销分析

| 场景 | 前置同步 | 后置同步 | 总计 |
|------|---------|---------|------|
| 无 CP 请求 | 0 | 0 | 0 |
| 有 pending CP 请求 | 每 4 步 1 次 all-reduce（announce） | 0 | 每 4 步 1 次 |
| 有 active CP 请求在 running | 0 | 每步 1 次 all-reduce（confirm） | 每步 1 次 |
| 有 active CP 请求 + preemption 检查 | 0 | 每步 2 次 all-reduce（confirm + preempt） | 每步 2 次 |

相比现有设计（每 4 步 2-3 次 all-reduce），新设计在有 active 请求时每步 1-2 次 all-reduce，频率更高但单次开销相同（小 tensor CPU all-reduce，微秒级）。考虑到长请求的 prefill/decode 计算本身就是毫秒级，这个开销可以忽略。

---

## 8. 边界情况处理

### 8.1 长请求首次 prefill

长请求的 prefill token 数很大（如 100K），可能超过单步的 token_budget。vLLM 的 chunked prefill 机制会将其分多步完成。

**处理**：只要长请求被调度了（哪怕只调度了部分 token），就算"被调度"。后置同步检查的是 `req_id in scheduler_output.num_scheduled_tokens`，不关心调度了多少 token。

### 8.2 长请求在 running 中被 preempt

`schedule()` 处理 running queue 时，如果 KV cache 不足，会 preempt 优先级最低的请求。如果长请求被 preempt：

- 本 rank 的 SchedulerOutput 中不会包含该请求
- 后置同步时 `_schedule_confirm_tensor[i] = 0`
- all-reduce MIN 后所有 rank 都会得到 0
- 所有 rank 统一 preempt

### 8.3 某个 rank 的 schedule() 完全没调度任何请求

可能发生在该 rank 的 running queue 已满且都是短请求的情况。此时长请求留在 waiting 中未被调度。后置同步会检测到不一致并回滚。

### 8.4 请求完成的同步

长请求在所有 rank 上应该同时完成（因为 decode 是同步的）。但如果因为网络延迟等原因，某个 rank 先检测到完成：

- 完成的请求不会出现在下一步的 SchedulerOutput 中
- 后置同步自然会处理这种情况

---

## 9. 去掉 Batch Reorder，改用 Index-Based Gather/Scatter

### 9.1 为什么去掉 reorder

现有设计中 `reorder_batch_to_split_cp_and_normal` 将 CP 请求移到 batch 前部，目的是让 attention backend 通过位置区分 CP/非 CP 请求。但这有几个问题：

1. **破坏 FCFS 语义**：`apply_permutation` 改变了 `gpu_input_batch` 的持久状态，后续 step 的遍历顺序被打乱
2. **实现复杂**：需要对 `gpu_input_batch`、`block_table` 等多个数据结构同步执行 permutation
3. **不必要**：CP 请求通常只有 1-2 个，为了它们重排整个 batch（可能几百个请求）不划算

### 9.2 替代方案：Index-Based

不改变 batch 中请求的物理顺序，只记录 CP 请求的索引位置，attention 时按需 gather/scatter。

**SchedulerOutput 新增字段**：

```python
@dataclass
class SchedulerOutput:
    ...
    # 替代 reorder：CP 请求在 batch 中的索引（所有 rank 按 req_id 排序保证一致）
    cp_req_indices: list[int] | None = None
```

**Model Runner 层**：

```python
def _prepare_inputs(self, scheduler_output):
    ...
    # 不再 reorder，只记录 CP 请求的位置
    if scheduler_output.num_cp_request > 0:
        cp_req_ids = sorted(scheduler_output.cp_rank_to_req_id)
        self.cp_req_indices = [
            self.input_batch.req_id_to_index[rid] for rid in cp_req_ids
        ]
    else:
        self.cp_req_indices = None
```

**Attention Backend 层**：

```python
def forward(self, query, key, value, ...):
    if self.cp_req_indices is None or len(self.cp_req_indices) == 0:
        # 无 CP 请求，走正常路径
        return normal_flash_attn(query, key, value, ...)
    
    num_reqs = query.shape[0]  # varlen 展开后是 token 维度，这里简化表示
    cp_indices = self.cp_req_indices
    non_cp_indices = [i for i in range(num_reqs) if i not in cp_indices_set]
    
    # 1. CP 请求：gather → 跨 rank 通信 → scatter 回
    cp_query = query[cp_token_ranges]  # 按 token range gather
    cp_query_all = get_dycp_group().all_gather(cp_query, dim=1)
    cp_out, cp_lse = flash_attn_varlen_func(
        q=cp_query_all, k=local_cp_kv, v=local_cp_kv,
        seqused_k=dycp_local_seq_lens,
        causal=False, return_softmax_lse=True,
    )
    cp_final = cp_lse_ag_out_rs(cp_out, cp_lse, get_dycp_group())
    output[cp_token_ranges] = cp_final
    
    # 2. 非 CP 请求：正常 attention（原地）
    output[non_cp_token_ranges] = normal_flash_attn(
        query[non_cp_token_ranges], key[non_cp_token_ranges], ...
    )
    
    return output
```

### 9.3 `dycp_local_seq_lens` 的计算

不再依赖"前 N 个是 CP 请求"的假设，改为按 index 填充：

```python
# 旧设计：依赖 reorder 后的位置
dycp_local_seq_lens[:num_dycp_reqs] = get_dcp_local_seq_lens(...)
dycp_local_seq_lens[num_dycp_reqs:] = seq_lens[num_dycp_reqs:]

# 新设计：按 index 填充
dycp_local_seq_lens = seq_lens.clone()
for idx in cp_req_indices:
    dycp_local_seq_lens[idx] = get_dcp_local_seq_lens(
        seq_lens[idx], cp_world_size, cp_rank, interleave_size
    )
```

### 9.4 开销对比

| 操作 | Reorder 方案 | Index-Based 方案 |
|------|-------------|-----------------|
| 每步固定开销 | `apply_permutation` 对整个 batch | 无 |
| Attention 额外开销 | 无（连续内存） | gather/scatter CP 请求的 token（通常 1-2 个请求） |
| 实现复杂度 | 高（需要同步多个数据结构） | 低（只需维护 index 列表） |
| 对 FCFS 的影响 | 有（改变 batch 持久状态） | 无 |

CP 请求数量通常很少（1-2 个），gather/scatter 的 tensor 很小，开销远低于对整个 batch 做 permutation。

---

## 10. 实现计划

### 10.1 修改文件

| 文件 | 改动 |
|------|------|
| `vllm/v1/core/sched/cp_sync.py` | 新增 `sync_announce()`；新增 `sync_schedule_confirm()`（三态） |
| `vllm/v1/core/sched/cp_aware_scheduler.py` | 新增 `post_schedule_cp_sync()`；简化 `run_cp_sync()` 为仅 announce；新增 soft/hard rollback |
| `vllm/v1/engine/core.py` | 在 `schedule()` 之后调用 `post_schedule_cp_sync()` |
| `vllm/v1/core/sched/output.py` | 新增 `cp_req_indices` 字段 |
| `vllm/v1/worker/gpu_model_runner.py` | 去掉 `reorder_batch` 调用；改用 `cp_req_indices` 计算 `dycp_local_seq_lens` |
| `vllm/v1/attention/backends/flash_attn.py` | 用 index-based gather/scatter 替代基于位置的 CP/非 CP 分支 |
| `vllm/v1/attention/backends/utils.py` | 删除 `reorder_batch_to_split_cp_and_normal` |
| `vllm/v1/worker/gpu_input_batch.py` | 删除 `apply_permutation`（仅用于 reorder 的情况） |
| `vllm/v1/worker/block_table.py` | 删除 `apply_permutation`（仅用于 reorder 的情况） |

### 10.2 新增接口

```python
# cp_sync.py
class CPSyncProtocol:
    def sync_announce(self, pending_ids: list[str]) -> list[str]:
        """仅 announce 阶段，返回所有 rank 都收到的请求 ID。"""
        ...
    
    def sync_schedule_confirm(self, active_ids: list[str], scheduled: list[bool]) -> list[str]:
        """后置确认，返回所有 rank 都调度了的请求 ID。"""
        ...
    
    def sync_preemption(self, active_ids: list[str], needs_preempt: list[bool]) -> list[str]:
        """Preemption 同步（保持不变）。"""
        ...

# cp_aware_scheduler.py
class CPAwareScheduler:
    def post_schedule_cp_sync(self, output: SchedulerOutput) -> SchedulerOutput:
        """schedule() 之后的后置同步。"""
        ...
    
    def _rollback_cp_requests(self, output: SchedulerOutput, rollback_ids: list[str]) -> SchedulerOutput:
        """回滚未通过共识的请求。"""
        ...
```

### 10.3 调用链

```python
# engine/core.py
def _process_engine_step(self):
    # 前置：仅 announce（按间隔）
    self._maybe_run_cp_announce()
    
    # 调度
    scheduler_output = self.scheduler.schedule()
    
    # 后置：确认 + 回滚（每步，如果有 active CP 请求）
    if hasattr(self.scheduler, 'post_schedule_cp_sync'):
        scheduler_output = self.scheduler.post_schedule_cp_sync(scheduler_output)
    
    # 下发
    ...
```

---

## 11. 总结

新设计的核心思想：**不要试图预测 `schedule()` 的行为，而是观察它的结果并做共识。**

- 前置同步仅保留 announce（确认请求到达），去掉 vote（不再预测可调度性）
- 后置同步基于 SchedulerOutput 做三态 all-reduce MIN 共识（SCHEDULED / NOT_SCHEDULED / PREEMPTED）
- soft rollback（budget 不足）：不重置 `num_computed_tokens`，下步从相同位置重试
- hard rollback（KV preempt）：所有 rank 统一重置 `num_computed_tokens = 0`
- 去掉 batch reorder，改用 index-based gather/scatter，保持 FCFS 语义不被破坏

这种设计将"是否能调度"的判断完全交给 `schedule()` 本身，同步层只负责"确认一致性"，职责更清晰，一致性保证更强。
