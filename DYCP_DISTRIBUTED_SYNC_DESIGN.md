# Dynamic CP 最小化架构修改方案：Distributed CP Sync

## 1. 背景与动机

### 1.1 原始 vLLM 0.16.0 DP 架构

vLLM 0.16.0 的数据并行（DP）架构采用**完全独立的多进程模型**：

```
DP_0: DPEngineCoreProc_0 → Scheduler_0 → MultiprocExecutor_0 → Workers
DP_1: DPEngineCoreProc_1 → Scheduler_1 → MultiprocExecutor_1 → Workers
DP_2: DPEngineCoreProc_2 → Scheduler_2 → MultiprocExecutor_2 → Workers
```

每个 DP rank 是一个独立进程，拥有独立的 Scheduler、KVCacheManager 和 Executor。DP 之间仅通过每 32 步一次的 all-reduce 同步全局完成状态。

### 1.2 当前 DYCP 实现的问题

当前 DYCP 实现为了实现动态 Context Parallel，对架构做了重大重构：

- 新增 `DomainEngineCoreProc` 替代 `DPEngineCoreProc`
- 新增 `DomainMultiprocExecutor` 替代 `MultiprocExecutor`
- 新增 `CrossDPScheduler` 替代 `Scheduler`
- 新增 `CrossDPKVCacheManager` 替代 `KVCacheManager`

这将原来"每个 DP 一个独立进程"的架构改为"一个进程管理多个 DP"，架构侵入性大，与上游 vLLM 差异显著，维护成本高。

### 1.3 本方案的目标

**保留原始 per-DP 独立进程架构**，通过轻量级的分布式协同协议实现相同的 DYCP 功能，最小化对 vLLM 架构的修改。

---

## 2. 核心原理

### 2.1 DYCP 的本质需求

Dynamic CP 的核心功能是：

1. **请求分类**：基于 prefill token 数阈值，将请求分为长序列（CP）和短序列（DP）
2. **短序列**：路由到负载最低的单个 DP，正常执行
3. **长序列**：分布到所有 DP 上并行处理（Context Parallel），每个 DP 只处理序列的一部分

### 2.2 为什么需要跨 DP 协调

长序列走 CP 时，存在三个必须跨 DP 协调的约束：

| 约束 | 说明 |
|------|------|
| **KV cache 原子分配** | 长序列的 KV cache 分布在多个 DP 上，必须所有 DP 都有足够 blocks 才能调度 |
| **同步执行** | CP 请求必须在所有涉及的 DP 上在同一个 step 执行（attention 计算后需要跨 rank 通信） |
| **同步 Preemption** | 任一 DP 需要 preempt CP 请求时，所有 DP 必须同时释放 |

### 2.3 本方案的核心思路

**用分布式共识协议替代集中式调度器。**

当前方案通过一个 `CrossDPScheduler` 在单进程内管理所有 DP 来天然满足上述约束。本方案的替代思路是：

- 每个 DP 保持独立的 Scheduler
- 通过**批量同步的两阶段提交协议**（基于 all-reduce）在 DP 之间达成共识
- 只有所有 DP 都同意时，CP 请求才被激活调度

```
当前方案（集中式）：
  一个 CrossDPScheduler 决定所有 DP 的调度 → 天然一致

本方案（分布式）：
  每个 DP 独立 Scheduler + 批量同步协议 → 协议保证一致
```

---

## 3. 架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (DPLBAsyncMPClient)                  │
│                                                                  │
│  add_request_async():                                           │
│    if 长请求 → 广播到所有 Engine                                  │
│    if 短请求 → 路由到负载最低的 Engine                             │
└────────────┬──────────────────────────────────┬─────────────────┘
             │                                  │
     ┌───────▼───────┐                 ┌───────▼───────┐
     │DP_0 EngineCore│                 │DP_1 EngineCore│
     │               │                 │               │
     │CPAwareScheduler◄── CP Sync ────►CPAwareScheduler│
     │               │  (all-reduce)   │               │
     │ KVCacheManager│                 │ KVCacheManager│
     │               │                 │               │
     │MultiprocExec. │                 │MultiprocExec. │
     └───────┬───────┘                 └───────┬───────┘
             │                                  │
     ┌───────▼───────┐                 ┌───────▼───────┐
     │   Workers     │                 │   Workers     │
     │ (TP group)    │◄── DYCP comm ──►│ (TP group)    │
     └───────────────┘                 └───────────────┘
```

### 3.2 组件职责

| 组件 | 职责 | 修改程度 |
|------|------|---------|
| `DPLBAsyncMPClient` | CP-aware 请求路由（长请求广播，短请求负载均衡） | 小改 |
| `CPAwareScheduler` | 继承 Scheduler，增加 CP 请求状态管理和本地 token 份额分配 | 新增 |
| `CPSyncProtocol` | 分布式两阶段提交协议，基于 dp_group all-reduce | 新增 |
| `DPEngineCoreProc` | busy loop 中增加 CP sync 调用点 | 极小改 |
| `KVCacheManager` | 不修改，每个 DP 只分配自己那部分的 blocks | 不变 |
| `MultiprocExecutor` | 不修改 | 不变 |

---

## 4. 详细设计

### 4.1 CPSyncProtocol — 分布式共识协议

**文件**：`vllm/v1/core/sched/cp_sync.py`

#### 4.1.1 设计原则

- **批量同步**：每 N 步（默认 N=4）执行一次同步，而非每步都同步
- **零开销退化**：无 pending CP 请求时完全不触发同步
- **复用现有通信**：使用已有的 `dp_group`（torch.distributed ProcessGroup）

#### 4.1.2 协议流程

```
每 N 步触发一次 CP Sync：

Phase 1 — 公告（Announce）
  各 DP 广播自己已收到的 pending CP 请求 ID
  all-reduce MIN → 结果为 1 表示所有 DP 都已收到该请求

Phase 2 — 投票（Vote）
  各 DP 检查本地是否有足够 blocks 分配给自己的 token 份额
  all-reduce MIN → 结果为 1 表示所有 DP 都能分配

结果：
  两轮都通过的请求被批准激活
```

#### 4.1.3 Preemption 同步

```
与 CP Sync 合并执行：

  各 DP 标记自己需要 preempt 的 CP 请求
  all-reduce MAX → 结果为 1 表示至少一个 DP 需要 preempt
  所有 DP 同步 preempt 该请求
```

#### 4.1.4 关键接口

```python
class CPSyncProtocol:
    def __init__(self, dp_group, cp_world_size, cp_rank, sync_interval=4):
        ...

    def should_sync(self) -> bool:
        """判断当前步是否为同步点"""

    def sync_cp_schedule(
        self, pending_request_ids: list[str], can_schedule: list[bool]
    ) -> list[str]:
        """两阶段提交，返回被批准的请求 ID 列表"""

    def sync_preemption(
        self, active_request_ids: list[str], needs_preempt: list[bool]
    ) -> list[str]:
        """Preemption 同步，返回需要 preempt 的请求 ID 列表"""
```

#### 4.1.5 性能分析

| 场景 | 开销 |
|------|------|
| 无 CP 请求 | 零（不触发任何 all-reduce） |
| 有 pending CP 请求 | 每 4 步 2 次 all-reduce（小 tensor，微秒级） |
| 有 active CP 请求需要 preemption 检查 | 每 4 步额外 1 次 all-reduce |

对比：现有的 finish-sync 已经每 32 步做一次 all-reduce，CP sync 的额外开销可忽略。

---

### 4.2 CPAwareScheduler — CP 感知调度器

**文件**：`vllm/v1/core/sched/cp_aware_scheduler.py`

#### 4.2.1 继承关系

```python
class CPAwareScheduler(Scheduler):
    """继承原始 Scheduler，增加 CP 感知能力"""
```

不替换 Scheduler，而是继承它。所有原始调度逻辑（running 队列管理、token budget、preemption 等）完全复用。

#### 4.2.2 请求状态机

```
                    add_request()
                         │
                         ▼
              ┌─── is_long_request? ───┐
              │                        │
             YES                      NO
              │                        │
              ▼                        ▼
     pending_cp_requests        waiting queue
              │                   (原始流程)
              │
     run_cp_sync() [每 N 步]
              │
              ▼
     ┌── all DPs agree? ──┐
     │                     │
    YES                   NO
     │                     │
     ▼                     │
  active_cp_requests       │
  → waiting queue          │
  → running queue          │
  → schedule()             │
     │                     │
     ▼                     │
  完成 or preempt ─────────┘
```

#### 4.2.3 本地 Token 份额计算

CP 请求的 KV cache 分布在所有 DP 上。每个 DP 只分配自己负责的那部分：

```python
def _get_local_cp_tokens(self, total_tokens: int) -> int:
    base = total_tokens // self.cp_world_size
    remainder = total_tokens % self.cp_world_size
    return base + (1 if self.cp_rank < remainder else 0)
```

例如：total_tokens=1000, cp_world_size=4
- rank 0: 250 tokens
- rank 1: 250 tokens
- rank 2: 250 tokens
- rank 3: 250 tokens

#### 4.2.4 SchedulerOutput CP 元数据

调度完成后，为 output 添加 CP 元数据供 Worker 使用：

```python
output.cp_rank = self.cp_rank                    # 本 DP 的 CP rank
output.num_cp_request = len(cp_req_ids)          # CP 请求数量
output.cp_rank_to_req_id = cp_req_ids            # 哪些请求是 CP 请求
output.req_id_to_cp_size = {req_id: cp_ranks}    # 每个 CP 请求的 rank 列表
output.cp_rank_scheduled_tokens = {req_id: cp_world_size}  # CP size
```

Worker 通过这些元数据知道哪些请求需要 CP 通信。

---

### 4.3 DPEngineCoreProc 修改

**文件**：`vllm/v1/engine/core.py`

#### 4.3.1 修改点

在 busy loop 中增加一个 CP sync 调用点：

```python
def run_busy_loop(self):
    while True:
        self._process_input_queue()

        # 新增：批量 CP 同步
        self._maybe_run_cp_sync()

        executed = self._process_engine_step()
        # ... 其余不变
```

#### 4.3.2 _maybe_run_cp_sync 实现

```python
def _maybe_run_cp_sync(self) -> None:
    scheduler = self.scheduler
    if (
        hasattr(scheduler, "cp_sync")
        and scheduler.cp_sync is not None
        and scheduler.has_pending_cp_requests()
        and scheduler.cp_sync.should_sync()
    ):
        scheduler.run_cp_sync()
```

使用 `hasattr` 检查确保向后兼容 — 如果 scheduler 不是 CPAwareScheduler，此方法为空操作。

---

### 4.4 Frontend CP-Aware 路由

**文件**：`vllm/v1/engine/core_client.py`

#### 4.4.1 路由逻辑

在 `DPLBAsyncMPClient` 中覆盖 `add_request_async`：

```python
async def add_request_async(self, request):
    if self._is_cp_request(request):
        # 长请求：广播到所有 engine
        for engine in self.core_engines:
            await self._send_input(ADD, request, engine)
    else:
        # 短请求：路由到负载最低的 engine（原始行为）
        chosen = self.get_core_engine_for_request(request)
        await self._send_input(ADD, request, chosen)
```

#### 4.4.2 长请求判断

```python
def _is_cp_request(self, request) -> bool:
    if self.cp_world_size <= 1:
        return False
    if request.prompt_token_ids is None:
        return False
    return len(request.prompt_token_ids) >= self.long_request_threshold
```

#### 4.4.3 Abort 处理

CP 请求被 abort 时，需要通知所有 engine：

```python
async def abort_requests_async(self, request_ids):
    if self.cp_world_size > 1:
        # 广播 abort 到所有 engine
        for eng in self.core_engines:
            await self._abort_requests(request_ids, eng)
    else:
        # 原始行为
        ...
```

---

## 5. CP 请求完整生命周期

```
时间线：
  Step 0    Step 1    Step 2    Step 3    Step 4    Step 5    ...
    │         │         │         │         │         │
    ▼         ▼         ▼         ▼         ▼         ▼

1. Frontend 收到长请求，广播到所有 DP
   DP_0: add_request → pending_cp_requests
   DP_1: add_request → pending_cp_requests

2. Step 1-3: 请求在 pending 状态等待同步点

3. Step 4: 到达 sync_interval，触发 CP Sync
   DP_0: can_allocate? → YES
   DP_1: can_allocate? → YES
   all-reduce MIN → [1, 1] → 批准！
   DP_0: activate → waiting → running
   DP_1: activate → waiting → running

4. Step 5+: 正常调度执行
   DP_0: schedule() → 分配 local_tokens 的 blocks → SchedulerOutput(cp_rank=0)
   DP_1: schedule() → 分配 local_tokens 的 blocks → SchedulerOutput(cp_rank=1)
   Workers: attention 计算 → CP 通信（all-reduce across DYCP group）

5. 完成：
   DP_0: update_from_output → 检测完成 → 清理 active_cp_requests
   DP_1: update_from_output → 检测完成 → 清理 active_cp_requests
```

---

## 6. 同步保证分析

### 6.1 CP 请求到达时序

**问题**：DP_0 在 Step N 收到请求，DP_1 在 Step N+1 收到。

**保证**：Phase 1（Announce）使用 all-reduce MIN。只有当所有 DP 都标记为 1（已收到）时，结果才为 1。如果 DP_1 还没收到，它标记为 0，MIN 结果为 0，请求不会被批准。下一个 sync 点再次检查。

### 6.2 同步执行保证

**问题**：CP 请求必须在所有 DP 上同一 step 执行。

**保证**：
- 所有 DP 在同一个 sync 点批准请求（all-reduce 是同步操作）
- 批准后，请求进入 running 队列
- 由于所有 DP 的 busy loop 是锁步的（通过现有的 finish-sync all-reduce 保证），running 队列中的请求会在同一 step 被调度

### 6.3 Token 数一致性

**问题**：各 DP 必须为 CP 请求调度相同数量的 tokens。

**保证**：`_get_local_cp_tokens()` 是确定性函数。给定相同的 `total_tokens` 和 `cp_rank`，所有 DP 独立计算出相同的结果。由于请求状态（`num_computed_tokens`）在所有 DP 上同步更新，输入相同则输出相同。

### 6.4 Preemption 一致性

**问题**：一个 DP 内存不足需要 preempt，其他 DP 可能还有空间。

**保证**：Preemption 同步使用 all-reduce MAX。任一 DP 标记为 1（需要 preempt），MAX 结果为 1，所有 DP 同步 preempt。

---

## 7. 与当前方案的对比

### 7.1 架构侵入性

| 维度 | 当前方案（CrossDPScheduler） | 本方案（Distributed Sync） |
|------|--------------------------|--------------------------|
| 新增类 | 4 个（替换原有类） | 2 个（继承/新增） |
| 修改文件 | 6+ 个核心文件 | 2 个核心文件小改 |
| 进程模型 | 改变（1 进程管多 DP） | 保留（每 DP 独立进程） |
| Executor | 新增 DomainMultiprocExecutor | 不变 |
| KVCacheManager | 新增 CrossDPKVCacheManager | 不变 |

### 7.2 性能对比

| 维度 | 当前方案 | 本方案 |
|------|---------|--------|
| 调度决策延迟 | 零（单进程） | 每 4 步 1 次 all-reduce（~10μs on NVLink） |
| CP 请求启动延迟 | 即时 | 最多 4 步（~4 个 decode step） |
| 无 CP 请求时开销 | 有（Domain 层开销） | 零（退化为原始 DP） |
| 内存开销 | 多个 block pool 实例 | 复用现有 block pool |

### 7.3 可维护性

| 维度 | 当前方案 | 本方案 |
|------|---------|--------|
| 与上游 vLLM 差异 | 大（核心架构不同） | 小（继承扩展） |
| 合并上游更新难度 | 高 | 低 |
| 故障隔离 | Domain 故障影响所有 DP | 各 DP 独立 |
| 调试难度 | 单进程，较易 | 分布式，较难 |

---

## 8. 配置参数

本方案复用现有配置参数，无需新增：

| 参数 | 位置 | 说明 |
|------|------|------|
| `dycp_size` | `parallel_config` | CP world size（每个 domain 内的 DP 数） |
| `long_request_threshold` | `scheduler_config` | 长/短请求分界阈值（token 数） |
| `num_cp_seqs` | `scheduler_config` | 最大并发 CP 请求数 |

CPSyncProtocol 的 `sync_interval`（默认 4）可通过代码常量调整，后续可暴露为配置项。

---

## 9. 文件清单

### 9.1 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `vllm/v1/core/sched/cp_sync.py` | ~170 | CPSyncProtocol 分布式共识协议 |
| `vllm/v1/core/sched/cp_aware_scheduler.py` | ~220 | CPAwareScheduler，继承 Scheduler |

### 9.2 修改文件

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `vllm/v1/engine/core.py` | +14 行 | DPEngineCoreProc 增加 `_maybe_run_cp_sync()` |
| `vllm/v1/engine/core_client.py` | +67 行 | DPLBAsyncMPClient 增加 CP 路由和广播 |

### 9.3 不需要修改的文件（与当前方案的关键区别）

- `vllm/v1/executor/multiproc_executor.py` — 无需 DomainMultiprocExecutor
- `vllm/v1/core/cross_dp_kv_cache_manager.py` — 无需，复用现有 KVCacheManager
- `vllm/v1/core/block_pool.py` — 无需 DPBlockPool

---

## 10. 局限性与后续工作

### 10.1 当前局限

1. **CP 启动延迟**：最多 4 步延迟（sync_interval=4）。对于长序列的 prefill 阶段，这个延迟相对于序列处理时间可忽略。
2. **分布式调试**：CP 同步问题需要跨进程调试，比单进程方案更复杂。
3. **Preemption 粒度**：当前实现中，任一 DP 触发 preemption 会导致所有 DP 都 preempt，可能过于保守。

### 10.2 后续优化方向

1. **自适应 sync_interval**：根据 pending CP 请求数量动态调整同步频率
2. **Preemption 优化**：区分"必须 preempt"和"建议 preempt"，减少不必要的 preemption
3. **负载感知路由优化**：将 CP 请求的负载也纳入短请求路由的考量
4. **Chunked Prefill 支持**：CP 请求的 chunked prefill 需要额外的 token 分配协调
