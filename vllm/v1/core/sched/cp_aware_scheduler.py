# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CP-Aware Scheduler for Dynamic Context Parallel.

Extends the existing Scheduler with CP awareness while preserving the
per-DP independent process architecture. CP requests are coordinated
via CPSyncProtocol rather than a centralized CrossDPScheduler.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.cp_sync import CPSyncProtocol
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = init_logger(__name__)


class CPAwareScheduler(Scheduler):
    """Scheduler with CP awareness for distributed DYCP.

    Each DP rank runs its own CPAwareScheduler. CP requests are coordinated
    via a distributed sync protocol using the existing dp_group.

    Key differences from base Scheduler:
    - Classifies requests as long (CP) or short (DP) based on token threshold
    - CP requests enter a pending state until all DPs confirm readiness
    - Only allocates local portion of KV cache for CP requests
    - Adds CP metadata to SchedulerOutput
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        dp_group: "ProcessGroup | None" = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        self.cp_world_size = vllm_config.parallel_config.dycp_size
        self.cp_rank = (
            vllm_config.parallel_config.data_parallel_rank % self.cp_world_size
        )
        self.long_request_threshold = (
            vllm_config.scheduler_config.long_request_threshold
        )
        self.max_cp_requests = vllm_config.scheduler_config.num_cp_seqs

        # CP request state management.
        self.pending_cp_requests: dict[str, Request] = {}
        self.active_cp_requests: dict[str, Request] = {}

        # CP sync protocol (None if dp_group not provided, e.g., single DP).
        self.cp_sync: CPSyncProtocol | None = None
        if dp_group is not None and self.cp_world_size > 1:
            self.cp_sync = CPSyncProtocol(
                dp_group=dp_group,
                cp_world_size=self.cp_world_size,
                cp_rank=self.cp_rank,
                sync_interval=4,
            )

    # ------------------------------------------------------------------
    # Request classification and routing
    # ------------------------------------------------------------------

    def _is_long_request(self, request: Request) -> bool:
        """Classify request as long (CP) based on prefill token count."""
        num_prefill_tokens = request.num_tokens - request.num_output_tokens
        return num_prefill_tokens >= self.long_request_threshold

    def _get_local_cp_tokens(self, total_tokens: int) -> int:
        """Calculate how many tokens this rank stores for a CP request."""
        base = total_tokens // self.cp_world_size
        remainder = total_tokens % self.cp_world_size
        return base + (1 if self.cp_rank < remainder else 0)

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        """Add request, routing to pending_cp or normal waiting queue."""
        if self.cp_world_size <= 1 or not self._is_long_request(request):
            request.cp_ranks = [self.cp_rank]
            super().add_request(request)
        else:
            request.cp_ranks = list(range(self.cp_world_size))
            self.pending_cp_requests[request.request_id] = request
            self.requests[request.request_id] = request
            logger.debug(
                "CP request %s added to pending (tokens=%d, rank=%d)",
                request.request_id,
                request.num_tokens,
                self.cp_rank,
            )

    def has_pending_cp_requests(self) -> bool:
        """Check if there are CP requests waiting for sync."""
        return len(self.pending_cp_requests) > 0

    def get_num_unfinished_requests(self) -> int:
        return (
            super().get_num_unfinished_requests()
            + len(self.pending_cp_requests)
        )

    # ------------------------------------------------------------------
    # CP Sync: activate pending CP requests via distributed consensus
    # ------------------------------------------------------------------

    def run_cp_sync(self) -> None:
        """Execute CP sync protocol to activate pending requests.

        Called from DPEngineCoreProc busy loop at sync intervals.
        """
        if self.cp_sync is None or not self.pending_cp_requests:
            return

        pending_ids = sorted(self.pending_cp_requests.keys())

        # Check local schedulability: can we allocate blocks for our portion?
        can_schedule: list[bool] = []
        num_free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        for req_id in pending_ids:
            request = self.pending_cp_requests[req_id]
            local_tokens = self._get_local_cp_tokens(
                request.num_tokens - request.num_computed_tokens
            )
            num_blocks_needed = (local_tokens + self.block_size - 1) // self.block_size
            can_schedule.append(num_blocks_needed <= num_free)

        # Distributed consensus.
        approved_ids = self.cp_sync.sync_cp_schedule(pending_ids, can_schedule)

        # Activate approved requests.
        for req_id in approved_ids:
            self._activate_cp_request(req_id)

        # Also sync preemption for active CP requests.
        self._sync_cp_preemption()

    def _activate_cp_request(self, request_id: str) -> None:
        """Move a CP request from pending to active (running queue)."""
        request = self.pending_cp_requests.pop(request_id)
        self.active_cp_requests[request_id] = request
        # Add to the normal waiting queue so schedule() picks it up.
        self.waiting.add_request(request)
        logger.debug(
            "CP request %s activated on rank %d",
            request_id,
            self.cp_rank,
        )

    def _sync_cp_preemption(self) -> None:
        """Check and propagate preemption needs for active CP requests."""
        if not self.active_cp_requests or self.cp_sync is None:
            return

        active_ids = sorted(self.active_cp_requests.keys())
        needs_preempt: list[bool] = []
        num_free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        for req_id in active_ids:
            # A rank needs preemption if it can't allocate the next decode block.
            local_tokens = self._get_local_cp_tokens(1)  # decode: 1 token
            num_blocks_needed = (local_tokens + self.block_size - 1) // self.block_size
            needs_preempt.append(num_blocks_needed > num_free)

        preempted_ids = self.cp_sync.sync_preemption(active_ids, needs_preempt)

        for req_id in preempted_ids:
            self._preempt_cp_request(req_id)

    def _preempt_cp_request(self, request_id: str) -> None:
        """Preempt a CP request: remove from running, free blocks."""
        if request_id not in self.active_cp_requests:
            return
        request = self.active_cp_requests.pop(request_id)
        # Remove from running queue if present.
        if request in self.running:
            self.running.remove(request)
        # Free KV cache blocks.
        self.kv_cache_manager.free(request)
        # Move back to pending for retry.
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        self.pending_cp_requests[request.request_id] = request
        logger.debug(
            "CP request %s preempted on rank %d",
            request_id,
            self.cp_rank,
        )

    # ------------------------------------------------------------------
    # Schedule override: add CP metadata to output
    # ------------------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        """Schedule requests, adding CP metadata to output."""
        output = super().schedule()

        # Annotate output with CP metadata.
        output.cp_rank = self.cp_rank

        # Count CP requests and build metadata.
        cp_req_ids: list[str] = []
        req_id_to_cp_size: dict[str, list[int]] = {}

        for req_id in output.num_scheduled_tokens:
            if req_id in self.active_cp_requests:
                request = self.active_cp_requests[req_id]
                cp_req_ids.append(req_id)
                req_id_to_cp_size[req_id] = request.cp_ranks

        output.num_cp_request = len(cp_req_ids)
        output.cp_rank_to_req_id = cp_req_ids if cp_req_ids else None
        output.req_id_to_cp_size = req_id_to_cp_size if req_id_to_cp_size else None

        # For CP requests, adjust scheduled tokens to local portion.
        if output.cp_rank_scheduled_tokens is None:
            output.cp_rank_scheduled_tokens = {}
        for req_id in cp_req_ids:
            output.cp_rank_scheduled_tokens[req_id] = self.cp_world_size

        return output

    # ------------------------------------------------------------------
    # Update from output: handle CP request completion
    # ------------------------------------------------------------------

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Update scheduler state from model output."""
        result = super().update_from_output(scheduler_output, model_runner_output)

        # Check if any active CP requests have finished.
        finished_cp = [
            req_id
            for req_id in list(self.active_cp_requests.keys())
            if req_id not in self.requests
        ]
        for req_id in finished_cp:
            del self.active_cp_requests[req_id]

        return result
