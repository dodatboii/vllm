# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CP-Aware Scheduler for Dynamic Context Parallel.

Extends the existing Scheduler with CP awareness while preserving the
per-DP independent process architecture. CP requests are coordinated
via CPSyncProtocol.

Design rationale: the client layer (DPEngineCoreClient) broadcasts CP requests
to ALL DP ranks before they reach the scheduler, so every rank is guaranteed to
hold the same CP request. No pre-schedule announce/vote phase is needed.
Coordination is a single post-schedule all-reduce MIN that agrees on whether
each rank successfully scheduled each CP request in the current step.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.cp_sync import (
    NOT_SCHEDULED,
    PREEMPTED,
    SCHEDULED,
    CPSyncProtocol,
)
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
    - CP requests are activated immediately on arrival (client guarantees
      broadcast to all ranks, so no announce phase is needed)
    - Only allocates local portion of KV cache for CP requests
    - Adds CP metadata to SchedulerOutput
    - Post-schedule all-reduce MIN agrees on confirmed/rollback decisions
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

        # Active CP requests: request_id -> Request.
        # CP requests are added here immediately on arrival; no pending state.
        self.active_cp_requests: dict[str, Request] = {}

        # Tracks CP requests preempted by schedule() in the current step.
        self._preempted_this_step: set[str] = set()

        # CP sync protocol (None if dp_group not provided, e.g., single DP).
        self.cp_sync: CPSyncProtocol | None = None
        if dp_group is not None and self.cp_world_size > 1:
            self.cp_sync = CPSyncProtocol(
                dp_group=dp_group,
                cp_world_size=self.cp_world_size,
                cp_rank=self.cp_rank,
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
        """Add request, routing CP requests directly to active state."""
        if self.cp_world_size <= 1 or not self._is_long_request(request):
            request.cp_ranks = [self.cp_rank]
            super().add_request(request)
        else:
            # Client guarantees CP requests are broadcast to all ranks, so
            # activate immediately without a pending/announce phase.
            request.cp_ranks = list(range(self.cp_world_size))
            self.active_cp_requests[request.request_id] = request
            self.requests[request.request_id] = request
            self.waiting.add_request(request)
            logger.debug(
                "CP request %s activated on rank %d (tokens=%d)",
                request.request_id,
                self.cp_rank,
                request.num_tokens,
            )

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Override to track CP requests preempted by schedule()."""
        if request.request_id in self.active_cp_requests:
            self._preempted_this_step.add(request.request_id)
        super()._preempt_request(request, timestamp)

    # ------------------------------------------------------------------
    # Schedule override: add CP metadata to output
    # ------------------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        """Schedule requests, adding CP metadata to output."""
        self._preempted_this_step.clear()
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
    # Post-schedule CP sync: consensus based on actual schedule result
    # ------------------------------------------------------------------

    def post_schedule_cp_sync(self, output: SchedulerOutput) -> SchedulerOutput:
        """Post-schedule sync: consensus on actual scheduling results."""
        if not self.active_cp_requests or self.cp_sync is None:
            self._preempted_this_step.clear()
            return output

        active_ids = sorted(self.active_cp_requests.keys())

        status: list[int] = []
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

        # Pass sorted confirmed CP req IDs to workers for index computation.
        if confirmed:
            output.cp_req_ids_sorted = sorted(confirmed)
        else:
            output.cp_req_ids_sorted = None

        # If this rank has no tokens to execute but peer ranks do have CP
        # tokens, signal workers to run a dummy forward pass for collective ops.
        if output.total_num_scheduled_tokens == 0 and confirmed:
            output.none_tokens_in_peer_sched = True

        return output

    def _soft_rollback(
        self, output: SchedulerOutput, rollback_ids: list[str]
    ) -> SchedulerOutput:
        """Remove from output without resetting num_computed_tokens."""
        for req_id in rollback_ids:
            if req_id in output.num_scheduled_tokens:
                num_tokens = output.num_scheduled_tokens.pop(req_id)
                output.total_num_scheduled_tokens -= num_tokens
                self._remove_req_from_output(output, req_id)

                request = self.active_cp_requests[req_id]
                self.kv_cache_manager.free(request)

                request.status = RequestStatus.WAITING
                if request in self.running:
                    self.running.remove(request)
                self.waiting.prepend_request(request)

        return output

    def _hard_rollback(
        self, output: SchedulerOutput, rollback_ids: list[str]
    ) -> SchedulerOutput:
        """Full rollback: all ranks preempt, reset num_computed_tokens=0."""
        for req_id in rollback_ids:
            if req_id in output.num_scheduled_tokens:
                num_tokens = output.num_scheduled_tokens.pop(req_id)
                output.total_num_scheduled_tokens -= num_tokens
                self._remove_req_from_output(output, req_id)

            request = self.active_cp_requests[req_id]

            if req_id not in self._preempted_this_step:
                # schedule() did not preempt this rank; do it manually.
                self.kv_cache_manager.free(request)
                request.num_computed_tokens = 0
                request.status = RequestStatus.PREEMPTED
                if request in self.running:
                    self.running.remove(request)

            # Keep in active_cp_requests; re-queue for the next step.
            if request not in self.waiting:
                self.waiting.prepend_request(request)

        return output

    def _remove_req_from_output(
        self, output: SchedulerOutput, req_id: str
    ) -> None:
        """Remove a request from all SchedulerOutput fields."""
        output.scheduled_new_reqs = [
            r for r in output.scheduled_new_reqs if r.req_id != req_id
        ]

        cached = output.scheduled_cached_reqs
        if req_id in cached.req_ids:
            idx = cached.req_ids.index(req_id)
            cached.req_ids.pop(idx)
            cached.new_token_ids.pop(idx)
            cached.new_block_ids.pop(idx)
            cached.num_computed_tokens.pop(idx)
            cached.num_output_tokens.pop(idx)
            cached.resumed_req_ids.discard(req_id)
            if req_id in cached.all_token_ids:
                del cached.all_token_ids[req_id]

        if output.cp_rank_scheduled_tokens and req_id in output.cp_rank_scheduled_tokens:
            del output.cp_rank_scheduled_tokens[req_id]
        if output.cp_rank_to_req_id and req_id in output.cp_rank_to_req_id:
            output.cp_rank_to_req_id.remove(req_id)
        if output.req_id_to_cp_size and req_id in output.req_id_to_cp_size:
            del output.req_id_to_cp_size[req_id]
        output.num_cp_request = max(0, output.num_cp_request - 1)

        if output.scheduled_spec_decode_tokens and req_id in output.scheduled_spec_decode_tokens:
            del output.scheduled_spec_decode_tokens[req_id]

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
