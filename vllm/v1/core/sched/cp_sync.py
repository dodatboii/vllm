# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed CP Sync Protocol for Dynamic Context Parallel.

Implements a post-schedule consensus protocol for coordinating CP request
scheduling across independent DP ranks. Uses the existing dp_group all-reduce.

Design rationale: CP requests are broadcast to ALL DP ranks by the client
layer (DPEngineCoreClient.add_request_async), so every rank is guaranteed to
receive the same CP request. No announce/vote phase is needed. The only
coordination required is a post-schedule all-reduce MIN to agree on whether
each rank successfully scheduled each CP request in a given step.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed

from vllm.logger import init_logger

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = init_logger(__name__)

# Maximum number of CP requests that can be synced in one round.
MAX_CP_SYNC_SLOTS = 32

# Three-state encoding for post-schedule consensus.
SCHEDULED = 2
NOT_SCHEDULED = 1
PREEMPTED = 0


class CPSyncProtocol:
    """Post-schedule consensus protocol for CP request scheduling.

    Uses the existing dp_group all-reduce MIN to agree on whether each active
    CP request was successfully scheduled on ALL ranks in the current step.

    Three-state encoding per request:
        SCHEDULED (2)     - this rank scheduled the request
        NOT_SCHEDULED (1) - this rank did not schedule it (budget exhausted)
        PREEMPTED (0)     - this rank preempted it (KV cache eviction)

    After all-reduce MIN:
        min >= SCHEDULED  -> confirmed: execute on all ranks
        min >= NOT_SCHEDULED -> soft rollback: re-queue without resetting KV
        min == PREEMPTED  -> hard rollback: all ranks evict and reset
    """

    def __init__(
        self,
        dp_group: ProcessGroup,
        cp_world_size: int,
        cp_rank: int,
    ):
        self.dp_group = dp_group
        self.cp_world_size = cp_world_size
        self.cp_rank = cp_rank

        # Pre-allocated tensor for post-schedule all-reduce (avoids per-call
        # allocation on the hot path).
        self._confirm_tensor = torch.zeros(
            MAX_CP_SYNC_SLOTS, dtype=torch.int32, device="cpu"
        )

    def sync_schedule_confirm(
        self,
        active_ids: list[str],
        status: list[int],
    ) -> tuple[list[str], list[str], list[str]]:
        """Post-schedule consensus using three-state encoding.

        Args:
            active_ids: Active CP request IDs (sorted, identical across ranks).
            status: Per-request status on this rank
                    (SCHEDULED=2 / NOT_SCHEDULED=1 / PREEMPTED=0).

        Returns:
            (confirmed_ids, soft_rollback_ids, hard_rollback_ids)
        """
        num_slots = min(len(active_ids), MAX_CP_SYNC_SLOTS)
        if num_slots == 0:
            return [], [], []

        self._confirm_tensor.zero_()
        for i in range(num_slots):
            self._confirm_tensor[i] = status[i]

        torch.distributed.all_reduce(
            self._confirm_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.dp_group,
        )

        confirmed: list[str] = []
        soft_rollback: list[str] = []
        hard_rollback: list[str] = []
        for i in range(num_slots):
            val = self._confirm_tensor[i].item()
            if val >= SCHEDULED:
                confirmed.append(active_ids[i])
            elif val >= NOT_SCHEDULED:
                soft_rollback.append(active_ids[i])
            else:
                hard_rollback.append(active_ids[i])

        if soft_rollback or hard_rollback:
            logger.debug(
                "CP confirm: confirmed=%d soft_rollback=%d hard_rollback=%d"
                " on rank %d",
                len(confirmed),
                len(soft_rollback),
                len(hard_rollback),
                self.cp_rank,
            )

        return confirmed, soft_rollback, hard_rollback

    def sync_empty(self) -> None:
        """Participate in the sync all_reduce with no active CP requests.

        Called by ranks that have no active CP requests in a given step so
        that peer ranks which do have active requests are not blocked waiting
        for all participants in the collective operation.

        Fills the tensor with NOT_SCHEDULED (1) so that the MIN operation does
        not drive any active slot down to PREEMPTED (0), which would trigger
        spurious hard-rollbacks on the peer ranks.
        """
        self._confirm_tensor.fill_(NOT_SCHEDULED)
        torch.distributed.all_reduce(
            self._confirm_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=self.dp_group,
        )
