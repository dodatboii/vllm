# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed CP Sync Protocol for Dynamic Context Parallel.

Implements a batch-synchronized two-phase commit protocol for coordinating
CP request scheduling across independent DP ranks. Uses the existing dp_group
all-reduce mechanism.
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


class CPSyncProtocol:
    """Distributed consensus protocol for CP request scheduling.

    Uses the existing dp_group all-reduce to coordinate CP decisions
    across independent DP ranks. Operates in batch-sync mode: every
    sync_interval steps, pending CP requests are voted on.

    Protocol:
        Phase 1 (announce): all-reduce to confirm all DPs have received
                            the same set of pending CP requests.
        Phase 2 (vote):     all-reduce MIN to confirm all DPs can allocate
                            blocks for their portion.
        Result:             only requests approved by ALL DPs are activated.
    """

    def __init__(
        self,
        dp_group: ProcessGroup,
        cp_world_size: int,
        cp_rank: int,
        sync_interval: int = 4,
    ):
        self.dp_group = dp_group
        self.cp_world_size = cp_world_size
        self.cp_rank = cp_rank
        self.sync_interval = sync_interval
        self.step_counter = 0

        # Pre-allocated tensors for all-reduce (avoid per-call allocation).
        # Vote tensor: one slot per potential CP request in a sync round.
        self._vote_tensor = torch.zeros(
            MAX_CP_SYNC_SLOTS, dtype=torch.int32, device="cpu"
        )
        # Preemption tensor: same layout.
        self._preempt_tensor = torch.zeros(
            MAX_CP_SYNC_SLOTS, dtype=torch.int32, device="cpu"
        )
        # Announce tensor: tracks which request IDs are known to this rank.
        # Uses a hash-based approach: each pending request hashes to a slot.
        self._announce_tensor = torch.zeros(
            MAX_CP_SYNC_SLOTS, dtype=torch.int32, device="cpu"
        )

    def should_sync(self) -> bool:
        """Check if this step is a sync point."""
        self.step_counter += 1
        return self.step_counter % self.sync_interval == 0

    def sync_cp_schedule(
        self,
        pending_request_ids: list[str],
        can_schedule: list[bool],
    ) -> list[str]:
        """Two-phase commit for CP request scheduling.

        Args:
            pending_request_ids: CP request IDs pending on this rank.
            can_schedule: For each pending request, whether this rank
                          has enough KV cache blocks for its portion.

        Returns:
            List of request IDs approved for scheduling (all DPs agreed).
        """
        num_pending = len(pending_request_ids)
        if num_pending == 0:
            return []

        num_slots = min(num_pending, MAX_CP_SYNC_SLOTS)

        # Phase 1: Announce which requests this rank knows about.
        # Each rank sets a 1 for requests it has. all-reduce with MIN:
        # result is 1 only if ALL ranks have that request.
        self._announce_tensor.zero_()
        for i in range(num_slots):
            self._announce_tensor[i] = 1

        torch.distributed.all_reduce(
            self._announce_tensor[:num_slots],
            op=torch.distributed.ReduceOp.MIN,
            group=self.dp_group,
        )

        # Phase 2: For requests known to all ranks, vote on schedulability.
        self._vote_tensor.zero_()
        for i in range(num_slots):
            if self._announce_tensor[i].item() == 1 and can_schedule[i]:
                self._vote_tensor[i] = 1

        torch.distributed.all_reduce(
            self._vote_tensor[:num_slots],
            op=torch.distributed.ReduceOp.MIN,
            group=self.dp_group,
        )

        # Collect approved request IDs.
        approved: list[str] = []
        for i in range(num_slots):
            if self._vote_tensor[i].item() == 1:
                approved.append(pending_request_ids[i])

        if approved:
            logger.debug(
                "CP sync: approved %d/%d requests on rank %d",
                len(approved),
                num_pending,
                self.cp_rank,
            )

        return approved

    def sync_preemption(
        self,
        active_request_ids: list[str],
        needs_preempt: list[bool],
    ) -> list[str]:
        """Synchronize preemption decisions across all DPs.

        If ANY DP needs to preempt a CP request, ALL DPs must preempt it.
        Uses all-reduce with MAX to propagate preemption signals.

        Args:
            active_request_ids: Currently active CP request IDs.
            needs_preempt: For each active request, whether this rank
                           needs to preempt it (e.g., OOM).

        Returns:
            List of request IDs that ALL DPs must preempt.
        """
        num_active = len(active_request_ids)
        if num_active == 0:
            return []

        num_slots = min(num_active, MAX_CP_SYNC_SLOTS)

        self._preempt_tensor.zero_()
        for i in range(num_slots):
            if needs_preempt[i]:
                self._preempt_tensor[i] = 1

        torch.distributed.all_reduce(
            self._preempt_tensor[:num_slots],
            op=torch.distributed.ReduceOp.MAX,
            group=self.dp_group,
        )

        preempted: list[str] = []
        for i in range(num_slots):
            if self._preempt_tensor[i].item() == 1:
                preempted.append(active_request_ids[i])

        if preempted:
            logger.debug(
                "CP sync: preempting %d requests on rank %d",
                len(preempted),
                self.cp_rank,
            )

        return preempted
