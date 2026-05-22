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

# Three-state encoding for post-schedule consensus.
SCHEDULED = 2
NOT_SCHEDULED = 1
PREEMPTED = 0


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
        # Confirm tensor: post-schedule three-state consensus.
        self._confirm_tensor = torch.zeros(
            MAX_CP_SYNC_SLOTS, dtype=torch.int32, device="cpu"
        )

    def should_sync(self) -> bool:
        """Check if this step is a sync point."""
        self.step_counter += 1
        return self.step_counter % self.sync_interval == 0

    def sync_announce(
        self,
        pending_request_ids: list[str],
    ) -> list[str]:
        """Announce phase: confirm all ranks have received the same requests.

        Args:
            pending_request_ids: CP request IDs pending on this rank (sorted).

        Returns:
            List of request IDs known to ALL ranks.
        """
        num_slots = min(len(pending_request_ids), MAX_CP_SYNC_SLOTS)
        if num_slots == 0:
            return []

        self._announce_tensor.zero_()
        for i in range(num_slots):
            self._announce_tensor[i] = 1

        torch.distributed.all_reduce(
            self._announce_tensor[:num_slots],
            op=torch.distributed.ReduceOp.MIN,
            group=self.dp_group,
        )

        announced: list[str] = []
        for i in range(num_slots):
            if self._announce_tensor[i].item() == 1:
                announced.append(pending_request_ids[i])

        if announced:
            logger.debug(
                "CP announce: %d/%d requests known to all ranks on rank %d",
                len(announced),
                len(pending_request_ids),
                self.cp_rank,
            )

        return announced

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
            self._confirm_tensor[:num_slots],
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
