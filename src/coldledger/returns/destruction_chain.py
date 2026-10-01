"""Recoverable state chain for authorized return-destruction orders.

The reverse flow crosses six boundaries (authorization, quarantine,
approval, carrier handover, destruction certificate, financial write-off).
Each forward step and every compensation is an atomic, balanced *voucher
group* recorded as a saga event, so the chain can always be rebuilt and
resumed:

* external receipts (carrier handover, destruction certificate) are
  de-duplicated chain-wide -- a re-sent callback is a no-op, never a second
  posting;
* approval withdrawal compensates every voucher the chain has posted
  (quarantine stock move **and** the approval provision) as one group under
  both ledger locks, then emits a single event. It is impossible to observe
  a state where only one ledger was rolled back;
* every voucher reference is derived from ``(chain, stage, group, ledger)``,
  so a process crashing between the ledger commit and the saga append simply
  re-attaches to the already-posted voucher on retry/recovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from threading import RLock
from typing import Iterable

from coldledger.storage.event_store import EventStore

# --- stages ----------------------------------------------------------------

STAGE_AUTHORIZED = "authorized"
STAGE_QUARANTINED = "quarantined"
STAGE_APPROVED = "approved"
STAGE_HANDED_OVER = "handed_over"
STAGE_DESTROYED = "destroyed"
STAGE_WRITTEN_OFF = "written_off"
STAGE_CLOSED = "closed"
STAGE_CANCELED = "canceled"

FORWARD_STAGES = (
    STAGE_QUARANTINED,
    STAGE_APPROVED,
    STAGE_HANDED_OVER,
    STAGE_DESTROYED,
    STAGE_WRITTEN_OFF,
    STAGE_CLOSED,
)
TERMINAL_STAGES = frozenset({STAGE_CLOSED, STAGE_CANCELED})
RECEIPT_STAGES = frozenset({STAGE_HANDED_OVER, STAGE_DESTROYED})

_NEXT: dict[str, frozenset[str]] = {
    STAGE_AUTHORIZED: frozenset({STAGE_QUARANTINED}),
    STAGE_QUARANTINED: frozenset({STAGE_APPROVED}),
    STAGE_APPROVED: frozenset({STAGE_HANDED_OVER, STAGE_CANCELED}),
    STAGE_HANDED_OVER: frozenset({STAGE_DESTROYED}),
    STAGE_DESTROYED: frozenset({STAGE_WRITTEN_OFF}),
    STAGE_WRITTEN_OFF: frozenset({STAGE_CLOSED}),
    STAGE_CLOSED: frozenset(),
    STAGE_CANCELED: frozenset(),
}


class ChainError(ValueError):
    """Illegal state-chain command."""


# --- atomic, balanced voucher ledger ---------------------------------------

@dataclass(frozen=True)
class VoucherLeg:
    account: str
    amount: Decimal
    reason: str


@dataclass(frozen=True)
class Voucher:
    reference: str
    legs: tuple[VoucherLeg, ...]
    at: datetime
    reversal_of: str | None = None

    @property
    def balanced(self) -> bool:
        return sum((leg.amount for leg in self.legs), Decimal("0")) == 0


class VoucherLedger:
    """Double-entry ledger whose unit of posting is a whole balanced voucher.

    A reversal always negates *every* leg of the original voucher in one
    commit; there is no API that can reverse a single leg. Both ``post`` and
    ``reverse`` are idempotent on their voucher reference: a retried call
    with the same reference and identical content returns the prior voucher
    instead of posting again.
    """

    def __init__(self, name: str):
        self.name = name
        self._vouchers: dict[str, Voucher] = {}
        self._reversal_of: dict[str, str] = {}
        self._lock = RLock()

    def lock(self) -> RLock:
        return self._lock

    @staticmethod
    def _normalize(legs: Iterable[tuple[str, object, str]]) -> tuple[VoucherLeg, ...]:
        normalized = tuple(
            VoucherLeg(str(account), Decimal(str(amount)), reason)
            for account, amount, reason in legs
        )
        if not normalized:
            raise ValueError("voucher has no legs")
        if any(not leg.account or not leg.reason for leg in normalized):
            raise ValueError("account and reason required")
        if any(leg.amount == 0 for leg in normalized):
            raise ValueError("zero amount voucher leg")
        return normalized

    def post_voucher(
        self,
        reference: str,
        legs: Iterable[tuple[str, object, str]],
        at: datetime,
    ) -> Voucher:
        wanted = self._normalize(legs)
        if sum((leg.amount for leg in wanted), Decimal("0")) != 0:
            raise ValueError("unbalanced voucher")
        when = at.astimezone(timezone.utc)
        with self._lock:
            existing = self._vouchers.get(reference)
            if existing is not None:
                if existing.legs != wanted:
                    raise ValueError(f"voucher {reference} already posted with different legs")
                return existing
            voucher = Voucher(reference, wanted, when)
            self._vouchers[reference] = voucher
            return voucher

    def reverse_voucher(
        self,
        original_reference: str,
        reversal_reference: str,
        at: datetime,
        reason: str,
    ) -> Voucher:
        when = at.astimezone(timezone.utc)
        with self._lock:
            original = self._vouchers.get(original_reference)
            if original is None:
                raise KeyError(original_reference)
            prior_ref = self._reversal_of.get(original_reference)
            if prior_ref is not None:
                if prior_ref != reversal_reference:
                    raise ValueError(f"voucher {original_reference} already reversed")
                return self._vouchers[prior_ref]
            if reversal_reference in self._vouchers:
                raise ValueError(f"voucher {reversal_reference} already posted")
            legs = tuple(
                VoucherLeg(leg.account, -leg.amount, reason or leg.reason)
                for leg in original.legs
            )
            reversal = Voucher(reversal_reference, legs, when, original_reference)
            self._vouchers[reversal_reference] = reversal
            self._reversal_of[original_reference] = reversal_reference
            return reversal

    def voucher(self, reference: str) -> Voucher:
        with self._lock:
            return self._vouchers[reference]

    def vouchers(self) -> tuple[Voucher, ...]:
        with self._lock:
            return tuple(self._vouchers.values())

    def is_reversed(self, reference: str) -> bool:
        with self._lock:
            return reference in self._reversal_of

    def balance(self, account: str) -> Decimal:
        with self._lock:
            return sum(
                (leg.amount for v in self._vouchers.values() for leg in v.legs if leg.account == account),
                Decimal("0"),
            )

    def audit(self) -> None:
        with self._lock:
            for ref, voucher in self._vouchers.items():
                if not voucher.balanced:
                    raise ValueError(f"unbalanced voucher {ref}")
                if voucher.reversal_of is not None:
                    original = self._vouchers.get(voucher.reversal_of)
                    if original is None:
                        raise ValueError(f"reversal {ref} points at missing voucher")
                    if self._reversal_of.get(voucher.reversal_of) != ref:
                        raise ValueError(f"reversal {ref} not linked back")


# --- saga records -----------------------------------------------------------

@dataclass(frozen=True)
class GroupRecord:
    stage: str
    group: str
    direction: str  # "forward" | "reversal"
    voucher_refs: tuple[str, ...]
    at: datetime


@dataclass(frozen=True)
class ChainState:
    reference: str
    stage: str
    revision: int
    sku: str
    quantity: Decimal
    amount: Decimal
    actor: str
    updated_at: datetime | None
    groups: tuple[GroupRecord, ...] = ()
    receipts: tuple[tuple[str, str], ...] = ()
    auth_note: str = ""


@dataclass(frozen=True)
class Advancement:
    state: ChainState
    duplicate: bool
    voucher_refs: tuple[str, ...]


@dataclass(frozen=True)
class Reconciliation:
    reference: str
    consistent: bool
    stage: str
    expected: dict[str, Decimal] = field(default_factory=dict)
    actual: dict[str, Decimal] = field(default_factory=dict)
    differences: dict[str, Decimal] = field(default_factory=dict)
    voucher_refs: tuple[str, ...] = ()
    reversed_refs: tuple[str, ...] = ()


# --- stage posting specification -------------------------------------------

def _group_spec(stage: str, qty: Decimal, amount: Decimal) -> list[tuple[str, str, list[tuple[str, Decimal, str]]]]:
    """Return [(group, ledger, legs)] posted atomically when entering stage."""
    if stage == STAGE_QUARANTINED:
        return [("quarantine", "stock", [
            ("stock:on_hand", -qty, "returned goods quarantined"),
            ("stock:quarantine", qty, "returned goods quarantined"),
        ])]
    if stage == STAGE_APPROVED:
        return [("approval", "finance", [
            ("expense:destruction", amount, "destruction approved"),
            ("liability:destruction_provision", -amount, "destruction approved"),
        ])]
    if stage == STAGE_HANDED_OVER:
        return [("handover", "stock", [
            ("stock:quarantine", -qty, "goods handed to carrier"),
            ("stock:in_custody_carrier", qty, "goods handed to carrier"),
        ])]
    if stage == STAGE_DESTROYED:
        return [("destruction", "stock", [
            ("stock:in_custody_carrier", -qty, "goods destroyed"),
            ("stock:destroyed", qty, "goods destroyed"),
        ])]
    if stage == STAGE_WRITTEN_OFF:
        # One group across two ledgers, committed under both locks together.
        return [
            ("writeoff", "stock", [
                ("stock:destroyed", -qty, "destruction written off"),
                ("stock:written_off", qty, "destruction written off"),
            ]),
            ("writeoff", "finance", [
                ("liability:destruction_provision", amount, "destruction written off"),
                ("asset:inventory_value", -amount, "destruction written off"),
            ]),
        ]
    return []


class DestructionChains:
    """Event-sourced coordinator over one stock and one finance voucher ledger."""

    STREAM_PREFIX = "destruction:"

    def __init__(
        self,
        stock: VoucherLedger | None = None,
        finance: VoucherLedger | None = None,
        store: EventStore | None = None,
    ):
        self.stock = stock or VoucherLedger("stock")
        self.finance = finance or VoucherLedger("finance")
        self.store = store or EventStore()
        self._chains: dict[str, ChainState] = {}
        self._receipts: dict[str, tuple[str, str]] = {}
        self._lock = RLock()

    # -- ledger access -------------------------------------------------------

    def _ledger(self, name: str) -> VoucherLedger:
        return self.stock if name == "stock" else self.finance

    def _ref_ledger(self, ref: str) -> VoucherLedger:
        # forward refs end ``:<ledger>``; reversals end ``:<ledger>:reversal``
        token = ref.rsplit(":", 2)
        name = token[-2] if token[-1] == "reversal" else token[-1]
        return self._ledger(name)

    def _stream(self, reference: str) -> str:
        return self.STREAM_PREFIX + reference

    # -- open / query --------------------------------------------------------

    def open(
        self,
        reference: str,
        sku: str,
        quantity: object,
        amount: object,
        actor: str,
        at: datetime,
        auth_note: str = "",
    ) -> ChainState:
        qty = Decimal(str(quantity))
        amt = Decimal(str(amount))
        if qty <= 0 or amt < 0:
            raise ValueError("quantity must be positive, amount non-negative")
        with self._lock:
            if reference in self._chains:
                raise ChainError(f"chain {reference} already open")
            when = at.astimezone(timezone.utc)
            self.store.append(self._stream(reference), 0, "opened", {
                "reference": reference, "sku": sku, "quantity": str(qty),
                "amount": str(amt), "actor": actor, "at": when, "auth_note": auth_note,
            })
            state = ChainState(reference, STAGE_AUTHORIZED, 1, sku, qty, amt, actor, when, auth_note=auth_note)
            self._chains[reference] = state
            return state

    def state(self, reference: str) -> ChainState:
        with self._lock:
            try:
                return self._chains[reference]
            except KeyError:
                raise ChainError(f"unknown chain {reference}") from None

    # -- forward command -----------------------------------------------------

    def advance(
        self,
        reference: str,
        target: str,
        actor: str,
        at: datetime,
        receipt_id: str | None = None,
        reason: str = "",
    ) -> Advancement:
        when = at.astimezone(timezone.utc)
        with self._lock:
            state = self._chains.get(reference)
            if state is None:
                raise ChainError(f"unknown chain {reference}")

            if receipt_id is not None:
                owner = self._receipts.get(receipt_id)
                if owner is not None:
                    if owner != (reference, target):
                        raise ChainError(f"receipt {receipt_id} already consumed by {owner[0]}:{owner[1]}")
                    # Re-sent callback, including one arriving after the
                    # chain has already progressed: acknowledge, never repost.
                    return Advancement(state, True, ())
            if target in RECEIPT_STAGES and not receipt_id:
                raise ChainError(f"stage {target} requires a receipt")

            if target not in _NEXT.get(state.stage, frozenset()):
                raise ChainError(f"transition {state.stage} -> {target} not allowed")
            if target == STAGE_CANCELED:
                raise ChainError("cancellation must go through withdraw_approval")
            if state.stage in TERMINAL_STAGES:
                raise ChainError(f"chain {reference} is {state.stage}")
            if state.updated_at is not None and when < state.updated_at:
                raise ChainError("event time moved backward")

            spec = _group_spec(target, state.quantity, state.amount)
            voucher_refs = self._post_group(reference, target, spec, when)

            groups = state.groups
            if spec:
                groups += (GroupRecord(target, spec[0][0], "forward", tuple(voucher_refs), when),)
            receipts = state.receipts
            if receipt_id is not None:
                self._receipts[receipt_id] = (reference, target)
                receipts += ((receipt_id, target),)

            nxt = ChainState(
                reference, target, state.revision + 1, state.sku, state.quantity,
                state.amount, actor, when, groups, receipts, state.auth_note,
            )
            self.store.append(self._stream(reference), state.revision, "advanced", {
                "target": target, "actor": actor, "at": when, "reason": reason,
                "receipt_id": receipt_id,
                "groups": [
                    {"stage": g.stage, "group": g.group, "voucher_refs": list(g.voucher_refs)}
                    for g in groups if g.stage == target and g.direction == "forward"
                ],
            })
            self._chains[reference] = nxt
            return Advancement(nxt, False, tuple(voucher_refs))

    def submit_receipt(
        self,
        reference: str,
        stage: str,
        receipt_id: str,
        actor: str,
        at: datetime,
    ) -> Advancement:
        """Convenience entry point for carrier/certificate callbacks."""
        return self.advance(reference, stage, actor, at, receipt_id=receipt_id)

    def _post_group(
        self,
        reference: str,
        stage: str,
        spec: list[tuple[str, str, list[tuple[str, Decimal, str]]]],
        at: datetime,
    ) -> list[str]:
        # Fixed lock order across both ledgers; RLock re-entry makes the
        # idempotent voucher posts inside safe. The whole group is visible
        # all at once or not at all.
        refs: list[str] = []
        with self.stock.lock(), self.finance.lock():
            for group, ledger_name, legs in spec:
                ref = f"{reference}:{stage}:{group}:{ledger_name}"
                self._ledger(ledger_name).post_voucher(ref, legs, at)
                refs.append(ref)
        return refs

    # -- approval withdrawal (whole-chain compensation) ----------------------

    def withdraw_approval(
        self,
        reference: str,
        actor: str,
        at: datetime,
        reason: str,
    ) -> Advancement:
        """Reverse every posted voucher and cancel -- atomically as a group.

        Allowed only while the order is still ``approved`` (quarantine stock
        plus the provision are the only postings in flight). Once goods have
        been handed to the carrier a recall flow, not a rollback, is required.
        """
        when = at.astimezone(timezone.utc)
        note = reason or "approval withdrawn"
        with self._lock:
            state = self._chains.get(reference)
            if state is None:
                raise ChainError(f"unknown chain {reference}")
            if state.stage == STAGE_CANCELED:
                return Advancement(state, True, ())
            if state.stage != STAGE_APPROVED:
                raise ChainError(
                    f"withdrawal only allowed at {STAGE_APPROVED}, chain is {state.stage}"
                )
            if state.updated_at is not None and when < state.updated_at:
                raise ChainError("event time moved backward")

            # Reverse in the opposite order of posting, both ledgers locked:
            # finance provision first, then the quarantine stock move. A
            # crash/retry re-enters idempotently through deterministic refs.
            reversal_refs: list[str] = []
            with self.stock.lock(), self.finance.lock():
                for stage, group, ledger_name in (
                    (STAGE_APPROVED, "approval", "finance"),
                    (STAGE_QUARANTINED, "quarantine", "stock"),
                ):
                    original = f"{reference}:{stage}:{group}:{ledger_name}"
                    reversal = f"{original}:reversal"
                    self._ledger(ledger_name).reverse_voucher(original, reversal, when, note)
                    reversal_refs.append(reversal)

            groups = state.groups + (
                GroupRecord(STAGE_CANCELED, "approval_withdrawal", "reversal", tuple(reversal_refs), when),
            )
            nxt = ChainState(
                reference, STAGE_CANCELED, state.revision + 1, state.sku, state.quantity,
                state.amount, actor, when, groups, state.receipts, state.auth_note,
            )
            self.store.append(self._stream(reference), state.revision, "withdrawn", {
                "actor": actor, "at": when, "reason": note,
                "reversal_refs": reversal_refs,
            })
            self._chains[reference] = nxt
            return Advancement(nxt, False, tuple(reversal_refs))

    # -- reconciliation -------------------------------------------------------

    def reconcile(self, reference: str) -> Reconciliation:
        """Compare model-derived expected balances against ledger reality.

        Expected = legs of every forward stage the chain has reached, minus
        those compensated by withdrawal. Actual nets the legs of every
        voucher the saga references (reversal vouchers negate their originals).
        Any voucher posted under this chain's reference prefix but absent from
        the saga (the post-commit/crash-before-append window) is reported as
        an *unrecorded* posting and makes the chain inconsistent.
        """
        with self._lock:
            state = self._chains.get(reference)
            if state is None:
                raise ChainError(f"unknown chain {reference}")

            canceled = state.stage == STAGE_CANCELED
            reversed_groups = {"quarantine", "approval"}
            expected: dict[str, Decimal] = {}
            actual: dict[str, Decimal] = {}
            recorded_refs: set[str] = set()
            forward_refs: list[str] = []
            reversal_refs: list[str] = []

            for record in state.groups:
                if record.direction == "reversal":
                    reversal_refs.extend(record.voucher_refs)
                    for ref in record.voucher_refs:
                        recorded_refs.add(ref)
                        ledger = self._ref_ledger(ref)
                        for leg in ledger.voucher(ref).legs:
                            actual[leg.account] = actual.get(leg.account, Decimal("0")) + leg.amount
                    continue
                forward_refs.extend(record.voucher_refs)
                compensated_here = canceled and record.group in reversed_groups
                for _group, _ledger_name, legs in _group_spec(record.stage, state.quantity, state.amount):
                    for account, amount, _reason in legs:
                        if not compensated_here:
                            expected[account] = expected.get(account, Decimal("0")) + amount
                for ref in record.voucher_refs:
                    recorded_refs.add(ref)
                    ledger = self._ref_ledger(ref)
                    voucher = ledger.voucher(ref)
                    if compensated_here and not ledger.is_reversed(ref):
                        raise ChainError(f"compensated voucher {ref} has no reversal")
                    for leg in voucher.legs:
                        actual[leg.account] = actual.get(leg.account, Decimal("0")) + leg.amount

            # Orphan postings: committed to a ledger but never recorded in
            # the saga. Their legs must not silently match expected.
            prefix = f"{reference}:"
            for ledger in (self.stock, self.finance):
                for voucher in ledger.vouchers():
                    if voucher.reference.startswith(prefix) and voucher.reference not in recorded_refs:
                        for leg in voucher.legs:
                            actual[leg.account] = actual.get(leg.account, Decimal("0")) + leg.amount

            accounts = set(expected) | set(actual)
            differences = {
                a: actual.get(a, Decimal("0")) - expected.get(a, Decimal("0"))
                for a in sorted(accounts)
                if actual.get(a, Decimal("0")) - expected.get(a, Decimal("0")) != 0
            }
            return Reconciliation(
                reference=reference,
                consistent=not differences,
                stage=state.stage,
                expected=expected,
                actual=actual,
                differences=differences,
                voucher_refs=tuple(forward_refs),
                reversed_refs=tuple(reversal_refs),
            )

    # -- recovery -------------------------------------------------------------

    def recover(self, reference: str) -> ChainState:
        """Rebuild chain state from its event stream (e.g. after restart)."""
        with self._lock:
            state = self._fold(reference)
            self._chains[reference] = state
            for receipt_id, stage in state.receipts:
                owner = self._receipts.setdefault(receipt_id, (reference, stage))
                if owner != (reference, stage):
                    raise ChainError(f"receipt {receipt_id} ownership conflict on recovery")
            return state

    def recover_all(self) -> tuple[ChainState, ...]:
        streams = [s for s in self.store.all_streams() if s.startswith(self.STREAM_PREFIX)]
        out = [self.recover(s[len(self.STREAM_PREFIX):]) for s in sorted(streams)]
        return tuple(out)

    def _fold(self, reference: str) -> ChainState:
        events = self.store.read(self._stream(reference))
        if not events:
            raise ChainError(f"unknown chain {reference}")
        state: ChainState | None = None
        for event in events:
            if event.sequence == 1:
                if event.kind != "opened":
                    raise ChainError("stream must begin with opened")
                p = event.payload
                state = ChainState(
                    reference, STAGE_AUTHORIZED, 1, p["sku"], Decimal(p["quantity"]),
                    Decimal(p["amount"]), p["actor"], p["at"], auth_note=p.get("auth_note", ""),
                )
                continue
            assert state is not None
            if event.kind == "advanced":
                p = event.payload
                groups = state.groups + tuple(
                    GroupRecord(
                        g["stage"], g["group"], "forward", tuple(g["voucher_refs"]), p["at"]
                    )
                    for g in p["groups"]
                )
                receipts = state.receipts
                if p.get("receipt_id"):
                    receipts += ((p["receipt_id"], p["target"]),)
                state = ChainState(
                    reference, p["target"], event.sequence, state.sku, state.quantity,
                    state.amount, p["actor"], p["at"], groups, receipts, state.auth_note,
                )
            elif event.kind == "withdrawn":
                p = event.payload
                groups = state.groups + (GroupRecord(
                    STAGE_CANCELED, "approval_withdrawal", "reversal",
                    tuple(p["reversal_refs"]), p["at"],
                ))
                state = ChainState(
                    reference, STAGE_CANCELED, event.sequence, state.sku, state.quantity,
                    state.amount, p["actor"], p["at"], groups, state.receipts, state.auth_note,
                )
            else:
                raise ChainError(f"unknown event kind {event.kind}")
        assert state is not None
        return state
