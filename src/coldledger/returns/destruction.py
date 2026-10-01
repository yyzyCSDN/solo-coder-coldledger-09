from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from threading import RLock
from typing import Iterable

from ..finance.claims_ledger import ClaimsLedger
from ..inventory.ledger import InventoryLedger

REQUESTED = "requested"
AUTHORIZED = "authorized"
QUARANTINED = "quarantined"
APPROVED = "approved"
HANDED_OVER = "handed_over"
DESTROYED = "destroyed"
CERTIFIED = "certified"
WRITTEN_OFF = "written_off"
CLOSED = "closed"
WITHDRAWN = "withdrawn"

STATES = [
    REQUESTED,
    AUTHORIZED,
    QUARANTINED,
    APPROVED,
    HANDED_OVER,
    DESTROYED,
    CERTIFIED,
    WRITTEN_OFF,
    CLOSED,
    WITHDRAWN,
]

REQUEST = "request"
AUTHORIZATION = "authorization"
QUARANTINE = "quarantine"
APPROVAL = "approval"
CARRIER_HANDOVER = "carrier_handover"
DESTRUCTION = "destruction"
CERTIFICATION = "certification"
WRITE_OFF = "write_off"
CLOSE = "close"
APPROVAL_WITHDRAWAL = "approval_withdrawal"

STEP_TARGETS = {
    REQUEST: REQUESTED,
    AUTHORIZATION: AUTHORIZED,
    QUARANTINE: QUARANTINED,
    APPROVAL: APPROVED,
    CARRIER_HANDOVER: HANDED_OVER,
    DESTRUCTION: DESTROYED,
    CERTIFICATION: CERTIFIED,
    WRITE_OFF: WRITTEN_OFF,
    CLOSE: CLOSED,
    APPROVAL_WITHDRAWAL: WITHDRAWN,
}

STEP_ALIASES = {
    "open": REQUEST,
    "create": REQUEST,
    "requested": REQUEST,
    "authorize": AUTHORIZATION,
    "authorized": AUTHORIZATION,
    "quarantined": QUARANTINE,
    "approve": APPROVAL,
    "approved": APPROVAL,
    "handover": CARRIER_HANDOVER,
    "handed_over": CARRIER_HANDOVER,
    "carrier_handed_over": CARRIER_HANDOVER,
    "destroy": DESTRUCTION,
    "destroyed": DESTRUCTION,
    "certificate": CERTIFICATION,
    "certified": CERTIFICATION,
    "destruction_certified": CERTIFICATION,
    "writeoff": WRITE_OFF,
    "written_off": WRITE_OFF,
    "accounts_written_off": WRITE_OFF,
    "closed": CLOSE,
    "withdraw": APPROVAL_WITHDRAWAL,
    "withdraw_approval": APPROVAL_WITHDRAWAL,
    "approval_withdrawn": APPROVAL_WITHDRAWAL,
    "withdrawn": APPROVAL_WITHDRAWAL,
}

STATE_ALIASES = {
    "carrier_handed_over": HANDED_OVER,
    "destruction_certified": CERTIFIED,
    "accounts_written_off": WRITTEN_OFF,
    "approval_withdrawn": WITHDRAWN,
}

ALLOWED: dict[str, frozenset[str]] = {
    REQUESTED: frozenset({AUTHORIZED}),
    AUTHORIZED: frozenset({QUARANTINED}),
    QUARANTINED: frozenset({APPROVED}),
    APPROVED: frozenset({HANDED_OVER, WITHDRAWN}),
    HANDED_OVER: frozenset({DESTROYED}),
    DESTROYED: frozenset({CERTIFIED}),
    CERTIFIED: frozenset({WRITTEN_OFF}),
    WRITTEN_OFF: frozenset({CLOSED}),
    CLOSED: frozenset(),
    WITHDRAWN: frozenset(),
}

AVAILABLE_ACCOUNT = "available"
QUARANTINE_ACCOUNT = "quarantine"
RESERVE_ACCOUNT = "write_off_reserve"
ACCRUAL_ACCOUNT = "destruction_accrual"
LOSS_ACCOUNT = "destruction_loss"


def normalize_step(step: str) -> str:
    key = str(step).strip()
    return STEP_ALIASES.get(key, key)


def normalize_state(state: str) -> str:
    key = str(state).strip()
    return STATE_ALIASES.get(key, key)


@dataclass(frozen=True)
class Receipt:
    receipt_id: str
    step: str
    actor: str
    at: datetime
    quantity: Decimal | None = None
    amount: Decimal | None = None
    carrier: str | None = None
    handoff_ref: str | None = None
    certificate_id: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class DestructionState:
    reference: str
    state: str
    revision: int = 0
    owner: str | None = None
    updated_at: datetime | None = None

    def validate(self) -> None:
        if normalize_state(self.state) not in STATES:
            raise ValueError("unknown destruction state")
        if self.revision < 0:
            raise ValueError("negative revision")


@dataclass(frozen=True)
class Command:
    reference: str
    expected_revision: int
    target: str
    actor: str
    at: datetime
    receipt_id: str = ""
    reason: str = ""


@dataclass(frozen=True)
class DestructionEvent:
    sequence: int
    reference: str
    step: str
    before: str
    after: str
    revision: int
    receipt_id: str
    actor: str
    at: datetime
    quantity: Decimal | None = None
    amount: Decimal | None = None
    carrier: str | None = None
    handoff_ref: str | None = None
    certificate_id: str | None = None
    reason: str = ""


Transition = DestructionEvent


def _decimal(value: Decimal | str | int | float | None) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _time(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _required_text(value: str | None, label: str) -> str:
    if not value or not str(value).strip():
        raise ValueError(f"{label} required")
    return str(value).strip()


def _positive(value: Decimal | str | int | float | None, label: str) -> Decimal:
    amount = _decimal(value)
    if amount is None or amount <= 0:
        raise ValueError(f"{label} must be positive")
    return amount


def event_for(current: DestructionState | None, receipt: Receipt, reference: str) -> DestructionEvent:
    step = normalize_step(receipt.step)
    if step not in STEP_TARGETS:
        raise ValueError("unknown destruction step")
    _required_text(reference, "reference")
    _required_text(receipt.receipt_id, "receipt_id")
    _required_text(receipt.actor, "actor")
    at = _time(receipt.at)
    quantity = _decimal(receipt.quantity)
    amount = _decimal(receipt.amount)

    if step == REQUEST:
        if current is not None:
            raise ValueError("destruction case already exists")
        quantity = _positive(quantity, "quantity")
        if amount is not None and amount <= 0:
            raise ValueError("amount must be positive")
        return DestructionEvent(
            1, reference, step, "", REQUESTED, 1, receipt.receipt_id,
            receipt.actor, at, quantity, amount, reason=receipt.reason,
        )

    if current is None:
        raise ValueError("destruction case not opened")
    if reference != current.reference:
        raise ValueError("reference mismatch")
    if current.updated_at is not None and at < _time(current.updated_at):
        raise ValueError("event time moved backward")

    target = STEP_TARGETS[step]
    if target not in ALLOWED.get(current.state, frozenset()):
        raise ValueError(f"transition {current.state} -> {target} not allowed")

    if quantity is not None or (amount is not None and step != APPROVAL):
        raise ValueError("quantity and amount are accepted only at their originating step")
    if step == APPROVAL:
        amount = _positive(amount, "amount")
    if step == CARRIER_HANDOVER:
        carrier = _required_text(receipt.carrier, "carrier")
        handoff_ref = _required_text(receipt.handoff_ref, "handoff_ref")
    else:
        carrier = receipt.carrier
        handoff_ref = receipt.handoff_ref
    if step == CERTIFICATION:
        certificate_id = _required_text(receipt.certificate_id, "certificate_id")
    else:
        certificate_id = receipt.certificate_id
    if step == APPROVAL_WITHDRAWAL:
        _required_text(receipt.reason, "reason")

    sequence = current.revision + 1
    return DestructionEvent(
        sequence,
        reference,
        step,
        current.state,
        target,
        sequence,
        receipt.receipt_id,
        receipt.actor,
        at,
        None,
        amount,
        carrier,
        handoff_ref,
        certificate_id,
        receipt.reason,
    )


def open_case(
    reference: str,
    receipt_id: str,
    actor: str,
    at: datetime,
    quantity: Decimal | str | int | float,
    amount: Decimal | str | int | float | None = None,
) -> tuple[DestructionState, DestructionEvent]:
    receipt = Receipt(receipt_id, REQUEST, actor, at, quantity=quantity, amount=amount)
    event = event_for(None, receipt, reference)
    return state_from_event(event), event


def state_from_event(event: DestructionEvent) -> DestructionState:
    if event.step != REQUEST or event.before or event.after != REQUESTED or event.revision != 1:
        raise ValueError("invalid opening event")
    return DestructionState(event.reference, event.after, 1, event.actor, event.at)


def apply_event(current: DestructionState | None, event: DestructionEvent) -> DestructionState:
    if event.step == REQUEST:
        if current is not None:
            raise ValueError("opening event for existing case")
        return state_from_event(event)

    if current is None:
        raise ValueError("destruction case not opened")
    if event.reference != current.reference:
        raise ValueError("mixed references")
    if event.sequence != current.revision + 1 or event.revision != current.revision + 1:
        raise ValueError("revision gap")
    if event.before != current.state:
        raise ValueError("event before-state mismatch")
    if event.after not in ALLOWED.get(current.state, frozenset()):
        raise ValueError("event contains illegal transition")
    if current.updated_at is not None and _time(event.at) < _time(current.updated_at):
        raise ValueError("event time moved backward")
    return replace(
        current,
        state=event.after,
        revision=event.revision,
        owner=event.actor,
        updated_at=event.at,
    )


def replay(initial: DestructionState | None, events: Iterable[DestructionEvent]) -> DestructionState | None:
    state = initial
    for event in sorted(events, key=lambda e: (e.reference, e.sequence)):
        state = apply_event(state, event)
    return state


def apply(current: DestructionState, command: Command) -> tuple[DestructionState, DestructionEvent]:
    current.validate()
    target = normalize_state(command.target)
    if target not in STATES:
        raise ValueError("invalid target state")
    if command.reference != current.reference:
        raise ValueError("reference mismatch")
    if command.expected_revision != current.revision:
        raise RuntimeError("optimistic concurrency conflict")
    if target == current.state:
        raise ValueError("no-op transition")
    if target not in ALLOWED.get(current.state, frozenset()):
        raise ValueError(f"transition {current.state} -> {target} not allowed")
    at = _time(command.at)
    if current.updated_at is not None and at < _time(current.updated_at):
        raise ValueError("event time moved backward")

    step = next((step for step, state_name in STEP_TARGETS.items() if state_name == target), "")
    sequence = current.revision + 1
    event = DestructionEvent(
        sequence, current.reference, step, current.state, target, sequence,
        command.receipt_id, command.actor, at, reason=command.reason,
    )
    return apply_event(current, event), event


def can_transition(current: DestructionState, target: str) -> bool:
    return normalize_state(target) in ALLOWED.get(current.state, frozenset())


def next_states(current: DestructionState) -> tuple[str, ...]:
    return tuple(sorted(ALLOWED.get(current.state, frozenset())))


def _same_receipt(event: DestructionEvent, receipt: Receipt) -> bool:
    return (
        event.actor == receipt.actor
        and _time(event.at) == _time(receipt.at)
        and event.quantity == _decimal(receipt.quantity)
        and event.amount == _decimal(receipt.amount)
        and event.carrier == receipt.carrier
        and event.handoff_ref == receipt.handoff_ref
        and event.certificate_id == receipt.certificate_id
        and event.reason == receipt.reason
    )


class DestructionWorkflow:
    """Event-sourced saga for returns destruction.

    The state event is durably recorded before external ledger side effects run.
    Every ledger side effect has a deterministic idempotency key, so retrying a
    duplicate receipt after a crash completes missing postings without creating
    a second charge, movement, or reversal.
    """

    def __init__(
        self,
        inventory: InventoryLedger | None = None,
        finance: ClaimsLedger | None = None,
        events: Iterable[DestructionEvent] = (),
    ):
        self.inventory = inventory or InventoryLedger()
        self.finance = finance or ClaimsLedger()
        self._lock = RLock()
        self._states: dict[str, DestructionState] = {}
        self._events: dict[str, list[DestructionEvent]] = {}
        self._receipts: dict[str, DestructionEvent] = {}
        self._external_refs: dict[tuple[str, str], str] = {}
        for event in sorted(events, key=lambda e: (e.reference, e.sequence)):
            self._ingest(event)

    def open_case(
        self,
        reference: str,
        receipt_id: str,
        actor: str,
        at: datetime,
        quantity: Decimal | str | int | float,
        amount: Decimal | str | int | float | None = None,
    ) -> DestructionState:
        return self._submit(
            reference,
            Receipt(receipt_id, REQUEST, actor, at, quantity=quantity, amount=amount),
        )

    def authorize(self, reference: str, receipt_id: str, actor: str, at: datetime) -> DestructionState:
        return self._submit(reference, Receipt(receipt_id, AUTHORIZATION, actor, at))

    def quarantine(self, reference: str, receipt_id: str, actor: str, at: datetime) -> DestructionState:
        return self._submit(reference, Receipt(receipt_id, QUARANTINE, actor, at))

    def approve(
        self,
        reference: str,
        receipt_id: str,
        actor: str,
        at: datetime,
        amount: Decimal | str | int | float,
    ) -> DestructionState:
        return self._submit(reference, Receipt(receipt_id, APPROVAL, actor, at, amount=amount))

    def hand_over(
        self,
        reference: str,
        receipt_id: str,
        actor: str,
        at: datetime,
        carrier: str,
        handoff_ref: str,
    ) -> DestructionState:
        receipt = Receipt(
            receipt_id, CARRIER_HANDOVER, actor, at, carrier=carrier, handoff_ref=handoff_ref
        )
        return self._submit(reference, receipt)

    def destroy(self, reference: str, receipt_id: str, actor: str, at: datetime) -> DestructionState:
        return self._submit(reference, Receipt(receipt_id, DESTRUCTION, actor, at))

    def certify(
        self,
        reference: str,
        receipt_id: str,
        actor: str,
        at: datetime,
        certificate_id: str,
    ) -> DestructionState:
        receipt = Receipt(receipt_id, CERTIFICATION, actor, at, certificate_id=certificate_id)
        return self._submit(reference, receipt)

    def write_off(self, reference: str, receipt_id: str, actor: str, at: datetime) -> DestructionState:
        return self._submit(reference, Receipt(receipt_id, WRITE_OFF, actor, at))

    def close(self, reference: str, receipt_id: str, actor: str, at: datetime) -> DestructionState:
        return self._submit(reference, Receipt(receipt_id, CLOSE, actor, at))

    def withdraw_approval(
        self, reference: str, receipt_id: str, actor: str, at: datetime, reason: str
    ) -> DestructionState:
        receipt = Receipt(receipt_id, APPROVAL_WITHDRAWAL, actor, at, reason=reason)
        return self._submit(reference, receipt)

    def state(self, reference: str) -> DestructionState:
        with self._lock:
            return self._states[reference]

    def events(self, reference: str) -> tuple[DestructionEvent, ...]:
        with self._lock:
            return tuple(self._events.get(reference, ()))

    def all_events(self) -> tuple[DestructionEvent, ...]:
        with self._lock:
            return tuple(
                event
                for reference in sorted(self._events)
                for event in self._events[reference]
            )

    def recover(self) -> dict[str, DestructionState]:
        with self._lock:
            for reference in sorted(self._events):
                self._recover_case(reference)
            return dict(self._states)

    def pending_effects(self, reference: str) -> tuple[str, ...]:
        with self._lock:
            pending: list[str] = []
            for event in self._events.get(reference, ()):
                for operation in self._operations(event):
                    if not self._committed(operation):
                        pending.append(operation[-1] if operation[0] == "post" else operation[3])
            return tuple(pending)

    def is_consistent(self) -> bool:
        with self._lock:
            return all(not self.pending_effects(reference) for reference in self._events)

    def _submit(self, reference: str, receipt: Receipt) -> DestructionState:
        step = normalize_step(receipt.step)
        receipt = replace(receipt, step=step)
        with self._lock:
            current = self._states.get(reference)
            if current is not None:
                self._recover_case(reference)

            previous = self._receipts.get(receipt.receipt_id)
            if previous is not None:
                if previous.reference != reference or previous.step != step:
                    raise ValueError("receipt_id already used for a different step")
                if not _same_receipt(previous, receipt):
                    raise ValueError("duplicate receipt with conflicting payload")
                self._recover_case(reference)
                return self._states[reference]

            event = event_for(current, receipt, reference)
            self._ingest(event)
            try:
                self._run_effects(event)
            except Exception:
                # The intent remains durable. Replaying the same receipt resumes
                # the saga instead of posting a partial second transaction.
                raise
            return self._states[reference]

    def _ingest(self, event: DestructionEvent) -> None:
        rows = self._events.setdefault(event.reference, [])
        if event.sequence != len(rows) + 1:
            raise ValueError("event sequence gap")
        if event.receipt_id in self._receipts:
            raise ValueError("duplicate receipt_id")

        external_keys: list[tuple[str, str]] = []
        if event.handoff_ref:
            external_keys.append(("handoff_ref", event.handoff_ref))
        if event.certificate_id:
            external_keys.append(("certificate_id", event.certificate_id))
        for key in external_keys:
            if key in self._external_refs:
                if key[0] == "handoff_ref":
                    raise ValueError("duplicate carrier handoff reference")
                raise ValueError("duplicate destruction certificate")

        current = self._states.get(event.reference)
        next_state = apply_event(current, event)
        rows.append(event)
        self._states[event.reference] = next_state
        self._receipts[event.receipt_id] = event
        self._external_refs.update((key, event.reference) for key in external_keys)

    def _recover_case(self, reference: str) -> None:
        for event in self._events.get(reference, ()):
            self._run_effects(event)

    def _case_quantity(self, reference: str) -> Decimal:
        for event in self._events[reference]:
            if event.quantity is not None:
                return event.quantity
        raise ValueError("case quantity missing")

    def _case_amount(self, reference: str) -> Decimal:
        for event in self._events[reference]:
            if event.amount is not None and event.step == APPROVAL:
                return event.amount
        raise ValueError("approved destruction amount missing")

    def _operations(self, event: DestructionEvent):
        reference = event.reference
        if event.step == QUARANTINE:
            quantity = self._case_quantity(reference)
            return [
                ("post", self.inventory, AVAILABLE_ACCOUNT, self._ref(reference, "quarantine:available"), -quantity, "quarantine returned stock"),
                ("post", self.inventory, QUARANTINE_ACCOUNT, self._ref(reference, "quarantine:quarantine"), quantity, "quarantine returned stock"),
            ]
        if event.step == APPROVAL:
            amount = event.amount
            return [
                ("post", self.finance, RESERVE_ACCOUNT, self._ref(reference, "approval:reserve"), amount, "approve destruction write-off"),
                ("post", self.finance, ACCRUAL_ACCOUNT, self._ref(reference, "approval:accrual"), -amount, "approve destruction write-off"),
            ]
        if event.step == WRITE_OFF:
            quantity = self._case_quantity(reference)
            amount = self._case_amount(reference)
            return [
                ("post", self.inventory, QUARANTINE_ACCOUNT, self._ref(reference, "write-off:inventory"), -quantity, "destroy returned stock"),
                ("post", self.finance, RESERVE_ACCOUNT, self._ref(reference, "write-off:reserve"), -amount, "settle destruction reserve"),
                ("post", self.finance, ACCRUAL_ACCOUNT, self._ref(reference, "write-off:accrual"), amount, "settle destruction accrual"),
                ("post", self.finance, LOSS_ACCOUNT, self._ref(reference, "write-off:loss"), -amount, "recognize destruction loss"),
            ]
        if event.step == APPROVAL_WITHDRAWAL:
            return [
                ("reverse", self.inventory, self._ref(reference, "quarantine:available"), self._ref(reference, "quarantine:available:reversal"), "approval withdrawn"),
                ("reverse", self.inventory, self._ref(reference, "quarantine:quarantine"), self._ref(reference, "quarantine:quarantine:reversal"), "approval withdrawn"),
                ("reverse", self.finance, self._ref(reference, "approval:reserve"), self._ref(reference, "approval:reserve:reversal"), "approval withdrawn"),
                ("reverse", self.finance, self._ref(reference, "approval:accrual"), self._ref(reference, "approval:accrual:reversal"), "approval withdrawn"),
            ]
        return []

    def _run_effects(self, event: DestructionEvent) -> None:
        operations = self._operations(event)
        if operations and all(operation[0] == "reverse" for operation in operations):
            # Compensation is validated as a complete set first. A retry resumes
            # from deterministic reversal keys rather than leaving one ledger
            # compensated while a later ledger's original entry is missing.
            for operation in operations:
                _, ledger, original_ref, reversal_ref, reason = operation
                original = self._entry(ledger, original_ref)
                if original is None:
                    raise ValueError(f"original posting missing: {original_ref}")
                existing = [entry for entry in ledger.entries() if entry.reversal_of == original.sequence]
                if existing and (
                    existing[0].reference != reversal_ref
                    or existing[0].quantity != -original.quantity
                    or existing[0].reason != reason
                ):
                    raise ValueError(f"conflicting reversal: {reversal_ref}")

        for operation in operations:
            if operation[0] == "post":
                _, ledger, account, reference, quantity, reason = operation
                self._post_once(ledger, account, reference, quantity, reason)
            else:
                _, ledger, original_ref, reversal_ref, reason = operation
                self._reverse_once(ledger, original_ref, reversal_ref, reason)

    def _committed(self, operation) -> bool:
        if operation[0] == "post":
            return self._entry(operation[1], operation[3]) is not None
        _, ledger, original_ref, reversal_ref, _ = operation
        original = self._entry(ledger, original_ref)
        if original is None:
            return False
        return any(entry.reversal_of == original.sequence for entry in ledger.entries())

    @staticmethod
    def _ref(reference: str, suffix: str) -> str:
        return f"{reference}:{suffix}"

    @staticmethod
    def _entry(ledger, reference: str):
        matches = [entry for entry in ledger.entries() if entry.reference == reference]
        if len(matches) > 1:
            raise ValueError("duplicate ledger side effect")
        return matches[0] if matches else None

    def _post_once(self, ledger, account: str, reference: str, quantity: Decimal, reason: str):
        existing = self._entry(ledger, reference)
        if existing is not None:
            if existing.account != account or existing.quantity != quantity or existing.reason != reason:
                raise ValueError(f"conflicting idempotent posting: {reference}")
            return existing
        return ledger.post(account, reference, quantity, reason)

    def _reverse_once(self, ledger, original_ref: str, reversal_ref: str, reason: str):
        original = self._entry(ledger, original_ref)
        if original is None:
            raise ValueError(f"original posting missing: {original_ref}")
        for entry in ledger.entries():
            if entry.reversal_of == original.sequence:
                if entry.reference != reversal_ref or entry.quantity != -original.quantity or entry.reason != reason:
                    raise ValueError(f"conflicting reversal: {reversal_ref}")
                return entry
        return ledger.reverse(original.sequence, reversal_ref, reason)
