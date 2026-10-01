import unittest
from datetime import datetime, timezone
from decimal import Decimal

from coldledger.finance.claims_ledger import ClaimsLedger
from coldledger.inventory.ledger import InventoryLedger
from coldledger.returns.destruction import (
    AUTHORIZED,
    APPROVED,
    CERTIFIED,
    CLOSED,
    DESTROYED,
    HANDED_OVER,
    QUARANTINED,
    REQUESTED,
    WRITTEN_OFF,
    ACCRUAL_ACCOUNT,
    AVAILABLE_ACCOUNT,
    DestructionWorkflow,
    QUARANTINE_ACCOUNT,
    LOSS_ACCOUNT,
    RESERVE_ACCOUNT,
    apply_event,
    event_for,
    open_case,
)

UTC = timezone.utc


class DestructionWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.at = datetime(2026, 10, 1, tzinfo=UTC)
        self.inventory = InventoryLedger()
        self.finance = ClaimsLedger()
        self.inventory.post(AVAILABLE_ACCOUNT, "seed", 10, "seed stock")
        self.workflow = DestructionWorkflow(self.inventory, self.finance)

    def complete_case(self, receipt_prefix="r"):
        w = self.workflow
        w.open_case("R", receipt_prefix + "1", "owner", self.at, 10, None)
        w.authorize("R", receipt_prefix + "2", "owner", self.at)
        w.quarantine("R", receipt_prefix + "3", "owner", self.at)
        w.approve("R", receipt_prefix + "4", "owner", self.at, "100.50")
        w.hand_over("R", receipt_prefix + "5", "carrier-desk", self.at, "CAR", "H-1")
        w.destroy("R", receipt_prefix + "6", "carrier", self.at)
        w.certify("R", receipt_prefix + "7", "auditor", self.at, "CERT-1")
        w.write_off("R", receipt_prefix + "8", "accountant", self.at)
        return w.close("R", receipt_prefix + "9", "accountant", self.at)

    def test_full_recoverable_chain(self):
        state = self.complete_case()
        self.assertEqual(
            state.state,
            CLOSED,
        )
        self.assertEqual(
            [event.after for event in self.workflow.events("R")],
            [REQUESTED, AUTHORIZED, QUARANTINED, APPROVED, HANDED_OVER, DESTROYED, CERTIFIED, WRITTEN_OFF, CLOSED],
        )
        self.assertEqual(self.inventory.balance(AVAILABLE_ACCOUNT), Decimal("0"))
        self.assertEqual(self.inventory.balance(QUARANTINE_ACCOUNT), Decimal("0"))
        self.assertEqual(self.finance.balance(RESERVE_ACCOUNT), Decimal("0"))
        self.assertEqual(self.finance.balance(ACCRUAL_ACCOUNT), Decimal("0"))
        self.assertEqual(self.finance.balance(LOSS_ACCOUNT), Decimal("-100.50"))

    def test_duplicate_receipt_is_idempotent(self):
        self.complete_case()
        entries_before = len(self.inventory.entries()) + len(self.finance.entries())
        state = self.workflow.certify("R", "r7", "auditor", self.at, "CERT-1")
        entries_after = len(self.inventory.entries()) + len(self.finance.entries())
        self.assertEqual(state.state, CLOSED)
        self.assertEqual(entries_before, entries_after)

    def test_duplicate_handoff_reference_rejected(self):
        self.complete_case()
        other = self.workflow
        other.open_case("S", "s1", "owner", self.at, 1)
        other.authorize("S", "s2", "owner", self.at)
        other.quarantine("S", "s3", "owner", self.at)
        other.approve("S", "s4", "owner", self.at, 10)
        with self.assertRaisesRegex(ValueError, "duplicate carrier handoff reference"):
            other.hand_over("S", "s5", "carrier-desk", self.at, "CAR", "H-1")

    def test_approval_withdrawal_reverses_all_related_entries(self):
        w = self.workflow
        w.open_case("R", "r1", "owner", self.at, 10)
        w.authorize("R", "r2", "owner", self.at)
        w.quarantine("R", "r3", "owner", self.at)
        w.approve("R", "r4", "owner", self.at, "100.50")
        state = w.withdraw_approval("R", "r5", "owner", self.at, "incorrect disposition")
        self.assertEqual(state.state, "withdrawn")
        self.assertEqual(self.inventory.balance(AVAILABLE_ACCOUNT), Decimal("10"))
        self.assertEqual(self.inventory.balance(QUARANTINE_ACCOUNT), Decimal("0"))
        self.assertEqual(self.finance.balance(RESERVE_ACCOUNT), Decimal("0"))
        self.assertEqual(self.finance.balance(ACCRUAL_ACCOUNT), Decimal("0"))
        self.assertEqual(len(self.finance.entries()), 4)

    def test_events_without_side_effects_can_be_recovered(self):
        _, opened = open_case("R", "r1", "owner", self.at, 10)
        state = apply_event(None, opened)
        events = [opened]
        for receipt_id, step, amount in [
            ("r2", "authorization", None),
            ("r3", "quarantine", None),
            ("r4", "approval", "100"),
        ]:
            from coldledger.returns.destruction import Receipt

            event = event_for(
                state,
                Receipt(receipt_id, step, "owner", self.at, amount=amount),
                "R",
            )
            state = apply_event(state, event)
            events.append(event)

        workflow = DestructionWorkflow(self.inventory, self.finance, events)
        self.assertTrue(workflow.pending_effects("R"))
        workflow.recover()
        self.assertFalse(workflow.pending_effects("R"))
        self.assertEqual(self.inventory.balance(QUARANTINE_ACCOUNT), Decimal("10"))
        self.assertEqual(self.finance.balance(RESERVE_ACCOUNT), Decimal("100"))
        workflow.recover()
        self.assertEqual(len(self.inventory.entries()), 3)
        self.assertEqual(len(self.finance.entries()), 2)


if __name__ == "__main__":
    unittest.main()
