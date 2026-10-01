import unittest
from datetime import datetime, timezone
from decimal import Decimal

from coldledger.returns.destruction_chain import (
    DestructionChains,
    VoucherLedger,
    ChainError,
    STAGE_AUTHORIZED,
    STAGE_QUARANTINED,
    STAGE_APPROVED,
    STAGE_HANDED_OVER,
    STAGE_DESTROYED,
    STAGE_WRITTEN_OFF,
    STAGE_CLOSED,
    STAGE_CANCELED,
)
from coldledger.storage.event_store import EventStore

UTC = timezone.utc
T = lambda h: datetime(2026, 10, 1, h, tzinfo=UTC)


def happy_path(chains, ref="R-1", amount="1000.00"):
    chains.open(ref, "SKU-X", "7", amount, "qa", T(8), auth_note="cold excursion")
    chains.advance(ref, STAGE_QUARANTINED, "wh", T(9))
    chains.advance(ref, STAGE_APPROVED, "mgr", T(10), reason="disposition: destroy")
    chains.submit_receipt(ref, STAGE_HANDED_OVER, "POD-77", "carrier", T(11))
    chains.submit_receipt(ref, STAGE_DESTROYED, "CERT-9", "incinerator", T(12))
    chains.advance(ref, STAGE_WRITTEN_OFF, "fin", T(13))
    chains.advance(ref, STAGE_CLOSED, "fin", T(14))
    return ref


class DestructionChainTests(unittest.TestCase):
    def setUp(self):
        self.chains = DestructionChains()

    # -- full flow: inventory and finance stay aligned ----------------------

    def test_happy_path_reconciles_to_zero(self):
        ref = happy_path(self.chains)
        state = self.chains.state(ref)
        self.assertEqual(state.stage, STAGE_CLOSED)
        rep = self.chains.reconcile(ref)
        self.assertTrue(rep.consistent, rep.differences)
        self.assertEqual(rep.differences, {})
        # Goods leave on-hand once quarantined and never return: -7 on_hand,
        # +7 written_off, intermediate transit accounts all back to zero.
        self.assertEqual(self.chains.stock.balance("stock:on_hand"), Decimal("-7"))
        self.assertEqual(self.chains.stock.balance("stock:quarantine"), Decimal("0"))
        self.assertEqual(self.chains.stock.balance("stock:in_custody_carrier"), Decimal("0"))
        self.assertEqual(self.chains.stock.balance("stock:destroyed"), Decimal("0"))
        self.assertEqual(self.chains.stock.balance("stock:written_off"), Decimal("7"))
        # Finance voucher postings balance.
        self.chains.finance.audit()
        self.chains.stock.audit()
        self.assertEqual(
            self.chains.finance.balance("liability:destruction_provision"), Decimal("0")
        )

    def test_stage_gates_receipts_and_skips(self):
        ref = happy_path(self.chains)  # noqa: F841
        c = DestructionChains()
        c.open("R-2", "SKU-X", "3", "300", "qa", T(8))
        # carrier/certificate stages require an external receipt
        with self.assertRaises(ChainError):
            c.advance("R-2", STAGE_HANDED_OVER, "carrier", T(11))
        c.advance("R-2", STAGE_QUARANTINED, "wh", T(9))
        with self.assertRaises(ChainError):
            c.advance("R-2", STAGE_DESTROYED, "inc", T(10))

    # -- idempotent duplicate receipts ---------------------------------------

    def test_duplicate_receipt_is_idempotent(self):
        c = self.chains
        c.open("R-3", "SKU-X", "5", "500", "qa", T(8))
        c.advance("R-3", STAGE_QUARANTINED, "wh", T(9))
        c.advance("R-3", STAGE_APPROVED, "mgr", T(10))
        first = c.submit_receipt("R-3", STAGE_HANDED_OVER, "POD-1", "carrier", T(11))
        stock_vouchers_before = len(c.stock.vouchers())
        # callback retried after timeout / re-delivered queue message
        second = c.submit_receipt("R-3", STAGE_HANDED_OVER, "POD-1", "carrier", T(11))
        third = c.submit_receipt("R-3", STAGE_HANDED_OVER, "POD-1", "carrier", T(12))
        self.assertFalse(first.duplicate)
        self.assertTrue(second.duplicate)
        self.assertTrue(third.duplicate)
        self.assertEqual(len(c.stock.vouchers()), stock_vouchers_before)
        self.assertEqual(c.state("R-3").stage, STAGE_HANDED_OVER)
        self.assertEqual(len(c.state("R-3").receipts), 1)
        self.assertTrue(c.reconcile("R-3").consistent)

    def test_receipt_reuse_across_chains_rejected(self):
        c = self.chains
        for ref in ("A", "B"):
            c.open(ref, "SKU-X", "1", "10", "qa", T(8))
            c.advance(ref, STAGE_QUARANTINED, "wh", T(9))
            c.advance(ref, STAGE_APPROVED, "mgr", T(10))
        c.submit_receipt("A", STAGE_HANDED_OVER, "POD-X", "carrier", T(11))
        with self.assertRaises(ChainError):
            c.submit_receipt("B", STAGE_HANDED_OVER, "POD-X", "carrier", T(11))

    # -- approval withdrawal rolls back BOTH ledgers as one group ------------

    def test_withdrawal_compensates_all_postings_atomically(self):
        c = self.chains
        c.open("R-4", "SKU-X", "4", "400.00", "qa", T(8))
        c.advance("R-4", STAGE_QUARANTINED, "wh", T(9))
        c.advance("R-4", STAGE_APPROVED, "mgr", T(10))
        before = (
            {v.reference for v in c.stock.vouchers()},
            {v.reference for v in c.finance.vouchers()},
        )
        result = c.withdraw_approval("R-4", "mgr", T(15), "customer recalled the request")
        self.assertEqual(c.state("R-4").stage, STAGE_CANCELED)
        # two reversals, one per ledger, emitted by a single saga event
        self.assertEqual(len(result.voucher_refs), 2)
        # stock move fully undone ...
        self.assertEqual(c.stock.balance("stock:on_hand"), Decimal("0"))
        self.assertEqual(c.stock.balance("stock:quarantine"), Decimal("0"))
        # ... and the provision simultaneously, never one without the other
        self.assertEqual(c.finance.balance("expense:destruction"), Decimal("0"))
        self.assertEqual(c.finance.balance("liability:destruction_provision"), Decimal("0"))
        rep = c.reconcile("R-4")
        self.assertTrue(rep.consistent, rep.differences)
        self.assertEqual(len(rep.reversed_refs), 2)
        after = (
            {v.reference for v in c.stock.vouchers()},
            {v.reference for v in c.finance.vouchers()},
        )
        self.assertEqual(len(after[0] - before[0]), 1)
        self.assertEqual(len(after[1] - before[1]), 1)

    def test_withdrawal_idempotent_and_forward_blocked(self):
        c = self.chains
        c.open("R-5", "SKU-X", "2", "20", "qa", T(8))
        c.advance("R-5", STAGE_QUARANTINED, "wh", T(9))
        c.advance("R-5", STAGE_APPROVED, "mgr", T(10))
        first = c.withdraw_approval("R-5", "mgr", T(15), "x")
        again = c.withdraw_approval("R-5", "mgr", T(16), "x")
        self.assertFalse(first.duplicate)
        self.assertTrue(again.duplicate)
        # no additional reversals on retry
        self.assertEqual(len([v for v in c.stock.vouchers() if v.reversal_of]), 1)
        with self.assertRaises(ChainError):
            c.submit_receipt("R-5", STAGE_HANDED_OVER, "POD-z", "carrier", T(17))

    def test_backdated_withdrawal_rejected(self):
        c = self.chains
        c.open("R-6b", "SKU-X", "2", "20", "qa", T(8))
        c.advance("R-6b", STAGE_QUARANTINED, "wh", T(9))
        c.advance("R-6b", STAGE_APPROVED, "mgr", T(10))
        with self.assertRaises(ChainError):
            c.withdraw_approval("R-6b", "mgr", T(9), "backdated")
        self.assertEqual(c.state("R-6b").stage, STAGE_APPROVED)

    def test_withdrawal_blocked_after_handover(self):
        c = self.chains
        c.open("R-6", "SKU-X", "2", "20", "qa", T(8))
        c.advance("R-6", STAGE_QUARANTINED, "wh", T(9))
        c.advance("R-6", STAGE_APPROVED, "mgr", T(10))
        c.submit_receipt("R-6", STAGE_HANDED_OVER, "POD-6", "carrier", T(11))
        with self.assertRaises(ChainError):
            c.withdraw_approval("R-6", "mgr", T(12), "too late")
        # failed withdrawal changed nothing
        self.assertEqual(c.state("R-6").stage, STAGE_HANDED_OVER)
        self.assertTrue(c.reconcile("R-6").consistent)

    # -- recovery from the event stream --------------------------------------

    def test_recovery_rebuilds_state_and_dedupe_registry(self):
        c = self.chains
        happy_path(c, "R-7", "250")
        store = c.store
        fresh = DestructionChains(stock=c.stock, finance=c.finance, store=store)
        state = fresh.recover("R-7")
        self.assertEqual(state.stage, STAGE_CLOSED)
        self.assertEqual(state.revision, c.state("R-7").revision)
        self.assertEqual(len(state.receipts), 2)
        # receipt registry restored: a late re-delivery is still a no-op
        late = fresh.submit_receipt("R-7", STAGE_DESTROYED, "CERT-9", "incinerator", T(15))
        self.assertTrue(late.duplicate)
        all_states = fresh.recover_all()
        self.assertEqual({s.reference for s in all_states}, {"R-7"})

    def test_orphan_posting_after_crash_is_detected(self):
        # Simulate ledger commit succeeding but saga event append missing:
        # post the deterministic quarantine voucher reference directly.
        c = self.chains
        c.open("R-8", "SKU-X", "9", "900", "qa", T(8))
        ref = "R-8:quarantined:quarantine:stock"
        c.stock.post_voucher(ref, [
            ("stock:on_hand", Decimal("-9"), "returned goods quarantined"),
            ("stock:quarantine", Decimal("9"), "returned goods quarantined"),
        ], T(9))
        rep = c.reconcile("R-8")
        self.assertFalse(rep.consistent)
        self.assertIn("stock:quarantine", rep.differences)


class VoucherLedgerTests(unittest.TestCase):
    def test_post_is_idempotent_and_rejects_leg_level_tampering(self):
        lg = VoucherLedger("t")
        legs = [("a", "10", "in"), ("b", "-10", "out")]
        v1 = lg.post_voucher("V1", legs, T(8))
        v2 = lg.post_voucher("V1", legs, T(9))
        self.assertIs(v1, v2)
        with self.assertRaises(ValueError):
            lg.post_voucher("V1", [("a", "11", "in"), ("b", "-11", "out")], T(9))
        with self.assertRaises(ValueError):
            lg.post_voucher("V2", [("a", "10", "in"), ("b", "-9", "out")], T(9))

    def test_reversal_negates_every_leg_once(self):
        lg = VoucherLedger("t")
        lg.post_voucher("V1", [("a", "10", "in"), ("b", "-4", "x"), ("c", "-6", "y")], T(8))
        r1 = lg.reverse_voucher("V1", "V1R", T(9), "undo")
        r2 = lg.reverse_voucher("V1", "V1R", T(9), "undo")
        self.assertIs(r1, r2)
        self.assertEqual(lg.balance("a"), Decimal("0"))
        self.assertEqual(lg.balance("b"), Decimal("0"))
        with self.assertRaises(ValueError):
            lg.reverse_voucher("V1", "V1R-other", T(9), "undo")


if __name__ == "__main__":
    unittest.main()
