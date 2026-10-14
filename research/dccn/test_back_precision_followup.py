"""Check precision interventions against independent shared-model changes."""
import copy
import unittest

import torch

from config import OfdmConfig
from quant import ActQ, QConfig
from run_back_precision_followup import BackPrecisionReceiver, HERE, point_seed
from struct_quant_model import StructuredQDCCN


class PrecisionBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = OfdmConfig(nbits=1, long_cp=True)
        cls.state = torch.load(HERE / "ckpt/sq_b1_longcp_s_r0_slim_w8a8_pdpfix_seed1.pt",
                               map_location="cpu", weights_only=True)
        torch.manual_seed(202610027)
        cls.inputs = torch.randn(5, cls.cfg.N_SYM, cls.cfg.S, dtype=torch.complex64)

    def pair(self, variant):
        probe = BackPrecisionReceiver(self.cfg, rank=0, eq_dense=True, slim=True, qc=QConfig()).eval()
        probe.load_state_dict(self.state)
        probe.configure(variant)
        original = StructuredQDCCN(self.cfg, rank=0, eq_dense=True, slim=True, qc=QConfig()).eval()
        original.load_state_dict(self.state)
        original.set_transform_backend("fft")
        return probe, original

    def compare(self, probe, original):
        with torch.inference_mode():
            torch.testing.assert_close(probe(self.inputs), original(self.inputs), rtol=0, atol=0)
        for name, value in probe.state_dict().items():
            self.assertTrue(torch.equal(value, self.state[name]))

    def test_fft_baseline(self):
        self.compare(*self.pair("baseline_fft"))

    def test_matrix_baseline(self):
        probe, original = self.pair("baseline_matrix")
        original.set_transform_backend("matrix")
        self.compare(probe, original)

    def test_weight_only_restoration(self):
        probe, original = self.pair("float_w")
        original.eq.m_back.qc = copy.copy(original.qc)
        original.eq.m_back.qc.quant_w = False
        self.compare(probe, original)

    def test_full_block_restoration(self):
        probe, original = self.pair("float_w_in_out")
        original.eq.m_back.qc = copy.copy(original.qc)
        original.eq.m_back.qc.quant_w = original.eq.m_back.qc.quant_a = False
        self.compare(probe, original)

    def test_output_only_a12(self):
        probe, original = self.pair("back_out_a12")
        original.eq.m_back.q_out.qr.bits = original.eq.m_back.q_out.qi.bits = 12
        changed = [n for n, m in probe.named_modules() if isinstance(m, ActQ) and m.bits != 8]
        self.assertEqual(changed, ["eq.m_back.q_out.qr", "eq.m_back.q_out.qi"])
        self.compare(probe, original)

    def test_variant_reset(self):
        probe, original = self.pair("div_back_in_out_a16")
        probe.configure("baseline_fft")
        self.assertTrue(all(m.bits == 8 for m in probe.modules() if isinstance(m, ActQ)))
        self.compare(probe, original)

    def test_cohort_namespaces(self):
        seeds = {point_seed(2026100201, tag, seed, channel, snr)
                 for tag in ("b1_longcp", "b2_longcp", "b2_shortcp")
                 for seed in (1, 2, 3) for channel in ("EVA", "ETU") for snr in (12, 16, 20, 24)}
        self.assertEqual(len(seeds), 72)


if __name__ == "__main__":
    unittest.main()
