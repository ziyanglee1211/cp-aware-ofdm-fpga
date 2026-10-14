"""Development-only tests for continuation identity, RNG resume, and pairing.

No formal training or final-test cohort is opened by this test module.
"""
import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import run_continued_qat_control as runner
from quant import ActQ
from run_back_precision_followup import BackPrecisionReceiver
from quant import QConfig


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original = runner.HERE / "ckpt/sq_b1_longcp_s_r0_slim_w8a8_pdpfix_seed1.pt"
        cls.original_hash = runner.file_hash(cls.original)
        cls.state = torch.load(cls.original, map_location="cpu", weights_only=True)
        runner.seed_all(101101101)
        cls.inputs = torch.randn(5, 7, 80, dtype=torch.complex64)

    @classmethod
    def tearDownClass(cls):
        if runner.file_hash(cls.original) != cls.original_hash:
            raise AssertionError("Original checkpoint changed during development tests")

    def model(self, arm="A1", backend="matrix"):
        model = runner.make_model("b1_longcp", self.state, backend)
        if backend == "matrix":
            runner.train_mode(model, arm)
        return model

    def optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=.95)
        return optimizer, scheduler

    def test_zero_fft_identity_matches_existing_evaluator(self):
        model = self.model(backend="fft")
        reference = BackPrecisionReceiver(model.cfg, rank=0, eq_dense=True, slim=True, qc=QConfig()).eval()
        reference.load_state_dict(self.state, strict=True)
        reference.configure("baseline_fft")
        with torch.inference_mode():
            torch.testing.assert_close(model(self.inputs), reference(self.inputs), rtol=0, atol=0)
        self.assertEqual(runner.state_hash(model.state_dict()), runner.state_hash(self.state))

    def test_only_a12_output_bits_change_and_match_existing_evaluator(self):
        model = runner.make_model("b1_longcp", self.state, "fft", output_a12=True)
        reference = BackPrecisionReceiver(model.cfg, rank=0, eq_dense=True, slim=True, qc=QConfig()).eval()
        reference.load_state_dict(self.state, strict=True)
        reference.configure("back_out_a12")
        changed = tuple(name for name, row in runner.activation_ranges(model).items() if row["bits"] != 8)
        self.assertEqual(changed, runner.BOUNDARY)
        with torch.inference_mode():
            torch.testing.assert_close(model(self.inputs), reference(self.inputs), rtol=0, atol=0)
        self.assertEqual(runner.state_hash(model.state_dict()), runner.state_hash(self.state))

    def test_strict_loading_rejects_missing_buffer(self):
        incomplete = copy.deepcopy(self.state)
        del incomplete["eq.m_back.q_out.qr.amax"]
        with self.assertRaises(RuntimeError):
            runner.make_model("b1_longcp", incomplete)

    def test_initialization_never_loads_struct_or_recalibrates(self):
        with mock.patch.object(runner.StructuredQDCCN, "load_struct", side_effect=AssertionError("load_struct called")) as structural:
            with mock.patch.object(runner.StructuredQDCCN, "calibrate", side_effect=AssertionError("calibrate called")) as calibration:
                model = runner.make_model("b1_longcp", self.state, "fft")
                self.assertFalse(model.training)
                self.assertTrue(all(not module.training for module in model.modules() if isinstance(module, ActQ)))
                self.assertEqual(model.eq.transform_backend, "fft")
                self.assertEqual(model.basic.transform_backend, "fft")
                structural.assert_not_called()
                calibration.assert_not_called()

    def test_observer_running_and_frozen_modes(self):
        observer = ActQ(8)
        observer(torch.tensor([1.0]))
        observer.eval()
        observer(torch.tensor([5.0]))
        self.assertEqual(float(observer.amax), 1.0)
        observer.train()
        observer(torch.tensor([5.0]))
        self.assertEqual(float(observer.amax), 5.0)
        frozen = self.model("A1F")
        normal = self.model("A1")
        self.assertTrue(all(not module.training for module in frozen.modules() if isinstance(module, ActQ)))
        self.assertTrue(all(module.training for module in normal.modules() if isinstance(module, ActQ)))
        self.assertEqual(frozen.eq.transform_backend, "matrix")
        self.assertEqual(frozen.basic.transform_backend, "matrix")

    def test_frozen_ranges_allow_parameter_adaptation(self):
        model = self.model("A1F")
        before = runner.activation_ranges(model)
        optimizer, scheduler = self.optimizer(model)
        self.assertEqual(len(optimizer.state), 0)
        runner.seed_all(102102102)
        tx = runner.OfdmTxTorch(model.cfg, torch.device("cpu"))
        received, labels, _, _ = runner.draw_training(tx, runner.channels_for(model.cfg), 4)
        loss, _ = runner.optimization_step(model, optimizer, scheduler, received, labels)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(runner.activation_ranges(model), before)
        self.assertNotEqual(runner.state_hash(model.state_dict()), runner.state_hash(self.state))
        self.assertGreater(len(optimizer.state), 0)

    def test_complete_safe_checkpoint_reproduces_next_update(self):
        model = self.model()
        optimizer, scheduler = self.optimizer(model)
        tx = runner.OfdmTxTorch(model.cfg, torch.device("cpu"))
        channels = runner.channels_for(model.cfg)
        runner.seed_all(103103103)
        received, labels, hashes, _ = runner.draw_training(tx, channels, 4)
        runner.optimization_step(model, optimizer, scheduler, received, labels)
        full = runner.full_checkpoint(model, optimizer, scheduler, {
            "step": 1, "training_stimulus_chain_sha256": runner.advance_chain(runner.CHAIN_INITIAL, 1, hashes),
            "training_slots": 4, "elapsed_seconds": 0.1, "optimization_seconds": 0.1})
        with tempfile.TemporaryDirectory(prefix="continued_qat_unit_") as directory:
            path = Path(directory) / "complete.pt"
            runner.exclusive_torch_save(path, full)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            with self.assertRaises(FileExistsError):
                runner.exclusive_torch_save(path, full)
        received, labels, next_hashes, _ = runner.draw_training(tx, channels, 4)
        runner.optimization_step(model, optimizer, scheduler, received, labels)
        expected = runner.state_hash(model.state_dict())
        resumed = self.model()
        resumed.load_state_dict(loaded["model"], strict=True)
        resumed_optimizer, resumed_scheduler = self.optimizer(resumed)
        resumed_optimizer.load_state_dict(loaded["optimizer"])
        resumed_scheduler.load_state_dict(loaded["scheduler"])
        runner.restore_rng(loaded["rng"])
        received, labels, resumed_hashes, _ = runner.draw_training(tx, channels, 4)
        self.assertEqual(next_hashes, resumed_hashes)
        runner.optimization_step(resumed, resumed_optimizer, resumed_scheduler, received, labels)
        self.assertEqual(runner.state_hash(resumed.state_dict()), expected)
        self.assertEqual(scheduler.state_dict(), resumed_scheduler.state_dict())

    def test_cpu_numpy_python_rng_roundtrip(self):
        runner.seed_all(104104104)
        state = runner.capture_rng()
        expected = (torch.rand(3), np.random.rand(3), random.random())
        runner.restore_rng(state)
        actual = (torch.rand(3), np.random.rand(3), random.random())
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(np.array_equal(expected[1], actual[1]))
        self.assertEqual(expected[2], actual[2])

    def test_a1_and_a1f_training_stimulus_sequences_are_identical(self):
        chains = []
        for arm in ("A1", "A1F"):
            model = self.model(arm)
            optimizer, scheduler = self.optimizer(model)
            tx = runner.OfdmTxTorch(model.cfg, torch.device("cpu"))
            channels = runner.channels_for(model.cfg)
            runner.seed_all(105105105)
            chain = runner.CHAIN_INITIAL
            for step in (1, 2):
                received, labels, hashes, _ = runner.draw_training(tx, channels, 4)
                chain = runner.advance_chain(chain, step, hashes)
                runner.optimization_step(model, optimizer, scheduler, received, labels)
            chains.append(chain)
        self.assertEqual(chains[0], chains[1])

    def test_bit_level_pairing_and_slot_counts(self):
        labels = torch.tensor([[False, True, False], [True, True, False]])
        predictions = {"A0": torch.tensor([[True, True, False], [True, False, False]]),
                       "A1": labels.clone(), "A2": torch.tensor([[False, False, False], [True, False, True]])}
        arrays, arms, contrasts = runner.paired_counts(predictions, labels, [["A1", "A2"],
            ["A1", "A0"], ["A2", "A0"]], bootstrap=10, seed=106106106)
        self.assertEqual(arms["A0"]["error_bits"], 2)
        self.assertEqual(arms["A1"]["error_bits"], 0)
        self.assertEqual(arms["A2"]["error_bits"], 3)
        self.assertEqual(contrasts["A1_minus_A2"]["corrected_decisions"], 3)
        self.assertEqual(contrasts["A1_minus_A2"]["harmed_decisions"], 0)
        for candidate, reference in (("A1", "A2"), ("A1", "A0"), ("A2", "A0")):
            key = candidate + "_minus_" + reference
            np.testing.assert_array_equal(arrays[key + "__flips"],
                arrays[key + "__corrected"] + arrays[key + "__harmed"])
            np.testing.assert_array_equal(arrays[candidate + "__slot_errors"] - arrays[reference + "__slot_errors"],
                arrays[key + "__harmed"] - arrays[key + "__corrected"])
            np.testing.assert_array_equal(arrays[candidate + "__pred"], predictions[candidate].numpy())

    def test_output_exclusivity_and_final_test_gate(self):
        with tempfile.TemporaryDirectory(prefix="continued_qat_unit_") as directory:
            root = Path(directory)
            protocol = {"output_root": str(root)}
            runner.exclusive_directory(root / "new", protocol)
            with self.assertRaises(FileExistsError):
                runner.exclusive_directory(root / "new", protocol)
            with self.assertRaises(ValueError):
                runner.exclusive_directory(root, protocol)
        with mock.patch.object(runner, "load_final") as load:
            with self.assertRaises(ValueError):
                runner.run_evaluation({}, "digest", "protocol", Path("unused"), Path("unused"))
            load.assert_not_called()

    def test_inference_preserves_final_short_batch_and_rejects_training_mode(self):
        class BatchProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []

            def forward(self, value):
                self.batch_sizes.append(len(value))
                first = torch.zeros(len(value), 3, 1)
                return torch.stack((first, first + .75), dim=-1)

        probe = BatchProbe().eval()
        predictions = runner.infer_predictions(probe, torch.zeros(65, 7, 80), 64)
        self.assertEqual(probe.batch_sizes, [64, 1])
        self.assertEqual(tuple(predictions.shape), (65, 3, 1))
        self.assertTrue(bool(predictions.all()))
        probe.train()
        with self.assertRaises(ValueError):
            runner.infer_predictions(probe, torch.zeros(4, 7, 80), 64)


if __name__ == "__main__":
    unittest.main()
