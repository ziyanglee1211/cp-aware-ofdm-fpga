"""Small independent regressions for the isolated precision implementation."""
import json
import re
import unittest
from pathlib import Path
import numpy as np
import close_evidence as audit

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "pdpfix_opt_20260928"


def code_only(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


class LocalA12Tests(unittest.TestCase):
    def test_only_nominated_core_boundary_changes(self):
        before = (BASE/"receiver_opt_rev_t.cpp").read_text(encoding="utf-8")
        before = before.replace('#include <stdint.h>', '#include <stdint.h>\n#include "local_a12_quant.h"')
        before = before.replace('const int8_t *Tqr, const int8_t *Tqi', 'const int16_t *Tqr, const int16_t *Tqi')
        before = before.replace('static int8_t Tqr[QN],  Tqi[QN];', 'static int16_t Tqr[QN], Tqi[QN];')
        before = before.replace('EQ_MB_QOUT_RE', 'LOCAL_A12_RE').replace('EQ_MB_QOUT_IM', 'LOCAL_A12_IM')
        before = before.replace('Tqr[f * NN + o] = q8(', 'Tqr[f * NN + o] = q12(')
        before = before.replace('Tqi[f * NN + o] = q8(', 'Tqi[f * NN + o] = q12(')
        self.assertEqual(code_only(before), code_only((HERE/"receiver_local_a12.cpp").read_text(encoding="utf-8")))

    def test_fixed_golden_and_native_arrays(self):
        z = np.load(HERE/"transfer_audit/fixed_vectors.npz")
        for name in ("rev_t", "a12"):
            self.assertEqual(z[name+"_c"].shape, (60,320))
            self.assertTrue(np.array_equal(z[name+"_c"], z[name+"_torch"]))
            self.assertTrue(np.isfinite(z[name+"_margin"]).all())

    def test_complete_paired_cohort(self):
        manifest = audit.load(HERE/"transfer_audit/manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(set(manifest["conditions"])), 8)
        for condition in manifest["conditions"]:
            z = np.load(HERE/"transfer_audit"/(condition+".npz"))
            self.assertEqual(z["labels"].shape, (2000,320))
            for name in ("c_a12", "c_rev_t", "torch_a12", "torch_rev_t"):
                self.assertTrue(np.array_equal((z[name+"_pred"] != z["labels"]).sum(1), z[name+"_errors"]))

    def test_rounding_and_scales(self):
        row = audit.load(HERE/"transfer_audit/quantizer.json")
        self.assertEqual(row["samples"], 116393)
        self.assertEqual(row["mismatches"], 0)
        self.assertEqual(len(row["scales_exact_float32"]),2)

    def test_hls_schedule_and_memory_blocks(self):
        old = audit.hls(BASE/"proj_rev_t/sol_opt/syn/report/receiver_slot_opt_csynth.xml")
        new = audit.hls(HERE/"proj_a12/sol_opt/syn/report/receiver_slot_opt_csynth.xml")
        self.assertEqual(old["latency"],new["latency"])
        self.assertEqual(new["latency"],58115)
        self.assertEqual(old["resources"]["BRAM_18K"],new["resources"]["BRAM_18K"])
        rpt = (HERE/"proj_a12/sol_opt/syn/report/receiver_slot_opt_csynth.rpt").read_text()
        for name in ("Tqr_U", "Tqi_U"):
            self.assertRegex(rpt, r"\|"+name+r"\s*\|Tqr_RAM_AUTO_1R1W\s*\|\s*1\|\s*0\|\s*0\|\s*0\|\s*448\|\s*16\|")

    def test_baseline_physical_parser(self):
        resources = audit.utilization(BASE/"board_t_fabric/reports/system_utilization.rpt")
        self.assertEqual(resources["bram18_equivalents"],274)
        self.assertEqual(resources["slices"],12687)
        timing = audit.timing(BASE/"board_t_fabric/reports/system_timing_summary.rpt")
        self.assertEqual(timing["wns_ns"],.361)
        self.assertEqual(timing["no_output_delay_ports"],8)

    def test_static_source_binding(self):
        row = audit.load(HERE/"static_audit.json")
        self.assertEqual(row["static_chain_status"],"pass")
        self.assertTrue(all(audit.sha(audit.ROOT/p["path"])==p["sha256"] for p in row["artifacts"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
