# Recorded environments and portability

The locked continuation protocol records Python 3.9.23, torch 2.8.0+cpu, NumPy 2.0.2, CPU and two threads. MATLAB figure rendering records R2021b. HLS and routed implementation evidence records Vitis HLS/Vivado 2022.2 for XC7Z020CLG484-1 at a 10 ns clock target. These are observed historical settings, not a claim that arbitrary versions are bit-equivalent.

The release integrity checker needs only Python's standard library. The per-slot recount needs NumPy. Historical research scripts also import PyTorch and, for some plotting/diagnostic routes, Matplotlib; install a suitable PyTorch CPU environment using the platform's official instructions. The complete archive includes starting and selected final checkpoints. Do not load untrusted PyTorch pickle files obtained from unrelated sources; verify this package's hashes first.

The original main inference implementation and native C bridge were tested on Windows. In particular, the channel loader uses a Windows path separator and the native bridge uses `__declspec(dllexport)`. A Linux/macOS inference rerun requires a reviewed, documented portability patch; the numerical audit does not require that bridge. Frozen original code is deliberately not silently patched under an old protocol hash.

Historical protocol manifests contain the author's original project paths. Use `provenance/original_file_hashes.json` to locate the same file by release-relative path. Re-running an experiment on a new machine requires a fresh output namespace and a new local protocol/path binding; do not alter a historical protocol and claim it is the original locked record.

No automatic environment export from the user's full machine was included, so this document does not expose unrelated installed packages, accounts, tokens or private paths. The local audit report records what was actually tested during packaging.
