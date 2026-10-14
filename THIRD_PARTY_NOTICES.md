# Third-party materials

## Deep-Waveform channel coefficient matrices

`research/code/dl_ofdm-master/dev/py/3gpp/AM_EPA.csv`, `AM_EVA.csv`, and `AM_ETU.csv` were copied byte-for-byte from the author's locally retained Deep-Waveform source distribution. Their hashes are in the provenance manifest.

Upstream project identified in that distribution: https://github.com/zhongyuanzhao/dl_ofdm

Its MIT license, copyright 2018–2021 Zhongyuan Zhao, is preserved at `research/code/dl_ofdm-master/LICENSE`. Do not credit these coefficient matrices as newly created by this manuscript's authors. The present original research code implements and evaluates the receiver variants described by the manuscript; it is not a claim to have invented Deep-Waveform.

## External tools not redistributed

PyTorch, NumPy, Matplotlib, MATLAB, Vitis HLS, Vivado and compiler toolchains are external dependencies. Their executables, installation trees, licensed runtime libraries and generated vendor IP are not included. Users must obtain them under their respective terms.

The archived Fig.9 renderer in `figures/reference_scripts` mentions the original workstation's external panel-alignment helper. That helper is not bundled or required by the self-contained native-FIG export utility. It is a historical drawing script, not a fully self-contained third-party runtime distribution.
