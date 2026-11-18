# ISTA-DASLab's Coder release: the list of kept experts

`engine/data/coder_mask.txt` lists, for each of the model's 48 layers, the 256 of 512 routed experts that ISTA-DASLab's
expert-pruned release keeps. It is taken from that release's allocation list:

- Repository: https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF, revision
  `bba88856f8c3c0c9e40e45f82d05e228b814ac43` (2026-10-19)
- File: `tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.rco-allocation.txt` (SHA-256
  `9a04497b9133f76682550981772de8cdcde52ee89b164fa0d1cc18efdd30ef31`), its second section (per expert, by the index in the
  unpruned model)
- License of the repository: Apache License 2.0, the full text is in `LICENSE-Apache-2.0.txt` beside this file.

Our change: the list is rewritten as one line a layer (the layer, then its kept experts) by `scripts/coder_mask.py`, which
keeps an expert when its `gate`, `up` and `down` rows all have a type. No weights of that release are included. The
experts were chosen by Deep Algorithms and Systems Lab, Institute of Science and Technology Austria, with RCO
(https://arxiv.org/abs/2605.00649, https://github.com/IST-DASLab/RCO).
