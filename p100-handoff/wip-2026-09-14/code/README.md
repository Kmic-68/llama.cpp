The staged source states that produced commits 13b24fe27, fccdafca1, b67848c64, dce17bf1b and the
same-GPU copy guard are in git history; only what git does not carry is kept here:

- `peer-v1-rejected.patch` — the peer-copy fix variant that was measured and **not** kept: it
  records a fresh marker on the destination, which serialises the two directions of the exchange
  (-2.9% tg256, -3.3% MTP, against -0.7%/-0.6% for the variant that shipped).
- `temp-race-injection.patch` — delays one device's all-reduce ADD with dummy sgemms on its own
  stream, and puts the two copy guards behind `GGML_CUDA_PEER_WAIT_DST` / `GGML_CUDA_SAMEDEV_WAIT_DST`.
  Makes the copy race deterministic. Never commit it.
- `temp-knobs-and-nodehash-hook.patch` — the per-node output hashing used for the whole-pass
  determinism runs, plus the precision knobs. Also temporary: the hook synchronizes per node, which
  hides cross-device races.

Apply either patch to the tree at the commit named in its header, build into `build-opt`, and copy
`build-opt/bin` to a snapshot directory before switching back (binaries carry a RUNPATH into
`build-opt`, so run them with `LD_LIBRARY_PATH` set to the snapshot).
