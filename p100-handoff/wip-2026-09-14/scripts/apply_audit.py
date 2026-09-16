#!/usr/bin/env python3
"""Insert the 2026-09-14/15 findings into the audit document and update the claims they overturn."""
import sys

AUDIT = "p100-handoff/release-sync/docs/AUDIT-2026-09-12.md"
NEW = "p100-handoff/wip-2026-09-14/AUDIT-0915-draft.md"

s = open(AUDIT).read()
new = open(NEW).read()
if "@@" in new:
    sys.exit("placeholders left in " + NEW + ":\n" + "\n".join(l for l in new.splitlines() if "@@" in l))
if "Found after the audit (2026-09-14/15)" in s:
    sys.exit("already inserted")

# 1. the "lead" paragraph of the 09-13 section is now measured, and the norm kernels were stressed
lead_old = """**A lead, not a finding.** Upstream's quantized prefill matmuls on Pascal also run COMPUTE_16F
GEMMs with `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, over far longer sums (5120-17408 terms). Whether the
same family switch applies at those shapes is unmeasured. Separately, upstream's `norm_f32` (in
place) and `group_norm_f32` use the load-then-store-in-a-later-pass pattern that raced here; the
served model uses neither (RMS norm only)."""
lead_new = """**A lead, not a finding.** Upstream's quantized prefill matmuls on Pascal also run COMPUTE_16F
GEMMs with `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, over far longer sums (5120-17408 terms). Whether the
same family switch applies at those shapes is unmeasured. Separately, upstream's `norm_f32` (in
place) and `group_norm_f32` use the load-then-store-in-a-later-pass pattern that raced here; the
served model uses neither (RMS norm only). **Both leads were followed on 2026-09-14/15 — see the
next section: the matmul one was real and is fixed, the kernel one did not reproduce.**"""
if lead_old not in s:
    sys.exit("could not find the lead paragraph to update")
s = s.replace(lead_old, lead_new)

# 2. the scope-table row for the f16 all-reduce gains a pointer
row_old = "| f16 tensor-parallel all-reduce | LOSSLESS (proved) |"
row_new = "| f16 tensor-parallel all-reduce | LOSSLESS (proved), but applied too widely until 2026-09-15 (see below) |"
if row_old in s:
    s = s.replace(row_old, row_new, 1)

# 3. insert the new findings section before "### Fixed after the audit, verified at runtime"
anchor = "### Fixed after the audit, verified at runtime"
if anchor not in s:
    sys.exit("could not find the insertion anchor")
s = s.replace(anchor, new.rstrip() + "\n\n" + anchor, 1)

open(AUDIT, "w").write(s)
print("audit updated:", AUDIT)
