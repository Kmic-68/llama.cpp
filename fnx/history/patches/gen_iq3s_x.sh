#!/bin/bash
# gen_iq3s_x.sh: writes iq3s_x.cuh beside itself: ggml's vec_dot_iq3_s_q8_1 with its int8 dot products in the xmad form (idea 67,
# moe_x_patch.py copies the file to the engine's sources). Fails unless exactly the two ggml_cuda_dp4a calls were there (the name stays
# in two lines of the header).
I=$(dirname "$(readlink -f "$0")")
V=/srv/build/strata-p100/build-cuda12/_deps/strata_llamacpp-src/ggml/src/ggml-cuda/vecdotq.cuh
O=$I/iq3s_x.cuh
{
  echo "// iq3s_x.cuh: ggml's vec_dot_iq3_s_q8_1 (ggml-cuda/vecdotq.cuh) with its two ggml_cuda_dp4a calls a step in p100_compat.h's"
  echo "// xmad form (fnx_dp4a_xmad); nothing else differs (gen_iq3s_x.sh makes it from that file; idea 67, moe_x_patch.py). On sm_60"
  echo "// ggml_cuda_dp4a is the compiler's own from C. The same sums."
  awk '/^static __device__ __forceinline__ float vec_dot_iq3_s_q8_1\(/{p=1} p{print} p&&/^}/{exit}' "$V" |
    sed 's/vec_dot_iq3_s_q8_1(/vec_dot_iq3_s_q8_1_x(/; s/ggml_cuda_dp4a(/fnx_dp4a_xmad(/g'
} > "$O"
n=$(grep -c 'fnx_dp4a_xmad(' "$O"); m=$(grep -c 'ggml_cuda_dp4a' "$O"); h=$(grep -c 'vec_dot_iq3_s_q8_1_x(' "$O")
echo "iq3s_x.cuh: $(wc -l < "$O") vrstic, xmad $n, ggml $m, glava $h"
[ "$n" = 2 ] && [ "$m" = 2 ] && [ "$h" = 1 ] && tail -1 "$O" | grep -q '^}$'
