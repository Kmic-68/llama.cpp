# toolchain.sh: sourced by build.sh and setup.sh (which define say and die). pick_toolchain sets NVCC, CUDA_VER and
# HOSTCXX from the environment (CUDA=<toolkit root>, HOSTCXX=<compiler>) and stops with a plain message when the toolkit is
# not CUDA 12.x (CUDA 13 dropped sm_60) or the host compiler is newer than that toolkit accepts.
pick_toolchain() {
  if [ -n "${CUDA:-}" ]; then
    NVCC=$CUDA/bin/nvcc
    export PATH=$CUDA/bin:$PATH
  else
    NVCC=$(command -v nvcc || true)
  fi
  [ -n "$NVCC" ] && [ -x "$NVCC" ] || die "no nvcc (install CUDA 12.x, or set CUDA=<toolkit root>)"
  CUDA_VER=$("$NVCC" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')
  case "$CUDA_VER" in
    12.*) ;;
    *) die "CUDA $CUDA_VER: a 12.x toolkit is needed (CUDA 13 has no sm_60); ours is 12.4.131" ;;
  esac
  if [ -z "${HOSTCXX:-}" ]; then
    HOSTCXX=$(command -v g++-13 || command -v g++ || true)
  fi
  [ -n "$HOSTCXX" ] && [ -x "$(command -v "$HOSTCXX")" ] || die "no host C++ compiler (install g++-13, or set HOSTCXX)"
  HOSTCXX=$(command -v "$HOSTCXX")
  # nvcc refuses a newer gcc than its release supports: 12.0-12.3 up to gcc 12, 12.4-12.7 up to 13, 12.8 and later up to 14
  local gcc_major cuda_minor gcc_max id
  gcc_major=$("$HOSTCXX" -dumpversion | cut -d. -f1)
  cuda_minor=${CUDA_VER#12.}
  gcc_max=12
  [ "$cuda_minor" -ge 4 ] && gcc_max=13
  [ "$cuda_minor" -ge 8 ] && gcc_max=14
  id=$("$HOSTCXX" --version 2>/dev/null || true)
  if [[ "$id" != *clang* ]] && [ "$gcc_major" -gt "$gcc_max" ]; then
    die "$HOSTCXX is gcc $gcc_major, CUDA $CUDA_VER takes gcc $gcc_max at most: install g++-$gcc_max and set HOSTCXX=g++-$gcc_max"
  fi
  command -v cmake >/dev/null || die "no cmake (3.22 or newer)"
  command -v git >/dev/null || die "no git"
}
