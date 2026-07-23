#!/bin/bash
# corex_nvcc_wrapper.sh
# Drop-in replacement for `nvcc` when building lc0's CUDA backend on the
# Iluvatar CoreX (ivcore11) GPU environment.
#
# The CoreX SDK does NOT ship nvcc (red line: no nvcc). Instead the CUDA
# frontend is clang++ invoked with `-x ivcore --cuda-gpu-arch=ivcore11`.
# lc0's meson.build drives .cu compilation through an `nvcc`-style program,
# probing it with `-h` / `--dryrun` and finally compiling with
#   nvcc <arch flags> <-Xcompiler ...> -I <...> --std=c++NN -c IN -o OUT
#
# This wrapper answers those probes and translates a real compile command
# into an equivalent CoreX clang++ invocation, dropping nvcc-only flags
# (-arch/-gencode/-code/-Xcompiler/-Wno-deprecated-gpu-targets/...).

set -o pipefail

COREX=/usr/local/corex
CLANG=${COREX}/bin/clang++

# ---- probe modes -----------------------------------------------------------
for a in "$@"; do
  case "$a" in
    -h|--help)
      # Minimal help. Deliberately advertise NO sm_XX / -arch=native /
      # -arch=all-major so meson's flag-detection falls through to a benign
      # branch; the arch flags it would emit are dropped below anyway.
      echo "corex_nvcc_wrapper (clang++ -x ivcore for ivcore11)"
      echo "Usage: nvcc [options] -c <file>.cu -o <file>.o"
      exit 0
      ;;
    --dryrun)
      # meson uses --dryrun to (a) detect the highest supported C++ std and
      # (b) scan for __CUDA_ARCH__ tokens. Succeed silently; std probing
      # picks the first (c++20) and CUTLASS stays disabled (max_cuda=0).
      exit 0
      ;;
  esac
done

# ---- real compile: translate args ------------------------------------------
args=()
next_is_value_to_drop=0
next_is_isystem=0

while [ $# -gt 0 ]; do
  tok="$1"
  if [ "$next_is_value_to_drop" -eq 1 ]; then
    next_is_value_to_drop=0
    shift; continue
  fi
  case "$tok" in
    -Xcompiler)
      # host-compiler passthrough; keep -fPIC, drop MSVC/other host flags
      shift
      hf="$1"
      case "$hf" in
        -fPIC|-fpic) args+=("-fPIC") ;;
        *) : ;;  # drop -MT/-MD/-MTd/-MDd and friends
      esac
      shift; continue
      ;;
    -ccbin|-Xptxas|-Xlinker|-Xcudafe|-Xnvlink)
      next_is_value_to_drop=1; shift; continue ;;
    -ccbin=*|-Xptxas=*) shift; continue ;;
    -arch=*|-code=*|-gencode|-gencode=*|--generate-code|--generate-code=*) \
      # a bare -gencode / --generate-code takes a following "arch=..,code=.." value
      case "$tok" in -gencode|--generate-code) next_is_value_to_drop=1 ;; esac
      shift; continue ;;
    -Wno-deprecated-gpu-targets|--use_fast_math|-lineinfo|--generate-line-info) \
      shift; continue ;;
    -maxrregcount=*|--threads|-rdc=*|--relocatable-device-code=*) \
      case "$tok" in --threads) next_is_value_to_drop=1 ;; esac
      shift; continue ;;
    --std=*) args+=("-std=${tok#--std=}"); shift; continue ;;
    -isystem=*) args+=("-isystem" "${tok#-isystem=}"); shift; continue ;;
    *) args+=("$tok"); shift; continue ;;
  esac
done

exec "$CLANG" \
  -x ivcore \
  --cuda-path="${COREX}" \
  --cuda-gpu-arch=ivcore11 \
  -Wno-unknown-cuda-version \
  -fPIC \
  "${args[@]}" \
  -I"${COREX}/include"
