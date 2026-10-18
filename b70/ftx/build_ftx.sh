#!/usr/bin/env bash
# Build the FTX CPU expert tier for the B70, CPU only (needs the image's oneAPI icpx, torch XPU and a host C++ compiler).
#   1. _ftx.so          GPU side (ft_xpu.sycl: torch.ops.ftx.split / combine), spir64. Same flags as the campaign's
#                       bf.sh, which built the measured opt/ftx/_ftx.so (2026-10-15 23:36).
#   2. ft_tier_host.so  host side (ft_tier_ext.cpp + ft_core.h, AVX2), built exactly the way ftx_tier._build_host()
#                       builds it at run time (torch.utils.cpp_extension, -O3 -mavx2 -mfma -mf16c -march=znver3), into
#                       FTX_BUILD so the server finds it up to date and does not recompile.
# ft_tier_ext.cpp and ft_core.h are kernels/cpu_avx2/ (glm-flash-lite main at 0bd10f3) plus the FTX_SRC=usm path
# (planar Mats on the GPU store's USM blobs, tier_add_layer_usm, tier_validate; branch b70-lowram). The default
# FTX_SRC=ckpt path computes the same as before, but the source changed, so ft_tier_host.so has a new hash.
#   KDIR       directory holding these sources and receiving _ftx.so; must be the path ftx_tier.py uses (/opt/glmopt/ftx)
#   OPTDIR     directory holding ftx_tier.py (/opt/glmopt)
#   FTX_BUILD  host extension build directory (/opt/glmopt-build/ct)
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -euo pipefail
KDIR=${KDIR:-/opt/glmopt/ftx}; OPTDIR=${OPTDIR:-/opt/glmopt}; export FTX_BUILD=${FTX_BUILD:-/opt/glmopt-build/ct}
T=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
cd "$KDIR"
icpx -fsycl -fsycl-targets=spir64 -O3 -fPIC -std=c++17 -shared -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I"$T/include" -I"$T/include/torch/csrc/api/include" -x c++ ft_xpu.sycl -x none -o _ftx.so.tmp \
  -L"$T/lib" -Wl,-rpath,"$T/lib" -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu
mv -f _ftx.so.tmp _ftx.so && ls -la _ftx.so
python3 -c "
import sys; sys.path.insert(0, '$OPTDIR'); import ftx_tier
ftx_tier._KDIR = '$KDIR'; h = ftx_tier._build_host()
print('host ext ok', [n for n in dir(h) if n.startswith('tier')])"
ls -la "$FTX_BUILD"
