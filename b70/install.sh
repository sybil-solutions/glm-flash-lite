#!/usr/bin/env bash
# Install the GLM-5.3-Flash B70 FTX stack into the image, at the paths the campaign bind-mounted (run G262-I0s800c1k).
# Base image: ghcr.io/sybil-solutions/qwen38-flash-next-b70-offload@sha256:d30df691d6ab890b95f7c6179e4b677a0d759018e39b20e93bf0efdb8780450c
# (sglang 0.5.20-xpu + exl3xpu/trellis-serve, torch XPU, oneAPI 2026.0). CPU only, no GPU needed.
#   /opt/glmopt                    opt/*.py (post-import hooks, FTX tier) + ftx/ sources and _ftx.so
#   /opt/glmopt-build/ct           prebuilt FTX host extension (ft_tier_host.so)
#   /moe/_moe_n128.so              MoE library (EXL3_MOE_LIB)
#   /opt/trellis-serve/xpu/exl3xpu sglang_plugin.py (= plugin/sglang_plugin.tier.py), nvtier.py, mla_unfuse.py
#   site-packages                  sglang file replacements (sglang-patches/) + zz_glm_opt.pth
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
OPT=/opt/glmopt; BUILD=/opt/glmopt-build; MOE=/moe
P=/opt/trellis-serve/xpu/exl3xpu
S=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
SG=$S/sglang/srt

mkdir -p "$OPT/ftx" "$BUILD" "$MOE"
cp "$SRC"/opt/glm_opt.py "$SRC"/opt/ftx_tier.py "$SRC"/opt/mhc_triton.py "$SRC"/opt/pp_host.py "$SRC"/opt/test_ftx_usm.py "$OPT/"
cp "$SRC"/ftx/ft_xpu.sycl "$SRC"/ftx/ft_tier_ext.cpp "$SRC"/ftx/ft_core.h "$OPT/ftx/"
cp "$SRC"/opt/zz_glm_opt.pth "$S/zz_glm_opt.pth"

# exl3xpu plugin files (the campaign mounted these over the base image's copies)
test -d "$P"
cp "$SRC"/plugin/sglang_plugin.tier.py "$P/sglang_plugin.py"
cp "$SRC"/plugin/nvtier.py "$P/nvtier.py"
cp "$SRC"/plugin/mla_unfuse.py "$P/mla_unfuse.py"

# sglang file replacements: every target must already exist in the base (same sglang build the campaign ran)
while read -r f dst; do
  test -f "$SG/$dst" || { echo "missing sglang file $SG/$dst" >&2; exit 1; }
  cp "$SRC/sglang-patches/$f" "$SG/$dst"
done <<'MAP'
deepseek_v2_override.py arg_groups/model_overrides/deepseek_v2.py
xpu_chunk_delta_h.py hardware_backend/xpu/kernels/fla/chunk_delta_h.py
dsa_backend_kpool.py layers/attention/dsa/dsa_backend_kpool.py
dsa_backend.py layers/attention/dsa_backend.py
dsa_indexer_kpool.py layers/attention/dsa/dsa_indexer_kpool.py
glm5_next.py models/glm5_next.py
MAP

# kernels (CPU-only builds with the image's icpx)
OUT=$MOE/_moe_n128.so bash "$SRC/kernels/moe-glm-ftx/build.sh"
KDIR=$OPT/ftx OPTDIR=$OPT FTX_BUILD=$BUILD/ct bash "$SRC/ftx/build_ftx.sh"

# serving helpers
mkdir -p /opt/glm-b70
cp -r "$SRC"/serve "$SRC"/README.md /opt/glm-b70/
chmod +x /opt/glm-b70/serve/entrypoint.sh
echo "glm-b70 installed"
