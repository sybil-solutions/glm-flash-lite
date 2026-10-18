"""CPU-only image check (no GPU, no weights): the MoE and FTX libraries load next to torch the way the server loads
them, the FTX host extension is prebuilt and loads without recompiling, the post-import hooks and the exl3xpu plugin
files import, and the sglang file replacements are the packaged ones."""
import hashlib, os, sys, time
import torch

torch.ops.load_library(os.environ["EXL3_MOE_LIB"])
torch.ops.load_library("/opt/glmopt/ftx/_ftx.so")
assert hasattr(torch.ops.ftx, "split") and hasattr(torch.ops.ftx, "combine"), "ftx ops missing"

sys.path.insert(0, "/opt/glmopt")
import ftx_tier  # noqa: E402
t0 = time.time()
h = ftx_tier._build_host()
assert all(hasattr(h, n) for n in ("tier_init", "tier_add_layer", "tier_start", "tier_stats", "tier_add_layer_usm",
                                   "tier_validate")), dir(h)
dt = time.time() - t0
assert dt < 20, f"host extension took {dt:.0f}s: it was rebuilt instead of reused"

import glm_opt  # noqa: E402,F401  (installed through zz_glm_opt.pth as well)
assert any(type(f).__name__ == "_F" for f in sys.meta_path), "glm_opt import hook not installed"
import exl3xpu.nvtier  # noqa: E402,F401
import exl3xpu.mla_unfuse  # noqa: E402,F401
src = open("/opt/trellis-serve/xpu/exl3xpu/sglang_plugin.py", "rb").read()
assert b"EXL3_MOE_STAGE_MIN_M" in src, "sglang_plugin.py is not the GLM tier plugin"
import sglang.srt.models.glm5_next  # noqa: E402,F401
print(f"glm-b70 check ok: moe lib, _ftx.so, ft_tier_host ({dt:.1f}s, cached), glm_opt hooks, exl3xpu plugin, glm5_next")
