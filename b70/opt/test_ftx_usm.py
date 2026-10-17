"""test_ftx_usm.py [LAYER] [NEXP] [TRIALS]: CPU-only check of FTX_SRC=usm on real GLM-5.3-Flash weights (no GPU, no
server). Packs experts [0, NEXP) of one layer with exl3xpu.moe_offload.pack_expert (the function that fills the GPU
store's USM host copy) into plain host memory, points the CPU tier at the blobs (tier_add_layer_usm) and runs
tier_validate against today's path (swizzled checkpoint copy): bit-identity over random 1-4 token calls in every mode,
then ms per expert at M = 1 with 8 picks, for the auto kernel (bands for I16) and for blocks only.
Run inside the image (CPU only):
  docker run --rm --entrypoint python3 --cpuset-cpus 4-23 -v $W:/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw:ro \\
    -e EXL3_MODEL_PATH=/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw IMG /opt/glmopt/test_ftx_usm.py 3 64
Note: plain malloc memory may be THP-backed where the real USM host copy is not; the server's FTX_VALIDATE=1 line times
the real USM pages."""
import os, sys, time
sys.path.insert(0, "/opt/glmopt")
os.environ.setdefault("FTX_BUILD", "/opt/glmopt-build/ct")
import torch
import ftx_tier as F
from exl3xpu.moe_offload import pack_expert

L = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NE = int(sys.argv[2]) if len(sys.argv) > 2 else 64
TRIALS = int(sys.argv[3]) if len(sys.argv) > 3 else 48
Hd, I, K = 4096, 2048, 3
BLOB = (Hd // 16) * (2 * I // 16) * 96 + (I // 16) * (Hd // 16) * 96 + (3 * Hd + 3 * I) * 2   # 9,474,048
H = F._build_host()
ck = F._Ckpt(os.environ["EXL3_MODEL_PATH"])
pre = f"model.language_model.layers.{L}.mlp.experts."
t0 = time.time()
blobs = torch.empty((NE, BLOB), dtype=torch.uint8)
for e in range(NE):
    m = {r: {s: ck.get(f"{pre}{e}.{r}_proj.{s}") for s in ("trellis", "suh", "svh")} for r in ("gate", "up", "down")}
    pack_expert(m["gate"], m["up"], m["down"], K, out=blobs[e])
ptrs, st, tr = F._ckpt_layer(ck, L, NE)
print(f"layer {L}: {NE} experts packed ({NE * BLOB / 1e9:.2f} GB) in {time.time() - t0:.0f}s", flush=True)
cpus = F._cpus()
ctrl = torch.zeros(8, dtype=torch.int64); hx = torch.zeros(4 * Hd, dtype=torch.float16)
picks = torch.zeros(4 * 8 * 3, dtype=torch.int32); hout = torch.zeros(4 * Hd, dtype=torch.float32)
H.tier_init(len(cpus), cpus, ctrl, hx, picks, hout, -1, Hd, I, 1, 0)
ok = True
for kern, name in ((0, "usm auto (bands for I16)"), (1, "usm blocks")):
    H.tier_add_layer_usm(0, blobs.data_ptr(), NE, BLOB, kern)
    n, bad, maxd, maxr, ms_ref, ms_live = H.tier_validate(0, ptrs, st["suh_g"], st["svh_g"], st["suh_u"], st["svh_u"],
                                                          st["suh_d"], st["svh_d"], NE, TRIALS, 1234 + L)
    print(f"{name}: {int(n)} outputs, {int(bad)} not bit-identical (max |d| {maxd:.3g}, max |y| {maxr:.3g}); "
          f"ms/expert ckpt-swz {ms_ref:.4f} vs {ms_live:.4f} ({(ms_live / ms_ref - 1) * 100:+.1f} %), {len(cpus)} cpus", flush=True)
    ok = ok and bad == 0
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
