"""CPU expert tier for GLM-5.3-Flash on the B70 (XPU): decode MoE picks that miss the device expert cache are split
between the GPU (zero-copy read + LRU admission) and an AVX2 CPU worker (glm-flash-lite kernels/cpu_avx2, MIT) by a
cost model. GPU side: /opt/glmopt/ftx/_ftx.so (ft_xpu.sycl: torch.ops.ftx.split / combine). Host side: ft_tier_ext.cpp
built with torch.utils.cpp_extension (cached under /opt/glmopt-build).

Per decode MoE call (M <= FTX_MAXM rows, not the MTP layer):
  ids2, w2 = ftx.split(ids, w, x, ...)   # CPU picks -> id -1 / weight 0, job published to USM host memory
  y = store.forward_cached(key, x, ids2, w2)
  ftx.combine(y, ...)                    # bounded spin on the worker's done seq, y += fp32 partial

Expert source for the CPU worker (FTX_SRC):
  ckpt (default, run G262-I0s800c1k): a CPU copy built from the checkpoint (native exllamav3 trellis layout) and re-laid
       out block-contiguous by tier_add_layer (swz=1): ~2.7 GB per layer, 114 GB for the 42 layers, a second host copy
       next to the GPU expert store's USM host copy.
  usm  no copy: the worker reads the GPU expert store's USM host blobs in place (exl3xpu PLANAR4 blob layout, ft_core.h
       layer_from_blobs), so every expert exists once in host RAM. I16 calls (every call where no expert has more than
       2 tokens: all of C1) run as row bands over the blob rows, other calls as 128-output blocks; both are
       bit-identical to the ckpt path (FTX_USM_KERN: 0 auto, 1 blocks only, 2 bands only = not bit-identical for AFFINE).

Env: FTX=1 (enable), FTX_CPUS (default "4-23"), FTX_MAXM (4), FTX_TZC (0.40 ms per zero-copy miss), FTX_THIT (0.023),
     FTX_A (0.11), FTX_B (0.104), FTX_TOK (0.2), FTX_MAXN (32), FTX_FORCE_N (-1), FTX_MAX_SPIN (400000 polls),
     FTX_MODE (-1 auto | 0 AFFINE | 1 EXACT | 2 I16), FTX_STATS_EVERY (2000 calls), FTX_SRC (ckpt | usm), FTX_USM_KERN (0),
     FTX_VALIDATE (0; 1 = at init, compare the tier's layers bit for bit with today's ckpt path on random calls, see
     validate()), FTX_VALIDATE_LAYERS (first,last FTX layer by model layer number), FTX_VALIDATE_NEXP (32),
     FTX_VALIDATE_TRIALS (48), FTX_VALIDATE_STRICT (1: a mismatch turns the tier off)
"""
import json, os, re, sys, time, threading
import torch

S = {"init": False, "failed": False, "calls": 0}
MODEL = os.environ.get("EXL3_MODEL_PATH", "")
_KDIR = "/opt/glmopt/ftx"
_H = None


def _log(*a):
    print("FTX:", *a, file=sys.stderr, flush=True)


def _pp_rank():
    try:
        from sglang.srt.distributed import get_pp_group
        return get_pp_group().rank_in_group
    except Exception:
        return 0


def _cpus():
    out = []
    spec = os.environ.get(f"FTX_CPUS_PP{_pp_rank()}", os.environ.get("FTX_CPUS", "4-23"))
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    allowed = os.sched_getaffinity(0)
    return [c for c in out if c in allowed]


def _build_host():
    global _H
    if _H is None:
        from torch.utils.cpp_extension import load
        d = os.environ.get("FTX_BUILD", "/opt/glmopt-build/ct")
        os.makedirs(d, exist_ok=True)
        _H = load(name="ft_tier_host", sources=[os.path.join(_KDIR, "ft_tier_ext.cpp")], build_directory=d,
                  extra_cflags=["-O3", "-mavx2", "-mfma", "-mf16c", "-march=znver3", "-std=c++17", "-I" + _KDIR],
                  extra_ldflags=["-lpthread"], verbose=False)
    return _H


class _Ckpt:
    """Expert tensors straight from the checkpoint's safetensors (mmap)."""

    def __init__(self, path):
        from safetensors import safe_open
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.where = idx
        self.path = path
        self.files = {}
        self._so = safe_open

    def get(self, name):
        f = self.where[name]
        if f not in self.files:
            self.files[f] = self._so(os.path.join(self.path, f), framework="pt", device="cpu")
        return self.files[f].get_tensor(name)


def _layer_id(key):
    m = re.search(r"layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def _ckpt_layer(ck, L, E):
    """Checkpoint tensors of model layer L's experts: (ptrs int64 [E, 3] native trellis addresses, fp16 scale dict,
    the trellis tensors that must stay alive while ptrs are used)."""
    pre = f"model.language_model.layers.{L}.mlp.experts."
    tr, sc = [], {k: [] for k in ("suh_g", "svh_g", "suh_u", "svh_u", "suh_d", "svh_d")}
    for e in range(E):
        row = []
        for pj, tag in (("gate_proj", "g"), ("up_proj", "u"), ("down_proj", "d")):
            t = ck.get(f"{pre}{e}.{pj}.trellis").contiguous()
            row.append(t)
            sc["suh_" + tag].append(ck.get(f"{pre}{e}.{pj}.suh").to(torch.float16))
            sc["svh_" + tag].append(ck.get(f"{pre}{e}.{pj}.svh").to(torch.float16))
        tr.append(row)
    ptrs = torch.tensor([[t.data_ptr() for t in row] for row in tr], dtype=torch.int64)
    st = {k: torch.stack(v).contiguous() for k, v in sc.items()}
    return ptrs, st, tr


def validate(H, store, keys_l):
    """FTX_VALIDATE=1: for the chosen layers, compare the tier's layer (as added: usm or ckpt) with today's ckpt path
    (a swizzled copy of experts [0, FTX_VALIDATE_NEXP) built from the checkpoint) on random 1-4 token calls in every
    mode, bit for bit, and time both at M = 1 with 8 picks. Returns True if every output was bit-identical."""
    nexp = int(os.environ.get("FTX_VALIDATE_NEXP", "32"))
    trials = int(os.environ.get("FTX_VALIDATE_TRIALS", "48"))
    want = os.environ.get("FTX_VALIDATE_LAYERS", "")
    ls = sorted(keys_l)
    pick = [int(v) for v in want.split(",") if v.strip()] if want else sorted({ls[0], ls[-1]})
    ck = _Ckpt(MODEL)
    ok = True
    for L in pick:
        if L not in keys_l:
            _log(f"validate: layer {L} is not an FTX layer, skipped"); continue
        li = keys_l[L]
        ptrs, st, tr = _ckpt_layer(ck, L, nexp)
        t0 = time.time()
        n, bad, maxd, maxr, ms_ref, ms_live = H.tier_validate(li, ptrs, st["suh_g"], st["svh_g"], st["suh_u"], st["svh_u"],
                                                              st["suh_d"], st["svh_d"], nexp, trials, 1234 + L)
        _log("validate", json.dumps({"layer": L, "src": os.environ.get("FTX_SRC", "ckpt"), "outputs": int(n),
                                     "not_bit_identical": int(bad), "max_abs_diff": maxd, "max_abs_ref": round(maxr, 3),
                                     "ms_per_expert_ckpt_swz": round(ms_ref, 4), "ms_per_expert_live": round(ms_live, 4),
                                     "s": round(time.time() - t0, 1)}))
        ok = ok and bad == 0
        del ptrs, st, tr
    ck.files.clear()
    return ok


def init(store):
    """Build the CPU tier for every MoE layer of this rank's store (called once, eager, before graph capture)."""
    t0 = time.time()
    H = _build_host()
    src = os.environ.get("FTX_SRC", "ckpt")
    if src not in ("ckpt", "usm"):
        raise ValueError(f"FTX_SRC={src!r}: ckpt or usm")
    usm = src == "usm"
    if usm and int(getattr(store, "K", 3)) != 3:
        raise ValueError(f"FTX_SRC=usm: the CPU kernels are K=3 only (store K={store.K})")
    if usm and not hasattr(store, "host"):
        raise ValueError("FTX_SRC=usm needs the USM expert store (EXL3_NVTIER=0)")
    torch.ops.load_library(os.path.join(_KDIR, "_ftx.so"))
    dev = store.dev
    E, Hd, I = store.E, store.H, store.I
    maxm = int(os.environ.get("FTX_MAXM", "4"))
    X = store.X
    # CPU picks are id -1: only the pair-mode MoE path skips them (align mode would index cnt[-1]); keep pair mode
    # for every row count the tier handles
    if maxm > 4:
        X.moe_set_pair_max_m(maxm)
        _log(f"moe pair mode up to M={maxm}")
    S["ctrl"] = X.host_alloc(8 * 8).view(torch.int64); S["ctrl"].zero_()
    S["hx"] = X.host_alloc(maxm * Hd * 2).view(torch.float16)
    S["picks"] = X.host_alloc(maxm * 8 * 3 * 4).view(torch.int32)
    S["hout"] = X.host_alloc(maxm * Hd * 4).view(torch.float32)
    keys = sorted(store.layer_index, key=lambda k: store.layer_index[k])
    nl = len(keys)
    cpus = _cpus()
    H.tier_init(len(cpus), cpus, S["ctrl"], S["hx"], S["picks"], S["hout"], int(os.environ.get("FTX_MODE", "-1")),
                Hd, I, nl, 0 if usm else 1)
    ck = None if usm else _Ckpt(MODEL)
    kern = int(os.environ.get("FTX_USM_KERN", "0"))
    S["keys"] = set()
    keys_l = {}
    for key in keys:
        li = store.layer_index[key]
        L = _layer_id(key)
        if L is None or key.startswith("mtp.") or "decoder" in key:
            _log("skip (GPU path):", key); continue
        S["keys"].add(key)
        keys_l[L] = li
        if usm:
            h = store.host[key]          # uint8 [E, blob] USM host memory, kept by the store for the server's lifetime
            assert h.shape == (E, store.blob) and h.is_contiguous(), (tuple(h.shape), store.blob)
            H.tier_add_layer_usm(li, h.data_ptr(), E, store.blob, kern)
            continue
        ptrs, st, tr = _ckpt_layer(ck, L, E)
        H.tier_add_layer(li, ptrs, st["suh_g"], st["svh_g"], st["suh_u"], st["svh_u"], st["suh_d"], st["svh_d"])
        del tr, ptrs   # tier_add_layer copied the trellis (swz); scales are kept by the extension
    if ck is not None:
        ck.files.clear()
    if os.environ.get("FTX_VALIDATE", "0") == "1":
        if not validate(H, store, keys_l) and os.environ.get("FTX_VALIDATE_STRICT", "1") == "1":
            raise RuntimeError("FTX_VALIDATE: the tier's outputs are not bit-identical to the ckpt path")
    g = lambda k, d: float(os.environ.get(k, d))
    S["pol"] = torch.tensor([g("FTX_TZC", 0.40), g("FTX_THIT", 0.023), g("FTX_A", 0.11), g("FTX_B", 0.104),
                             g("FTX_TOK", 0.2), g("FTX_MAXN", 32), g("FTX_FORCE_N", -1), 0.0], dtype=torch.float32, device=dev)
    S["score"] = torch.zeros(store.max_layers * E, dtype=torch.float32, device=dev)
    S["seq"] = torch.zeros(1, dtype=torch.int64, device=dev)
    S["dflag"] = torch.zeros(1, dtype=torch.int64, device=dev)
    S["stats"] = torch.zeros(8, dtype=torch.int64, device=dev)
    # poll mode in bits 40..47: 2 = uncached-hint loads of the host done flag (plain loads see a stale cached value)
    S["max_spin"] = int(os.environ.get("FTX_MAX_SPIN", "400000")) | (int(os.environ.get("FTX_POLL", "2")) << 40)
    S["maxm"] = maxm
    S["ptr"] = {k: S[k].data_ptr() for k in ("ctrl", "hx", "picks", "hout")}
    H.tier_start(cpus[0])
    ev = float(os.environ.get("FTX_STATS_S", "60"))
    if ev > 0:   # graph replay bypasses moe_decode(): report from a timer thread
        def _rep():
            while True:
                time.sleep(ev)
                try: summary()
                except Exception as e: _log("stats failed", repr(e)); return
        threading.Thread(target=_rep, daemon=True, name="ftx-stats").start()
    S["H"], S["store"], S["init"] = H, store, True
    _log(f"pp{_pp_rank()} tier up: {len(S['keys'])}/{nl} layers, E={E}, cpus={cpus[0]}-{cpus[-1]} ({len(cpus)}), src {src}"
         f"{f' (kern {kern}, reads the USM store in place)' if usm else ''}, swz {H.tier_stats()[5] / 1e9:.1f} GB, "
         f"{time.time() - t0:.0f}s")


def moe_decode(store, key, flat, ids, w):
    """Returns y or None (tier not applicable)."""
    if S["failed"] or flat.shape[0] > S.get("maxm", int(os.environ.get("FTX_MAXM", "4"))) or key.startswith("mtp."):
        return None
    if not S["init"]:
        if torch.xpu.is_current_stream_capturing():
            return None
        try:
            init(store)
        except Exception as e:  # never take the server down: log and fall back to the GPU path
            import traceback; traceback.print_exc()
            _log("init failed, tier off:", repr(e)); S["failed"] = True
            return None
    if store is not S["store"] or key not in S["keys"]:
        return None
    ids = ids if ids.dtype == torch.int32 and ids.is_contiguous() else ids.to(torch.int32).contiguous()
    w = w if w.dtype == torch.float32 and w.is_contiguous() else w.to(torch.float32).contiguous()
    x = flat.contiguous()
    P = S["ptr"]
    ids2, w2 = torch.ops.ftx.split(ids, w, x, store.slot_of_dev, store.layer_index[key], store.E, S["score"], S["pol"],
                                   P["ctrl"], P["hx"], P["picks"], S["seq"], S["dflag"], S["stats"])
    y = store.forward_cached(key, x, ids2, w2)
    torch.ops.ftx.combine(y, S["dflag"], P["ctrl"], P["hout"], S["stats"], S["max_spin"])
    S["calls"] += 1
    ev = int(os.environ.get("FTX_STATS_EVERY", "4000"))
    if ev and S["calls"] % ev == 0 and not torch.xpu.is_current_stream_capturing():
        summary()
    return y


def summary():
    st = S["stats"].tolist()
    hs = S["H"].tier_stats()
    d = {"layer_calls": st[3], "hits": st[0], "misses": st[1], "cpu_picks": st[2],
         "hit_rate": round(st[0] / max(1, st[0] + st[1]), 3), "cpu_share_of_misses": round(st[2] / max(1, st[1]), 3),
         "combine_waits": st[5], "spin_polls_mean": round(st[4] / max(1, st[5]), 1), "spin_timeouts": st[6],
         "cpu_jobs": hs[0], "cpu_experts": hs[1], "cpu_busy_ms_per_job": round(hs[4] / 1e6 / max(1, hs[0]), 3)}
    _log("stats", json.dumps(d))
    return d


def install(mod):
    """Hook the plugin's EXL3 MoE method (exl3xpu.sglang_plugin) once its classes are built."""
    orig_build = mod._build_classes

    def _build_classes():
        out = orig_build()
        cls = mod.__dict__.get("Exl3XpuMoEMethod") or out[0].__dict__.get("Exl3XpuMoEMethod")
        if cls is None:
            _log("Exl3XpuMoEMethod not found; tier not installed")
            return out
        orig_apply = cls.apply

        def apply(self, layer, dispatch_output):
            x = dispatch_output.hidden_states
            flat = x.reshape(-1, x.shape[-1])
            if flat.shape[0] <= int(os.environ.get("FTX_MAXM", "4")) and mod._MOE_CACHE:
                from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
                tk = dispatch_output.topk_output
                store = layer.exl3_moe_store
                tw = tk.topk_weights if self.routed_scale == 1.0 else tk.topk_weights * self.routed_scale
                store.prefill_reset()
                y = moe_decode(store, self.key, flat, tk.topk_ids.reshape(flat.shape[0], -1), tw.reshape(flat.shape[0], -1))
                if y is not None:
                    return StandardCombineInput(hidden_states=y.view_as(x))
            return orig_apply(self, layer, dispatch_output)

        cls.apply = apply
        _log("installed on", cls.__name__)
        return out

    mod._build_classes = _build_classes
