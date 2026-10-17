# Low-RAM mode: one host copy of the experts (`FTX_SRC=usm`)

Branch `b70-lowram`. This is opt-in. With no new env set, the server behaves exactly as run `G262-I0s800c1k`.

## The problem

Run `G262-I0s800c1k` holds every routed expert twice in host RAM:

| copy | what reads it | size |
|---|---|---|
| GPU expert store, USM host memory (`ExpertStore.host`, exl3xpu blob layout, PLANAR4) | VRAM cache fills, GPU zero-copy misses, staged prefill | 42 x 288 x 9,474,048 B = **114.6 GB**, not charged to the cgroup |
| FTX CPU tier, swizzled copy (`tier_add_layer`, swz=1, built from the checkpoint) | the AVX2 CPU worker (decode misses that FTX gives to the CPU) | 42 x 288 x 3 x 3,145,728 B = **114.2 GB** anon, plus 0.45 GB of fp16 scales |
| everything else (torch, SGLang, Triton, page cache while loading) | | about 30 GB of cgroup peak (8 GB anon) |

Total: about 259 GB (cgroup peak 144.3 GB plus 114.6 GB of USM). The goal is a 180 GB host or smaller, with decode and
prefill no slower than today if possible.

## Options

The measured stats used below come from the FTX tier at C1 in the earlier campaign:

- hit_rate 0.46-0.51, so about 3.9 misses per layer call (8 picks)
- cpu_share_of_misses 0.85, so about 3.3 CPU experts per job
- cpu_busy_ms_per_job 0.54-0.69, which is about 23-29 ms of CPU work per token, against 57 ms per token at 17.5 tok/s
- FTX_TZC 0.40 ms per GPU zero-copy miss
- C004: 0.114-0.120 ms per expert on the swizzled copy against 0.165-0.172 ms on checkpoint order with 128-output
  block units (+45 %)

### A. Drop the USM copy (VRAM cache and GPU misses from the NVMe store, prefill staged from NVMe)

- **RAM.** About 145 GB: the 114.2 GB swizzled copy plus about 30 GB.
- **Decode.**
  - The 15 % of misses the GPU takes today, about 0.6 per layer call or about 25 per token, become exact NVMe reads
    on the critical path. That is 9.47 MB at 20-24 GB/s, 0.40-0.47 ms, plus about 0.2 ms first byte, against 0.40 ms
    of zero-copy today. Expect +4-6 ms per token, or 15.9-16.3 tok/s (-7 to -10 %).
  - The VRAM cache would admit only through NVMe. Replacing LRU write-through admission with a static hot set loses
    hit rate, which sends more work to an already loaded CPU.
  - Routing every miss to the CPU instead costs +0.6 experts per job and the adaptive cache.
- **Prefill.** Every 1,024-token chunk touches nearly all 12,096 experts: 114.6 GB per chunk. Today that streams from
  RAM at about 21 GB/s (5.4 s per chunk at 8k). From NVMe at 20-24 GB/s it is the same or slower, and only if the
  array sustains that while decode reads compete.
- **Effort.** High, and none of it is testable without the GPU:
  - an exact-miss wait kernel in `moe-glm-ftx` (port of qwen `exact_miss.h`, itself new)
  - an NVMe service and a static hot set
  - NVMe staging for prefill
  - FTX cost-model terms for NVMe

### B. Drop the swizzled CPU copy: the CPU tier reads the USM blobs in place (chosen, implemented)

- **RAM.** About 145 GB peak: 114.6 GB USM plus about 30 GB. Steady state is about 125 GB: USM plus about 8-10 GB anon.
  The 0.45 GB of CPU-side scale tensors also goes, because the scales are read from the blob. This fits a 180 GB host
  with more than 30 GB of margin, and probably a 160 GB host. The container `--memory` can drop from 260g to about
  48-64g.
- **Prefill.** Unchanged. It is the same GPU path, and it already reads the USM copy.
- **Decode.** Only the CPU kernel's input changes.
  - The blob is not the checkpoint order. Gate and up share tile rows (24 KB rows), and words are PLANAR4 within
    groups of 4 tiles.
  - The CPU code de-planarizes each 384-byte group into L1 (9 permutes + 6 blends per tile, small next to the
    tile's 32 decode8 steps). It then runs the unchanged tile kernels.
  - Calls with at most 2 tokens per expert (I16, all of C1) run as **row bands**. A unit is 8 tile rows of one matrix,
    8 contiguous runs of 12 KB (gate, up) or 24 KB (down). This avoids the 768-byte-per-row strided pattern behind the
    +45 % of C004.
  - Other calls run as 128-output blocks.
  - Expected CPU cost per job: +5-20 % (de-planarize, band partials of 1 MB per matrix in L2/L3, one more barrier).
    That is +1-5 ms per token if the CPU is fully on the critical path, or 16.2-17.4 tok/s (0 to -7 %). The cost model
    moves picks back to the GPU as the CPU slows, which limits the loss.
  - Worst case, if bands gain nothing over blocks on xe USM pages: +45 %, about -14 %.
  - Decode stays **bit-identical** (proved on CPU, see below), so quality cannot move.
- **Effort.** Small. Everything is CPU code that is testable without a GPU: `ft_core.h` planar Mats,
  `tier_add_layer_usm`, `tier_validate` and the `ftx_tier.py` switch. No GPU kernel changes.

### C. Hybrid: keep USM, and give the CPU a swizzled copy of only the most-missed experts

- **RAM.** 114.6 GB + f x 114.2 GB + 30 GB. For a 180 GB host, f is at most about 0.3 (35 GB).
- **Decode.** The CPU may only take misses on its subset. The top 30 % of experts by miss count cover perhaps 55-65 %
  of misses, against 85 % taken today, so about 20-30 points of misses move to GPU zero-copy at 0.40 ms each. That is
  about +0.4 ms per call net of the CPU work saved, or about -15 %.
- **Effort.** Medium: a per-expert CPU eligibility mask in `ftx.split`, and a miss-count profile.
- **Verdict.** It only beats B if B's kernel turns out more than about 30 % slower than the swizzled copy. Even then,
  "B plus retuned `FTX_B`" is the cheaper fix.

### Choice

B wins on all three axes:

- It uses the same RAM as A (about 145 GB) with no new I/O on the critical path.
- Prefill is untouched.
- Outputs are bit-identical.
- It is the smallest change, and all of it is testable without a GPU.

To go below about 130 GB later, B combines with A's machinery. USM would keep only a hot subset, and cold experts would
come from `/mnt/nvx/glm53/glm53_experts.bin` (same blob order and record size, `record = model_layer*288 + e`, L=45).
The CPU tier already reads the blob format, so it could take NVMe-filled pages with no further layout work.

## What changed (all behind `FTX_SRC=usm`; default `ckpt` = today)

| file | change |
|---|---|
| `ftx/ft_core.h` | `Mat.planar`, `Mat.rs`. `deplanar4` (AVX2) + `deplanar4_ref`. Planar instantiations of `gemv_block`, `gemv_block_i16`, `gemv_band`, `gemv_band_i16`: `if constexpr`, so the default instantiations compile as before. `blob_bytes_k3`, `layer_from_blobs` |
| `ftx/ft_tier_ext.cpp` | `swizzle_copy` (the old inline loop, unchanged). `tier_add_layer_usm(li, base, E, blob, kern)`. `forward_layer`: usm + I16 gives `moe_forward_band`, everything else gives `moe_forward`. `tier_validate` (bit-identity and timing against a swizzled checkpoint copy) |
| `opt/ftx_tier.py` | `FTX_SRC`, `FTX_USM_KERN`, `FTX_VALIDATE*`. The checkpoint loading is factored into `_ckpt_layer` (same tensors, same order) |
| `opt/test_ftx_usm.py` | CPU-only check on real weights inside the image |
| `ftx/test_usm.cpp`, `test_usm_fixture.py`, `test_usm.sh` | CPU-only synthetic check: layout, bit-identity, speed |
| `serve/probe_exact.py` | A/B output bit-identity probe between two server arms |
| `serve/ramwatch.sh` | MemAvailable + cgroup current/peak + PSS anon/shmem/file, once a second |
| `serve/selfcheck.py`, `install.sh`, `ftx/build_ftx.sh` | new symbols checked, test installed, notes |

Env (new):

- `FTX_SRC=ckpt|usm` (default `ckpt`).
- `FTX_USM_KERN=0|1|2`: 0 = auto (bands for I16, blocks otherwise; bit-identical), 1 = blocks only (bit-identical),
  2 = bands always (AFFINE/EXACT then re-associate fp32 sums).
- `FTX_VALIDATE=1`: at tier init, `tier_validate` on `FTX_VALIDATE_LAYERS` (default: first and last FTX layer) with
  `FTX_VALIDATE_NEXP=32` experts and `FTX_VALIDATE_TRIALS=48`. With `FTX_VALIDATE_STRICT=1` (default) a mismatch
  turns the tier off.

`FTX_SRC=usm` needs `EXL3_NVTIER=0` (the USM store) and K=3. If either is missing, init refuses and the tier is off.

## Tested so far (no GPU; macOS, clang x86_64 under Rosetta)

`b70/ftx/test_usm.sh` with 12 random experts, 24 trials of 1-4 tokens (up to 4 tokens per expert), and blobs packed by
a verbatim copy of `exl3xpu.moe_offload.pack_expert`:

```
1. deplanar4 vs scalar reference: identical (2000 random groups)
2. blob layout vs checkpoint: 1179648 tiles, every tile and scale vector identical
3. forward vs today's swz path (24 trials, 1-4 tokens, max tokens per expert 4, max |y| 64.8, non-finite 0):
   AFFINE outputs  245760 | native-in-place diff 0 | usm block diff 0 | usm auto diff 0 | usm band diff 245112 (max |d| 0.0306)
   EXACT  outputs  245760 | native-in-place diff 0 | usm block diff 0 | usm auto diff 0 | usm band diff 243442 (max |d| 0.0323)
   I16    outputs  245760 | native-in-place diff 0 | usm block diff 0 | usm auto diff 0 | usm band diff 0 (max |d| 0)
ALL OK
```

`ft_tier_ext.cpp` passes a clang `-fsyntax-only -Wall` check against the torch headers.

**Not tested:**

- g++ in the image. The band I16 identity relies on the same fma contraction as the block path; `FTX_VALIDATE`
  settles this on the host.
- The torch extension build and the Python path in a real server.
- Speed on the EPYC. The Rosetta timings are meaningless.
- How the xe driver maps USM host pages for the CPU (cacheable, 4 KiB or huge).
- RAM and kit numbers.

## GPU-host test plan (omarchy, card 83:00.0)

All numbers below are pass criteria.

0. **Image.** Build it from this branch's head. Only the pin changes.

   ```bash
   git -C local-ai-images pull
   docker build --build-arg GLM_COMMIT=<b70-lowram head sha> -t glm53-b70-lowram local-ai-images/glm53-flash-b70-ftx
   docker run --rm glm53-b70-lowram check        # "glm-b70 check ok", host extension not rebuilt
   ```

1. **CPU only, synthetic, speed.** No GPU needed. The server can be down.

   ```bash
   docker run --rm --cpuset-cpus 4-23 --entrypoint bash glm53-b70-lowram -c \
     'cd /opt/glm-b70-src/ftx && ./test_usm.sh /tmp/t 20 24 40 64'
   ```

   Pass:
   - "ALL OK", with `usm block diff 0` and `usm auto diff 0` in every mode.
   - Speed line: usm band at most 1.10x swz.
   - Note usm block / swz too. It tells you whether `FTX_USM_KERN=1` is a fallback.

2. **CPU only, real weights** (layer 3, 64 experts, 0.6 GB):

   ```bash
   docker run --rm --cpuset-cpus 4-23 --entrypoint python3 -e FTX_CPUS=4-23 \
     -v /data/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw:/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw:ro \
     glm53-b70-lowram /opt/glmopt/test_ftx_usm.py 3 64
   ```

   Pass: "0 not bit-identical" for auto and blocks, and "ALL OK". Record the ms/expert ratio. This uses plain malloc
   memory, not USM.

3. **Server A/B, same session, same flags, fresh container per arm.** Use the measured command from `README.md` with
   these differences:
   - arm `ckpt`: `-e FTX_SRC=ckpt -e FTX_VALIDATE=1`, `--memory 260g` (today)
   - arm `usm`: `-e FTX_SRC=usm -e FTX_VALIDATE=1`, `--memory 64g --memory-swap 64g`

   For each arm:

   ```bash
   bash b70/serve/ramwatch.sh glm53-b70 ram_$ARM.csv &     # start before docker run (baseline row)
   docker run -d --name glm53-b70 ... glm53-b70-lowram
   # wait for /health
   python3 b70/serve/probe_exact.py run http://127.0.0.1:30260 probe_$ARM.json   # first requests after start
   docker logs glm53-b70 2>&1 | grep -E 'FTX: (validate|tier up|stats)'
   python3 bench/sweep.py --url http://127.0.0.1:30260 --card intel-arc-pro-b70-32gb --template glm \
     --config "G262-lowram $ARM" --no-early-exit --out kit_$ARM.json
   ```

   Then `python3 b70/serve/probe_exact.py compare probe_ckpt.json probe_usm.json`.

   Pass:
   - **Bit-exactness.**
     - `FTX: validate` lines show `not_bit_identical 0` for both layers in the usm arm.
     - probe compare prints `IDENTICAL` (6 prompts, token ids and top-5 logprobs).
     - FTX stats show `spin_timeouts 0` in both arms. A timeout makes that call's output differ by design.
   - **RAM** (ramwatch):
     - usm arm `cg_peak_gb` at most 40 (ckpt: about 144).
     - Steady `held_gb` at most 15 (ckpt: about 122).
     - System MemAvailable drop from baseline: peak at most 150 GB, steady at most 135 GB (ckpt: about 259).
     - Uncharged = drop - cg_current ≈ 114.6 GB in both arms. This is the USM store.
   - **FTX stats at C1:**
     - usm `cpu_busy_ms_per_job` at most 1.10x ckpt, and the `ms_per_expert_live / ms_per_expert_ckpt_swz` ratio in
       the validate line at most 1.10.
     - `hit_rate` and `cpu_share_of_misses` within ±0.03 of ckpt.
   - **Kit** (`bench/sweep.py`, the protocol of `results/G262-I0s800c1k/kit_sweep.json`):
     - decode C1 at least 16.6 tok/s (95 % of 17.5), target at least 17.0
     - C2 and C4 aggregate within 5 % of 21.5 and 24.5
     - prefill 8k / 16k / 32k / 64k within 3 % of 189 / 174 / 147 / 112 (the path is unchanged)

4. **If decode misses its target.**
   - If blocks beat bands on the box, rerun the usm arm with `FTX_USM_KERN=1`.
   - Set `FTX_B` to the new measured `cpu_busy_ms_per_job` slope. The cost model then moves misses back to the GPU.
   - If USM pages read slowly from the CPU (validate ratio well above 1.45, the C004 native-order cost), the xe host
     mapping is the issue. Report it; the next step is a hugepage-backed host allocation for the store
     (`host_alloc` replacement), which is a change to the exl3xpu plugin.

5. **Optional 180 GB proof.** Repeat the usm arm with the host's free RAM limited to 180 GB, for example with a
   `--memory 180g` systemd slice around the docker daemon or a tmpfs filler sized to total - 180 GB. Take care: never
   let the filler push the box into OOM. Run the probe plus the C1 decode only.
