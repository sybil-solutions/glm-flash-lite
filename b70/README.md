# b70: GLM-5.3-Flash on one Intel Arc Pro B70, experts in RAM, FTX CPU expert tier

This directory packages the serving stack for GLM-5.3-Flash EXL3 3.05 bpw on one Intel Arc Pro B70 32 GB, with the
routed experts in host RAM and a CPU expert tier (FTX). It is the stack measured as run `G262-I0s800c1k` on omarchy on
2026-10-17. Everything the campaign bind-mounted into the container is here as source, so an image can be built
without host mounts. Only the weights are mounted (`/models`, read-only).

- Engine: SGLang 0.5.20-xpu with the exl3xpu (trellis-serve) EXL3 plugin, from the base image
  `ghcr.io/sybil-solutions/qwen38-flash-next-b70-offload@sha256:d30df691d6ab890b95f7c6179e4b677a0d759018e39b20e93bf0efdb8780450c`.
  The campaign ran that exact image with `--entrypoint python3`.
- Weights: `turboderp/GLM-5.3-Flash-exl3`, branch `3.05bpw`, revision `332ab457b709b7ba30dd9a448be5de03b80a7ac9`
  (125.26 GB), at `/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw`.
- Image definition: `sybil-solutions/local-ai-images`, directory `glm53-flash-b70-ftx/`.

## Measured (run G262-I0s800c1k)

Setup: omarchy, 1x Arc Pro B70 32 GB (PCI 83:00.0, 230 W cap), AMD EPYC 7443P (24 cores), 512 GB 8-channel DDR4.
Container flags: `--memory 260g --cpuset-cpus 4-23,28-47`, 800 VRAM expert slots (7.58 GB), FTX workers on CPUs 4-23,
context 262,144 tokens, 270,336-token fp8 KV cache, `--max-running-requests 4`, chunked prefill 1024. The numbers come
from the kit protocol (`bench/sweep.py v1`, `results/G262-I0s800c1k/kit_sweep.json`). Decode runs each answer to its
natural end, with no output cap.

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8,192 | 189.4 tok/s | - | 1 | 270,336 fp8 | 1 |
| 16,384 | 173.6 tok/s | - | 1 | 270,336 fp8 | 1 |
| 32,768 | 147.2 tok/s | - | 1 | 270,336 fp8 | 1 |
| 65,536 | 111.6 tok/s | - | 1 | 270,336 fp8 | 1 |
| short prompt | - | 17.5 tok/s | C1 | 270,336 fp8 | 1 |
| short prompt | - | 21.5 tok/s aggregate (12.6 per stream) | C2 | 270,336 fp8 | 1 |
| short prompt | - | 20.7 tok/s aggregate (8.0 per stream) | C3 | 270,336 fp8 | 1 |
| short prompt | - | 24.5 tok/s aggregate (6.9 per stream) | C4 | 270,336 fp8 | 1 |

TTFT for a short prompt is 3.6-9.3 s (per-round maximum). Over the run, FTX counted a device expert-cache hit rate of
0.514, with the CPU taking 85.6 % of misses at 0.54 ms per CPU job and no combine spin timeouts.

Memory: the container cgroup peaked at 144.3 GB (122 GB anon, most of it the FTX copy of the experts at 114.2 GB). The
packed experts in USM host memory (42 layers x 2.729 GB = 114.6 GB) are allocated by the xe driver, which does not charge
them to the cgroup. Plan on about 260 GB of free host RAM, the `--memory` the run used. Nobody has measured a
system-wide MemAvailable drop for this config yet.

Long context (fixed after G262-I0s800c1k): the XPU decode indexer called `sgl_kernel.fp8_paged_mqa_logits` on the
index-K pool pages, which hold 64 fp8 keys followed by 64 fp32 scales (the layout the kpool compress kernels write and
`gather_index_k_scale_prefix_into` reads). The XPU kernel reads each token as 128 key bytes plus 4 scale bytes
interleaved, so decode picked its 2,048 history tokens close to at random once a prompt passed about 3k tokens. A
needle test (thinking off, temperature 0) failed from 5k tokens up and returned only the first digits of the code.
`_get_topk_paged_xpu` now repacks the pooled pages the batch references into the interleaved layout (static shapes, so
decode graphs still capture). A torch reference over the same pages now matches the kernel logits (correlation 1.000,
top-512 overlap 512/512 on all 11 indexer layers), and the needle passes at 7k, 18k, 36k and 68k tokens. Prefill was
not affected: its ragged path gathers keys and scales separately. Note that GLM-5.3's chat template has no
`enable_thinking` switch; a thinking-off request needs the prompt rendered with an empty `<think></think>`.

**Low-RAM mode (branch `b70-lowram`, not yet measured on the GPU):** `-e FTX_SRC=usm` makes the FTX CPU worker read
the GPU expert store's USM host copy in place instead of building its own 114 GB copy. The experts then exist once in
host RAM, about 145 GB peak in total, with outputs bit-identical to the default. Design, estimates and the GPU test plan:
[`LOWRAM.md`](LOWRAM.md). The default (`FTX_SRC=ckpt`) is the measured configuration.

Prefill is slow because the DPAS prefill MoE kernel is off for M < 256 (`GLM_MOE_PREFILL_MIN_M=256`) and prefill reads
the experts zero-copy over PCIe. Decode is bound by expert misses.

## How it works

- `exl3xpu` keeps every routed expert of the 42 MoE layers packed in USM host memory. 800 device slots act as an LRU
  expert cache that the kernels read zero-copy on a miss (`EXL3_MOE_SLOTS=800`, `EXL3_MOE_CACHE=1`, `EXL3_NVTIER=0`).
- FTX (`opt/ftx_tier.py`, `ftx/`) applies to decode MoE calls (M <= 4 rows). After the router runs, `ftx.split` (a SYCL
  kernel) splits the picks that miss the device cache between the GPU and an AVX2 CPU worker, using a cost model. CPU
  picks get id -1 and weight 0, and the MoE kernel skips them. The job is published to USM host memory. After the GPU
  experts finish, `ftx.combine` spins (bounded) on the worker's done sequence and adds the CPU partial. The CPU worker
  (`ft_tier_ext.cpp` / `ft_core.h`, the same as `kernels/cpu_avx2/`) keeps its own block-contiguous copy of the experts
  (about 2.7 GB per layer).
- `opt/glm_opt.py`, loaded through `zz_glm_opt.pth`, installs post-import hooks:
  - Triton mHC pre/post (`mhc_triton.py`).
  - The MoE prefill threshold.
  - The FTX install on `Exl3XpuMoEMethod`.
  - NextN/MTP key mapping and fixes.
  - A DSA top-k share fix.
  - Skipping the single-rank TP sampler sync.
  - Host-path PP hand-off (`pp_host.py`), which is unused at PP 1.
- `plugin/`: the GLM tier variant of the exl3xpu SGLang plugin, plus `nvtier.py` and `mla_unfuse.py`.
- `sglang-patches/`: replacements for six SGLang files (GLM5-Next model, DSA backend/indexer kpool, XPU FLA chunk kernel,
  DeepSeek-V2 arg overrides). Each one differs from the base image's copy.

## Layout and install paths

| source | installed at (where the campaign mounted it) |
|---|---|
| `opt/*.py` | `/opt/glmopt/` |
| `opt/zz_glm_opt.pth` | `site-packages/zz_glm_opt.pth` |
| `ftx/ft_xpu.sycl`, `ft_tier_ext.cpp`, `ft_core.h` + built `_ftx.so` | `/opt/glmopt/ftx/` |
| built `ft_tier_host.so` (FTX host extension) | `/opt/glmopt-build/ct/` |
| `kernels/moe-glm-ftx/` -> built `_moe_glm_ftx.so` | `/moe/_moe_n128.so` (`EXL3_MOE_LIB`) |
| `plugin/sglang_plugin.tier.py`, `nvtier.py`, `mla_unfuse.py` | `/opt/trellis-serve/xpu/exl3xpu/{sglang_plugin,nvtier,mla_unfuse}.py` |
| `sglang-patches/*.py` | `site-packages/sglang/srt/...` (map in `install.sh`) |
| `serve/` | `/opt/glm-b70/serve/` (entrypoint, model view, FTX CPU picker, self check) |

`install.sh` performs the whole install inside the base image. It needs only the CPU, building with the image's oneAPI
`icpx`. `kernels/moe-glm-ftx/build.sh` and `ftx/build_ftx.sh` use the exact flags that built the measured binaries.
Rebuilding them CPU-only in the base image on 2026-10-17 gave byte-identical files:

| binary | sha256 (campaign = rebuild) | built by |
|---|---|---|
| `_moe_glm_ftx.so` | `f7a219857eb9f86302325a898a53af29ccdb5a763154fb1af163e57b36f16115` | `kernels/moe-glm-ftx/build.sh` |
| `_ftx.so` | `3c4200bebb1503bf3b41fd3eb5216caef02c12b31350936fc92bd14435d75b48` | `ftx/build_ftx.sh` step 1 (no per-kernel split) |
| `ft_tier_host.so` | `ff2721979e4e435c4da824ffd5b3c5005e974f89d68f1f5473852c99fcb33862` | `ftx/build_ftx.sh` step 2 (`ftx_tier._build_host`) |

On branch `b70-lowram` the FTX host sources gained the `FTX_SRC=usm` path, so `ft_tier_host.so` builds to a new hash.
`_moe_glm_ftx.so` and `_ftx.so` are unchanged.

The host extension is compiled with `-march=znver3` (AVX2/FMA/F16C), as measured. It runs on Zen 3 or newer. Other x86
CPUs with AVX2 may hit an illegal instruction.

## Launch (no host bind mounts except the weights)

```bash
IMG=ghcr.io/sybil-solutions/glm53-flash-b70-ftx@sha256:<digest>      # built from local-ai-images/glm53-flash-b70-ftx
hf download turboderp/GLM-5.3-Flash-exl3 --revision 332ab457b709b7ba30dd9a448be5de03b80a7ac9 \
  --local-dir /data/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw
R=$(readlink -f /dev/dri/by-path/pci-0000:83:00.0-render)            # the B70's render node
docker run -d --name glm53-b70 --device "$R:$R:rwm" -e ZE_AFFINITY_MASK=0 \
  --memory 260g --memory-swap 260g --shm-size 8g --ulimit memlock=-1:-1 --cpuset-cpus 4-23,28-47 \
  -p 127.0.0.1:30260:30260 \
  -v /data/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw:/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw:ro "$IMG"
```

With no arguments, the entrypoint runs the measured command:

```
python3 -m sglang.launch_server --model-path /models/turboderp-GLM-5.3-Flash-exl3-3.05bpw --quantization exl3
  --trust-remote-code --device xpu --host 0.0.0.0 --port 30260 --served-model-name glm53-flash --pp-size 1
  --reasoning-parser glm45 --tool-call-parser glm47 --disable-shared-experts-fusion --kv-cache-dtype fp8_e4m3
  --dtype bfloat16 --context-length 262144 --mem-fraction-static 0.88 --chunked-prefill-size 1024
  --max-total-tokens 270336 --max-running-requests 4 --max-mamba-cache-size 24 --page-size 64
  --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 2 4
```

The image sets the run's environment as defaults. The base image's own ENV is inherited unchanged, as in the campaign.

```
EXL3_MODEL_PATH=/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw EXL3_MOE_LIB=/moe/_moe_n128.so EXL3_MOE_GU=8 EXL3_MOE_DN=4
EXL3_MOE_SLOTS=800 EXL3_MOE_CACHE=1 EXL3_TIER_H=4096 EXL3_TIER_I=2048 EXL3_TIER_E=288 EXL3_TIER_DENSE=3 EXL3_TIER_L=42
EXL3_NVTIER=0 EXL3_NVTIER_PRIOR= EXL3_QSA_XPU=0 EXL3_NGRAM_HOST=0 EXL3_HC_XPU=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0 EXL3_TORCH_MEM_FRACTION=0.95
TORCHINDUCTOR_COMPILE_THREADS=1 HF_HUB_OFFLINE=1 NEOReadDebugKeys=1 EnableSharedSystemUsmSupport=1
EnableRecoverablePageFaults=1 GLM_MOE_PREFILL_MIN_M=256 EXL3_MOE_STAGE_MIN_M=128 FTX=1 FTX_CPUS=auto
```

`FTX_CPUS=auto` resolves to the first hardware thread of every core in the container's cpuset. With
`--cpuset-cpus 4-23,28-47` on the EPYC 7443P that is `4-23`, the measured value. With any other argv, the entrypoint
execs it as given, and an argument equal to the checkpoint path is replaced by the model view (see below).
`docker run IMG check` is the CPU-only self check.

## Differences from the campaign container (not yet measured on a GPU)

- Checkpoint index. The HF index at 332ab457 does not list the 22 DSA kpool tensors in `kpool_aux.safetensors`. On the
  campaign host they were appended to the index as `model-00016-of-00016.safetensors`, a byte-identical copy of
  `kpool_aux.safetensors`. The image does not write to `/models`. Instead, `serve/model_view.py` builds a symlink view
  under `/tmp/glm53-model` whose index maps those tensors to `kpool_aux.safetensors`, then points `--model-path` and
  `EXL3_MODEL_PATH` at it. The tensors loaded are the same, but the view has not yet been started on a B70.
- The Triton cache was a host directory (`/mnt/nvx/b7g-triton:/tcache`). In the image it lives in the container, so the
  first start compiles the Triton kernels.
- Left out because this config does not use them: `opt/b70x/` (second-B70 expert server), the NVMe expert store
  (`EXL3_NVTIER=1`), the probe/profiling mounts, and `build/ct2` (another host-extension build).
- `kern/_moe_glm.so` (the pre-FTX MoE library from 2026-10-15 15:23, sha256 `79fb4d1c...0c6402`) is not used by this
  config. Its source is `kernels/moe-glm-ftx/exl3_moe.n128.pre-ftx.sycl`, which is `exl3_moe.n128.sycl` without the
  three `ftx:` hunks. Copying it to the name `exl3_moe.n128.sycl` and building it with the campaign's `kern/build.sh`
  flags (the `build.sh` flags without the FTX reordering) reproduces it byte for byte. The source file name is embedded
  in the binary, so building it under its `.pre-ftx` name gives a different hash.
