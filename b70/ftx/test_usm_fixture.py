"""test_usm_fixture.py OUT_DIR [E] [SEED]: CPU-only fixture for test_usm.cpp. Writes E random GLM-5.3-Flash-shaped experts
(H 4096, I 2048, K 3) twice:
  native.bin  checkpoint layout, per expert: gate, up, down trellis (int16 [k/16][n/16][48]), then fp16
              suh_g[H] svh_g[I] suh_u[H] svh_u[I] suh_d[I] svh_d[H]
  blob.bin    the exl3xpu expert blob of each expert, packed by exl3xpu.moe_offload.pack_expert (the function that
              fills the GPU expert store's USM host copy); a verbatim copy of it is used when exl3xpu is not installed
Random int16 trellis words are valid 3-bit trellis streams, so every expert decodes to finite weights."""
import json, os, sys

import torch

try:
    from exl3xpu.moe_offload import pack_expert, planar4   # the image's copy (authoritative)
    SRC = "exl3xpu.moe_offload"
except Exception:   # verbatim from qwen-serve qwen3.8-flash-next/b70-offload/plugin/exl3xpu/moe_offload.py
    SRC = "embedded copy"

    def planar4(tr: torch.Tensor, K: int) -> torch.Tensor:
        assert K in (3, 5, 7), "PLANAR4 packing implemented for odd K (D = K words per 32-value period)"
        r, n, _ = tr.shape
        w = tr.contiguous().view(torch.int32).view(r, n // 4, 4, 8, K)
        return w.permute(0, 1, 4, 2, 3).contiguous()

    def pack_expert(gate, up, down, K, out=None):
        b = lambda z: z.contiguous().view(torch.uint8).flatten()     # noqa: E731
        gu = torch.cat([gate["trellis"], up["trellis"]], dim=1)
        parts = [b(planar4(gu, K)), b(planar4(down["trellis"], K)), b(gate["suh"].half()), b(up["suh"].half()),
                 b(gate["svh"].half()), b(up["svh"].half()), b(down["suh"].half()), b(down["svh"].half())]
        if out is None:
            return torch.cat(parts)
        o = 0
        for p in parts:
            out[o:o + p.numel()].copy_(p)
            o += p.numel()
        assert o == out.numel(), (o, out.numel())
        return out


def main():
    out = sys.argv[1]
    E = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    H, I, K = 4096, 2048, 3
    g = torch.Generator().manual_seed(seed)
    os.makedirs(out, exist_ok=True)
    blob_bytes = None
    with open(os.path.join(out, "native.bin"), "wb") as fn, open(os.path.join(out, "blob.bin"), "wb") as fb:
        for e in range(E):
            mats = {}
            for name, k, n in (("gate", H, I), ("up", H, I), ("down", I, H)):
                tr = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * K), dtype=torch.int32, generator=g).to(torch.int16)
                # EXL3 suh/svh: per-channel sign flips times a scale; any finite fp16 works for the test
                suh = (torch.randn(k, generator=g) * 0.4 + torch.sign(torch.randn(k, generator=g))).half()
                svh = (torch.randn(n, generator=g) * 0.02 + 0.05 * torch.sign(torch.randn(n, generator=g))).half()
                mats[name] = {"trellis": tr, "suh": suh, "svh": svh}
            for name in ("gate", "up", "down"):
                fn.write(mats[name]["trellis"].contiguous().numpy().tobytes())
            for name in ("gate", "up", "down"):
                fn.write(mats[name]["suh"].numpy().tobytes())
                fn.write(mats[name]["svh"].numpy().tobytes())
            blob = pack_expert(mats["gate"], mats["up"], mats["down"], K)
            blob_bytes = blob.numel()
            fb.write(blob.numpy().tobytes())
    json.dump({"E": E, "H": H, "I": I, "K": K, "blob": blob_bytes, "seed": seed, "pack_expert": SRC},
              open(os.path.join(out, "meta.json"), "w"))
    print(f"fixture: {E} experts, blob {blob_bytes} B, pack_expert from {SRC} -> {out}")


if __name__ == "__main__":
    main()
