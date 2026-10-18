// FreeToken CPU tier: host side (torch CPU extension). Owns the ft_mul1 pool and a worker thread that
// serves one MoE job at a time through pinned, GPU-mapped buffers written by the GPU split kernel
// (ft_tier_cu.cu):
//   ctrl[0] job seq (GPU writes last)   ctrl[1] done seq (worker writes last)
//   ctrl[2] layer   ctrl[3] ntok   ctrl[4] npicks
//   hx    [ntok, H] fp16 expert input      picks [npicks][3] = (token, expert, weight f32 bits)
//   hout  [ntok, H] fp32 weighted routed partial of the CPU picks (the GPU adds it into the MoE output)
// Weights: tier_add_layer (FTX_SRC=ckpt, default) builds a block-contiguous CPU copy of the checkpoint trellis
// (swz); tier_add_layer_usm (FTX_SRC=usm) reads the GPU expert store's USM host blobs in place (PLANAR4,
// ft_core.h layer_from_blobs), so the experts exist once in host RAM.
#include <torch/extension.h>
#include "ft_core.h"

namespace ft {

struct Stats { std::atomic<long> jobs{0}, experts{0}, tokens{0}, empty{0}; std::atomic<long long> busy_ns{0}, lat_ns{0}; };

struct Tier
{
    Pool pool;
    std::vector<Layer> layers;
    std::vector<torch::Tensor> keep;
    volatile int64_t* ctrl = nullptr;
    const uint16_t* hx = nullptr;
    const int32_t* picks = nullptr;
    float* hout = nullptr;
    int mode = -1, H = 4096, I = 2048, swz = 1;
    int usm = 0;        // layers point into the GPU store's USM blobs (planar)
    int usm_kern = 0;   // usm: 0 auto (row bands for I16 calls, blocks otherwise: bit-identical), 1 blocks, 2 bands
    size_t swz_bytes = 0;
    std::thread th;
    std::atomic<bool> quit{false}, running{false};
    Stats st;
    std::vector<float> xf;
};

Tier* T = nullptr;

void tier_init(int64_t threads, std::vector<int64_t> cpus, torch::Tensor ctrl, torch::Tensor hx, torch::Tensor picks, torch::Tensor hout,
               int64_t mode, int64_t H, int64_t I, int64_t nlayers, int64_t swz)
{
    TORCH_CHECK(T == nullptr, "tier already initialised");
    init_perm(); init_tables();
    T = new Tier();
    std::vector<int> c(cpus.begin(), cpus.end());
    cpu_set_t saved; sched_getaffinity(0, sizeof(saved), &saved);
    T->pool.start(int(threads), c);          // pins the calling thread to c[0]; the worker thread takes that role
    sched_setaffinity(0, sizeof(saved), &saved);   // give the Python thread its affinity back
    T->ctrl = reinterpret_cast<volatile int64_t*>(ctrl.data_ptr<int64_t>());
    T->hx = reinterpret_cast<const uint16_t*>(hx.data_ptr());
    T->picks = picks.data_ptr<int32_t>();
    T->hout = hout.data_ptr<float>();
    T->keep = { ctrl, hx, picks, hout };
    T->mode = int(mode); T->H = int(H); T->I = int(I); T->swz = int(swz);
    T->layers.resize(nlayers);
}

// CPU-side block-contiguous copy ([n/128][k/16][8 tiles]) of experts [0, E) of L, repointing their Mats to it: every
// 128-output work unit streams one contiguous run (C004: 0.114-0.120 vs 0.165-0.172 ms/expert on the native order).
// Plain (unpinned) huge-page memory; the pinned native home copy stays the GPU's. Returns the bytes allocated.
size_t swizzle_copy(Layer& L, int E, uint8_t** out_buf = nullptr)
{
    const int H = L.H, I = L.I;
    const size_t mb = size_t(H / 16) * (I / 16) * TILE_BYTES;           // bytes per matrix (all three equal)
    uint8_t* buf = static_cast<uint8_t*>(big_alloc(mb * 3 * E));
    if (out_buf) *out_buf = buf;
    std::vector<std::thread> ts;
    const int nt = 16;
    for (int w = 0; w < nt; ++w)
        ts.emplace_back([&, w] {
            for (int e = w; e < E; e += nt)
                for (int k = 0; k < 3; ++k)
                {
                    Mat& m = k == 0 ? L.ex[e].g : k == 1 ? L.ex[e].u : L.ex[e].d;
                    uint8_t* dst = buf + (size_t(e) * 3 + k) * mb;
                    const int tk_n = m.k / 16, tn = m.n / 16;
                    for (int b = 0; b < tn / 8; ++b)
                        for (int tk = 0; tk < tk_n; ++tk)
                            std::memcpy(dst + (size_t(b) * tk_n + tk) * 768, m.tr + (size_t(tk) * tn + b * 8) * TILE_BYTES, 768);
                    m.tr = dst; m.swz = 1;
                }
        });
    for (auto& t : ts) t.join();
    return mb * 3 * E;
}

// ptrs: int64 [E, 3] host addresses of gate/up/down trellis (native layout); scales fp16 CPU [E, k] / [E, n]
void tier_add_layer(int64_t li, torch::Tensor ptrs, torch::Tensor suh_g, torch::Tensor svh_g, torch::Tensor suh_u, torch::Tensor svh_u,
                    torch::Tensor suh_d, torch::Tensor svh_d)
{
    TORCH_CHECK(T, "init first");
    const int E = int(ptrs.size(0)), H = T->H, I = T->I;
    for (auto& t : { suh_g, svh_g, suh_u, svh_u, suh_d, svh_d })
        TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == at::kHalf && t.is_contiguous() && t.size(0) == E, "scales: CPU fp16 contiguous [E, .]");
    TORCH_CHECK(suh_g.size(1) == H && svh_g.size(1) == I && suh_d.size(1) == I && svh_d.size(1) == H, "scale shapes");
    auto p = ptrs.accessor<int64_t, 2>();
    Layer& L = T->layers[li];
    L.H = H; L.I = I; L.ex.resize(E);
    auto u16 = [](const torch::Tensor& t, int e) { return reinterpret_cast<const uint16_t*>(t.data_ptr()) + size_t(e) * t.size(1); };
    for (int e = 0; e < E; ++e)
    {
        L.ex[e].g = { reinterpret_cast<const uint8_t*>(p[e][0]), u16(suh_g, e), u16(svh_g, e), H, I };
        L.ex[e].u = { reinterpret_cast<const uint8_t*>(p[e][1]), u16(suh_u, e), u16(svh_u, e), H, I };
        L.ex[e].d = { reinterpret_cast<const uint8_t*>(p[e][2]), u16(suh_d, e), u16(svh_d, e), I, H };
    }
    for (auto& t : { ptrs, suh_g, svh_g, suh_u, svh_u, suh_d, svh_d }) T->keep.push_back(t);
    if (T->swz) T->swz_bytes += swizzle_copy(L, E);
}

// FTX_SRC=usm: layer li reads the GPU expert store's USM host copy in place. base = address of expert 0's blob,
// experts `blob` bytes apart (ExpertStore.host[key], uint8 [E, blob]). No copy; the caller keeps the store alive.
void tier_add_layer_usm(int64_t li, int64_t base, int64_t E, int64_t blob, int64_t kern)
{
    TORCH_CHECK(T, "init first");
    TORCH_CHECK(!T->swz, "tier_add_layer_usm: tier_init with swz = 0");
    TORCH_CHECK(layer_from_blobs(T->layers[li], reinterpret_cast<const uint8_t*>(base), int(E), size_t(blob), T->H, T->I),
                "tier_add_layer_usm: blob ", blob, " B is not an exl3xpu K=3 blob for H=", T->H, " I=", T->I,
                " (want ", blob_bytes_k3(T->H, T->I), ")");
    T->usm = 1; T->usm_kern = int(kern);
}

// mode is resolved (0 AFFINE, 1 EXACT, 2 I16). ckpt layers: moe_forward (today). usm layers: row bands for I16 calls
// (contiguous 12-24 KB runs of the blob rows; bit-identical to moe_forward), 128-output blocks otherwise.
void forward_layer(const Layer& L, const float* x, int ntok, const std::vector<std::vector<std::pair<int, float>>>& route, float* out, int mode)
{
    const bool band = T->usm && (T->usm_kern == 2 || (T->usm_kern == 0 && mode == 2));
    if (band) moe_forward_band(T->pool, L, x, ntok, route, out, mode);
    else moe_forward(T->pool, L, x, ntok, route, out, mode);
}

void run_job(int li, int ntok, int np)
{
    const int H = T->H;
    T->xf.resize(size_t(std::max(ntok, 1)) * H);
    for (size_t i = 0; i < size_t(ntok) * H; ++i) T->xf[i] = h2f(T->hx[i]);
    std::vector<std::vector<std::pair<int, float>>> route(ntok);
    for (int k = 0; k < np; ++k)
    {
        const int t = T->picks[k * 3], e = T->picks[k * 3 + 1]; float w; std::memcpy(&w, &T->picks[k * 3 + 2], 4);
        route[t].push_back({ e, w });
    }
    if (np == 0) { std::memset(T->hout, 0, size_t(ntok) * H * 4); T->st.empty++; return; }
    int maxm = 0;
    { std::vector<int> cnt(T->layers[li].ex.size(), 0); for (auto& r : route) for (auto& pr : r) maxm = std::max(maxm, ++cnt[pr.first]); }
    const int mode = T->mode >= 0 ? T->mode : (maxm <= 2 ? 2 : 0);
    forward_layer(T->layers[li], T->xf.data(), ntok, route, T->hout, mode);
    T->st.experts += np; T->st.tokens += ntok;
}

void worker_main(int cpu0)
{
    Pool::pin(cpu0);
    int64_t last = T->ctrl[0];
    long idle = 0;
    while (!T->quit.load(std::memory_order_relaxed))
    {
        const int64_t s = __atomic_load_n(const_cast<int64_t*>(T->ctrl), __ATOMIC_ACQUIRE);
        if (s == last)
        {
            if (++idle < 400000) _mm_pause(); else std::this_thread::sleep_for(std::chrono::microseconds(20));
            continue;
        }
        idle = 0;
        const double t0 = now();
        run_job(int(T->ctrl[2]), int(T->ctrl[3]), int(T->ctrl[4]));
        T->st.busy_ns += (long long) ((now() - t0) * 1e9);
        T->st.jobs++;
        last = s;
        __atomic_store_n(const_cast<int64_t*>(T->ctrl + 1), s, __ATOMIC_RELEASE);
    }
}

void tier_start(int64_t cpu0)
{
    TORCH_CHECK(T && !T->running, "init first / already running");
    T->running = true;
    T->th = std::thread(worker_main, int(cpu0));
}

void tier_stop()
{
    if (!T) return;
    T->quit = true;
    if (T->th.joinable()) T->th.join();
    T->pool.stop();
}

// Synchronous forward for tests (worker must not be running): x fp32 [m, H], sel int32 [m, k], w fp32 [m, k]
torch::Tensor tier_forward(int64_t li, torch::Tensor x, torch::Tensor sel, torch::Tensor w, int64_t mode)
{
    TORCH_CHECK(T && !T->running, "tier_forward: worker running");
    const int m = int(x.size(0)), k = int(sel.size(1));
    auto out = torch::zeros({ m, T->H }, torch::kFloat);
    std::vector<std::vector<std::pair<int, float>>> route(m);
    auto sa = sel.accessor<int32_t, 2>(); auto wa = w.accessor<float, 2>();
    for (int t = 0; t < m; ++t) for (int j = 0; j < k; ++j) if (sa[t][j] >= 0) route[t].push_back({ sa[t][j], wa[t][j] });
    int maxm = 0;
    { std::vector<int> cnt(T->layers[li].ex.size(), 0); for (auto& r : route) for (auto& pr : r) maxm = std::max(maxm, ++cnt[pr.first]); }
    const int md = mode >= 0 ? int(mode) : (maxm <= 2 ? 2 : 0);
    forward_layer(T->layers[li], x.data_ptr<float>(), m, route, out.data_ptr<float>(), md);
    return out;
}

// Check layer li (as added: usm or ckpt) against today's FTX path on the same experts: a swizzled copy of experts
// [0, nexp) built from the checkpoint tensors (ptrs int64 [E, 3] native trellis addresses, fp16 scales [E, .], as for
// tier_add_layer), run through moe_forward. Random fp16-valued inputs, 1-4 tokens, 8 distinct experts per token from
// [0, nexp) (shared experts give m >= 2), every mode (auto, AFFINE, EXACT, I16). Worker must not be running.
// Returns { outputs compared, outputs not bit-identical, max |diff|, max |ref|, ms/expert today (swz copy, M=1, 8 picks),
//           ms/expert live layer (same calls) }.
std::vector<double> tier_validate(int64_t li, torch::Tensor ptrs, torch::Tensor suh_g, torch::Tensor svh_g, torch::Tensor suh_u,
                                  torch::Tensor svh_u, torch::Tensor suh_d, torch::Tensor svh_d, int64_t nexp, int64_t trials, int64_t seed)
{
    TORCH_CHECK(T && !T->running, "tier_validate: init first, before tier_start");
    TORCH_CHECK(nexp >= 8 && nexp <= ptrs.size(0), "tier_validate: 8 <= nexp <= E");
    const int H = T->H, I = T->I, NE = int(nexp);
    auto p = ptrs.accessor<int64_t, 2>();
    auto u16 = [](const torch::Tensor& t, int e) { return reinterpret_cast<const uint16_t*>(t.data_ptr()) + size_t(e) * t.size(1); };
    Layer R; R.H = H; R.I = I; R.ex.resize(NE);
    for (int e = 0; e < NE; ++e)
    {
        R.ex[e].g = { reinterpret_cast<const uint8_t*>(p[e][0]), u16(suh_g, e), u16(svh_g, e), H, I };
        R.ex[e].u = { reinterpret_cast<const uint8_t*>(p[e][1]), u16(suh_u, e), u16(svh_u, e), H, I };
        R.ex[e].d = { reinterpret_cast<const uint8_t*>(p[e][2]), u16(suh_d, e), u16(svh_d, e), I, H };
    }
    uint8_t* buf = nullptr;
    swizzle_copy(R, NE, &buf);
    const Layer& Lv = T->layers[li];
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::normal_distribution<float> nd(0.f, 0.5f);
    std::uniform_real_distribution<float> ud(0.05f, 1.f);
    double n = 0, bad = 0, maxd = 0, maxr = 0;
    for (int it = 0; it < trials; ++it)
    {
        const int ntok = 1 + it % 4;
        std::vector<float> x(size_t(ntok) * H);
        for (auto& v : x) v = r16(nd(rng));
        std::vector<std::vector<std::pair<int, float>>> route(ntok);
        for (int t = 0; t < ntok; ++t)
        {
            std::vector<int> ids(NE); for (int e = 0; e < NE; ++e) ids[e] = e;
            std::shuffle(ids.begin(), ids.end(), rng);
            for (int k = 0; k < 8; ++k) route[t].push_back({ ids[k], ud(rng) });
        }
        int maxm = 0;
        { std::vector<int> cnt(NE, 0); for (auto& r : route) for (auto& pr : r) maxm = std::max(maxm, ++cnt[pr.first]); }
        for (int mode : { -1, 0, 1, 2 })
        {
            const int md = mode >= 0 ? mode : (maxm <= 2 ? 2 : 0);
            std::vector<float> a(size_t(ntok) * H), b(size_t(ntok) * H);
            moe_forward(T->pool, R, x.data(), ntok, route, a.data(), md);
            forward_layer(Lv, x.data(), ntok, route, b.data(), md);
            for (size_t i = 0; i < a.size(); ++i)
            {
                n += 1; maxr = std::max(maxr, double(std::fabs(a[i])));
                if (std::memcmp(&a[i], &b[i], 4)) { bad += 1; maxd = std::max(maxd, double(std::fabs(a[i] - b[i]))); }
            }
        }
    }
    // speed: M = 1, 8 picks, auto mode (I16), as a C1 decode CPU job
    double tr = 0, tl = 0; const int reps = 40;
    std::vector<float> x(H), y(H);
    for (int it = 0; it < 2 * reps; ++it)
    {
        for (auto& v : x) v = r16(nd(rng));
        std::vector<int> ids(NE); for (int e = 0; e < NE; ++e) ids[e] = e;
        std::shuffle(ids.begin(), ids.end(), rng);
        std::vector<std::vector<std::pair<int, float>>> route(1);
        for (int k = 0; k < 8; ++k) route[0].push_back({ ids[k], ud(rng) });
        const double t0 = now();
        if (it % 2) forward_layer(Lv, x.data(), 1, route, y.data(), 2); else moe_forward(T->pool, R, x.data(), 1, route, y.data(), 2);
        (it % 2 ? tl : tr) += now() - t0;
    }
    free(buf);
    return { n, bad, maxd, maxr, tr * 1e3 / (reps * 8), tl * 1e3 / (reps * 8) };
}

std::vector<int64_t> tier_stats()
{
    if (!T) return {};
    return { T->st.jobs.load(), T->st.experts.load(), T->st.tokens.load(), T->st.empty.load(), (int64_t) T->st.busy_ns.load(), (int64_t) T->swz_bytes };
}

}  // namespace ft

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tier_init", &ft::tier_init);
    m.def("tier_add_layer", &ft::tier_add_layer);
    m.def("tier_add_layer_usm", &ft::tier_add_layer_usm);
    m.def("tier_validate", &ft::tier_validate);
    m.def("tier_start", &ft::tier_start);
    m.def("tier_stop", &ft::tier_stop);
    m.def("tier_forward", &ft::tier_forward);
    m.def("tier_stats", &ft::tier_stats);
}
