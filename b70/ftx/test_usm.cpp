// test_usm.cpp: CPU-only check of the FTX_SRC=usm CPU tier path (ft_core.h planar Mats on exl3xpu blobs) against
// today's FTX path (swizzled checkpoint copy), on the fixture written by test_usm_fixture.py.
//   1. deplanar4 (AVX2) == deplanar4_ref (scalar) on random bytes
//   2. layout: every tile of every blob matrix, deplanarized, equals the checkpoint tile; blob scales == checkpoint scales
//   3. MoE forward, bit for bit: usm blocks and usm auto (row bands for I16) vs today's swz blocks, every mode, 1-4 tokens
//      (also usm bands for AFFINE / EXACT, reported, not required: fp32 re-association)
//   4. speed (meaningful on the target x86 host only): ms per expert, M = 1, 8 picks, I16, swz vs usm block vs usm band
// Build + run: test_usm.sh (Linux g++ / macOS clang x86_64 under Rosetta).
#ifdef __APPLE__
// macOS has no CPU affinity API or THP: the pool runs unpinned (correctness only)
#include <pthread.h>
#include <sys/mman.h>
typedef struct { unsigned long bits[16]; } cpu_set_t;
#define CPU_ZERO(s) ((void)(s))
#define CPU_SET(c, s) ((void)(c), (void)(s))
static inline int pthread_setaffinity_np(pthread_t, size_t, const cpu_set_t*) { return 0; }
#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 0
#endif
#endif
#include "ft_core.h"

#include <cinttypes>

namespace {

std::vector<uint8_t> slurp(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

}  // namespace

int main(int argc, char** argv)
{
    const std::string dir = argc > 1 ? argv[1] : ".";
    const int threads = argc > 2 ? atoi(argv[2]) : 4;
    const int trials = argc > 3 ? atoi(argv[3]) : 24;
    init_perm(); init_tables();
    const int H = 4096, I = 2048;
    const size_t mbytes = size_t(H / 16) * (I / 16) * TILE_BYTES;     // 3,145,728 per matrix
    const size_t scb = size_t(3 * H + 3 * I) * 2;
    const size_t per_native = 3 * mbytes + scb;
    const size_t blob = blob_bytes_k3(H, I);
    std::vector<uint8_t> nat = slurp(dir + "/native.bin"), blb = slurp(dir + "/blob.bin");
    const int E = int(nat.size() / per_native);
    CHECK(nat.size() == per_native * E && blb.size() == blob * E, "fixture sizes %zu %zu (E=%d, blob=%zu)", nat.size(), blb.size(), E, blob);
    CHECK(blob == 9474048, "blob_bytes_k3(4096, 2048) = %zu, want 9,474,048 (packed NVMe store record)", blob);
    if (fails) return 1;
    printf("fixture: %d experts, blob %zu B, %d threads\n", E, blob, threads);

    // 1. deplanar4
    {
        std::mt19937 rng(7);
        alignas(32) uint8_t src[384], a[384], b[384];
        int bad = 0;
        for (int it = 0; it < 2000; ++it)
        {
            for (auto& v : src) v = uint8_t(rng());
            deplanar4(src, a); deplanar4_ref(src, b);
            bad += std::memcmp(a, b, 384) != 0;
        }
        CHECK(bad == 0, "deplanar4 != deplanar4_ref in %d of 2000 groups", bad);
        printf("1. deplanar4 vs scalar reference: %s\n", bad ? "MISMATCH" : "identical (2000 random groups)");
    }

    // layers: N native (checkpoint, in place), A today's (swizzled copy), P usm (blob, in place)
    Layer N, A, P;
    N.H = A.H = H; N.I = A.I = I; N.ex.resize(E);
    for (int e = 0; e < E; ++e)
    {
        const uint8_t* b = nat.data() + per_native * e;
        const uint16_t* sc = reinterpret_cast<const uint16_t*>(b + 3 * mbytes);
        // native.bin scale order: suh_g svh_g suh_u svh_u suh_d svh_d
        const uint16_t *suh_g = sc, *svh_g = sc + H, *suh_u = sc + H + I, *svh_u = sc + 2 * H + I, *suh_d = sc + 2 * H + 2 * I, *svh_d = sc + 2 * H + 3 * I;
        N.ex[e].g = { b, suh_g, svh_g, H, I };
        N.ex[e].u = { b + mbytes, suh_u, svh_u, H, I };
        N.ex[e].d = { b + 2 * mbytes, suh_d, svh_d, I, H };
    }
    std::vector<uint8_t> swz(3 * mbytes * size_t(E));
    A = N;
    for (int e = 0; e < E; ++e)   // the permutation of ft_tier_ext.cpp swizzle_copy (= ft_core swizzle_layer)
        for (int k = 0; k < 3; ++k)
        {
            Mat& m = k == 0 ? A.ex[e].g : k == 1 ? A.ex[e].u : A.ex[e].d;
            uint8_t* dst = swz.data() + (size_t(e) * 3 + k) * mbytes;
            const int tk_n = m.k / 16, tn = m.n / 16;
            for (int b = 0; b < tn / 8; ++b)
                for (int tk = 0; tk < tk_n; ++tk)
                    std::memcpy(dst + (size_t(b) * tk_n + tk) * 768, m.tr + (size_t(tk) * tn + b * 8) * TILE_BYTES, 768);
            m.tr = dst; m.swz = 1;
        }
    CHECK(layer_from_blobs(P, blb.data(), E, blob, H, I), "layer_from_blobs refused the fixture geometry");
    CHECK(!layer_from_blobs(N, blb.data(), E, blob + 64, H, I), "layer_from_blobs accepted a wrong blob size");

    // 2. layout
    {
        long bad_tiles = 0, bad_sc = 0, tiles = 0;
        alignas(32) uint8_t t4[384];
        for (int e = 0; e < E; ++e)
            for (int k = 0; k < 3; ++k)
            {
                const Mat& n = k == 0 ? N.ex[e].g : k == 1 ? N.ex[e].u : N.ex[e].d;
                const Mat& p = k == 0 ? P.ex[e].g : k == 1 ? P.ex[e].u : P.ex[e].d;
                const int tk_n = n.k / 16, tn = n.n / 16;
                for (int tk = 0; tk < tk_n; ++tk)
                    for (int t = 0; t < tn; t += 4)
                    {
                        deplanar4(p.tr + tk * p.rs + size_t(t) * TILE_BYTES, t4);
                        bad_tiles += std::memcmp(t4, n.tr + (size_t(tk) * tn + t) * TILE_BYTES, 384) != 0;
                        tiles += 4;
                    }
                bad_sc += std::memcmp(n.suh, p.suh, size_t(n.k) * 2) != 0;
                bad_sc += std::memcmp(n.svh, p.svh, size_t(n.n) * 2) != 0;
            }
        CHECK(bad_tiles == 0 && bad_sc == 0, "layout: %ld tile groups, %ld scale vectors differ", bad_tiles, bad_sc);
        printf("2. blob layout vs checkpoint: %ld tiles, %s\n", tiles, bad_tiles || bad_sc ? "MISMATCH" : "every tile and scale vector identical");
    }
    if (fails) return 1;

    // 3. forward, bit for bit
    Pool pool;
    std::vector<int> cpus; for (int c = 0; c < threads; ++c) cpus.push_back(c);
    pool.start(threads, cpus);
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 0.5f);
    std::uniform_real_distribution<float> ud(0.05f, 1.f);
    const char* mname[3] = { "AFFINE", "EXACT", "I16" };
    long n_out[3] = { 0, 0, 0 }, bad_blk[3] = { 0, 0, 0 }, bad_auto[3] = { 0, 0, 0 }, bad_band[3] = { 0, 0, 0 }, bad_nat[3] = { 0, 0, 0 };
    double maxd_band[3] = { 0, 0, 0 }, maxref = 0;
    int nonfinite = 0, maxm_seen = 0;
    for (int it = 0; it < trials; ++it)
    {
        const int ntok = 1 + it % 4;
        std::vector<float> x(size_t(ntok) * H);
        for (auto& v : x) v = r16(nd(rng));
        std::vector<std::vector<std::pair<int, float>>> route(ntok);
        for (int t = 0; t < ntok; ++t)
        {
            std::vector<int> ids(E); for (int e = 0; e < E; ++e) ids[e] = e;
            std::shuffle(ids.begin(), ids.end(), rng);
            for (int k = 0; k < 8; ++k) route[t].push_back({ ids[k], ud(rng) });
        }
        int maxm = 0;
        { std::vector<int> cnt(E, 0); for (auto& r : route) for (auto& pr : r) maxm = std::max(maxm, ++cnt[pr.first]); }
        maxm_seen = std::max(maxm_seen, maxm);
        for (int md = 0; md < 3; ++md)
        {
            const size_t no = size_t(ntok) * H;
            std::vector<float> a(no), nn(no), b(no), c(no), d(no);
            moe_forward(pool, A, x.data(), ntok, route, a.data(), md);           // today
            moe_forward(pool, N, x.data(), ntok, route, nn.data(), md);          // checkpoint order, in place
            moe_forward(pool, P, x.data(), ntok, route, b.data(), md);           // usm, blocks
            moe_forward_band(pool, P, x.data(), ntok, route, c.data(), md);      // usm, bands
            // usm auto (FTX_USM_KERN=0): bands for I16, blocks otherwise
            if (md == 2) d = c; else d = b;
            for (size_t i = 0; i < no; ++i)
            {
                nonfinite += !std::isfinite(a[i]);
                maxref = std::max(maxref, double(std::fabs(a[i])));
                bad_nat[md] += std::memcmp(&a[i], &nn[i], 4) != 0;
                bad_blk[md] += std::memcmp(&a[i], &b[i], 4) != 0;
                bad_auto[md] += std::memcmp(&a[i], &d[i], 4) != 0;
                if (std::memcmp(&a[i], &c[i], 4)) { ++bad_band[md]; maxd_band[md] = std::max(maxd_band[md], double(std::fabs(a[i] - c[i]))); }
            }
            n_out[md] += long(no);
        }
    }
    printf("3. forward vs today's swz path (%d trials, 1-4 tokens, max tokens per expert %d, max |y| %.3g, non-finite %d):\n",
           trials, maxm_seen, maxref, nonfinite);
    for (int md = 0; md < 3; ++md)
        printf("   %-6s outputs %7ld | native-in-place diff %ld | usm block diff %ld | usm auto diff %ld | usm band diff %ld (max |d| %.3g)\n",
               mname[md], n_out[md], bad_nat[md], bad_blk[md], bad_auto[md], bad_band[md], maxd_band[md]);
    CHECK(nonfinite == 0, "reference produced %d non-finite outputs (fixture scales)", nonfinite);
    for (int md = 0; md < 3; ++md)
    {
        CHECK(bad_nat[md] == 0, "%s: checkpoint-order blocks differ from swz blocks", mname[md]);
        CHECK(bad_blk[md] == 0, "%s: usm blocks differ from today's path", mname[md]);
        CHECK(bad_auto[md] == 0, "%s: usm auto differs from today's path", mname[md]);
    }

    // 4. speed (C1 decode CPU job: 1 token, 8 picks, I16)
    {
        const int reps = argc > 4 ? atoi(argv[4]) : 30;
        double t[3] = { 0, 0, 0 };
        std::vector<float> x(H), y(H);
        for (int it = 0; it < reps * 3; ++it)
        {
            for (auto& v : x) v = r16(nd(rng));
            std::vector<int> ids(E); for (int e = 0; e < E; ++e) ids[e] = e;
            std::shuffle(ids.begin(), ids.end(), rng);
            std::vector<std::vector<std::pair<int, float>>> route(1);
            for (int k = 0; k < 8; ++k) route[0].push_back({ ids[k], ud(rng) });
            const double t0 = now();
            if (it % 3 == 0) moe_forward(pool, A, x.data(), 1, route, y.data(), 2);
            else if (it % 3 == 1) moe_forward(pool, P, x.data(), 1, route, y.data(), 2);
            else moe_forward_band(pool, P, x.data(), 1, route, y.data(), 2);
            t[it % 3] += now() - t0;
        }
        printf("4. ms per expert (M=1, 8 picks, I16, %d threads, %d experts in the pool): swz %.4f | usm block %.4f | usm band %.4f\n",
               threads, E, t[0] * 1e3 / (reps * 8), t[1] * 1e3 / (reps * 8), t[2] * 1e3 / (reps * 8));
    }
    pool.stop();
    printf(fails ? "FAILED (%d)\n" : "ALL OK\n", fails);
    return fails ? 1 : 0;
}
