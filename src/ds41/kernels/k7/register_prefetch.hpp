// A scalar register pipeline (a ring of kPrefetchDepth stages) shared by the CUDA producer and CPU model.
// It only changes load timing: every accumulator consumes steps 0..Steps-1.
#pragma once

#ifdef __CUDACC__
#define K7_PIPELINE_INLINE __device__ __forceinline__
#else
#define K7_PIPELINE_INLINE inline
#endif

namespace strata::ds41::kernels::k7_detail {

template <int Tokens>
struct Stage {
    float weight;
    float values[Tokens];
};

template <int Tokens, typename Fma>
K7_PIPELINE_INLINE void consume(const Stage<Tokens>& stage, int step, int row,
                                float (&dots)[Tokens], float (&squares)[Tokens], Fma fma) {
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float value = stage.values[token];
        dots[token] = fma(value, stage.weight, dots[token]);
        if (row < 4 && (step & 3) == row)
            squares[token] = fma(value, value, squares[token]);
    }
}

/// How many steps a lane loads ahead: enough loads in flight to stream the cold 1.97 MB weight (a lane holds
/// Depth stages of 1 + Tokens floats in registers, so fewer for more tokens; m = 8 measured slower at 4 than at 2).
template <int Tokens>
constexpr int kPrefetchDepth = Tokens <= 2 ? 16 : Tokens <= 4 ? 8 : 2;

template <int Tokens, int Steps, typename Load, typename Fma>
K7_PIPELINE_INLINE void register_prefetch(Load load, Fma fma, int row,
                                        float (&dots)[Tokens], float (&squares)[Tokens]) {
    constexpr int Depth = kPrefetchDepth<Tokens>;
    static_assert(Tokens >= 1 && Tokens <= 8 && Steps >= Depth && Steps % Depth == 0);
    if constexpr (Depth == 2) {
        // The original two-step pipeline, not unrolled (m = 8 measured slower fully unrolled).
        Stage<Tokens> current0 = load(0);
        Stage<Tokens> current1 = load(1);
#pragma unroll 1
        for (int step = 0; step < Steps - 2; step += 2) {
            // Fetch two future stride-256 positions before consuming this pair.
            const Stage<Tokens> next0 = load(step + 2);
            const Stage<Tokens> next1 = load(step + 3);
            consume(current0, step, row, dots, squares, fma);
            consume(current1, step + 1, row, dots, squares, fma);
            current0 = next0;
            current1 = next1;
        }
        // Separate drain: there is no speculative load past the last legal value.
        consume(current0, Steps - 2, row, dots, squares, fma);
        consume(current1, Steps - 1, row, dots, squares, fma);
    } else {
        // A ring of Depth stages in registers (fully unrolled, so every index is a constant). Step s is consumed in
        // order 0..Steps-1; its slot is refilled with step s + Depth, never past the last legal step.
        Stage<Tokens> ring[Depth];
#pragma unroll
        for (int s = 0; s < Depth; ++s) ring[s] = load(s);
#pragma unroll
        for (int step = 0; step < Steps; ++step) {
            const Stage<Tokens> current = ring[step % Depth];
            if (step + Depth < Steps) ring[step % Depth] = load(step + Depth);
            consume(current, step, row, dots, squares, fma);
        }
    }
}

}  // namespace strata::ds41::kernels::k7_detail
#undef K7_PIPELINE_INLINE
