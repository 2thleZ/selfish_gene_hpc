#pragma once

#include <cuda_runtime.h>

// Fast Hash-based RNG (Stateless)
// Based on PCG Hash / Jarzynski's wang_hash variants
// Good balance of speed and statistical quality for GPU simulations

__host__ __device__ __forceinline__ unsigned int hash_pcg(unsigned int state) {
  unsigned int state_h = state * 747796405u + 2891336453u;
  unsigned int word =
      ((state_h >> ((state_h >> 28u) + 4u)) ^ state_h) * 277803737u;
  return (word >> 22u) ^ word;
}

// Combine inputs into a single seed
__host__ __device__ __forceinline__ unsigned int
hash_coords(unsigned int x, unsigned int y, unsigned int step,
            unsigned int seed) {
  unsigned int s = seed;
  s = hash_pcg(s ^ x);
  s = hash_pcg(s ^ y);
  s = hash_pcg(s ^ step);
  return s;
}

// Returns uniform float [0, 1]
__host__ __device__ __forceinline__ float
rng_float(unsigned int x, unsigned int y, unsigned int step, unsigned int seed,
          int sub_step) {
  unsigned int h = hash_coords(x, y, step, seed);
  h = hash_pcg(h ^ sub_step); // Mix in sub-step (e.g., 0=death, 1=rep, 2=mut)

  // Convert to float [0, 1]
  // Standard trick: (h & 0xFFFFFF) / 16777216.0f
  // But direct cast divide is fine for simulation:
  return (h & 0xFFFFFF) / 16777216.0f;
}

// Gaussian approximation
__host__ __device__ __forceinline__ float
rng_normal(unsigned int x, unsigned int y, unsigned int step, unsigned int seed,
           int sub_step) {
  // Simple Irwin-Hall (sum of 3 uniforms - 1.5) -> Approx Normal(0, 0.25)
  // Fast and good enough for biological drift
  float u1 = rng_float(x, y, step, seed, sub_step);
  float u2 = rng_float(x, y, step, seed, sub_step + 1234);
  float u3 = rng_float(x, y, step, seed, sub_step + 5678);
  return (u1 + u2 + u3 - 1.5f); // Range [-1.5, 1.5]
}
