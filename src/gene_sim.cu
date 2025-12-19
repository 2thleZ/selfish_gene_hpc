#include "gene_sim.hpp"
#include "rng.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/count.h> // Added by user
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// data layout (float4 per cell)
// x: replication rate
// y: death rate
// z: mutation rate
// w: state (state < 0 means empty)

#define BLOCK_SIZE 16

// device resources
float4 *d_grid_current = nullptr;
float4 *d_grid_next = nullptr;

// logic helper for trait clamping
__host__ __device__ float clamp(float v, float lo, float hi) {
  return fmaxf(lo, fminf(hi, v));
}

// Derived from Veritasium stats
// Base Rep = 0.04, Base Death = 0.02, Base Mut = 0.04 (per step)

// Naive Kernel removed/stubbed below

// shared memory kernel
// 16x16 tile + halo caching to reduce global memory bandwidth
__global__ void step_replicator_shared(const float4 *current, float4 *next,
                                       int width, int height,
                                       float crowding_factor, int current_pop,
                                       int seed, int step_idx, float base_rep,
                                       float base_death, float base_mut) {
  // shared tile with halo
  __shared__ float4 tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int x = blockIdx.x * blockDim.x + tx;
  int y = blockIdx.y * blockDim.y + ty;

  int lx = tx + 1;
  int ly = ty + 1;

  // cooperative load
  // loads center and halos into shared memory tile

  // Helper to get global index with wrap
  auto get_idx = [&](int gx, int gy) {
    return ((gy + height) % height) * width + ((gx + width) % width);
  };

  // Load Center
  tile[ly][lx] = current[get_idx(x, y)];

  // Load Halos
  if (ty == 0)
    tile[0][lx] = current[get_idx(x, y - 1)]; // Top
  if (ty == blockDim.y - 1)
    tile[ly + 1][lx] = current[get_idx(x, y + 1)]; // Bottom
  if (tx == 0)
    tile[ly][0] = current[get_idx(x - 1, y)]; // Left
  if (tx == blockDim.x - 1)
    tile[ly][lx + 1] = current[get_idx(x + 1, y)]; // Right

  // Corners
  if (tx == 0 && ty == 0)
    tile[0][0] = current[get_idx(x - 1, y - 1)];
  if (tx == blockDim.x - 1 && ty == 0)
    tile[0][lx + 1] = current[get_idx(x + 1, y - 1)];
  if (tx == 0 && ty == blockDim.y - 1)
    tile[ly + 1][0] = current[get_idx(x - 1, y + 1)];
  if (tx == blockDim.x - 1 && ty == blockDim.y - 1)
    tile[ly + 1][lx + 1] = current[get_idx(x + 1, y + 1)];

  __syncthreads();

  if (x >= width || y >= height)
    return;

  // cell logic using shared memory and hash-based RNG
  float4 my_state = tile[ly][lx];
  float4 new_state = make_float4(0.0f, 0.0f, 0.0f, -1.0f);

  bool is_alive = (my_state.w >= 0.0f);
  int rng_sub = 0; // sub-step index for unique RNG per event

  if (is_alive) {
    float r_death = rng_float(x, y, step_idx, seed, rng_sub++);

    if (r_death < my_state.y) {
      // death event check
    } else {
      new_state = my_state; // survival
    }
  } else {
    float r_spon = rng_float(x, y, step_idx, seed, rng_sub++);
    if (r_spon < 0.01f) {
      new_state = make_float4(base_rep, base_death, base_mut, 1.0f);
    } else {
      int start_dir = (int)(rng_float(x, y, step_idx, seed, rng_sub++) * 8.0f);
      for (int i = 0; i < 8; ++i) {
        int dir = (start_dir + i) % 8;
        int neighbors_offset[8][2] = {{-1, -1}, {0, -1}, {1, -1}, {-1, 0},
                                      {1, 0},   {-1, 1}, {0, 1},  {1, 1}};

        int flow_dx = neighbors_offset[dir][0];
        int flow_dy = neighbors_offset[dir][1];

        // neighbor identification and replication check
        float4 neighbor = tile[ly + flow_dy][lx + flow_dx];
        bool n_alive = (neighbor.w >= 0.0f);

        if (n_alive) {
          float resource_factor = 1.0f - ((float)current_pop / crowding_factor);
          if (resource_factor < 0.0f)
            resource_factor = 0.0f;

          float eff_rep_rate = neighbor.x * resource_factor;

          if (rng_float(x, y, step_idx, seed, rng_sub++) < eff_rep_rate) {
            new_state = neighbor;
            // trait mutation check
            if (rng_float(x, y, step_idx, seed, rng_sub++) < neighbor.z) {
              float noise_scale = 0.02f;
              // Gaussian approx or uniform
              new_state.x +=
                  (rng_float(x, y, step_idx, seed, rng_sub++) - 0.5f) *
                  noise_scale;
              new_state.y +=
                  (rng_float(x, y, step_idx, seed, rng_sub++) - 0.5f) *
                  noise_scale;
              new_state.z +=
                  (rng_float(x, y, step_idx, seed, rng_sub++) - 0.5f) *
                  noise_scale;

              new_state.x = clamp(new_state.x, 0.0f, 1.0f);
              new_state.y = clamp(new_state.y, 0.0f, 1.0f);
              new_state.z = clamp(new_state.z, 0.0f, 1.0f);
              new_state.w += 0.1f;
            }
            break;
          }
        }
      }
    }
  }

  next[y * width + x] = new_state;
}

// grid initialization kernel to empty state
__global__ void init_grid_kernel(float4 *grid, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  // State -1.0f = Empty
  grid[y * width + x] = make_float4(0.0f, 0.0f, 0.0f, -1.0f);
}

// 1. Naive Kernel Stub (Disabled)
__global__ void step_replicator_naive(const float4 *current, float4 *next,
                                      int width, int height,
                                      float crowding_factor, int current_pop,
                                      int seed, int step_idx, float base_rep,
                                      float base_death, float base_mut) {
  // Stubbed out for now
}

// host-level simulation management
void init_simulation(int width, int height, int seed) {
  size_t bytes = width * height * sizeof(float4);
  CHECK_CUDA(cudaMalloc(&d_grid_current, bytes));
  CHECK_CUDA(cudaMalloc(&d_grid_next, bytes));

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  init_grid_kernel<<<grid, block>>>(d_grid_current, width, height);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Init Kernel Launch Error: %s\n", cudaGetErrorString(err));
  }
  init_grid_kernel<<<grid, block>>>(d_grid_current, width, height);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

void cleanup_simulation() {
  if (d_grid_current)
    cudaFree(d_grid_current);
  if (d_grid_next)
    cudaFree(d_grid_next);
}

// thrust count functor for active population
struct is_alive_functor {
  __host__ __device__ bool operator()(const float4 &x) { return x.w >= 0.0f; }
};

// thrust reduction functor for trait summation
#include <thrust/tuple.h>

struct sum_traits_functor {
  __host__ __device__ thrust::tuple<float, float, float>
  operator()(const float4 &x) {
    if (x.w >= 0.0f) {
      return thrust::make_tuple(x.x, x.y, x.z);
    } else {
      return thrust::make_tuple(0.0f, 0.0f, 0.0f);
    }
  }
};

struct tuple_sum {
  __host__ __device__ thrust::tuple<float, float, float>
  operator()(const thrust::tuple<float, float, float> &a,
             const thrust::tuple<float, float, float> &b) {
    return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                              thrust::get<1>(a) + thrust::get<1>(b),
                              thrust::get<2>(a) + thrust::get<2>(b));
  }
};
#include <thrust/transform_reduce.h>

TraitStats step_simulation(float *current_host, int width, int height,
                           SimConfig &config, KernelVariant variant,
                           int step_idx) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // population and trait statistic gathering
  thrust::device_ptr<float4> t_ptr(d_grid_current);
  int current_pop = thrust::count_if(
      thrust::device, t_ptr, t_ptr + width * height, is_alive_functor());

  // Sum traits
  thrust::tuple<float, float, float> sums = thrust::transform_reduce(
      thrust::device, t_ptr, t_ptr + width * height, sum_traits_functor(),
      thrust::make_tuple(0.0f, 0.0f, 0.0f), tuple_sum());

  // simulation step execution
  cudaEventRecord(start);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  if (variant == KernelVariant::Naive) {
    // Naive kernel temporarily disabled during optimization refactor
    // step_replicator_naive<<<grid, block>>>(...);
    printf("Naive kernel disabled. Running Shared kernel instead.\n");
    step_replicator_shared<<<grid, block>>>(
        d_grid_current, d_grid_next, width, height,
        (float)config.crowding_factor, current_pop, config.seed, step_idx,
        config.base_rep_rate, config.base_death_rate, config.base_mut_rate);
  } else {
    step_replicator_shared<<<grid, block>>>(
        d_grid_current, d_grid_next, width, height,
        (float)config.crowding_factor, current_pop, config.seed, step_idx,
        config.base_rep_rate, config.base_death_rate, config.base_mut_rate);
  }

  CHECK_CUDA(cudaGetLastError());

  // buffer swap
  float4 *temp = d_grid_current;
  d_grid_current = d_grid_next;
  d_grid_next = temp;

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float millis = 0;
  cudaEventElapsedTime(&millis, start, stop);

  // host memory copy-back for visualization
  if (current_host != nullptr) {
    CHECK_CUDA(cudaMemcpy(current_host, d_grid_current,
                          width * height * sizeof(float4),
                          cudaMemcpyDeviceToHost));
  }

  TraitStats stats;
  stats.population = current_pop;
  stats.step_time_sec = millis / 1000.0f;
  stats.avg_rep = (current_pop > 0) ? thrust::get<0>(sums) / current_pop : 0.0f;
  stats.avg_death =
      (current_pop > 0) ? thrust::get<1>(sums) / current_pop : 0.0f;
  stats.avg_mut = (current_pop > 0) ? thrust::get<2>(sums) / current_pop : 0.0f;

  if (step_idx % 100 == 0) {
    printf("Step %d: Pop=%d, Rep=%.3f, Death=%.3f, Mut=%.3f, Time=%.3f ms\n",
           step_idx, current_pop, stats.avg_rep, stats.avg_death, stats.avg_mut,
           millis);
  }

  return stats;
}

// trait-to-color mapping for biological visualization
void map_traits_to_color(float rep, float death, float mut, uint8_t &r,
                         uint8_t &g, uint8_t &b) {
  // Mapping:
  // Rep Rate -> Red intensity (Higher = More Red)
  // Death Rate -> Green INVERSE (Lower Death = More Green/Healthy)
  // Mutation -> Blue (Higher Mut = More Blue)

  // Normalize based on some expected ranges
  // Rep: 0.04 base, max maybe 0.1?
  // Death: 0.02 base, max maybe 0.1?

  float n_rep = rep * 20.0f;              // 0.05 -> 1.0
  float n_death = 1.0f - (death * 20.0f); // 0.05 -> 0.0
  float n_mut = mut * 20.0f;

  r = (uint8_t)(clamp(n_rep, 0.0f, 1.0f) * 255.0f);
  g = (uint8_t)(clamp(n_death, 0.0f, 1.0f) * 255.0f);
  b = (uint8_t)(clamp(n_mut, 0.0f, 1.0f) * 255.0f);
}
