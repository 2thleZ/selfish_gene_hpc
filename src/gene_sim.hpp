#pragma once

#include <cstdint>
#include <string>

// Simulation Configuration
struct SimConfig {
  int width;
  int height;
  int steps;
  int seed;

  // Biological Parameters
  float base_rep_rate;
  float base_death_rate;
  float base_mut_rate;

  // Resource Limits
  int crowding_factor; // C

  // Visualization
  bool save_frames;
  int frame_interval;
};

enum class KernelVariant { Naive, Shared };

struct TraitStats {
  int population;
  float avg_rep;
  float avg_death;
  float avg_mut;
  float step_time_sec;
};

// host-level simulation management
void init_simulation(int width, int height, int seed);
void cleanup_simulation();

// simulation step execution and statistic gathering
TraitStats step_simulation(float *current_host, int width, int height,
                           SimConfig &config, KernelVariant variant,
                           int step_idx);

// Helper for visualization mapping
void map_traits_to_color(float rep, float death, float mut, uint8_t &r,
                         uint8_t &g, uint8_t &b);
