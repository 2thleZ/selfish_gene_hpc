#include "gene_sim.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct ParsedArgs {
  SimConfig config;
  std::string kernel;
};

// command line argument parsing for simulation configuration
ParsedArgs parse_args(int argc, char *argv[]) {
  ParsedArgs args;
  // Defaults matching veritasium context somewhat
  args.config.width = 1024;
  args.config.height = 1024;
  args.config.steps = 1000;
  args.config.seed = 1234;
  args.config.base_rep_rate = 0.04f;
  args.config.base_death_rate = 0.02f;
  args.config.base_mut_rate = 0.04f;
  args.config.crowding_factor = 1024 * 1024 / 2; // Half filled max
  args.config.save_frames = false;
  args.config.frame_interval = 10;
  args.kernel = "naive";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--width=") == 0)
      args.config.width = std::stoi(arg.substr(8));
    else if (arg.find("--height=") == 0)
      args.config.height = std::stoi(arg.substr(9));
    else if (arg.find("--steps=") == 0)
      args.config.steps = std::stoi(arg.substr(8));
    else if (arg.find("--crowding=") == 0)
      args.config.crowding_factor = std::stoi(arg.substr(11));
    else if (arg.find("--kernel=") == 0)
      args.kernel = arg.substr(9);
    else if (arg == "--save-frames")
      args.config.save_frames = true;
  }
  return args;
}

// ppm frame output for biological visualization
void write_ppm(const std::string &filename, int width, int height,
               const std::vector<float> &host_data) {
  std::ofstream ofs(filename, std::ios::binary);
  ofs << "P6\n" << width << " " << height << "\n255\n";

  std::vector<uint8_t> pixels(width * height * 3);
  for (int i = 0; i < width * height; ++i) {
    // host_data is float4 cast to float*, so stride is 4
    float rep = host_data[i * 4 + 0];
    float death = host_data[i * 4 + 1];
    float mut = host_data[i * 4 + 2];
    float state = host_data[i * 4 + 3];

    uint8_t r = 0, g = 0, b = 0;
    if (state >= 0.0f) {
      map_traits_to_color(rep, death, mut, r, g, b);
    }

    pixels[i * 3 + 0] = r;
    pixels[i * 3 + 1] = g;
    pixels[i * 3 + 2] = b;
  }

  ofs.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
}

int main(int argc, char *argv[]) {
  ParsedArgs args = parse_args(argc, argv);

  std::cout << "================================================\n";
  std::cout << " Selfish Gene: Evolutionary Stencil Simulation \n";
  std::cout << " Inspired by Richard Dawkins & Veritasium      \n";
  std::cout << "================================================\n";
  std::cout << "Grid: " << args.config.width << "x" << args.config.height
            << "\n";
  std::cout << "Kernel: " << args.kernel << "\n";
  std::cout << "Crowding Cap: " << args.config.crowding_factor << "\n";

  init_simulation(args.config.width, args.config.height, args.config.seed);

  std::vector<float> host_buffer;
  if (args.config.save_frames) {
    host_buffer.resize(args.config.width * args.config.height * 4);
    system("mkdir -p frames");
  }

  KernelVariant variant =
      (args.kernel == "shared") ? KernelVariant::Shared : KernelVariant::Naive;

  float total_time = 0.0f;

  std::ofstream csv_file("results.csv");
  csv_file << "step,pop,avg_rep,avg_death,avg_mut\n";

  for (int step = 0; step < args.config.steps; ++step) {
    bool save =
        args.config.save_frames && (step % args.config.frame_interval == 0);
    float *host_ptr = save ? host_buffer.data() : nullptr;

    // Changed return type from float to TraitStats
    TraitStats stats =
        step_simulation(host_ptr, args.config.width, args.config.height,
                        args.config, variant, step);
    total_time += stats.step_time_sec;

    csv_file << step << "," << stats.population << "," << stats.avg_rep << ","
             << stats.avg_death << "," << stats.avg_mut << "\n";

    if (save) {
      std::ostringstream ss;
      ss << "frames/step_" << std::setw(6) << std::setfill('0') << step
         << ".ppm";
      write_ppm(ss.str(), args.config.width, args.config.height, host_buffer);
    }
  }

  cleanup_simulation();

  std::cout << "Simulation Complete. Total GPU Time: " << total_time << " s\n";

  return 0;
}
