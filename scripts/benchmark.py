#!/usr/bin/env python3
import subprocess
import csv
import re
import os
import sys

# Configuration
BUILD_DIR = os.path.abspath("build")
EXECUTABLE = os.path.join(BUILD_DIR, "selfish_gene")
OUTPUT_FILE = "results.csv"

# Sweep parameters
GRID_SIZES = [1024, 2048, 4096, 8192] # 16384 might be too large for some GPUs/CPUs
STEPS = 1000
BACKENDS = ["cpu", "gpu"]
KERNELS = ["naive", "shared", "bitpacked", "multistep", "multistep64"]

def get_gpu_name():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], encoding="utf-8")
        return output.strip()
    except:
        return "Unknown GPU"

def run_benchmark(backend, kernel, width, height, steps):
    cmd = [
        EXECUTABLE,
        f"--backend={backend}",
        f"--width={width}",
        f"--height={height}",
        f"--steps={steps}"
    ]
    
    if backend == "gpu":
        cmd.append(f"--kernel={kernel}")
        
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse output
        # Example:
        # Results:
        #   Time: 1.3372 s
        #   CUpS: 12546600465.4225
        #   GCUpS: 12.5466
        #   B_eff: 25.0932 GB/s
        
        time_match = re.search(r"Time:\s+([\d\.]+)", output)
        cups_match = re.search(r"CUpS:\s+([\d\.]+)", output)
        gcups_match = re.search(r"GCUpS:\s+([\d\.]+)", output)
        beff_match = re.search(r"B_eff:\s+([\d\.]+)", output)
        
        if time_match and cups_match and gcups_match and beff_match:
            return {
                "time": float(time_match.group(1)),
                "cups": float(cups_match.group(1)),
                "gcups": float(gcups_match.group(1)),
                "b_eff": float(beff_match.group(1))
            }
        else:
            print("Error parsing output")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return None

def main():
    if not os.path.exists(EXECUTABLE):
        print(f"Error: Executable not found at {EXECUTABLE}")
        sys.exit(1)
        
    gpu_name = get_gpu_name()
    print(f"Detected GPU: {gpu_name}")
    
    results = []
    
    # CPU runs (only naive kernel equivalent, ignore kernel param)
    print("--- Benchmarking CPU ---")
    for size in GRID_SIZES:
        # Skip very large sizes for CPU to save time
        if size > 4096:
            continue
            
        metrics = run_benchmark("cpu", "naive", size, size, 100) # Fewer steps for CPU
        if metrics:
            results.append({
                "gpu_name": gpu_name,
                "backend": "cpu",
                "kernel": "scalar",
                "width": size,
                "height": size,
                "steps": 100,
                **metrics
            })

    # GPU runs
    print("--- Benchmarking GPU ---")
    for kernel in KERNELS:
        for size in GRID_SIZES:
            metrics = run_benchmark("gpu", kernel, size, size, STEPS)
            if metrics:
                results.append({
                    "gpu_name": gpu_name,
                    "backend": "gpu",
                    "kernel": kernel,
                    "width": size,
                    "height": size,
                    "steps": STEPS,
                    **metrics
                })
                
    # Save results
    with open(OUTPUT_FILE, "w", newline="") as f:
        fieldnames = ["gpu_name", "backend", "kernel", "width", "height", "steps", "time", "cups", "gcups", "b_eff"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
