# N-Body Simulation (CUDA)

[cite_start]A high-performance **N-body simulation** implemented in CUDA to model complex gravitational interactions between $N$ celestial bodies[cite: 3, 26, 27]. [cite_start]This project calculates planetary trajectories, predicts asteroid impacts, and evaluates defensive missile strategies [cite: 59-63].

---

## ## Parallelism Strategy

[cite_start]The simulation utilizes a **1D grid of $N$ blocks**, with each block representing one celestial body [cite: 111-137]. [cite_start]This maps the $O(N^2)$ gravity calculations directly to GPU hardware [cite: 111-137].

### ### 1. Acceleration Kernel (`run_step`)
[cite_start]Each block computes the total force exerted on its assigned body by the other $N-1$ bodies[cite: 69, 111].
* [cite_start]**Shared Memory**: Accumulates acceleration vectors (`ax`, `ay`, `az`) to reduce global memory traffic[cite: 115, 120, 121].
* [cite_start]**Atomic Operations**: Uses `atomicAdd` to safely aggregate partial forces from threads within the block, ensuring data integrity [cite: 125-127].
* [cite_start]**Synchronized Updates**: Thread 0 in each block updates the final velocity and position after all threads complete the force calculation to maintain consistency [cite: 130-137].



### ### 2. Device-Only Execution Loop
[cite_start]To maximize throughput, the 200,000-step simulation loop resides entirely on the GPU  [cite: 76, 201-210].
* **Zero-Copy Logic**: Data remains in VRAM for the duration of the simulation; no `cudaMemcpy` occurs inside the time-stepping loop.
* **State Management**: Initial parameters are cached in "original" device buffers to allow rapid resets when testing different counter-attack scenarios.

---

## ## Building and Running

### ### Compilation
The program is built using the NVIDIA CUDA Compiler (`nvcc`):
```bash
nvcc -O3 hw5.cu -o hw5
```

### ExecutionRun the simulation by providing an input file and a target output path1:Bash./hw5 <input_file> <output_file>
### Input Data FormatThe input file expects the following structure 2:ParameterDescriptionNTotal number of celestial bodies 3planet-id0-indexed ID of the target planet 4asteroid-idID of the asteroid 5qx qy qzInitial position coordinates (double) 6vx vy vzInitial velocity vectors (double) 7mInitial mass (double) 8typeBody type (device or civilian) 9

## Scalability & OptimizationMulti-GPU Scaling: On systems with multiple GPUs, performance can be maximized by distributing "Counter-Attack" scenarios across different devices, running independent simulations in parallel10.Scenario Management: Since the problem requires testing the destruction of multiple gravity devices, the code uses init_param kernels to reset the environment state quickly between trials.Efficiency: Avoids redundant host-to-device transfers by keeping all simulation state in global device memory.