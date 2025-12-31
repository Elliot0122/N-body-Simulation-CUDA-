# N-Body Simulation (CUDA)

A high-performance N-body simulation implemented in CUDA to model complex gravitational interactions between $N$ celestial bodies. This project calculates planetary trajectories, predicts asteroid impacts, and evaluates defensive missile strategies.

## Parallelism Strategy

The simulation utilizes a **1D grid of $N$ blocks**, with each block representing one celestial body. This maps the $O(N^2)$ gravity calculations directly to GPU hardware.

### 1. Acceleration Kernel (`run_step`)
Each block computes the total force exerted on its assigned body by the other $N-1$ bodies.
* **Shared Memory**: Accumulates acceleration vectors (`ax`, `ay`, `az`) to reduce global memory traffic.
* **Atomic Operations**: Uses `atomicAdd` to safely aggregate partial forces from threads within the block, ensuring data integrity.
* **Synchronized Updates**: Thread 0 in each block updates the final velocity and position after all threads complete the force calculation to maintain consistency.



### 2. Device-Only Execution Loop
To maximize throughput, the 200,000-step simulation loop resides entirely on the GPU.
* **Zero-Copy Logic**: Data remains in VRAM for the duration of the simulation; no `cudaMemcpy` occurs inside the time-stepping loop.
* **State Management**: Initial parameters are cached in "original" device buffers to allow rapid resets when testing different counter-attack scenarios.

## Building and Running

### Compilation
The program is built using the NVIDIA CUDA Compiler (`nvcc`):
```bash
srun -ppp --gres=gpu:2 ./hw5 input-file output-file
```

### Execution
Run the simulation by providing an input file and a target output path1:
```bash
./hw5 input_file output_file
```
### Input Data Format

The input file expects the following structure:

| Line | Parameter | Description |
| :--- | :--- | :--- |
| 2 | `N` | Total number of celestial bodies |
| 3 | `planet-id` | 0-indexed ID of the target planet |
| 4 | `asteroid-id` | ID of the asteroid |
| 5 | `qx qy qz` | Initial position coordinates (`double`) |
| 6 | `vx vy vz` | Initial velocity vectors (`double`) |
| 7 | `m` | Initial mass (`double`) |
| 8 | `type` | Body type (`device` or `civilian`) |

## Scalability & Optimization

* **Multi-GPU Scaling**: On systems with multiple GPUs, performance can be maximized by distributing "Counter-Attack" scenarios across different devices, running independent simulations in parallel.
* **Scenario Management**: Since the problem requires testing the destruction of multiple gravity devices, the code uses `init_param` kernels to reset the environment state quickly between trials.
* **Efficiency**: Avoids redundant host-to-device transfers by keeping all simulation state in global device memory.
