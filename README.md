# GPU Kernel Microarchitecture Profiler

A system-level profiling pipeline designed to quantify the performance impact of specific CUDA kernel microarchitecture patterns. By engineering intentional bottlenecks—such as uncoalesced memory access and atomic contention—this project measures the physical limits of NVIDIA A100 hardware on the ASPIRE2A Supercomputer.

## Project Objectives
* **Identify Bottlenecks**: Distinguish between compute-bound and memory-bound workloads.
* **Quantify Inefficiency**: Measure the exact performance penalty of non-optimal memory patterns.
* **Hardware Telemetry**: Log real-time GPU metrics including utilization, temperature, and memory footprint during high-stress runs.

---

## 🚀 Key Performance Results (NVIDIA A100)

By scaling workloads to **256 million elements** ($2^{28}$) and executing **500 iterations** per kernel, the following architectural impacts were quantified:

| Kernel Pattern | Optimization State | Avg Execution Time | Performance Impact |
|:--- |:--- |:--- |:--- |
| **Vector Add** | Coalesced (Optimal) | **~2.35 ms** | Baseline (1.0x) |
| **Uncoalesced Copy** | Strided (Non-optimal) | **~13.78 ms** | **5.8x Slowdown** |
| **Histogram** | Atomic Contention | **~12.87 ms** | **5.5x Slowdown** |
| **Matrix Mul** | Compute Intensive | **~472.30 ms** | Sustained Load |

---

## 🛠️ System Architecture

The profiler operates as an automated pipeline within an HPC environment using PBS Pro:

1. **Kernel Suite**: 7 custom CUDA C++ kernels targeting specific hardware units (Load/Store units, SM schedulers, Atomic engines).
2. **HPC Orchestration**: PBS job scripts manage resource allocation (16 CPUs, 1 A100 GPU, 110GB RAM) and execution flow.
3. **Telemetry Engine**: A background process utilizing `nvidia-smi` to log thermal and utilization data.
4. **Analysis Layer**: NVIDIA Nsight Systems (`nsys`) for nanosecond-precision kernel timing.

---
## Results

### Performance Benchmark Summary

All kernels were profiled on NVIDIA A100 40GB GPUs (NSCC ASPIRE2A) using NVIDIA Nsight Systems. Each kernel was executed 500 times to ensure statistical significance and sustained GPU load.

| Kernel | Avg Time (ms) | Total Time (ms) | Instances | Category |
|--------|---------------|-----------------|-----------|----------|
| vectorAdd | 2.35 | 1,176 | 500 | Baseline (Coalesced) |
| warpDivergence | 1.65 | 826 | 500 | Logic Branching |
| convolution | 1.66 | 829 | 500 | Memory Pattern |
| reduction | 4.49 | 2,247 | 500 | Sync Overhead |
| histogram | 12.88 | 6,439 | 500 | Atomic Contention |
| uncoalescedCopy | 13.79 | 6,894 | 500 | Memory Bottleneck |
| matMul | 473.30 | 47,330 | 100 | Compute-Bound |

### Key Findings

#### 1. Memory Coalescing Impact (5.9x Slowdown)

Uncoalesced memory access patterns caused a 587% performance degradation compared to coalesced access:

    vectorAdd (Coalesced):      2.35 ms avg
    uncoalescedCopy:           13.79 ms avg
    Slowdown Factor:            5.87x

This demonstrates that memory access patterns are critical for GPU performance optimization.

#### 2. Atomic Contention Overhead (5.5x Slowdown)

Atomic operations on shared memory addresses created severe serialization:

    vectorAdd (No Atomics):     2.35 ms avg
    histogram (With Atomics):  12.88 ms avg
    Slowdown Factor:            5.48x

Atomic contention is a major bottleneck in parallel reduction and histogram operations.

#### 3. Compute-Bound Workload Sustainability

Matrix multiplication kernel sustained GPU load for 47+ seconds:

    matMul Total Time:         47,330 ms
    Instances:                     100
    Avg Per Kernel:             473.3 ms
    GPU Utilization:              90%+

This validates the profiling methodology for sustained compute workloads.

#### 4. Warp Divergence Analysis

Branching logic showed workload-dependent impact:

    vectorAdd (No Branching):   2.35 ms avg
    warpDivergence (Branching): 1.65 ms avg

Note: Warp divergence impact varies based on problem size and branch complexity.

### GPU Telemetry Data

Real-time GPU metrics were logged at 1-second intervals during kernel execution:

    Timestamp            | Util (%) | Temp (C) | Memory (MiB)
    ---------------------|----------|----------|-------------
    2026/03/09 02:25:06  |    10    |    43    |      5
    2026/03/09 02:25:11  |    10    |    43    |      5

Note: Low utilization in telemetry is due to sub-second kernel execution times. Nsight Systems provides microsecond-level accuracy for precise measurements.

### Profiling Tools Used

    Tool                  | Purpose
    ----------------------|----------------------------------
    NVIDIA Nsight Systems | Kernel-level profiling (ns accuracy)
    nvidia-smi            | Real-time GPU telemetry (1s intervals)
    PBS Pro               | HPC job scheduling and resource allocation
    CUDA Events           | Precise kernel timing within code

### Hardware Specifications

    Component    | Specification
    -------------|---------------------------
    GPU          | NVIDIA A100 40GB
    Cluster      | NSCC ASPIRE2A
    CPU          | AMD EPYC 7713 (64 cores/node)
    Memory       | 110 GB per GPU (enforced ratio)
    Interconnect | Slingshot (multi-node capable)

### Reproducibility

All results can be reproduced by:

    1. Clone this repository
    2. Upload to ASPIRE2A or similar NVIDIA GPU cluster
    3. Update PBS script with your Project ID
    4. Run: qsub run_gpu_profiler.pbs
    5. Analyze: nsys stats results/*.nsys-rep

Raw profiling data (.nsys-rep files) and telemetry logs are included in the results/ and logs/ directories.
## 💻 Implementation Details

### Uncoalesced Memory Access Pattern
To force a memory bottleneck, the indexing math was modified to break coalescing. Instead of threads 0-31 reading adjacent addresses, they are forced to access strided memory locations, triggering 32 separate memory transactions per warp.

    __global__ void uncoalescedCopy(float *input, float *output, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            int stride = 32;
            int bad_idx = (idx % stride) * (N / stride) + (idx / stride);
            if (bad_idx < N) {
                output[bad_idx] = input[idx] * 2.0f;
            }
        }
    }

### Sustained Load Execution
Kernels are wrapped in a CPU-side loop to ensure the GPU maintains a steady state long enough for telemetry sensors to capture physical changes like temperature spikes.

    printf("Launching Stress Test (500 iterations)...\n");
    for(int iter = 0; iter < 500; iter++) {
        vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();

---

## 📊 Telemetry Data
During the **Matrix Multiplication** run, the pipeline recorded:
* **GPU Utilization**: Sustained 100% for 47+ seconds.
* **Thermal Response**: Temperature increase from **48°C to 68°C**.
* **Memory Footprint**: Peak allocation of **~3.5GB** for Vector Addition.

---

## How to Run (HPC Environment)

1. Load the CUDA environment:
    module load cuda/12.2.2

2. Compile the kernel suite:
    nvcc kernels/vector_add.cu -o results/vector_add

3. Submit the profiling job:
    qsub run_gpu_profiler.pbs

4. Extract results:
    nsys stats results/vector_add_report.nsys-rep
