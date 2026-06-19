# Fluid Dynamics Optimization on Parallel Architectures

A high-performance Computational Fluid Dynamics (CFD) solver implementing the **D2Q9 Lattice Boltzmann Method (LBM)** with a Bhatnagar-Gross-Krook (BGK) relaxation model. This repository evaluates the architectural and performance trade-offs of accelerating fluid dynamics simulation across both sequential CPU and parallel GPU hardware.

---

## Project Overview

The main challenge of LBM simulations is **Memory Bandwidth Saturation**. The project focuses on shifting from traditional, sub-optimal Object-Oriented design patterns to optimized parallel paradigms. 

Key structural optimizations implemented in this project include:
* **Structure of Arrays (SoA):** Eliminates stride penalties by flattening the 9 discrete direction populations into independent, linear memory planes, achieving **100% coalesced global memory transactions** under CUDA.
* **Fused Pull-Grid Paradigm:** Resolves fluid leakage bugs by refactoring the advection step from a *Push (Scatter)* model to a *Pull (Gather)* model, making the parallel execution thread-safe without expensive synchronization methods.
* **Spatial Cache Locality:** Exploits GPU's L1/L2 texture caches by tuning thread block dimensions (analyzing `32x1`, `32x8`, `16x16`, and `32x16` configurations) to minimize the perimeter-to-area ratio.

---

## Repository Structure

The repository contains three primary files, mapping out the evolution from baseline physics to parallel acceleration and visualization:

### 1. `naive_d2q9.cpp`
* **Role:** The sequential CPU reference implementation.
* **Description:** Implements the core D2Q9 LBM loop (Streaming, Boundary Bounce-Back, and BGK Collision) in standard C++. It utilizes the refactored **Fused Pull-Grid** layout to eliminate fluid logic leaks through thin solid boundaries, acting as the mathematical control variable for benchmarking.

### 2. `optimized_d2q9.cu`
* **Role:** The optimized GPU engine.
* **Description:** An optimized CUDA implementation that maps the fluid grid onto the GPU's SIMT architecture. It combines the streaming and collision operations into a single fused kernel (`lbm_optimized_pull_kernel`), forces maximum memory bus saturation using a strict SoA layout, and dumps binary state matrix frames into an local output directory.

### 3. `animate.py`
* **Role:** Visual simulation and dashboard pipeline.
* **Description:** A Python script utilizing `matplotlib` and `numpy` that parses the raw frame outputs from the two implementations. It handles 5 distinct geometric scenes (*Circle, Vertical Plate, Airfoil Wing, V-Cup Trap, and All Shapes Combined*), computing throughput metrics (**MLUPS**) and combining the matrix into a synchronized dashboard GIF.

---

## Execution & Dashboard Generation Instructions

In order to test the implemented design, execute the following terminal commands:


```bash
# Grant execution permissions
chmod +x run_simulation.sh

# Run the full simulation script
./run_simulation.sh
```
