# ==========================================
# D2Q9 Fluid Simulator Makefile
# ==========================================

# Compilers
CXX      := g++
NVCC     := nvcc

# Target Architectures & Compilation Flags
CXXFLAGS := -O3 -std=c++17 -march=native
NVFLAGS  := -O3 -arch=sm_80 -std=c++17

# Target Executable Names
CPU_TARGET := naive_d2q9
GPU_TARGET := optimized_d2q9

# Source Files
CPU_SRC    := naive_d2q9.cpp
GPU_SRC    := optimized_d2q9.cu

# Output Directory for Animation Frames
OUTPUT_DIR := output

.PHONY: all cpu gpu run-cpu run-gpu animate clean help

# Default target builds both binaries
all: cpu gpu

# Compile CPU implementation
cpu: $(CPU_SRC)
	@echo "Compiling Naive CPU implementation"
	$(CXX) $(CXXFLAGS) $(CPU_SRC) -o $(CPU_TARGET)
	@echo "CPU build complete: ./$(CPU_TARGET)"

# Compile GPU implementation targeting Ampere A100 (sm_80)
gpu: $(GPU_SRC)
	@echo "Compiling Optimized CUDA GPU implementation"
	$(NVCC) $(NVFLAGS) $(GPU_SRC) -o $(GPU_TARGET)
	@echo "GPU build complete: ./$(GPU_TARGET)"


# CPU
run-cpu: cpu
	mkdir -p $(OUTPUT_DIR)/cpu
	@echo "Running CPU simulation"
	./$(CPU_TARGET)
	@echo "Generating CPU simulation GIF"
	python3 animate.py --mode cpu

# GPU
run-gpu: gpu
	mkdir -p $(OUTPUT_DIR)/gpu
	@echo "Running GPU simulation"
	./$(GPU_TARGET)
	@echo "Generating GPU simulation GIF"
	python3 animate.py --mode gpu

# Animate simulation
animate:
	python3 animate.py --mode $(or $(MODE),gpu)


# Remove compiled binaries and temporary build artifacts
clean:
	@echo "Cleaning output"
	rm -f $(CPU_TARGET) $(GPU_TARGET)
	rm -rf $(OUTPUT_DIR)
	@echo "Cleanup complete."
