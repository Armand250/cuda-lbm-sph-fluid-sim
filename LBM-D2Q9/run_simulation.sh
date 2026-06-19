#!/bin/bash

set -e

INFO='\033[0;34m[INFO]\033[0m'
SUCCESS='\033[0;32m[SUCCESS]\033[0m'

echo -e "${INFO} Starting LBM Simulation"

# Clean previous output
echo -e "${INFO} Cleaning up workspace"
make clean

# Compile both CPU and GPU implementations
echo -e "${INFO} Building the architectures"
make all

# Execute the pipeline
echo -e "${INFO} Executing CPU simulation block and compiling the GIF"
make run-cpu

# Execute the parallel pipeline
echo -e "${INFO} Executing GPU simulation block and compiling the GIF"
make run-gpu

# Final validation check
echo -e "\n${SUCCESS} Simulation completed!"