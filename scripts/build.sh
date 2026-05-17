#!/bin/bash
# Usage: ./scripts/build.sh src/01_reductions/02_warpShuffleReduction
nvcc -I./common/include -I$1 $1/main.cu $1/kernel.cu common/src/*.cu -o ./bin/test_run

# INSTRUCTIONS:

# Make file executable by:          chmod +x scripts/build.sh
# If bin folder doesn't exist:      mkdir -p bin
# Run the compiled file:            ./bin/test_run