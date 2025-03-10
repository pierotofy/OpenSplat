#!/bin/bash

# $1 = cu124 --> 12.4
VER="${1:2:2}.${1:4:1}"

export CUDA_HOME=/usr/local/cuda-${VER}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}

echo "CUDA_HOME: ${CUDA_HOME}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "PATH: ${PATH}"