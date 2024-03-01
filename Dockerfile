ARG UBUNTU_VERSION=22.04

FROM ubuntu:${UBUNTU_VERSION}

ARG TORCH_VERSION=2.2.1
ARG CUDA_VERSION=12.1.1
ARG TORCH_CUDA_ARCH_LIST=7.0;7.5
ARG CMAKE_BUILD_TYPE=Release

SHELL ["/bin/bash", "-c"]

# Env variables
ENV DEBIAN_FRONTEND noninteractive

# Prepare directories
WORKDIR /code

# Copy everything
COPY . ./

# Install build dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    libopencv-dev \
    unzip \
    wget \
    sudo && \
    apt-get autoremove -y --purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install CUDA
RUN bash .github/workflows/cuda/Linux.sh ${CUDA_VERSION}

# Install libtorch
RUN wget --no-check-certificate -nv https://download.pytorch.org/libtorch/cu"${CUDA_VERSION%%.*}"$(echo $CUDA_VERSION | cut -d'.' -f2)/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcu"${CUDA_VERSION%%.*}"$(echo $CUDA_VERSION | cut -d'.' -f2).zip -O libtorch.zip && \
    unzip -q libtorch.zip -d . && \
    rm ./libtorch.zip

# Configure and build \
RUN source .github/workflows/cuda/Linux-env.sh cu"${CUDA_VERSION%%.*}"$(echo $CUDA_VERSION | cut -d'.' -f2) && \
    mkdir build && \
    cd build && \
    cmake .. \
    -GNinja \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCMAKE_PREFIX_PATH=/code/libtorch \
    -DCMAKE_INSTALL_PREFIX=/code/install \
    -DTORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} && \
    ninja