ARG UBUNTU_VERSION=22.04
ARG TORCH-VERSION=2.2.1
ARG CUDA_VERSION=12.1
ARG CMAKE-BUILD-TYPE=Release

FROM ubuntu:${UBUNTU_VERSION}

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
    wget

# Install CUDA
RUN bash .github/workflows/cuda/Linux.sh ${{ matrix.cuda-version }}
