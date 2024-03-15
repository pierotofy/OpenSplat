# 💦 OpenSplat 

A free and open source implementation of 3D gaussian splatting written in C++, focused on being portable, lean and fast.

![OpenSplat](https://github.com/pierotofy/OpenSplat/assets/1951843/3461e0e4-e134-4d6a-8a56-d89d00258e41)

OpenSplat takes camera poses + sparse points in [COLMAP](https://colmap.github.io/) or [nerfstudio](https://docs.nerf.studio/quickstart/custom_dataset.html) project format and computes a [scene file](https://drive.google.com/file/d/1w-CBxyWNXF3omA8B_IeOsRmSJel3iwyr/view?usp=sharing) (.ply) that can be later imported for viewing, editing and rendering in other [software](https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#open-source-implementations).

Commercial use allowed and encouraged under the terms of the [AGPLv3](https://www.tldrlegal.com/license/gnu-affero-general-public-license-v3-agpl-3-0). ✅

## Build (CUDA)

Requirements:

 * **CUDA**: Make sure you have the CUDA compiler (`nvcc`) in your PATH and that `nvidia-smi` is working. https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html 
 * **libtorch**: Visit https://pytorch.org/get-started/locally/ and select your OS, for package select "LibTorch". Make sure to match your version of CUDA if you want to leverage GPU support in libtorch.
 * **OpenCV**: `sudo apt install libopencv-dev` should do it.
 
 Then:

 ```bash
 git clone https://github.com/pierotofy/OpenSplat OpenSplat
 cd OpenSplat
 mkdir build && cd build
 cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. && make -j$(nproc)
 ```

 The software has been tested on Ubuntu 20.04 and Windows. With some changes it could run on macOS (help us by opening a PR?).

## Build (ROCm via HIP)
Requirements:

* **ROCm**: Make sure you have the ROCm installed at /opt/rocm. https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html
* **libtorch**: Visit https://pytorch.org/get-started/locally/ and select your OS, for package select "LibTorch". Make sure to match your version of ROCm (5.7) if you want to leverage AMD GPU support in libtorch.
* **OpenCV**: `sudo apt install libopencv-dev` should do it.

Then:

 ```bash
 git clone https://github.com/pierotofy/OpenSplat OpenSplat
 cd OpenSplat
 mkdir build && cd build
 export PYTORCH_ROCM_ARCH=gfx906
 cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ -DGPU_RUNTIME="HIP" -DHIP_ROOT_DIR=/opt/rocm -DOPENSPLAT_BUILD_SIMPLE_TRAINER=ON ..
 make
 ```
In addition, you can leverage Jinja to build the project
 ```
 cmake -GNinja -DCMAKE_PREFIX_PATH=/path/to/libtorch/ -DGPU_RUNTIME="HIP" -DHIP_ROOT_DIR=/opt/rocm -DOPENSPLAT_BUILD_SIMPLE_TRAINER=ON ..
 jinja
 ```

## Docker Build (CUDA)
Navigate to the root directory of OpenSplat repo that has Dockerfile and run the following command to build the Docker image:

```bash
docker build -t opensplat .
```

The `-t` flag and other `--build-arg` let you tag and further customize your image across different ubuntu versions, CUDA/libtorch stacks, and hardware accelerators. 
For example, to build an image with Ubuntu 22.04, CUDA 12.1.1, libtorch 2.2.1, and support for CUDA architectures 7.0 and 7.5, run the following command:

```bash
docker build \
  -t opensplat:ubuntu-22.04-cuda-12.1.1-torch-2.2.1 \
  --build-arg UBUNTU_VERSION=22.04 \
  --build-arg CUDA_VERSION=12.1.1 \
  --build-arg TORCH_VERSION=2.2.1 \
  --build-arg TORCH_CUDA_ARCH_LIST="7.0;7.5" \
  --build-arg CMAKE_BUILD_TYPE=Release .
```

## Docker Build (ROCm via HIP)
Navigate to the root directory of OpenSplat repo that has Dockerfile and run the following command to build the Docker image:
```bash
docker build \
  -t opensplat \
  -f Dockerfile.rocm .
```

The `-t` flag and other `--build-arg` let you tag and further customize your image across different ubuntu versions, CUDA/libtorch stacks, and hardware accelerators.
For example, to build an image with Ubuntu 22.04, CUDA 12.1.1, libtorch 2.2.1, ROCm 5.7.1, and support for ROCm architectures gfx906, run the following command:

```bash
docker build \
  -t opensplat:ubuntu-22.04-cuda-12.1.1-libtorch-2.2.1-rocm-5.7.1-llvm-16 \
  --build-arg UBUNTU_VERSION=22.04 \
  --build-arg CUDA_VERSION=12.1.1 \
  --build-arg TORCH_VERSION=2.2.1 \
  --build-arg ROCM_VERSION=5.7.1 \
  --build-arg PYTORCH_ROCM_ARCH="gfx906" \
  --build-arg CMAKE_BUILD_TYPE=Release .
```
Note: If you want to use ROCm 6.x, you need to switch to AMD version of pytorch docker as a base layer to build:
```bash
docker build \
  -t opensplat:ubuntu-22.04-libtorch-torch-2.1.2-rocm-6.0.2 \
  -f Dockerfile.rocm6 .
```

## Run

To get started, download a dataset and extract it to a folder: [ [banana](https://drive.google.com/file/d/1mUUZFDo2swd6CE5vwPPkjN63Hyf4XyEv/view?usp=sharing) ]  [ [train](https://drive.google.com/file/d/1-X741ecDczTRoMc3YenJLSFC9ulWXeNc/view?usp=sharing) ]  [ [truck](https://drive.google.com/file/d/1WWXo-GKo6d-yf-K1T1CswIdkdZrBNZ_e/view?usp=sharing) ] 

Then run:

```bash
./opensplat /path/to/banana -n 2000
[...]
Wrote splat.ply
```

The output `splat.ply` can then be dragged and dropped in one of the many [viewers](https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#viewers) such as  https://playcanvas.com/viewer. You can also edit/cleanup the scene using https://playcanvas.com/supersplat/editor

To run on your own data, choose the path to an existing [COLMAP](https://colmap.github.io/) or [nerfstudio](https://docs.nerf.studio/) project. The project must have sparse points included (random initialization is not supported, see https://github.com/pierotofy/OpenSplat/issues/7).

There's several parameters you can tune. To view the full list:

```bash
./opensplat --help
```

## Project Goals

We recently released OpenSplat, so there's lots of work to do.

 * Support for running on AMD cards (more testing needed)
 * Support for running on CPU-only
 * Improve speed / reduce memory usage
 * Distributed computation using multiple machines
 * Real-time training viewer output
 * Compressed scene outputs
 * Your ideas?

 https://github.com/pierotofy/OpenSplat/issues?q=is%3Aopen+is%3Aissue+label%3Aenhancement

## Contributing

We welcome contributions! Pull requests are welcome.

## GPU Memory Notes

A single gaussian takes ~2000 bytes of memory, so currenly you need ~2GB of GPU memory for each million gaussians.

## Credits

The methods used in OpenSplat are originally based on [splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html).

## License

The code in this repository, unless otherwise noted, is licensed under the AGPLv3.

The code from [splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) is originally licensed under the Apache 2.0 license and is © 2023 The Nerfstudio Team.
