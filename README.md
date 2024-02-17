[![GitHub stars](https://img.shields.io/github/stars/pierotofy/OpenSplat.svg?style=flat-square)](https://github.com/pierotofy/OpenSplat/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/pierotofy/OpenSplat.svg?style=flat-square)](https://github.com/pierotofy/OpenSplat/network)
[![GitHub issues](https://img.shields.io/github/issues/pierotofy/OpenSplat.svg?style=flat-square)](https://github.com/pierotofy/OpenSplat/issues)
[![GitHub license](https://img.shields.io/github/license/pierotofy/OpenSplat.svg?style=flat-square)](https://github.com/pierotofy/OpenSplat/blob/main/LICENSE)


# OpenSplat

A free and open source implementation of 3D gaussian splatting, written in C++. It's based on [splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) and focuses on being portable, lean and fast.

OpenSplat takes camera poses + sparse points and computes a scene file (.ply) that can be later imported for viewing, editing and rendering in other [software](https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#open-source-implementations).

## Build

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

 The software has been tested on Ubuntu 20.04, but with some or minimal changes, it should build on Windows and MacOS (help us by opening a PR?).

## Run

To get started, download a dataset and extract it to a folder: [[train](https://drive.google.com/file/d/1-X741ecDczTRoMc3YenJLSFC9ulWXeNc/view?usp=sharing)] | [[banana](https://drive.google.com/file/d/1mUUZFDo2swd6CE5vwPPkjN63Hyf4XyEv/view?usp=sharing)] 

Then run:

```bash
./opensplat /path/to/train
[...]
Wrote splat.ply
```

The output `splat.ply` can then be dragged and dropped in one of the many [viewers](https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#viewers) such as  https://playcanvas.com/viewer

To run on your own data, choose the path to an existing [nerfstudio](https://docs.nerf.studio/) project. The project must have sparse points included (random initialization is not supported, see https://github.com/pierotofy/OpenSplat/issues/7). You can generate nerfstudio projects from [COLMAP](https://github.com/colmap/colmap/) by using nerfstudio's `ns-process-data` command: https://docs.nerf.studio/quickstart/custom_dataset.html


We have plans to add support for reading COLMAP projects directly in the near future. See https://github.com/pierotofy/OpenSplat/issues/1

There's several parameters you can tune. To view the full list:

```bash
./opensplat --help
```

## Project Goals

We recently released OpenSplat, so there's lots of work to do.

 * Support for running on AMD cards
 * Support for running on CPU-only
 * Improve speed / reduce memory usage
 * Distributed computation using multiple machines
 * Real-time training viewer output
 * Compressed scene outputs
 * Your ideas?

 https://github.com/pierotofy/OpenSplat/issues?q=is%3Aopen+is%3Aissue+label%3Aenhancement

## Contributing

We welcome contributions! Pull requests are welcome.

## License

The code in this repository, unless otherwise noted, is licensed under the AGPLv3.

The code from [splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) is originally licensed under the Apache 2.0 license and is © 2023 The Nerfstudio Team.