# 💦 OpenSplat 

A free and open source implementation of 3D gaussian splatting, written in C++, focused on being portable, lean and fast.

![OpenSplat](https://github.com/pierotofy/OpenSplat/assets/1951843/3461e0e4-e134-4d6a-8a56-d89d00258e41)


OpenSplat takes camera poses + sparse points and computes a [scene file](https://drive.google.com/file/d/1w-CBxyWNXF3omA8B_IeOsRmSJel3iwyr/view?usp=sharing) (.ply) that can be later imported for viewing, editing and rendering in other [software](https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#open-source-implementations).

Commercial use allowed and encouraged under the terms of the [AGPLv3](https://www.tldrlegal.com/license/gnu-affero-general-public-license-v3-agpl-3-0). ✅

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

 The software has been tested on Ubuntu 20.04 and Windows. With some changes it could run on macOS (help us by opening a PR?).

## Run

To get started, download a dataset and extract it to a folder: [ [banana](https://drive.google.com/file/d/1mUUZFDo2swd6CE5vwPPkjN63Hyf4XyEv/view?usp=sharing) ] 

Then run:

```bash
./opensplat /path/to/banana -n 2000
[...]
Wrote splat.ply
```

The output `splat.ply` can then be dragged and dropped in one of the many [viewers](https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#viewers) such as  https://playcanvas.com/viewer. You can also edit/cleanup the scene using https://playcanvas.com/supersplat/editor

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

## Credits

The methods used in OpenSplat are originally based on [splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html).

## License

The code in this repository, unless otherwise noted, is licensed under the AGPLv3.

The code from [splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) is originally licensed under the Apache 2.0 license and is © 2023 The Nerfstudio Team.
