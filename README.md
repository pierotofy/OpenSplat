# OpenSplat

A free and open source implementation of 3D gaussian splatting, written in C++. It's based on [splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) but focused on being portable, lean and fast.

## Build

OpenSplat has been tested on Ubuntu 20.04, but should work on most platforms. With some or minimal changes, it should build on Windows and MacOS (help us by opening a PR?).

Requirements:

 * **CUDA**: Make sure you have the CUDA compiler (`nvcc`) in your PATH that `nvidia-smi` is working. https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html 
 * **libtorch**: Visit https://pytorch.org/get-started/locally/ and select your OS, for package select "LibTorch". Make sure to match your version of CUDA if you want to leverage GPU support in libtorch.
 * **OpenCV**: `sudo apt install libopencv-dev` should do it.
 
 Then:

 ```bash
 git clone https://github.com/pierotofy/OpenSplat OpenSplat
 cd OpenSplat
 mkdir build && cd build
 cmake .. && make -j$(nproc)
 ```

## Run

Download the trains dataset from from [here](TODO:URL).

You can run opensplat on most existing [nerfstudio](https://docs.nerf.studio/) projects so long as they have sparse points included. You can generate these from COLMAP by using nerfstudio's `ns-process-data` command: https://docs.nerf.studio/quickstart/custom_dataset.html

```bash
./opensplat /path/to/trains
```

The output `splat.ply` can then be dragged and dropped in one of the many [viewers](https://github.com/MrNeRF/awesome-3D-gaussian-splatting?tab=readme-ov-file#viewers) such as  https://playcanvas.com/viewer

We have plans to add support for reading COLMAP projects directly in the near future. See https://github.com/pierotofy/OpenSplat/issues/1

To view the list of parameters you can tune, run:

```bash
./opensplat --help
```

## Contributing

We welcome contributions! Pull requests are welcome.

## License

The code in this repository, unless otherwise noted, is licensed under the AGPLv3.

The code from [splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) is originally licensed under the Apache 2.0 license and is (C) 2023 The Nerfstudio Team.