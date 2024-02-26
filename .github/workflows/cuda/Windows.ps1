# CUDA major and minor version
$CUDA_VER_FULL = $args[0]
$CUDA_VER_ARR = $CUDA_VER_FULL.Split(".")
$CUDA_VER = "$($CUDA_VER_ARR[0]).$($CUDA_VER_ARR[1])"
$CUDA_VER_ID = "$($CUDA_VER_ARR[0])_$($CUDA_VER_ARR[1])"
$CUDA_VER_SHORT = "cu$($CUDA_VER_ARR[0])$($CUDA_VER_ARR[1])"

# Network installer url
if ( $CUDA_VER_ARR[0] -ge 11 ) {
$CUDA_URL = "https://developer.download.nvidia.com/compute/cuda/$CUDA_VER_FULL/network_installers/cuda_$($CUDA_VER_FULL)_windows_network.exe"
} else {
$CUDA_URL = "https://developer.download.nvidia.com/compute/cuda/$CUDA_VER/Prod/network_installers/cuda_$($CUDA_VER_FULL)_windows_network.exe"
}

# Installer arguments
$CUDA_INSTALL_ARGS = "-s"

# Required packages
$CUDA_PACKAGES = "nvcc", "cuobjdump", "nvprune", "cupti", "cublas_dev", "cudart", "cufft_dev", "curand_dev", "cusolver_dev", "cusparse_dev", "thrust", "npp_dev", "nvrtc_dev", "nvml_dev", "nvtx", "visual_studio_integration"
$CUDA_PACKAGES.ForEach({ $CUDA_INSTALL_ARGS += " $($_)_$($CUDA_VER)" })

# Download and install CUDA
echo "Downloading CUDA installer from $CUDA_URL"
Invoke-WebRequest $CUDA_URL -OutFile cuda.exe
echo "Installing CUDA..."
Start-Process -Wait -FilePath .\cuda.exe -ArgumentList "$CUDA_INSTALL_ARGS"
Remove-Item .\cuda.exe -Force

# Install NvToolsExt
$NVTOOLSEXT_PATH  = "C:\Program Files\NVIDIA Corporation\NvToolsExt"
if (Test-Path -Path "$NVTOOLSEXT_PATH " -PathType Container) {
Write-Output "Existing nvtools installation already found, continuing..."
return
}
function New-TemporaryDirectory() {
  New-TemporaryFile | ForEach-Object { Remove-Item $_; New-Item -ItemType Directory -Path $_ }
}
$NV_TOOLS_URL = "https://ossci-windows.s3.amazonaws.com/NvToolsExt.7z" # pytorch test-infra
$tmpToolsDl = New-TemporaryFile
Write-Output "Downloading NvTools, $NV_TOOLS_URL"
Invoke-WebRequest -Uri "$NV_TOOLS_URL" -OutFile "$tmpToolsDl"
$tmpExtractedNvTools = New-TemporaryDirectory
7z x "$tmpToolsDl" -o"$tmpExtractedNvTools"
Write-Output "Copying NvTools, '$tmpExtractedNvTools' -> '$NVTOOLSEXT_PATH'"
New-Item -Path "$NVTOOLSEXT_PATH "-ItemType "directory" -Force
Copy-Item -Recurse -Path "$tmpExtractedNvTools\*" -Destination "$NVTOOLSEXT_PATH"
Remove-Item "$tmpExtractedNvTools" -Recurse -Force
Remove-Item "$tmpToolsDl" -Recurse -Force