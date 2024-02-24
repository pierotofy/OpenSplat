
# Took from https://github.com/isl-org/Open3D

# CUDA major and minor version
$CUDA_VER_FULL = $args[0]
$CUDA_VER_ARR = $CUDA_VER_FULL.Split(".")
$CUDA_VER = "$($CUDA_VER_ARR[0]).$($CUDA_VER_ARR[1])"
$CUDA_VER_ID = "$($CUDA_VER_ARR[0])_$($CUDA_VER_ARR[1])"
$CUDA_VER_SHORT = "cu$($CUDA_VER_ARR[0])$($CUDA_VER_ARR[1])"

# Free up disk space
# Docker Cache
# docker rmi $(docker images -q -a)
# Android SDK
# if ($Env:ANDROID_HOME) {
# Remove-Item -Recurse -Force $Env:ANDROID_HOME -ErrorAction Ignore
# }
# if ($Env:ANDROID_NDK_HOME) {
# Remove-Item -Recurse -Force $Env:ANDROID_NDK_HOME -ErrorAction Ignore
# }
# # JVM
# if ($Env:JAVA_HOME_11_X64) {
# Remove-Item -Recurse -Force $Env:JAVA_HOME_11_X64 -ErrorAction Ignore
# }
# if ($Env:JAVA_HOME_8_X64) {
# Remove-Item -Recurse -Force $Env:JAVA_HOME_8_X64 -ErrorAction Ignore
# }
#Remove-Item -Force -Recurse -LiteralPath "C:\msys64\"
#Remove-Item -Force -Recurse -LiteralPath  "C:\mingw64\"

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

if ( !$? ) {
  exit 1
}

Remove-Item .\cuda.exe -Force
Get-PSDrive

# Add CUDA environment variables.
$CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$CUDA_VER"
echo "CUDA_PATH=$CUDA_PATH" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
echo "CUDA_PATH_V$CUDA_VER_ID=$CUDA_PATH" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
echo "CUDA_VER_SHORT=$CUDA_VER_SHORT" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
echo "$CUDA_PATH\bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
