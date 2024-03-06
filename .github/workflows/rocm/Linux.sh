#!/bin/bash

# ROCM major and minor version
ROCM_VER_FULL=${1}
ROCM_VER_ARR=($(echo ${ROCM_VER_FULL} | tr "." " "))
ROCM_VER="${ROCM_VER_ARR[0]}.${ROCM_VER_ARR[1]}"

if [[ ${2} == "ubuntu-20.04" ]]; then
  OS_CODE_NAME=focal
elif [[ ${2} == "ubuntu-22.04" ]]; then
  OS_CODE_NAME=jammy
else
  echo "Unrecognized OS=${2}"
  exit 1
fi

case ${ROCM_VER_FULL} in
  6.0.2)
    FILENAME=amdgpu-install_6.0.60002-1_all.deb
    URL=https://repo.radeon.com/amdgpu-install/${ROCM_VER_FULL}/ubuntu/${OS_CODE_NAME}
    ;;
  6.0.1)
    FILENAME=amdgpu-install_6.0.60001-1_all.deb
    URL=https://repo.radeon.com/amdgpu-install/${ROCM_VER_FULL}/ubuntu/${OS_CODE_NAME}
    ;;
  5.7.1)
    FILENAME=amdgpu-install_5.7.50701-1_all.deb
    URL=https://repo.radeon.com/amdgpu-install/${ROCM_VER_FULL}/ubuntu/${OS_CODE_NAME}
    ;;
  5.6.1)
    FILENAME=amdgpu-install_5.6.50601-1_all.deb
    URL=https://repo.radeon.com/amdgpu-install/${ROCM_VER_FULL}/ubuntu/${OS_CODE_NAME}
    ;;
  5.4.2)
    FILENAME=amdgpu-install_5.4.50402-1_all.deb
    URL=https://repo.radeon.com/amdgpu-install/${ROCM_VER_FULL}/ubuntu/${OS_CODE_NAME}
    ;;
  *)
    echo "Unrecognized ROCM_VERSION=${ROCM_VER}"
    exit 1
    ;;
esac

wget -nv ${URL}/${FILENAME}
sudo dpkg -i ${FILENAME}
sudo amdgpu-install -y --usecase=hip,rocm --no-dkms

sudo apt-get -qq update
sudo apt install -y hip-dev hipify-clang
sudo apt clean

rm -f ${FILENAME}