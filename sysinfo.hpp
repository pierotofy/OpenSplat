#ifndef SYSINFO_H
#define SYSINFO_H

#include <cstdint>

uint64_t physicalRamBytes();
uint64_t availableRamBytes();
// Free memory on the training device: GPU memory for CUDA/HIP, RAM otherwise
uint64_t freeDeviceMemoryBytes(bool gpu);
int currentPid();
bool processAlive(int pid);

#endif
