#include "sysinfo.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <fstream>
#include <string>
#endif

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#elif defined(USE_HIP)
#include <hip/hip_runtime_api.h>
#endif

uint64_t physicalRamBytes(){
#ifdef _WIN32
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) return st.ullTotalPhys;
#elif defined(__APPLE__)
    int64_t ram = 0;
    size_t size = sizeof(ram);
    if (sysctlbyname("hw.memsize", &ram, &size, nullptr, 0) == 0) return static_cast<uint64_t>(ram);
#elif defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) == 0) return static_cast<uint64_t>(si.totalram) * si.mem_unit;
#endif
    return 8ull << 30;
}

uint64_t availableRamBytes(){
#ifdef _WIN32
    MEMORYSTATUSEX st;
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) return st.ullAvailPhys;
#elif defined(__APPLE__)
    vm_size_t pageSize = 0;
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    mach_port_t host = mach_host_self();
    if (host_page_size(host, &pageSize) == KERN_SUCCESS &&
        host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) == KERN_SUCCESS){
        return static_cast<uint64_t>(vm.free_count + vm.inactive_count) * pageSize;
    }
#elif defined(__linux__)
    std::ifstream f("/proc/meminfo");
    std::string key;
    uint64_t kb = 0;
    while (f >> key >> kb){
        if (key == "MemAvailable:") return kb * 1024ull;
        f.ignore(256, '\n');
    }
    struct sysinfo si;
    if (sysinfo(&si) == 0) return static_cast<uint64_t>(si.freeram) * si.mem_unit;
#endif
    return physicalRamBytes() / 2;
}

uint64_t freeDeviceMemoryBytes(bool gpu){
    if (gpu){
#ifdef USE_CUDA
        size_t freeB = 0, totalB = 0;
        if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) return freeB;
#elif defined(USE_HIP)
        size_t freeB = 0, totalB = 0;
        if (hipMemGetInfo(&freeB, &totalB) == hipSuccess) return freeB;
#endif
    }
    return availableRamBytes();
}

int currentPid(){
#ifdef _WIN32
    return static_cast<int>(GetCurrentProcessId());
#else
    return static_cast<int>(getpid());
#endif
}

bool processAlive(int pid){
#ifdef _WIN32
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(pid));
    if (h == nullptr) return GetLastError() == ERROR_ACCESS_DENIED;
    DWORD code = 0;
    bool alive = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
    CloseHandle(h);
    return alive;
#else
    if (kill(pid, 0) == 0) return true;
    return errno == EPERM;
#endif
}
