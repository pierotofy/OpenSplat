#include <cmath>
// HIP CPU compatibility layer for missing functions / namespaces

float rsqrtf(float v){
    return 1.0f / std::sqrtf(v);
}