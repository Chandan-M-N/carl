#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <curand_kernel.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                         cudaGetErrorName(err_), __FILE__, __LINE__,           \
                         cudaGetErrorString(err_));                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__global__ void init(unsigned long long seed, curandState_t *states, long long n) {
    long long threadID = threadIdx.x + static_cast<long long>(blockDim.x) * blockIdx.x;
    if (threadID < n) {
        curand_init(seed, threadID, 0, &states[threadID]);
    }
}

__global__ void getRandNums(curandState_t *states, int *randNums, long long n) {
    long long threadID = threadIdx.x + static_cast<long long>(blockDim.x) * blockIdx.x;
    if (threadID >= n) {
        return;
    }
    float x = curand_uniform(&states[threadID]);
    float y = curand_uniform(&states[threadID]);
    float dx = fabsf(0.5f - x);
    float dy = fabsf(0.5f - y);
    float distance = sqrtf(dx * dx + dy * dy);

    randNums[threadID] = (distance < 0.5f) ? 1 : 0;
}

int main(int argc, char *argv[]) {
    // default number of blocks if not specified via command line
    long blocks = 256;
    // number of threads per block, max is 1024 on all supported GPUs
    const int bThreads = 512;

    if (argc >= 2) {
        errno = 0;
        char *endIn = NULL;
        long arg1 = std::strtol(argv[1], &endIn, 10);
        if (errno != 0 || endIn == argv[1] || *endIn != '\0' || arg1 <= 0 || arg1 > INT_MAX) {
            std::cerr << "Invalid number of blocks '" << argv[1]
                      << "', expected an integer between 1 and " << INT_MAX << std::endl;
            return EXIT_FAILURE;
        }
        blocks = arg1;
    } else {
        std::cout << "Number of blocks not specified, using default 256" << std::endl;
    }

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Device " << device << ": " << prop.name << " (compute capability "
              << prop.major << "." << prop.minor << ")" << std::endl;

    // total threads is needed for size of the arrays and later for monte-carlo approximation
    const long long tThreads = static_cast<long long>(blocks) * bThreads;
    std::cout << "Blocks: " << blocks << " Total Threads: " << tThreads << std::endl;

    curandState_t *states = NULL;
    CUDA_CHECK(cudaMallocManaged(&states, tThreads * sizeof(curandState_t)));
    init<<<blocks, bThreads>>>(static_cast<unsigned long long>(std::time(NULL)), states, tThreads);
    CUDA_CHECK(cudaGetLastError());

    int *randNums = NULL;
    CUDA_CHECK(cudaMallocManaged(&randNums, tThreads * sizeof(int)));
    getRandNums<<<blocks, bThreads>>>(states, randNums, tThreads);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    long long insidePoints = 0;
    for (long long i = 0; i < tThreads; i++) {
        insidePoints += randNums[i];
    }

    double pi = 4.0 * (static_cast<double>(insidePoints) / static_cast<double>(tThreads));
    std::cout << "Pi is approx: " << pi << std::endl;

    CUDA_CHECK(cudaFree(states));
    CUDA_CHECK(cudaFree(randNums));

    return EXIT_SUCCESS;
}
