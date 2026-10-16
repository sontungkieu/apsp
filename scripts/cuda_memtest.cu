// Independent GPU memory test (no PyTorch). One cudaMalloc of almost all free memory; per pass a kernel writes an
// address-derived 32-bit pattern, cudaDeviceSynchronize() (checked) ends the write, then a SEPARATE verify kernel
// recomputes the expected word from the index and records mismatches (atomic counter + first 64 records). The first
// mismatches are re-read on the host with cudaMemcpy and compared on the CPU.
//   nvcc -O2 -arch=sm_89 -o cuda_memtest scripts/cuda_memtest.cu && ./cuda_memtest [passes=4] [reserve_mib=600]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", \
    cudaGetErrorString(e), __FILE__, __LINE__); return 2; } } while (0)

__host__ __device__ inline uint32_t pat(uint64_t i, uint32_t seed, int inv) {
    uint32_t x = (uint32_t)(i * 2654435761ull) ^ (uint32_t)(i >> 32) ^ seed;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return inv ? ~x : x;
}

__global__ void wr(uint32_t* p, uint64_t n, uint32_t seed, int inv) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        p[i] = pat(i, seed, inv);
}

struct Rec { unsigned long long idx; uint32_t expect, actual; };

__global__ void vf(const uint32_t* p, uint64_t n, uint32_t seed, int inv, unsigned long long* cnt, Rec* rec) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        uint32_t e = pat(i, seed, inv), a = p[i];
        if (a != e) {
            unsigned long long k = atomicAdd(cnt, 1ull);
            if (k < 64) { rec[k].idx = i; rec[k].expect = e; rec[k].actual = a; }
        }
    }
}

int main(int argc, char** argv) {
    int passes = argc > 1 ? atoi(argv[1]) : 4;
    size_t reserve = (size_t)(argc > 2 ? atoi(argv[2]) : 600) << 20;
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
    size_t bytes = (fr - reserve) & ~((size_t)(1 << 20) - 1);
    uint64_t n = bytes / 4;
    uint32_t* d; CK(cudaMalloc(&d, bytes));
    unsigned long long* cnt; Rec* rec; CK(cudaMalloc(&cnt, 8)); CK(cudaMalloc(&rec, 64 * sizeof(Rec)));
    printf("{\"device\":\"%s\",\"total_bytes\":%zu,\"free_bytes\":%zu,\"tested_bytes\":%zu,\"base_ptr\":\"%p\",\"passes\":[\n",
           pr.name, tot, fr, bytes, (void*)d);
    unsigned long long grand = 0;
    for (int ps = 0; ps < passes; ++ps) {
        uint32_t seed = 0x9E3779B9u * (ps + 1); int inv = ps & 1;
        CK(cudaMemset(cnt, 0, 8));
        wr<<<pr.multiProcessorCount * 8, 256>>>(d, n, seed, inv);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        vf<<<pr.multiProcessorCount * 8, 256>>>(d, n, seed, inv, cnt, rec);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        unsigned long long c; CK(cudaMemcpy(&c, cnt, 8, cudaMemcpyDeviceToHost));
        Rec h[64]; CK(cudaMemcpy(h, rec, sizeof(h), cudaMemcpyDeviceToHost));
        grand += c;
        printf(" {\"pass\":%d,\"seed\":%u,\"inverted\":%d,\"mismatches\":%llu,\"first\":[", ps, seed, inv, c);
        int m = c < 64 ? (int)c : 64;
        for (int k = 0; k < m && k < 12; ++k) {
            uint32_t hv; CK(cudaMemcpy(&hv, d + h[k].idx, 4, cudaMemcpyDeviceToHost));   // host re-read
            printf("%s{\"byte_off\":%llu,\"gib\":%.4f,\"expect\":\"%08x\",\"actual_gpu\":\"%08x\",\"actual_host\":\"%08x\",\"xor\":\"%08x\"}",
                   k ? "," : "", h[k].idx * 4ull, h[k].idx * 4.0 / (1 << 30), h[k].expect, h[k].actual, hv,
                   h[k].expect ^ h[k].actual);
        }
        printf("]}%s\n", ps + 1 < passes ? "," : "");
        fflush(stdout);
    }
    printf("],\"total_mismatches\":%llu,\"verdict\":\"%s\"}\n", grand, grand ? "CORRUPTION" : "clean");
    cudaFree(d);
    return grand ? 1 : 0;
}
