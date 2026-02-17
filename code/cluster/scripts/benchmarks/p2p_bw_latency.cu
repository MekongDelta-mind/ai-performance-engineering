#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <string>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while (0)

int main(int argc, char** argv) {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        printf("Need at least 2 GPUs, found %d\n", device_count);
        return 0;
    }

    size_t bw_bytes = 256ull * 1024ull * 1024ull; // 256 MiB
    int bw_iters = 20;
    size_t lat_bytes = 4; // 4 bytes
    int lat_iters = 10000;

    // Optional overrides
    if (argc >= 2) bw_bytes = (size_t)atoll(argv[1]);
    if (argc >= 3) bw_iters = atoi(argv[2]);
    if (argc >= 4) lat_bytes = (size_t)atoll(argv[3]);
    if (argc >= 5) lat_iters = atoi(argv[4]);

    std::vector<void*> buffers(device_count, nullptr);

    for (int d = 0; d < device_count; ++d) {
        CHECK_CUDA(cudaSetDevice(d));
        CHECK_CUDA(cudaMalloc(&buffers[d], bw_bytes));
    }

    // Enable P2P where possible
    for (int i = 0; i < device_count; ++i) {
        for (int j = 0; j < device_count; ++j) {
            if (i == j) continue;
            int can = 0;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&can, i, j));
            if (can) {
                CHECK_CUDA(cudaSetDevice(i));
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                    fprintf(stderr, "Enable peer access %d->%d failed: %s\n", i, j, cudaGetErrorString(e));
                    return 1;
                }
                // Clear possible error
                cudaGetLastError();
            }
        }
    }

    printf("# src,dst,bw_gbs,lat_us,peer_access\n");

    for (int src = 0; src < device_count; ++src) {
        for (int dst = 0; dst < device_count; ++dst) {
            if (src == dst) continue;

            int can = 0;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&can, src, dst));
            if (!can) {
                printf("%d,%d,0,0,no\n", src, dst);
                continue;
            }

            // Measure bandwidth using dst device stream
            CHECK_CUDA(cudaSetDevice(dst));
            cudaStream_t stream;
            CHECK_CUDA(cudaStreamCreate(&stream));
            cudaEvent_t start, stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));

            // Warmup
            for (int w = 0; w < 3; ++w) {
                CHECK_CUDA(cudaMemcpyPeerAsync(buffers[dst], dst, buffers[src], src, bw_bytes, stream));
            }
            CHECK_CUDA(cudaStreamSynchronize(stream));

            CHECK_CUDA(cudaEventRecord(start, stream));
            for (int it = 0; it < bw_iters; ++it) {
                CHECK_CUDA(cudaMemcpyPeerAsync(buffers[dst], dst, buffers[src], src, bw_bytes, stream));
            }
            CHECK_CUDA(cudaEventRecord(stop, stream));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            double seconds = ms / 1000.0;
            double bytes = (double)bw_bytes * (double)bw_iters;
            double bw_gbs = (bytes / seconds) / 1e9;

            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            CHECK_CUDA(cudaStreamDestroy(stream));

            // Measure latency with small transfers
            CHECK_CUDA(cudaSetDevice(dst));
            cudaStream_t lstream;
            CHECK_CUDA(cudaStreamCreate(&lstream));
            cudaEvent_t lstart, lstop;
            CHECK_CUDA(cudaEventCreate(&lstart));
            CHECK_CUDA(cudaEventCreate(&lstop));

            // Warmup
            for (int w = 0; w < 10; ++w) {
                CHECK_CUDA(cudaMemcpyPeerAsync(buffers[dst], dst, buffers[src], src, lat_bytes, lstream));
            }
            CHECK_CUDA(cudaStreamSynchronize(lstream));

            CHECK_CUDA(cudaEventRecord(lstart, lstream));
            for (int it = 0; it < lat_iters; ++it) {
                CHECK_CUDA(cudaMemcpyPeerAsync(buffers[dst], dst, buffers[src], src, lat_bytes, lstream));
            }
            CHECK_CUDA(cudaEventRecord(lstop, lstream));
            CHECK_CUDA(cudaEventSynchronize(lstop));

            float lms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&lms, lstart, lstop));
            double lat_us = (lms * 1000.0) / (double)lat_iters;

            CHECK_CUDA(cudaEventDestroy(lstart));
            CHECK_CUDA(cudaEventDestroy(lstop));
            CHECK_CUDA(cudaStreamDestroy(lstream));

            printf("%d,%d,%.2f,%.3f,yes\n", src, dst, bw_gbs, lat_us);
        }
    }

    for (int d = 0; d < device_count; ++d) {
        CHECK_CUDA(cudaSetDevice(d));
        cudaFree(buffers[d]);
    }

    return 0;
}

