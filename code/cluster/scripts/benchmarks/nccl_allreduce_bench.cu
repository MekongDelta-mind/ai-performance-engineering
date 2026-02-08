#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

#define CUDACHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); \
  } \
} while(0)

#define NCCLCHECK(call) do { \
  ncclResult_t r = (call); \
  if (r != ncclSuccess) { \
    fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
    exit(1); \
  } \
} while(0)

int main(int argc, char** argv) {
  size_t min_bytes = 8;
  size_t max_bytes = 1ull << 30; // 1 GiB
  double factor = 2.0;
  int iters = 20;
  int warmup = 5;

  if (argc >= 2) min_bytes = (size_t)atoll(argv[1]);
  if (argc >= 3) max_bytes = (size_t)atoll(argv[2]);
  if (argc >= 4) factor = atof(argv[3]);
  if (argc >= 5) iters = atoi(argv[4]);

  int ndev = 0;
  CUDACHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    printf("Need at least 2 GPUs, found %d\n", ndev);
    return 0;
  }

  std::vector<ncclComm_t> comms(ndev);
  std::vector<cudaStream_t> streams(ndev);
  std::vector<void*> sendbufs(ndev), recvbufs(ndev);
  std::vector<cudaEvent_t> start(ndev), stop(ndev);

  NCCLCHECK(ncclCommInitAll(comms.data(), ndev, nullptr));

  for (int d = 0; d < ndev; ++d) {
    CUDACHECK(cudaSetDevice(d));
    CUDACHECK(cudaStreamCreate(&streams[d]));
    CUDACHECK(cudaMalloc(&sendbufs[d], max_bytes));
    CUDACHECK(cudaMalloc(&recvbufs[d], max_bytes));
    CUDACHECK(cudaEventCreate(&start[d]));
    CUDACHECK(cudaEventCreate(&stop[d]));
  }

  printf("# size_bytes,time_ms,alg_bw_gbs,bus_bw_gbs\n");

  for (size_t bytes = min_bytes; bytes <= max_bytes; bytes = (size_t)(bytes * factor)) {
    // Warmup
    for (int w = 0; w < warmup; ++w) {
      NCCLCHECK(ncclGroupStart());
      for (int d = 0; d < ndev; ++d) {
        CUDACHECK(cudaSetDevice(d));
        NCCLCHECK(ncclAllReduce(sendbufs[d], recvbufs[d], bytes / sizeof(float), ncclFloat, ncclSum, comms[d], streams[d]));
      }
      NCCLCHECK(ncclGroupEnd());
      for (int d = 0; d < ndev; ++d) {
        CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaStreamSynchronize(streams[d]));
      }
    }

    // Timed iterations
    float max_ms = 0.0f;
    for (int it = 0; it < iters; ++it) {
      for (int d = 0; d < ndev; ++d) {
        CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaEventRecord(start[d], streams[d]));
      }

      NCCLCHECK(ncclGroupStart());
      for (int d = 0; d < ndev; ++d) {
        CUDACHECK(cudaSetDevice(d));
        NCCLCHECK(ncclAllReduce(sendbufs[d], recvbufs[d], bytes / sizeof(float), ncclFloat, ncclSum, comms[d], streams[d]));
      }
      NCCLCHECK(ncclGroupEnd());

      for (int d = 0; d < ndev; ++d) {
        CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaEventRecord(stop[d], streams[d]));
      }

      max_ms = 0.0f;
      for (int d = 0; d < ndev; ++d) {
        CUDACHECK(cudaSetDevice(d));
        CUDACHECK(cudaEventSynchronize(stop[d]));
        float ms = 0.0f;
        CUDACHECK(cudaEventElapsedTime(&ms, start[d], stop[d]));
        if (ms > max_ms) max_ms = ms;
      }
    }

    double avg_ms = max_ms; // using max over GPUs for the last iter
    double seconds = avg_ms / 1000.0;
    double alg_bw = (bytes / seconds) / 1e9;
    double bus_bw = alg_bw * (2.0 * (ndev - 1) / ndev);

    printf("%zu,%.3f,%.2f,%.2f\n", bytes, avg_ms, alg_bw, bus_bw);
    if (bytes > max_bytes / factor) break; // avoid overflow
  }

  for (int d = 0; d < ndev; ++d) {
    CUDACHECK(cudaSetDevice(d));
    CUDACHECK(cudaFree(sendbufs[d]));
    CUDACHECK(cudaFree(recvbufs[d]));
    CUDACHECK(cudaEventDestroy(start[d]));
    CUDACHECK(cudaEventDestroy(stop[d]));
    CUDACHECK(cudaStreamDestroy(streams[d]));
    ncclCommDestroy(comms[d]);
  }

  return 0;
}
