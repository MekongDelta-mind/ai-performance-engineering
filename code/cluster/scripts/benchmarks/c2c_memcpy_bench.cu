#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static bool cuda_ok(cudaError_t e, const char* file, int line) {
  if (e == cudaSuccess) {
    return true;
  }
  fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
  return false;
}

#define CHECK_CUDA(call)                                                                             \
  do {                                                                                               \
    if (!cuda_ok((call), __FILE__, __LINE__)) {                                                     \
      return 1;                                                                                      \
    }                                                                                                \
  } while (0)

#define CHECK_CUDA_PAIR(call)                                                                        \
  do {                                                                                               \
    if (!cuda_ok((call), __FILE__, __LINE__)) {                                                     \
      return std::make_pair(0.0, 0.0);                                                              \
    }                                                                                                \
  } while (0)

static std::vector<size_t> parse_csv_sizes(const std::string& raw) {
  std::vector<size_t> out;
  size_t start = 0;
  while (start < raw.size()) {
    size_t end = raw.find(',', start);
    if (end == std::string::npos) end = raw.size();
    std::string tok = raw.substr(start, end - start);
    // trim
    while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.front()))) tok.erase(tok.begin());
    while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.back()))) tok.pop_back();
    if (!tok.empty()) {
      char* p = nullptr;
      unsigned long long v = std::strtoull(tok.c_str(), &p, 10);
      if (p == nullptr || *p != '\0') {
        fprintf(stderr, "Invalid size token: '%s'\n", tok.c_str());
        std::exit(2);
      }
      out.push_back(static_cast<size_t>(v));
    }
    start = end + 1;
  }
  return out;
}

static std::string json_escape(const std::string& s) {
  std::ostringstream oss;
  for (char c : s) {
    switch (c) {
      case '\\':
        oss << "\\\\";
        break;
      case '"':
        oss << "\\\"";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\r':
        oss << "\\r";
        break;
      case '\t':
        oss << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          oss << "\\u" << std::hex << (int)c;
        } else {
          oss << c;
        }
    }
  }
  return oss.str();
}

enum class HostMemType { kPageable, kPinned, kManaged };

static const char* host_mem_name(HostMemType t) {
  switch (t) {
    case HostMemType::kPageable:
      return "pageable";
    case HostMemType::kPinned:
      return "pinned";
    case HostMemType::kManaged:
      return "managed";
  }
  return "unknown";
}

struct HostBuf {
  void* ptr = nullptr;
  size_t bytes = 0;
  HostMemType type = HostMemType::kPageable;
};

static int alloc_host(HostMemType type, size_t bytes, HostBuf* out) {
  out->ptr = nullptr;
  out->bytes = bytes;
  out->type = type;
  if (bytes == 0) return 0;

  if (type == HostMemType::kPageable) {
    void* p = nullptr;
    int rc = posix_memalign(&p, 4096, bytes);
    if (rc != 0 || p == nullptr) {
      fprintf(stderr, "posix_memalign failed rc=%d\n", rc);
      return 2;
    }
    out->ptr = p;
    std::memset(out->ptr, 0xA5, bytes);
    return 0;
  }

  if (type == HostMemType::kPinned) {
    CHECK_CUDA(cudaHostAlloc(&out->ptr, bytes, cudaHostAllocDefault));
    std::memset(out->ptr, 0xA5, bytes);
    return 0;
  }

  if (type == HostMemType::kManaged) {
    CHECK_CUDA(cudaMallocManaged(&out->ptr, bytes, cudaMemAttachGlobal));
    // Touch on CPU to establish pages.
    std::memset(out->ptr, 0xA5, bytes);
    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
  }

  return 2;
}

static void free_host(const HostBuf& b) {
  if (!b.ptr) return;
  if (b.type == HostMemType::kPageable) {
    std::free(b.ptr);
    return;
  }
  if (b.type == HostMemType::kPinned) {
    cudaFreeHost(b.ptr);
    return;
  }
  if (b.type == HostMemType::kManaged) {
    cudaFree(b.ptr);
    return;
  }
}

struct CopyResult {
  std::string test;       // "bw" | "lat"
  std::string direction;  // "h2d" | "d2h"
  HostMemType host_mem;
  size_t size_bytes = 0;
  int iters = 0;
  double bw_gbps = 0.0;
  double lat_us = 0.0;
};

static std::pair<double, double> time_memcpy(cudaStream_t stream,
                                             cudaMemcpyKind kind,
                                             void* dst,
                                             const void* src,
                                             size_t bytes,
                                             int warmup,
                                             int iters) {
  for (int i = 0; i < warmup; i++) {
    CHECK_CUDA_PAIR(cudaMemcpyAsync(dst, src, bytes, kind, stream));
  }
  CHECK_CUDA_PAIR(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CHECK_CUDA_PAIR(cudaEventCreate(&start));
  CHECK_CUDA_PAIR(cudaEventCreate(&stop));
  CHECK_CUDA_PAIR(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; i++) {
    CHECK_CUDA_PAIR(cudaMemcpyAsync(dst, src, bytes, kind, stream));
  }
  CHECK_CUDA_PAIR(cudaEventRecord(stop, stream));
  CHECK_CUDA_PAIR(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CHECK_CUDA_PAIR(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA_PAIR(cudaEventDestroy(start));
  CHECK_CUDA_PAIR(cudaEventDestroy(stop));

  double secs = static_cast<double>(ms) / 1e3;
  double total_bytes = static_cast<double>(bytes) * static_cast<double>(iters);
  double gbps = (secs > 0.0) ? (total_bytes / secs / 1e9) : 0.0;
  double lat_us = (iters > 0 && secs > 0.0) ? (secs * 1e6 / static_cast<double>(iters)) : 0.0;
  return {gbps, lat_us};
}

int main(int argc, char** argv) {
  if (std::getenv("AISP_CLOCK_LOCKED") == nullptr || std::string(std::getenv("AISP_CLOCK_LOCKED")) != "1") {
    fprintf(stderr, "ERROR: AISP_CLOCK_LOCKED!=1. Run via scripts/run_with_gpu_clocks.sh.\n");
    return 3;
  }

  int device = 0;
  std::string run_id = "";
  std::string label = "";
  std::vector<size_t> bw_sizes = {4ull << 20, 64ull << 20, 1ull << 30};   // 4MiB,64MiB,1GiB
  std::vector<size_t> lat_sizes = {4ull, 4096ull, 65536ull};              // 4B,4KiB,64KiB
  int bw_iters = 20;
  int lat_iters = 20000;
  int warmup = 5;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&](const char* flag) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing value for %s\n", flag);
        std::exit(2);
      }
      return std::string(argv[++i]);
    };
    if (a == "--device") {
      device = std::stoi(need("--device"));
    } else if (a == "--run-id") {
      run_id = need("--run-id");
    } else if (a == "--label") {
      label = need("--label");
    } else if (a == "--bw-sizes") {
      bw_sizes = parse_csv_sizes(need("--bw-sizes"));
    } else if (a == "--lat-sizes") {
      lat_sizes = parse_csv_sizes(need("--lat-sizes"));
    } else if (a == "--bw-iters") {
      bw_iters = std::stoi(need("--bw-iters"));
    } else if (a == "--lat-iters") {
      lat_iters = std::stoi(need("--lat-iters"));
    } else if (a == "--warmup") {
      warmup = std::stoi(need("--warmup"));
    } else if (a == "-h" || a == "--help") {
      fprintf(stderr,
              "Usage: c2c_memcpy_bench [--device N] [--bw-sizes a,b,c] [--lat-sizes x,y,z]\n"
              "                        [--bw-iters N] [--lat-iters N] [--warmup N]\n");
      return 0;
    } else {
      fprintf(stderr, "Unknown arg: %s\n", a.c_str());
      return 2;
    }
  }

  CHECK_CUDA(cudaSetDevice(device));

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::string dev_name = prop.name ? std::string(prop.name) : std::string("unknown");

  size_t max_bytes = 0;
  for (auto s : bw_sizes) max_bytes = std::max(max_bytes, s);
  for (auto s : lat_sizes) max_bytes = std::max(max_bytes, s);
  if (max_bytes == 0) {
    fprintf(stderr, "No sizes provided.\n");
    return 2;
  }

  void* d_buf = nullptr;
  CHECK_CUDA(cudaMalloc(&d_buf, max_bytes));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  std::vector<CopyResult> records;
  std::vector<HostMemType> mem_types = {HostMemType::kPageable, HostMemType::kPinned, HostMemType::kManaged};

  for (HostMemType hmt : mem_types) {
    HostBuf host;
    int rc = alloc_host(hmt, max_bytes, &host);
    if (rc != 0) {
      fprintf(stderr, "Host alloc failed for %s\n", host_mem_name(hmt));
      cudaStreamDestroy(stream);
      cudaFree(d_buf);
      return rc;
    }

    // Bandwidth sweeps.
    for (size_t sz : bw_sizes) {
      auto [gbps_h2d, _lat_us_h2d] = time_memcpy(stream, cudaMemcpyHostToDevice, d_buf, host.ptr, sz, warmup, bw_iters);
      records.push_back(CopyResult{.test = "bw",
                                   .direction = "h2d",
                                   .host_mem = hmt,
                                   .size_bytes = sz,
                                   .iters = bw_iters,
                                   .bw_gbps = gbps_h2d,
                                   .lat_us = 0.0});

      auto [gbps_d2h, _lat_us_d2h] = time_memcpy(stream, cudaMemcpyDeviceToHost, host.ptr, d_buf, sz, warmup, bw_iters);
      records.push_back(CopyResult{.test = "bw",
                                   .direction = "d2h",
                                   .host_mem = hmt,
                                   .size_bytes = sz,
                                   .iters = bw_iters,
                                   .bw_gbps = gbps_d2h,
                                   .lat_us = 0.0});
    }

    // Latency sweeps.
    for (size_t sz : lat_sizes) {
      auto [_gbps_h2d, lat_us_h2d] = time_memcpy(stream, cudaMemcpyHostToDevice, d_buf, host.ptr, sz, warmup, lat_iters);
      records.push_back(CopyResult{.test = "lat",
                                   .direction = "h2d",
                                   .host_mem = hmt,
                                   .size_bytes = sz,
                                   .iters = lat_iters,
                                   .bw_gbps = 0.0,
                                   .lat_us = lat_us_h2d});

      auto [_gbps_d2h, lat_us_d2h] = time_memcpy(stream, cudaMemcpyDeviceToHost, host.ptr, d_buf, sz, warmup, lat_iters);
      records.push_back(CopyResult{.test = "lat",
                                   .direction = "d2h",
                                   .host_mem = hmt,
                                   .size_bytes = sz,
                                   .iters = lat_iters,
                                   .bw_gbps = 0.0,
                                   .lat_us = lat_us_d2h});
    }

    free_host(host);
  }

  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(d_buf));

  std::ostringstream oss;
  oss << "{";
  oss << "\"run_id\":\"" << json_escape(run_id) << "\",";
  oss << "\"label\":\"" << json_escape(label) << "\",";
  oss << "\"device\":" << device << ",";
  oss << "\"device_name\":\"" << json_escape(dev_name) << "\",";
  oss << "\"bw_sizes_bytes\":[";
  for (size_t i = 0; i < bw_sizes.size(); i++) {
    if (i) oss << ",";
    oss << static_cast<unsigned long long>(bw_sizes[i]);
  }
  oss << "],";
  oss << "\"lat_sizes_bytes\":[";
  for (size_t i = 0; i < lat_sizes.size(); i++) {
    if (i) oss << ",";
    oss << static_cast<unsigned long long>(lat_sizes[i]);
  }
  oss << "],";
  oss << "\"bw_iters\":" << bw_iters << ",";
  oss << "\"lat_iters\":" << lat_iters << ",";
  oss << "\"warmup\":" << warmup << ",";
  oss << "\"records\":[";
  for (size_t i = 0; i < records.size(); i++) {
    const auto& r = records[i];
    if (i) oss << ",";
    oss << "{";
    oss << "\"test\":\"" << r.test << "\",";
    oss << "\"direction\":\"" << r.direction << "\",";
    oss << "\"host_mem\":\"" << host_mem_name(r.host_mem) << "\",";
    oss << "\"size_bytes\":" << static_cast<unsigned long long>(r.size_bytes) << ",";
    oss << "\"iters\":" << r.iters << ",";
    if (r.test == "bw") {
      oss << "\"bw_gbps\":" << r.bw_gbps;
    } else {
      oss << "\"lat_us\":" << r.lat_us;
    }
    oss << "}";
  }
  oss << "]";
  oss << "}\n";

  std::fwrite(oss.str().data(), 1, oss.str().size(), stdout);
  return 0;
}
