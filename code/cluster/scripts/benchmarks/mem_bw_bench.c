#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t parse_u64(const char* s, const char* name) {
  if (!s || !*s) {
    fprintf(stderr, "Missing %s\n", name);
    exit(2);
  }
  errno = 0;
  char* end = NULL;
  unsigned long long v = strtoull(s, &end, 10);
  if (errno != 0 || end == NULL || *end != '\0') {
    fprintf(stderr, "Invalid %s: '%s'\n", name, s);
    exit(2);
  }
  return (uint64_t)v;
}

typedef struct {
  const uint8_t* src;
  uint8_t* dst;
  uint64_t off;
  uint64_t nbytes;
  uint64_t iters;
} thread_args_t;

static void* worker(void* arg) {
  thread_args_t* a = (thread_args_t*)arg;
  const uint8_t* src = a->src + a->off;
  uint8_t* dst = a->dst + a->off;
  for (uint64_t it = 0; it < a->iters; it++) {
    memcpy(dst, src, (size_t)a->nbytes);
  }
  return NULL;
}

int main(int argc, char** argv) {
  uint64_t bytes = 1024ull * 1024ull * 1024ull;
  uint64_t iters = 10;
  uint64_t threads = 16;
  uint64_t warmup = 2;

  for (int i = 1; i < argc; i++) {
    const char* a = argv[i];
    if (strcmp(a, "--bytes") == 0) {
      bytes = parse_u64(argv[++i], "--bytes");
    } else if (strcmp(a, "--iters") == 0) {
      iters = parse_u64(argv[++i], "--iters");
    } else if (strcmp(a, "--threads") == 0) {
      threads = parse_u64(argv[++i], "--threads");
    } else if (strcmp(a, "--warmup") == 0) {
      warmup = parse_u64(argv[++i], "--warmup");
    } else if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
      fprintf(stderr,
              "Usage: mem_bw_bench [--bytes N] [--iters N] [--threads N] [--warmup N]\n"
              "Prints a JSON object with memcpy throughput.\n");
      return 0;
    } else {
      fprintf(stderr, "Unknown arg: %s\n", a);
      return 2;
    }
  }

  if (threads == 0) threads = 1;
  if (bytes == 0) bytes = 1;

  // Cap threads to something reasonable by default.
  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncpu > 0 && threads > (uint64_t)ncpu) threads = (uint64_t)ncpu;

  void* src = NULL;
  void* dst = NULL;
  int rc1 = posix_memalign(&src, 4096, (size_t)bytes);
  int rc2 = posix_memalign(&dst, 4096, (size_t)bytes);
  if (rc1 != 0 || rc2 != 0 || src == NULL || dst == NULL) {
    fprintf(stderr, "posix_memalign failed rc1=%d rc2=%d\n", rc1, rc2);
    return 1;
  }

  memset(src, 0xA5, (size_t)bytes);
  memset(dst, 0x5A, (size_t)bytes);

  pthread_t* tids = (pthread_t*)calloc((size_t)threads, sizeof(pthread_t));
  thread_args_t* args = (thread_args_t*)calloc((size_t)threads, sizeof(thread_args_t));
  if (!tids || !args) {
    fprintf(stderr, "calloc failed\n");
    return 1;
  }

  uint64_t chunk = bytes / threads;
  // Keep all bytes covered; last thread takes remainder.
  for (uint64_t t = 0; t < threads; t++) {
    uint64_t off = t * chunk;
    uint64_t n = (t == threads - 1) ? (bytes - off) : chunk;
    args[t] = (thread_args_t){
        .src = (const uint8_t*)src,
        .dst = (uint8_t*)dst,
        .off = off,
        .nbytes = n,
        .iters = iters,
    };
  }

  // Warm up to fault pages and stabilize CPU frequency.
  for (uint64_t w = 0; w < warmup; w++) {
    for (uint64_t t = 0; t < threads; t++) {
      pthread_create(&tids[t], NULL, worker, &args[t]);
    }
    for (uint64_t t = 0; t < threads; t++) {
      pthread_join(tids[t], NULL);
    }
  }

  double t0 = now_s();
  for (uint64_t t = 0; t < threads; t++) {
    pthread_create(&tids[t], NULL, worker, &args[t]);
  }
  for (uint64_t t = 0; t < threads; t++) {
    pthread_join(tids[t], NULL);
  }
  double t1 = now_s();
  double elapsed = t1 - t0;

  double total_bytes = (double)bytes * (double)iters;
  double gbps = (elapsed > 0.0) ? (total_bytes / elapsed / 1e9) : 0.0;

  printf("{\"bytes\":%" PRIu64 ",\"iters\":%" PRIu64 ",\"threads\":%" PRIu64 ",\"warmup\":%" PRIu64
         ",\"elapsed_s\":%.6f,\"bw_gbps\":%.6f}\n",
         bytes,
         iters,
         threads,
         warmup,
         elapsed,
         gbps);

  free(tids);
  free(args);
  free(src);
  free(dst);
  return 0;
}

