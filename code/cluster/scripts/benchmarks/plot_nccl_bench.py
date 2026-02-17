#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt

CSV_PATH = '/tmp/p2p_test/nccl_allreduce.csv'

sizes = []
alg = []
bus = []

with open(CSV_PATH) as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        size, time_ms, alg_bw, bus_bw = line.strip().split(',')
        sizes.append(int(size))
        alg.append(float(alg_bw))
        bus.append(float(bus_bw))

plt.figure(figsize=(7,5))
plt.plot(sizes, alg, marker='o', label='Alg BW (GB/s)')
plt.plot(sizes, bus, marker='o', label='Bus BW (GB/s)')
plt.xscale('log', base=2)
plt.xlabel('Message size (bytes)')
plt.ylabel('Bandwidth (GB/s)')
plt.title('NCCL AllReduce Bandwidth vs Size')
plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('/tmp/p2p_test/nccl_allreduce_curve.png', dpi=150)

print('Wrote /tmp/p2p_test/nccl_allreduce_curve.png')
