#!/usr/bin/env python3
import csv
import math
from collections import defaultdict
import matplotlib.pyplot as plt

CSV_PATH = '/tmp/p2p_test/p2p_sweep.csv'

rows = []
with open(CSV_PATH) as f:
    r = csv.DictReader(f)
    for row in r:
        row['size_bytes'] = int(row['size_bytes'])
        row['src'] = int(row['src'])
        row['dst'] = int(row['dst'])
        row['bw_gbs'] = float(row['bw_gbs'])
        row['lat_us'] = float(row['lat_us'])
        rows.append(row)

# Bandwidth curves
bw_rows = [r for r in rows if r['test_type'] == 'bw']
by_pair = defaultdict(list)
by_size = defaultdict(list)
for r in bw_rows:
    key = f"{r['src']}->{r['dst']}"
    by_pair[key].append((r['size_bytes'], r['bw_gbs']))
    by_size[r['size_bytes']].append(r['bw_gbs'])

plt.figure(figsize=(7,5))
for pair, pts in sorted(by_pair.items()):
    pts = sorted(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    plt.plot(xs, ys, alpha=0.6, linewidth=1, label=pair)

# Average curve
sizes = sorted(by_size.keys())
avg = [sum(by_size[s]) / len(by_size[s]) for s in sizes]
plt.plot(sizes, avg, color='black', linewidth=2.5, label='avg')

plt.xscale('log', base=2)
plt.xlabel('Message size (bytes)')
plt.ylabel('Bandwidth (GB/s)')
plt.title('P2P Bandwidth vs Size')
plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.5)
plt.legend(ncol=3, fontsize=7)
plt.tight_layout()
plt.savefig('/tmp/p2p_test/p2p_bw_curve.png', dpi=150)

# Latency curves
lat_rows = [r for r in rows if r['test_type'] == 'lat']
by_pair = defaultdict(list)
by_size = defaultdict(list)
for r in lat_rows:
    key = f"{r['src']}->{r['dst']}"
    by_pair[key].append((r['size_bytes'], r['lat_us']))
    by_size[r['size_bytes']].append(r['lat_us'])

plt.figure(figsize=(7,5))
for pair, pts in sorted(by_pair.items()):
    pts = sorted(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    plt.plot(xs, ys, alpha=0.6, linewidth=1, label=pair)

sizes = sorted(by_size.keys())
avg = [sum(by_size[s]) / len(by_size[s]) for s in sizes]
plt.plot(sizes, avg, color='black', linewidth=2.5, label='avg')

plt.xscale('log', base=2)
plt.xlabel('Message size (bytes)')
plt.ylabel('Latency (us)')
plt.title('P2P Latency vs Size')
plt.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.5)
plt.legend(ncol=3, fontsize=7)
plt.tight_layout()
plt.savefig('/tmp/p2p_test/p2p_lat_curve.png', dpi=150)

print('Wrote /tmp/p2p_test/p2p_bw_curve.png')
print('Wrote /tmp/p2p_test/p2p_lat_curve.png')
