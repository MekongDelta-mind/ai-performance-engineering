# Cluster Health Suite Summary Comparison

- Baseline: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r1_extended_node1node2_cluster_health_suite_summary.json`
- Candidate: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r3_extended_node1node2_cluster_health_suite_summary.json`
- Threshold: 5.00%

## Metrics

| Metric | Baseline | Candidate | Abs diff | Pct diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `ib_read_bw.mlx5_0.avg_gbps` | 388.610 | 388.610 | 0.000 | 0.00% | OK |
| `ib_read_bw.mlx5_1.avg_gbps` | 388.530 | 388.530 | 0.000 | 0.00% | OK |
| `ib_read_bw.mlx5_4.avg_gbps` | 388.600 | 388.610 | 0.010 | 0.00% | OK |
| `ib_read_bw.mlx5_5.avg_gbps` | 388.620 | 388.610 | -0.010 | -0.00% | OK |
| `ib_send_bw.mlx5_0.avg_gbps` | 388.590 | 388.600 | 0.010 | 0.00% | OK |
| `ib_send_bw.mlx5_1.avg_gbps` | 388.520 | 388.520 | 0.000 | 0.00% | OK |
| `ib_send_bw.mlx5_4.avg_gbps` | 388.590 | 388.510 | -0.080 | -0.02% | OK |
| `ib_send_bw.mlx5_5.avg_gbps` | 388.600 | 388.600 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_0.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_1.avg_gbps` | 387.140 | 387.130 | -0.010 | -0.00% | OK |
| `ib_write_bw.mlx5_4.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_5.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `iperf3.fwd.gbps` | 7.898 | 7.617 | -0.281 | -3.56% | OK |
| `iperf3.rev.gbps` | 7.989 | 7.561 | -0.428 | -5.35% | REGRESSION |
| `nccl.all_gather_perf.max_busbw_gbps` | 442.320 | 436.040 | -6.280 | -1.42% | OK |
| `nccl.all_reduce_perf.max_busbw_gbps` | 593.500 | 652.910 | 59.410 | 10.01% | IMPROVEMENT |
| `nccl.alltoall_perf.max_busbw_gbps` | 372.630 | 347.870 | -24.760 | -6.64% | REGRESSION |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 575.490 | 598.070 | 22.580 | 3.92% | OK |
| `torchdist.max_busbw_gbps` | 397.157 | 460.830 | 63.672 | 16.03% | IMPROVEMENT |
