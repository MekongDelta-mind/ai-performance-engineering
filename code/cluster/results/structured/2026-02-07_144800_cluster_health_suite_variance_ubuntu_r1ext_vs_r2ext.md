# Cluster Health Suite Summary Comparison

- Baseline: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r1_extended_node1node2_cluster_health_suite_summary.json`
- Candidate: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r2_extended_node1node2_cluster_health_suite_summary.json`
- Threshold: 5.00%

## Metrics

| Metric | Baseline | Candidate | Abs diff | Pct diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `ib_read_bw.mlx5_0.avg_gbps` | 388.610 | 388.610 | 0.000 | 0.00% | OK |
| `ib_read_bw.mlx5_1.avg_gbps` | 388.530 | 388.600 | 0.070 | 0.02% | OK |
| `ib_read_bw.mlx5_4.avg_gbps` | 388.600 | 388.610 | 0.010 | 0.00% | OK |
| `ib_read_bw.mlx5_5.avg_gbps` | 388.620 | 388.610 | -0.010 | -0.00% | OK |
| `ib_send_bw.mlx5_0.avg_gbps` | 388.590 | 388.520 | -0.070 | -0.02% | OK |
| `ib_send_bw.mlx5_1.avg_gbps` | 388.520 | 388.600 | 0.080 | 0.02% | OK |
| `ib_send_bw.mlx5_4.avg_gbps` | 388.590 | 388.530 | -0.060 | -0.02% | OK |
| `ib_send_bw.mlx5_5.avg_gbps` | 388.600 | 388.590 | -0.010 | -0.00% | OK |
| `ib_write_bw.mlx5_0.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_1.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_4.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_5.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `iperf3.fwd.gbps` | 7.898 | 7.580 | -0.318 | -4.02% | OK |
| `iperf3.rev.gbps` | 7.989 | 7.659 | -0.330 | -4.13% | OK |
| `nccl.all_gather_perf.max_busbw_gbps` | 442.320 | 459.870 | 17.550 | 3.97% | OK |
| `nccl.all_reduce_perf.max_busbw_gbps` | 593.500 | 528.700 | -64.800 | -10.92% | REGRESSION |
| `nccl.alltoall_perf.max_busbw_gbps` | 372.630 | 372.690 | 0.060 | 0.02% | OK |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 575.490 | 680.630 | 105.140 | 18.27% | IMPROVEMENT |
| `torchdist.max_busbw_gbps` | 397.157 | 721.751 | 324.594 | 81.73% | IMPROVEMENT |
