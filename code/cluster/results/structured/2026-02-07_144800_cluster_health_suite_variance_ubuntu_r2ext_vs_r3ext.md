# Cluster Health Suite Summary Comparison

- Baseline: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r2_extended_node1node2_cluster_health_suite_summary.json`
- Candidate: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r3_extended_node1node2_cluster_health_suite_summary.json`
- Threshold: 5.00%

## Metrics

| Metric | Baseline | Candidate | Abs diff | Pct diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `ib_read_bw.mlx5_0.avg_gbps` | 388.610 | 388.610 | 0.000 | 0.00% | OK |
| `ib_read_bw.mlx5_1.avg_gbps` | 388.600 | 388.530 | -0.070 | -0.02% | OK |
| `ib_read_bw.mlx5_4.avg_gbps` | 388.610 | 388.610 | 0.000 | 0.00% | OK |
| `ib_read_bw.mlx5_5.avg_gbps` | 388.610 | 388.610 | 0.000 | 0.00% | OK |
| `ib_send_bw.mlx5_0.avg_gbps` | 388.520 | 388.600 | 0.080 | 0.02% | OK |
| `ib_send_bw.mlx5_1.avg_gbps` | 388.600 | 388.520 | -0.080 | -0.02% | OK |
| `ib_send_bw.mlx5_4.avg_gbps` | 388.530 | 388.510 | -0.020 | -0.01% | OK |
| `ib_send_bw.mlx5_5.avg_gbps` | 388.590 | 388.600 | 0.010 | 0.00% | OK |
| `ib_write_bw.mlx5_0.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_1.avg_gbps` | 387.140 | 387.130 | -0.010 | -0.00% | OK |
| `ib_write_bw.mlx5_4.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_5.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `iperf3.fwd.gbps` | 7.580 | 7.617 | 0.037 | 0.48% | OK |
| `iperf3.rev.gbps` | 7.659 | 7.561 | -0.098 | -1.28% | OK |
| `nccl.all_gather_perf.max_busbw_gbps` | 459.870 | 436.040 | -23.830 | -5.18% | REGRESSION |
| `nccl.all_reduce_perf.max_busbw_gbps` | 528.700 | 652.910 | 124.210 | 23.49% | IMPROVEMENT |
| `nccl.alltoall_perf.max_busbw_gbps` | 372.690 | 347.870 | -24.820 | -6.66% | REGRESSION |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 680.630 | 598.070 | -82.560 | -12.13% | REGRESSION |
| `torchdist.max_busbw_gbps` | 721.751 | 460.830 | -260.922 | -36.15% | REGRESSION |
