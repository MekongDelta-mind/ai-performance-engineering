# Cluster Health Suite Summary Comparison

- Baseline: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r1_base_node1node2_cluster_health_suite_summary.json`
- Candidate: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r3_base_node1node2_cluster_health_suite_summary.json`
- Threshold: 5.00%

## Metrics

| Metric | Baseline | Candidate | Abs diff | Pct diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `ib_write_bw.mlx5_0.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_1.avg_gbps` | 387.130 | 387.130 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_4.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_5.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `iperf3.fwd.gbps` | 7.877 | 7.661 | -0.216 | -2.75% | OK |
| `iperf3.rev.gbps` | 7.994 | 7.689 | -0.306 | -3.82% | OK |
| `nccl.all_gather_perf.max_busbw_gbps` | 664.180 | 441.430 | -222.750 | -33.54% | REGRESSION |
| `nccl.all_reduce_perf.max_busbw_gbps` | 842.000 | 535.910 | -306.090 | -36.35% | REGRESSION |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 680.160 | 577.840 | -102.320 | -15.04% | REGRESSION |
| `torchdist.max_busbw_gbps` | 553.358 | 421.160 | -132.198 | -23.89% | REGRESSION |
