# Cluster Health Suite Summary Comparison

- Baseline: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r1_base_node1node2_cluster_health_suite_summary.json`
- Candidate: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r2_base_node1node2_cluster_health_suite_summary.json`
- Threshold: 5.00%

## Metrics

| Metric | Baseline | Candidate | Abs diff | Pct diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `ib_write_bw.mlx5_0.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_1.avg_gbps` | 387.130 | 387.140 | 0.010 | 0.00% | OK |
| `ib_write_bw.mlx5_4.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_5.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `iperf3.fwd.gbps` | 7.877 | 7.819 | -0.058 | -0.74% | OK |
| `iperf3.rev.gbps` | 7.994 | 7.619 | -0.375 | -4.69% | OK |
| `nccl.all_gather_perf.max_busbw_gbps` | 664.180 | 460.800 | -203.380 | -30.62% | REGRESSION |
| `nccl.all_reduce_perf.max_busbw_gbps` | 842.000 | 526.810 | -315.190 | -37.43% | REGRESSION |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 680.160 | 575.840 | -104.320 | -15.34% | REGRESSION |
| `torchdist.max_busbw_gbps` | 553.358 | 712.114 | 158.756 | 28.69% | IMPROVEMENT |
