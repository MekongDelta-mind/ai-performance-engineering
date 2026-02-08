# Cluster Health Suite Summary Comparison

- Baseline: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r2_base_node1node2_cluster_health_suite_summary.json`
- Candidate: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r3_base_node1node2_cluster_health_suite_summary.json`
- Threshold: 5.00%

## Metrics

| Metric | Baseline | Candidate | Abs diff | Pct diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `ib_write_bw.mlx5_0.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_1.avg_gbps` | 387.140 | 387.130 | -0.010 | -0.00% | OK |
| `ib_write_bw.mlx5_4.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_5.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `iperf3.fwd.gbps` | 7.819 | 7.661 | -0.158 | -2.02% | OK |
| `iperf3.rev.gbps` | 7.619 | 7.689 | 0.069 | 0.91% | OK |
| `nccl.all_gather_perf.max_busbw_gbps` | 460.800 | 441.430 | -19.370 | -4.20% | OK |
| `nccl.all_reduce_perf.max_busbw_gbps` | 526.810 | 535.910 | 9.100 | 1.73% | OK |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 575.840 | 577.840 | 2.000 | 0.35% | OK |
| `torchdist.max_busbw_gbps` | 712.114 | 421.160 | -290.954 | -40.86% | REGRESSION |
