# Cluster Health Suite Summary Comparison

- Baseline: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_151437_cluster_health_nccl_only_variance_ubuntu_r2_base_node1node2_cluster_health_suite_summary.json`
- Candidate: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_151437_cluster_health_nccl_only_variance_ubuntu_r3_base_node1node2_cluster_health_suite_summary.json`
- Threshold: 5.00%

## Metrics

| Metric | Baseline | Candidate | Abs diff | Pct diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `nccl.all_gather_perf.max_busbw_gbps` | 427.520 | 427.720 | 0.200 | 0.05% | OK |
| `nccl.all_reduce_perf.max_busbw_gbps` | 548.320 | 554.390 | 6.070 | 1.11% | OK |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 681.470 | 576.460 | -105.010 | -15.41% | REGRESSION |
