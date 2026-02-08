# Cluster Health Suite Summary Comparison

- Baseline: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_151437_cluster_health_nccl_only_variance_ubuntu_r1_base_node1node2_cluster_health_suite_summary.json`
- Candidate: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_151437_cluster_health_nccl_only_variance_ubuntu_r3_base_node1node2_cluster_health_suite_summary.json`
- Threshold: 5.00%

## Metrics

| Metric | Baseline | Candidate | Abs diff | Pct diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `nccl.all_gather_perf.max_busbw_gbps` | 566.630 | 427.720 | -138.910 | -24.52% | REGRESSION |
| `nccl.all_reduce_perf.max_busbw_gbps` | 523.750 | 554.390 | 30.640 | 5.85% | IMPROVEMENT |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 575.870 | 576.460 | 0.590 | 0.10% | OK |
