# Cluster Health Suite Summary Comparison

- Baseline: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_152206_cluster_health_torchdist_only_variance_ubuntu_r1_base_node1node2_cluster_health_suite_summary.json`
- Candidate: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-07_152206_cluster_health_torchdist_only_variance_ubuntu_r2_base_node1node2_cluster_health_suite_summary.json`
- Threshold: 5.00%

## Metrics

| Metric | Baseline | Candidate | Abs diff | Pct diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `torchdist.max_busbw_gbps` | 460.116 | 333.747 | -126.369 | -27.46% | REGRESSION |
