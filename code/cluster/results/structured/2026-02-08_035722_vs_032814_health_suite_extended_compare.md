# Cluster Health Suite Summary Comparison

- Baseline: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-08_032814_neocloud_eval_full_fixed_health_suite_extended_node1node2_cluster_health_suite_summary.json`
- Candidate: `/home/ubuntu/ai-performance-engineering/code/cluster/results/structured/2026-02-08_035722_health_suite_variance_post_eval_full_fixed_r3_extended_node1node2_cluster_health_suite_summary.json`
- Threshold: 5.00%

## Metrics

| Metric | Baseline | Candidate | Abs diff | Pct diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `ib_read_bw.mlx5_0.avg_gbps` | 388.610 | 388.610 | 0.000 | 0.00% | OK |
| `ib_read_bw.mlx5_1.avg_gbps` | 388.530 | 388.610 | 0.080 | 0.02% | OK |
| `ib_read_bw.mlx5_4.avg_gbps` | 388.610 | 388.520 | -0.090 | -0.02% | OK |
| `ib_read_bw.mlx5_5.avg_gbps` | 388.530 | 388.530 | 0.000 | 0.00% | OK |
| `ib_send_bw.mlx5_0.avg_gbps` | 388.520 | 388.530 | 0.010 | 0.00% | OK |
| `ib_send_bw.mlx5_1.avg_gbps` | 388.520 | 388.590 | 0.070 | 0.02% | OK |
| `ib_send_bw.mlx5_4.avg_gbps` | 388.520 | 388.520 | 0.000 | 0.00% | OK |
| `ib_send_bw.mlx5_5.avg_gbps` | 388.520 | 388.520 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_0.avg_gbps` | 387.130 | 387.140 | 0.010 | 0.00% | OK |
| `ib_write_bw.mlx5_1.avg_gbps` | 387.140 | 387.140 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_4.avg_gbps` | 387.130 | 387.130 | 0.000 | 0.00% | OK |
| `ib_write_bw.mlx5_5.avg_gbps` | 387.130 | 387.130 | 0.000 | 0.00% | OK |
| `iperf3.fwd.gbps` | 7.620 | 7.499 | -0.121 | -1.59% | OK |
| `iperf3.rev.gbps` | 7.854 | 7.577 | -0.277 | -3.52% | OK |
| `nccl.all_gather_perf.max_busbw_gbps` | 655.630 | 655.350 | -0.280 | -0.04% | OK |
| `nccl.all_reduce_perf.max_busbw_gbps` | 839.390 | 838.890 | -0.500 | -0.06% | OK |
| `nccl.alltoall_perf.max_busbw_gbps` | 604.060 | 604.920 | 0.860 | 0.14% | OK |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 675.530 | 677.080 | 1.550 | 0.23% | OK |
| `torchdist.max_busbw_gbps` | 717.915 | 715.763 | -2.151 | -0.30% | OK |
