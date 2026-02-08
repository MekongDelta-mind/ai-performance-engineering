# Cluster Health Suite Variance Summary

## Runs

- `2026-02-08_035722_health_suite_variance_post_eval_full_fixed_r1_base` (base): `results/structured/2026-02-08_035722_health_suite_variance_post_eval_full_fixed_r1_base_node1node2_cluster_health_suite_summary.json`
- `2026-02-08_035722_health_suite_variance_post_eval_full_fixed_r2_base` (base): `results/structured/2026-02-08_035722_health_suite_variance_post_eval_full_fixed_r2_base_node1node2_cluster_health_suite_summary.json`
- `2026-02-08_035722_health_suite_variance_post_eval_full_fixed_r3_base` (base): `results/structured/2026-02-08_035722_health_suite_variance_post_eval_full_fixed_r3_base_node1node2_cluster_health_suite_summary.json`
- `2026-02-08_035722_health_suite_variance_post_eval_full_fixed_r1_extended` (extended): `results/structured/2026-02-08_035722_health_suite_variance_post_eval_full_fixed_r1_extended_node1node2_cluster_health_suite_summary.json`
- `2026-02-08_035722_health_suite_variance_post_eval_full_fixed_r2_extended` (extended): `results/structured/2026-02-08_035722_health_suite_variance_post_eval_full_fixed_r2_extended_node1node2_cluster_health_suite_summary.json`
- `2026-02-08_035722_health_suite_variance_post_eval_full_fixed_r3_extended` (extended): `results/structured/2026-02-08_035722_health_suite_variance_post_eval_full_fixed_r3_extended_node1node2_cluster_health_suite_summary.json`

## Stats (base)

| Metric | N | Mean | Stddev | Min | Max | CV% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ib_write_bw.mean_avg_gbps` | 3 | 387.131 | 0.004 | 387.125 | 387.135 | 0.00 |
| `ib_write_bw.mlx5_0.avg_gbps` | 3 | 387.137 | 0.005 | 387.130 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_1.avg_gbps` | 3 | 387.127 | 0.012 | 387.110 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_4.avg_gbps` | 3 | 387.130 | 0.000 | 387.130 | 387.130 | 0.00 |
| `ib_write_bw.mlx5_5.avg_gbps` | 3 | 387.130 | 0.000 | 387.130 | 387.130 | 0.00 |
| `iperf3.fwd.gbps` | 3 | 7.371 | 0.065 | 7.286 | 7.444 | 0.88 |
| `iperf3.rev.gbps` | 3 | 7.518 | 0.084 | 7.399 | 7.583 | 1.12 |
| `nccl.all_gather_perf.max_busbw_gbps` | 3 | 655.660 | 0.156 | 655.550 | 655.880 | 0.02 |
| `nccl.all_reduce_perf.max_busbw_gbps` | 3 | 839.207 | 0.829 | 838.410 | 840.350 | 0.10 |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 3 | 675.963 | 0.478 | 675.530 | 676.630 | 0.07 |
| `torchdist.max_busbw_gbps` | 3 | 717.089 | 1.215 | 715.704 | 718.662 | 0.17 |

## Stats (extended)

| Metric | N | Mean | Stddev | Min | Max | CV% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ib_read_bw.mean_avg_gbps` | 3 | 388.575 | 0.007 | 388.567 | 388.585 | 0.00 |
| `ib_read_bw.mlx5_0.avg_gbps` | 3 | 388.587 | 0.033 | 388.540 | 388.610 | 0.01 |
| `ib_read_bw.mlx5_1.avg_gbps` | 3 | 388.607 | 0.005 | 388.600 | 388.610 | 0.00 |
| `ib_read_bw.mlx5_4.avg_gbps` | 3 | 388.550 | 0.036 | 388.520 | 388.600 | 0.01 |
| `ib_read_bw.mlx5_5.avg_gbps` | 3 | 388.557 | 0.038 | 388.530 | 388.610 | 0.01 |
| `ib_send_bw.mean_avg_gbps` | 3 | 388.546 | 0.007 | 388.540 | 388.555 | 0.00 |
| `ib_send_bw.mlx5_0.avg_gbps` | 3 | 388.573 | 0.031 | 388.530 | 388.600 | 0.01 |
| `ib_send_bw.mlx5_1.avg_gbps` | 3 | 388.547 | 0.031 | 388.520 | 388.590 | 0.01 |
| `ib_send_bw.mlx5_4.avg_gbps` | 3 | 388.520 | 0.000 | 388.520 | 388.520 | 0.00 |
| `ib_send_bw.mlx5_5.avg_gbps` | 3 | 388.543 | 0.033 | 388.520 | 388.590 | 0.01 |
| `ib_write_bw.mean_avg_gbps` | 3 | 387.133 | 0.002 | 387.130 | 387.135 | 0.00 |
| `ib_write_bw.mlx5_0.avg_gbps` | 3 | 387.137 | 0.005 | 387.130 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_1.avg_gbps` | 3 | 387.137 | 0.005 | 387.130 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_4.avg_gbps` | 3 | 387.130 | 0.000 | 387.130 | 387.130 | 0.00 |
| `ib_write_bw.mlx5_5.avg_gbps` | 3 | 387.130 | 0.000 | 387.130 | 387.130 | 0.00 |
| `iperf3.fwd.gbps` | 3 | 7.503 | 0.134 | 7.341 | 7.668 | 1.78 |
| `iperf3.rev.gbps` | 3 | 7.574 | 0.221 | 7.302 | 7.843 | 2.92 |
| `nccl.all_gather_perf.max_busbw_gbps` | 3 | 655.423 | 0.148 | 655.290 | 655.630 | 0.02 |
| `nccl.all_reduce_perf.max_busbw_gbps` | 3 | 839.280 | 0.418 | 838.890 | 839.860 | 0.05 |
| `nccl.alltoall_perf.max_busbw_gbps` | 3 | 604.627 | 0.209 | 604.450 | 604.920 | 0.03 |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 3 | 676.353 | 0.588 | 675.640 | 677.080 | 0.09 |
| `torchdist.max_busbw_gbps` | 3 | 714.293 | 3.094 | 709.989 | 717.127 | 0.43 |

