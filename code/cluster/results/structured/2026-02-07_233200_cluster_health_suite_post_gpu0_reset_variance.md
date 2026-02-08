# Cluster Health Suite Variance Summary

## Runs

- `2026-02-07_233200_cluster_health_suite_post_gpu0_reset_r1_base` (base): `results/structured/2026-02-07_233200_cluster_health_suite_post_gpu0_reset_r1_base_node1node2_cluster_health_suite_summary.json`
- `2026-02-07_233200_cluster_health_suite_post_gpu0_reset_r2_base` (base): `results/structured/2026-02-07_233200_cluster_health_suite_post_gpu0_reset_r2_base_node1node2_cluster_health_suite_summary.json`
- `2026-02-07_233200_cluster_health_suite_post_gpu0_reset_r3_base` (base): `results/structured/2026-02-07_233200_cluster_health_suite_post_gpu0_reset_r3_base_node1node2_cluster_health_suite_summary.json`
- `2026-02-07_233200_cluster_health_suite_post_gpu0_reset_r1_extended` (extended): `results/structured/2026-02-07_233200_cluster_health_suite_post_gpu0_reset_r1_extended_node1node2_cluster_health_suite_summary.json`
- `2026-02-07_233200_cluster_health_suite_post_gpu0_reset_r2_extended` (extended): `results/structured/2026-02-07_233200_cluster_health_suite_post_gpu0_reset_r2_extended_node1node2_cluster_health_suite_summary.json`
- `2026-02-07_233200_cluster_health_suite_post_gpu0_reset_r3_extended` (extended): `results/structured/2026-02-07_233200_cluster_health_suite_post_gpu0_reset_r3_extended_node1node2_cluster_health_suite_summary.json`

## Stats (base)

| Metric | N | Mean | Stddev | Min | Max | CV% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ib_write_bw.mean_avg_gbps` | 3 | 387.135 | 0.004 | 387.130 | 387.137 | 0.00 |
| `ib_write_bw.mlx5_0.avg_gbps` | 3 | 387.133 | 0.005 | 387.130 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_1.avg_gbps` | 3 | 387.133 | 0.005 | 387.130 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_4.avg_gbps` | 3 | 387.137 | 0.005 | 387.130 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_5.avg_gbps` | 3 | 387.137 | 0.005 | 387.130 | 387.140 | 0.00 |
| `iperf3.fwd.gbps` | 3 | 7.680 | 0.133 | 7.538 | 7.857 | 1.73 |
| `iperf3.rev.gbps` | 3 | 7.809 | 0.179 | 7.556 | 7.936 | 2.29 |
| `nccl.all_gather_perf.max_busbw_gbps` | 3 | 655.220 | 0.815 | 654.130 | 656.090 | 0.12 |
| `nccl.all_reduce_perf.max_busbw_gbps` | 3 | 839.957 | 0.433 | 839.350 | 840.330 | 0.05 |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 3 | 676.140 | 0.346 | 675.690 | 676.530 | 0.05 |
| `torchdist.max_busbw_gbps` | 3 | 718.708 | 1.743 | 716.275 | 720.264 | 0.24 |

## Stats (extended)

| Metric | N | Mean | Stddev | Min | Max | CV% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ib_read_bw.mean_avg_gbps` | 3 | 388.559 | 0.022 | 388.535 | 388.587 | 0.01 |
| `ib_read_bw.mlx5_0.avg_gbps` | 3 | 388.583 | 0.038 | 388.530 | 388.610 | 0.01 |
| `ib_read_bw.mlx5_1.avg_gbps` | 3 | 388.553 | 0.033 | 388.530 | 388.600 | 0.01 |
| `ib_read_bw.mlx5_4.avg_gbps` | 3 | 388.563 | 0.033 | 388.540 | 388.610 | 0.01 |
| `ib_read_bw.mlx5_5.avg_gbps` | 3 | 388.537 | 0.005 | 388.530 | 388.540 | 0.00 |
| `ib_send_bw.mean_avg_gbps` | 3 | 388.534 | 0.018 | 388.520 | 388.560 | 0.00 |
| `ib_send_bw.mlx5_0.avg_gbps` | 3 | 388.543 | 0.040 | 388.510 | 388.600 | 0.01 |
| `ib_send_bw.mlx5_1.avg_gbps` | 3 | 388.543 | 0.033 | 388.520 | 388.590 | 0.01 |
| `ib_send_bw.mlx5_4.avg_gbps` | 3 | 388.527 | 0.005 | 388.520 | 388.530 | 0.00 |
| `ib_send_bw.mlx5_5.avg_gbps` | 3 | 388.523 | 0.005 | 388.520 | 388.530 | 0.00 |
| `ib_write_bw.mean_avg_gbps` | 3 | 387.133 | 0.001 | 387.132 | 387.135 | 0.00 |
| `ib_write_bw.mlx5_0.avg_gbps` | 3 | 387.130 | 0.000 | 387.130 | 387.130 | 0.00 |
| `ib_write_bw.mlx5_1.avg_gbps` | 3 | 387.130 | 0.000 | 387.130 | 387.130 | 0.00 |
| `ib_write_bw.mlx5_4.avg_gbps` | 3 | 387.133 | 0.005 | 387.130 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_5.avg_gbps` | 3 | 387.140 | 0.000 | 387.140 | 387.140 | 0.00 |
| `iperf3.fwd.gbps` | 3 | 7.578 | 0.098 | 7.493 | 7.715 | 1.29 |
| `iperf3.rev.gbps` | 3 | 7.706 | 0.150 | 7.599 | 7.918 | 1.94 |
| `nccl.all_gather_perf.max_busbw_gbps` | 3 | 655.233 | 0.560 | 654.760 | 656.020 | 0.09 |
| `nccl.all_reduce_perf.max_busbw_gbps` | 3 | 839.280 | 0.569 | 838.520 | 839.890 | 0.07 |
| `nccl.alltoall_perf.max_busbw_gbps` | 3 | 604.070 | 0.145 | 603.870 | 604.210 | 0.02 |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 3 | 676.137 | 0.094 | 676.020 | 676.250 | 0.01 |
| `torchdist.max_busbw_gbps` | 3 | 715.674 | 1.851 | 713.759 | 718.176 | 0.26 |

