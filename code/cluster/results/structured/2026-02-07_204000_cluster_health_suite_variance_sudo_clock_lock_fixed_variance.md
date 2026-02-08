# Cluster Health Suite Variance Summary

## Runs

- `2026-02-07_204000_cluster_health_suite_variance_sudo_clock_lock_fixed_r1_base` (base): `results/structured/2026-02-07_204000_cluster_health_suite_variance_sudo_clock_lock_fixed_r1_base_node1node2_cluster_health_suite_summary.json`
- `2026-02-07_204000_cluster_health_suite_variance_sudo_clock_lock_fixed_r2_base` (base): `results/structured/2026-02-07_204000_cluster_health_suite_variance_sudo_clock_lock_fixed_r2_base_node1node2_cluster_health_suite_summary.json`
- `2026-02-07_204000_cluster_health_suite_variance_sudo_clock_lock_fixed_r3_base` (base): `results/structured/2026-02-07_204000_cluster_health_suite_variance_sudo_clock_lock_fixed_r3_base_node1node2_cluster_health_suite_summary.json`
- `2026-02-07_204000_cluster_health_suite_variance_sudo_clock_lock_fixed_r1_extended` (extended): `results/structured/2026-02-07_204000_cluster_health_suite_variance_sudo_clock_lock_fixed_r1_extended_node1node2_cluster_health_suite_summary.json`
- `2026-02-07_204000_cluster_health_suite_variance_sudo_clock_lock_fixed_r2_extended` (extended): `results/structured/2026-02-07_204000_cluster_health_suite_variance_sudo_clock_lock_fixed_r2_extended_node1node2_cluster_health_suite_summary.json`
- `2026-02-07_204000_cluster_health_suite_variance_sudo_clock_lock_fixed_r3_extended` (extended): `results/structured/2026-02-07_204000_cluster_health_suite_variance_sudo_clock_lock_fixed_r3_extended_node1node2_cluster_health_suite_summary.json`

## Stats (base)

| Metric | N | Mean | Stddev | Min | Max | CV% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ib_write_bw.mean_avg_gbps` | 3 | 387.130 | 0.000 | 387.130 | 387.130 | 0.00 |
| `ib_write_bw.mlx5_0.avg_gbps` | 3 | 387.133 | 0.005 | 387.130 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_1.avg_gbps` | 3 | 387.133 | 0.005 | 387.130 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_4.avg_gbps` | 3 | 387.123 | 0.005 | 387.120 | 387.130 | 0.00 |
| `ib_write_bw.mlx5_5.avg_gbps` | 3 | 387.130 | 0.000 | 387.130 | 387.130 | 0.00 |
| `iperf3.fwd.gbps` | 3 | 7.555 | 0.134 | 7.428 | 7.740 | 1.77 |
| `iperf3.rev.gbps` | 3 | 7.675 | 0.197 | 7.416 | 7.895 | 2.57 |
| `nccl.all_gather_perf.max_busbw_gbps` | 3 | 427.543 | 0.621 | 426.990 | 428.410 | 0.15 |
| `nccl.all_reduce_perf.max_busbw_gbps` | 3 | 528.910 | 0.120 | 528.740 | 529.000 | 0.02 |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 3 | 575.487 | 0.100 | 575.380 | 575.620 | 0.02 |
| `torchdist.max_busbw_gbps` | 3 | 460.277 | 0.210 | 460.100 | 460.573 | 0.05 |

## Stats (extended)

| Metric | N | Mean | Stddev | Min | Max | CV% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ib_read_bw.mean_avg_gbps` | 3 | 388.549 | 0.022 | 388.517 | 388.565 | 0.01 |
| `ib_read_bw.mlx5_0.avg_gbps` | 3 | 388.577 | 0.033 | 388.530 | 388.600 | 0.01 |
| `ib_read_bw.mlx5_1.avg_gbps` | 3 | 388.557 | 0.038 | 388.530 | 388.610 | 0.01 |
| `ib_read_bw.mlx5_4.avg_gbps` | 3 | 388.510 | 0.127 | 388.330 | 388.600 | 0.03 |
| `ib_read_bw.mlx5_5.avg_gbps` | 3 | 388.553 | 0.033 | 388.530 | 388.600 | 0.01 |
| `ib_send_bw.mean_avg_gbps` | 3 | 388.304 | 0.383 | 387.762 | 388.577 | 0.10 |
| `ib_send_bw.mlx5_0.avg_gbps` | 3 | 388.553 | 0.040 | 388.520 | 388.610 | 0.01 |
| `ib_send_bw.mlx5_1.avg_gbps` | 3 | 388.570 | 0.036 | 388.520 | 388.600 | 0.01 |
| `ib_send_bw.mlx5_4.avg_gbps` | 3 | 387.503 | 1.537 | 385.330 | 388.600 | 0.40 |
| `ib_send_bw.mlx5_5.avg_gbps` | 3 | 388.590 | 0.000 | 388.590 | 388.590 | 0.00 |
| `ib_write_bw.mean_avg_gbps` | 3 | 387.076 | 0.075 | 386.970 | 387.130 | 0.02 |
| `ib_write_bw.mlx5_0.avg_gbps` | 3 | 386.917 | 0.302 | 386.490 | 387.130 | 0.08 |
| `ib_write_bw.mlx5_1.avg_gbps` | 3 | 387.133 | 0.005 | 387.130 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_4.avg_gbps` | 3 | 387.123 | 0.005 | 387.120 | 387.130 | 0.00 |
| `ib_write_bw.mlx5_5.avg_gbps` | 3 | 387.130 | 0.000 | 387.130 | 387.130 | 0.00 |
| `iperf3.fwd.gbps` | 3 | 7.487 | 0.091 | 7.367 | 7.587 | 1.21 |
| `iperf3.rev.gbps` | 3 | 7.689 | 0.198 | 7.409 | 7.834 | 2.58 |
| `nccl.all_gather_perf.max_busbw_gbps` | 3 | 427.150 | 0.232 | 426.930 | 427.470 | 0.05 |
| `nccl.all_reduce_perf.max_busbw_gbps` | 3 | 527.840 | 0.867 | 526.630 | 528.620 | 0.16 |
| `nccl.alltoall_perf.max_busbw_gbps` | 3 | 372.220 | 0.567 | 371.600 | 372.970 | 0.15 |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 3 | 575.710 | 0.248 | 575.360 | 575.890 | 0.04 |
| `torchdist.max_busbw_gbps` | 3 | 458.265 | 2.872 | 454.213 | 460.545 | 0.63 |

