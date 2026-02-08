# Cluster Health Suite Variance Summary

## Runs

- `2026-02-07_144800_cluster_health_suite_variance_ubuntu_r1_base` (base): `results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r1_base_node1node2_cluster_health_suite_summary.json`
- `2026-02-07_144800_cluster_health_suite_variance_ubuntu_r2_base` (base): `results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r2_base_node1node2_cluster_health_suite_summary.json`
- `2026-02-07_144800_cluster_health_suite_variance_ubuntu_r3_base` (base): `results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r3_base_node1node2_cluster_health_suite_summary.json`
- `2026-02-07_144800_cluster_health_suite_variance_ubuntu_r1_extended` (extended): `results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r1_extended_node1node2_cluster_health_suite_summary.json`
- `2026-02-07_144800_cluster_health_suite_variance_ubuntu_r2_extended` (extended): `results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r2_extended_node1node2_cluster_health_suite_summary.json`
- `2026-02-07_144800_cluster_health_suite_variance_ubuntu_r3_extended` (extended): `results/structured/2026-02-07_144800_cluster_health_suite_variance_ubuntu_r3_extended_node1node2_cluster_health_suite_summary.json`

## Stats (base)

| Metric | N | Mean | Stddev | Min | Max | CV% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ib_write_bw.mean_avg_gbps` | 3 | 387.138 | 0.001 | 387.137 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_0.avg_gbps` | 3 | 387.140 | 0.000 | 387.140 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_1.avg_gbps` | 3 | 387.133 | 0.005 | 387.130 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_4.avg_gbps` | 3 | 387.140 | 0.000 | 387.140 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_5.avg_gbps` | 3 | 387.140 | 0.000 | 387.140 | 387.140 | 0.00 |
| `iperf3.fwd.gbps` | 3 | 7.786 | 0.091 | 7.661 | 7.877 | 1.17 |
| `iperf3.rev.gbps` | 3 | 7.767 | 0.163 | 7.619 | 7.994 | 2.10 |
| `nccl.all_gather_perf.max_busbw_gbps` | 3 | 522.137 | 100.751 | 441.430 | 664.180 | 19.30 |
| `nccl.all_reduce_perf.max_busbw_gbps` | 3 | 634.907 | 146.484 | 526.810 | 842.000 | 23.07 |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 3 | 611.280 | 48.712 | 575.840 | 680.160 | 7.97 |
| `torchdist.max_busbw_gbps` | 3 | 562.211 | 118.946 | 421.160 | 712.114 | 21.16 |

## Stats (extended)

| Metric | N | Mean | Stddev | Min | Max | CV% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ib_read_bw.mean_avg_gbps` | 3 | 388.596 | 0.008 | 388.590 | 388.608 | 0.00 |
| `ib_read_bw.mlx5_0.avg_gbps` | 3 | 388.610 | 0.000 | 388.610 | 388.610 | 0.00 |
| `ib_read_bw.mlx5_1.avg_gbps` | 3 | 388.553 | 0.033 | 388.530 | 388.600 | 0.01 |
| `ib_read_bw.mlx5_4.avg_gbps` | 3 | 388.607 | 0.005 | 388.600 | 388.610 | 0.00 |
| `ib_read_bw.mlx5_5.avg_gbps` | 3 | 388.613 | 0.005 | 388.610 | 388.620 | 0.00 |
| `ib_send_bw.mean_avg_gbps` | 3 | 388.564 | 0.008 | 388.558 | 388.575 | 0.00 |
| `ib_send_bw.mlx5_0.avg_gbps` | 3 | 388.570 | 0.036 | 388.520 | 388.600 | 0.01 |
| `ib_send_bw.mlx5_1.avg_gbps` | 3 | 388.547 | 0.038 | 388.520 | 388.600 | 0.01 |
| `ib_send_bw.mlx5_4.avg_gbps` | 3 | 388.543 | 0.034 | 388.510 | 388.590 | 0.01 |
| `ib_send_bw.mlx5_5.avg_gbps` | 3 | 388.597 | 0.005 | 388.590 | 388.600 | 0.00 |
| `ib_write_bw.mean_avg_gbps` | 3 | 387.139 | 0.001 | 387.137 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_0.avg_gbps` | 3 | 387.140 | 0.000 | 387.140 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_1.avg_gbps` | 3 | 387.137 | 0.005 | 387.130 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_4.avg_gbps` | 3 | 387.140 | 0.000 | 387.140 | 387.140 | 0.00 |
| `ib_write_bw.mlx5_5.avg_gbps` | 3 | 387.140 | 0.000 | 387.140 | 387.140 | 0.00 |
| `iperf3.fwd.gbps` | 3 | 7.698 | 0.142 | 7.580 | 7.898 | 1.84 |
| `iperf3.rev.gbps` | 3 | 7.736 | 0.183 | 7.561 | 7.989 | 2.36 |
| `nccl.all_gather_perf.max_busbw_gbps` | 3 | 446.077 | 10.085 | 436.040 | 459.870 | 2.26 |
| `nccl.all_reduce_perf.max_busbw_gbps` | 3 | 591.703 | 50.724 | 528.700 | 652.910 | 8.57 |
| `nccl.alltoall_perf.max_busbw_gbps` | 3 | 364.397 | 11.686 | 347.870 | 372.690 | 3.21 |
| `nccl.reduce_scatter_perf.max_busbw_gbps` | 3 | 618.063 | 45.191 | 575.490 | 680.630 | 7.31 |
| `torchdist.max_busbw_gbps` | 3 | 526.579 | 140.434 | 397.157 | 721.751 | 26.67 |

