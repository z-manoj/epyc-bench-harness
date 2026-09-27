# Benchmark report: dev_llamacpp_accel

Host 2xf511w21hbkfa37ndpvkankat, kernel 6.8.0-1067-azure; tunables: smt_control=on, smt_active=1, governors=[], thp_enabled=madvise, thp_defrag=madvise, numa_balancing=0, online_cpus=0-15

| config | engine | precision | accel | topology | cores | binary | args |
|---|---|---|---|---|---|---|---|
| llamacpp_q8_zendnn | llamacpp | q8_0 | zendnn | 1x14c | 2-15 | /home/mkumar/llama.cpp/build/bin/llama-server | -t 14 -tb 14 -np 4 -c 16384 --cont-batching |
| llamacpp_q8_cpu | llamacpp | q8_0 | none | 1x14c | 2-15 | /home/mkumar/llama.cpp/build-nozendnn/bin/llama-server | -t 14 -tb 14 -np 4 -c 16384 --cont-batching |

## 1. Validity overview

| server_config_id | UNSTABLE | VALID | config_status |
|---|---|---|---|
| llamacpp_q8_cpu | 1 | 3 | DONE |
| llamacpp_q8_zendnn | 0 | 4 | DONE |

33 reps total; 24 SUCCESS. Failed or flagged reps:

| server_config_id | point_id | rep | status | flags | reasons |
|---|---|---|---|---|---|
| llamacpp_q8_zendnn | online_single_p128_o32_c1 | 1 | INVALID_BASELINE |  | stability gate failed: server-core util std pp 2.16 > 2 |
| llamacpp_q8_zendnn | online_single_p1024_o32_c1 | 1 | SUCCESS | WARN_MEM_GROWTH |  |
| llamacpp_q8_zendnn | online_multi_p128_o32_c4 | 3 | UNSTABLE_DRIFT |  | intra-rep drift tok/s 0.06451612893353532 tpot 0.006004356124573148 |
| llamacpp_q8_zendnn | batch_p512_o32_c4 | 1 | SUCCESS | WARN_PROMPT_TOKENS |  |
| llamacpp_q8_zendnn | batch_p512_o32_c4 | 2 | UNSTABLE_DRIFT | WARN_PROMPT_TOKENS | intra-rep drift tok/s 0.015748031452137913 tpot 0.3952921583109059 |
| llamacpp_q8_zendnn | batch_p512_o32_c4 | 3 | SUCCESS | WARN_PROMPT_TOKENS |  |
| llamacpp_q8_zendnn | batch_p512_o32_c4 | 4 | SUCCESS | WARN_PROMPT_TOKENS |  |
| llamacpp_q8_cpu | online_single_p128_o32_c1 | 1 | INVALID_BASELINE |  | stability gate failed: server-core util std pp 2.67 > 2 |
| llamacpp_q8_cpu | online_single_p1024_o32_c1 | 1 | SUCCESS | WARN_MEM_GROWTH |  |
| llamacpp_q8_cpu | online_multi_p128_o32_c4 | 1 | UNSTABLE_DRIFT |  | intra-rep drift tok/s 0.06451612896858105 tpot 0.0213139148508776 |
| llamacpp_q8_cpu | batch_p512_o32_c4 | 1 | SUCCESS | WARN_MEM_GROWTH,WARN_PROMPT_TOKENS |  |
| llamacpp_q8_cpu | batch_p512_o32_c4 | 2 | UNSTABLE_DRIFT | WARN_PROMPT_TOKENS | intra-rep drift tok/s 0.08130081300813002 tpot 0.03388790845379523 |
| llamacpp_q8_cpu | batch_p512_o32_c4 | 3 | UNSTABLE_DRIFT | WARN_PROMPT_TOKENS | intra-rep drift tok/s 0.08130081300813008 tpot 0.04349993434309825 |
| llamacpp_q8_cpu | batch_p512_o32_c4 | 4 | UNSTABLE_DRIFT | WARN_PROMPT_TOKENS | intra-rep drift tok/s 0.08130081298475343 tpot 0.03111153060449815 |
| llamacpp_q8_cpu | batch_p512_o32_c4 | 5 | SUCCESS | WARN_PROMPT_TOKENS |  |
| llamacpp_q8_cpu | batch_p512_o32_c4 | 6 | SUCCESS | WARN_PROMPT_TOKENS |  |
| llamacpp_q8_cpu | batch_p512_o32_c4 | 7 | UNSTABLE_DRIFT | WARN_PROMPT_TOKENS | intra-rep drift tok/s 0.08130081300813016 tpot 0.04005582334165009 |

## 2. Per-rep timelines and 3. rep overlays

Timeline panels: per-core CPU heatmap (50 ms, dashed lines delimit server cores), server-core utilisation with user/system split, server RSS/PSS and system memory, server-core frequency. Shading: gate (blue), warmup (orange), measure (green), cooldown (purple).

### llamacpp_q8_zendnn / online_single_p128_o32_c1

![overlay online_single_p128_o32_c1](img/overlay_llamacpp_q8_zendnn_online_single_p128_o32_c1.png)

![timeline rep 1](img/timeline_llamacpp_q8_zendnn_online_single_p128_o32_c1_rep1.png)

### llamacpp_q8_zendnn / online_single_p1024_o32_c1

![overlay online_single_p1024_o32_c1](img/overlay_llamacpp_q8_zendnn_online_single_p1024_o32_c1.png)

![timeline rep 1](img/timeline_llamacpp_q8_zendnn_online_single_p1024_o32_c1_rep1.png)

### llamacpp_q8_zendnn / online_multi_p128_o32_c4

![overlay online_multi_p128_o32_c4](img/overlay_llamacpp_q8_zendnn_online_multi_p128_o32_c4.png)

![timeline rep 3](img/timeline_llamacpp_q8_zendnn_online_multi_p128_o32_c4_rep3.png)

### llamacpp_q8_zendnn / batch_p512_o32_c4

![overlay batch_p512_o32_c4](img/overlay_llamacpp_q8_zendnn_batch_p512_o32_c4.png)

![timeline rep 1](img/timeline_llamacpp_q8_zendnn_batch_p512_o32_c4_rep1.png)

![timeline rep 2](img/timeline_llamacpp_q8_zendnn_batch_p512_o32_c4_rep2.png)

![timeline rep 3](img/timeline_llamacpp_q8_zendnn_batch_p512_o32_c4_rep3.png)

![timeline rep 4](img/timeline_llamacpp_q8_zendnn_batch_p512_o32_c4_rep4.png)

### llamacpp_q8_cpu / online_single_p128_o32_c1

![overlay online_single_p128_o32_c1](img/overlay_llamacpp_q8_cpu_online_single_p128_o32_c1.png)

![timeline rep 1](img/timeline_llamacpp_q8_cpu_online_single_p128_o32_c1_rep1.png)

### llamacpp_q8_cpu / online_single_p1024_o32_c1

![overlay online_single_p1024_o32_c1](img/overlay_llamacpp_q8_cpu_online_single_p1024_o32_c1.png)

![timeline rep 1](img/timeline_llamacpp_q8_cpu_online_single_p1024_o32_c1_rep1.png)

### llamacpp_q8_cpu / online_multi_p128_o32_c4

![overlay online_multi_p128_o32_c4](img/overlay_llamacpp_q8_cpu_online_multi_p128_o32_c4.png)

![timeline rep 1](img/timeline_llamacpp_q8_cpu_online_multi_p128_o32_c4_rep1.png)

### llamacpp_q8_cpu / batch_p512_o32_c4

![overlay batch_p512_o32_c4](img/overlay_llamacpp_q8_cpu_batch_p512_o32_c4.png)

![timeline rep 1](img/timeline_llamacpp_q8_cpu_batch_p512_o32_c4_rep1.png)

![timeline rep 2](img/timeline_llamacpp_q8_cpu_batch_p512_o32_c4_rep2.png)

![timeline rep 3](img/timeline_llamacpp_q8_cpu_batch_p512_o32_c4_rep3.png)

![timeline rep 4](img/timeline_llamacpp_q8_cpu_batch_p512_o32_c4_rep4.png)

![timeline rep 5](img/timeline_llamacpp_q8_cpu_batch_p512_o32_c4_rep5.png)

![timeline rep 6](img/timeline_llamacpp_q8_cpu_batch_p512_o32_c4_rep6.png)

![timeline rep 7](img/timeline_llamacpp_q8_cpu_batch_p512_o32_c4_rep7.png)

## 4. Cross-rep deviation (selected reps)

| config | point | class | output_tok_s | total_tok_s | ttft_p50_ms | tpot_p50_ms | itl_p50_ms | ttft_p95_ms | tpot_p95_ms | cpu_server_util_mean | rss_peak_mb |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llamacpp_q8_zendnn | online_single_p128_o32_c1 | VALID | 4.01 ± 0.0021 (cv 0.1%) | 20.2 ± 0.0106 (cv 0.1%) | 1003.3 ± 10.2 (cv 1.0%) | 225.4 ± 0.295 (cv 0.1%) | 225.0 ± 1.04 (cv 0.5%) | 1070.4 ± 42.1 (cv 3.9%) | 227.2 ± 0.988 (cv 0.4%) | 97.8 ± 0.188 (cv 0.2%) | 18625 ± 7.14 (cv 0.0%) |
| llamacpp_q8_zendnn | online_single_p1024_o32_c1 | VALID | 2.09 ± 0.0119 (cv 0.6%) | 69.1 ± 0.394 (cv 0.6%) | 8069.2 ± 27.3 (cv 0.3%) | 234.3 ± 1.59 (cv 0.7%) | 232.2 ± 3.31 (cv 1.4%) | 8172.1 ± 15.8 (cv 0.2%) | 237.2 ± 2.44 (cv 1.0%) | 98 ± 0.145 (cv 0.1%) | 19219 ± 0.986 (cv 0.0%) |
| llamacpp_q8_zendnn | online_multi_p128_o32_c4 | VALID | 11.9 ± 0.0217 (cv 0.2%) | 59.9 ± 0.109 (cv 0.2%) | 3747.8 ± 25.1 (cv 0.7%) | 226.0 ± 0.619 (cv 0.3%) | 221.1 ± 0.482 (cv 0.2%) | 3767.8 ± 28 (cv 0.7%) | 226.9 ± 0.274 (cv 0.1%) | 98.7 ± 0.184 (cv 0.2%) | 19435 ± 30.6 (cv 0.2%) |
| llamacpp_q8_zendnn | batch_p512_o32_c4 | VALID | 5.57 ± 0.0153 (cv 0.3%) | 94.5 ± 0.259 (cv 0.3%) | 15469 ± 25.3 (cv 0.2%) | 243.5 ± 2.02 (cv 0.8%) | 237.2 ± 1.54 (cv 0.6%) | 15837 ± 71.9 (cv 0.5%) | 248.8 ± 1.18 (cv 0.5%) | 98.5 ± 0.134 (cv 0.1%) | 19116 ± 101.0 (cv 0.5%) |
| llamacpp_q8_cpu | online_single_p128_o32_c1 | VALID | 3.36 ± 0.00602 (cv 0.2%) | 16.9 ± 0.0303 (cv 0.2%) | 2548.2 ± 4.15 (cv 0.2%) | 224.4 ± 0.769 (cv 0.3%) | 219.5 ± 0.562 (cv 0.3%) | 2591.1 ± 28.1 (cv 1.1%) | 227.3 ± 1.34 (cv 0.6%) | 98.6 ± 0.135 (cv 0.1%) | 10643 ± 0.158 (cv 0.0%) |
| llamacpp_q8_cpu | online_single_p1024_o32_c1 | VALID | 1.17 ± 0.00267 (cv 0.2%) | 38.7 ± 0.0881 (cv 0.2%) | 20108 ± 33.8 (cv 0.2%) | 231.5 ± 1.98 (cv 0.9%) | 226.5 ± 0.279 (cv 0.1%) | 20193 ± 61.3 (cv 0.3%) | 234.9 ± 1.15 (cv 0.5%) | 98.9 ± 0.139 (cv 0.1%) | 11245 ± 0.83 (cv 0.0%) |
| llamacpp_q8_cpu | online_multi_p128_o32_c4 | VALID | 7.71 ± 0.018 (cv 0.2%) | 38.8 ± 0.0907 (cv 0.2%) | 9631.5 ± 2.73 (cv 0.0%) | 225.1 ± 1.19 (cv 0.5%) | 220.6 ± 1.61 (cv 0.7%) | 9633.0 ± 2.37 (cv 0.0%) | 226.3 ± 2.43 (cv 1.1%) | 99.4 ± 0.0224 (cv 0.0%) | 11478 ± 0.0471 (cv 0.0%) |
| llamacpp_q8_cpu | batch_p512_o32_c4 | **UNSTABLE (!)** | 2.75 ± 0.0113 (cv 0.4%) | 46.6 ± 0.192 (cv 0.4%) | **35682 ± 2628.2 (cv 7.4%) (!)** | **347.3 ± 86 (cv 24.8%) (!)** | 237.5 ± 0.683 (cv 0.3%) | 38833 ± 76 (cv 0.2%) | **728.6 ± 416.0 (cv 57.1%) (!)** | 99.2 ± 0.0501 (cv 0.1%) | 10892 ± 23.2 (cv 0.2%) |

## 5. Scenario results

Only VALID / VALID_WITH_RERUNS points are plotted for online_single and batch; online_multi marks valid points with o and invalid with x.

![online_single_o32](img/online_single_o32.png)

![online_multi_p128_o32](img/online_multi_p128_o32.png)

![batch_o32](img/batch_o32.png)

| server_config_id | accel | point_id | classification | output_tok_s | total_tok_s | ttft_p50_ms | ttft_p95_ms | tpot_p50_ms | tpot_p95_ms | per_user_tok_s_p50 | output_tok_s_per_core | cpu_server_util_mean | rss_peak_mb | goodput_concurrency | goodput_tok_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llamacpp_q8_zendnn | zendnn | online_single_p128_o32_c1 | VALID | 4.01 | 20.2 | 1003.3 | 1070.4 | 225.4 | 227.2 | 4.44 | 0.286 | 97.8 | 18625 |  |  |
| llamacpp_q8_zendnn | zendnn | online_single_p1024_o32_c1 | VALID | 2.09 | 69.1 | 8069.2 | 8172.1 | 234.3 | 237.2 | 4.27 | 0.149 | 98 | 19219 |  |  |
| llamacpp_q8_zendnn | zendnn | online_multi_p128_o32_c4 | VALID | 11.9 | 59.9 | 3747.8 | 3767.8 | 226.0 | 226.9 | 4.43 | 0.85 | 98.7 | 19435 |  |  |
| llamacpp_q8_zendnn | zendnn | batch_p512_o32_c4 | VALID | 5.57 | 94.5 | 15469 | 15837 | 243.5 | 248.8 | 4.11 | 0.398 | 98.5 | 19116 |  |  |
| llamacpp_q8_cpu | none | online_single_p128_o32_c1 | VALID | 3.36 | 16.9 | 2548.2 | 2591.1 | 224.4 | 227.3 | 4.46 | 0.24 | 98.6 | 10643 |  |  |
| llamacpp_q8_cpu | none | online_single_p1024_o32_c1 | VALID | 1.17 | 38.7 | 20108 | 20193 | 231.5 | 234.9 | 4.32 | 0.0838 | 98.9 | 11245 |  |  |
| llamacpp_q8_cpu | none | online_multi_p128_o32_c4 | VALID | 7.71 | 38.8 | 9631.5 | 9633.0 | 225.1 | 226.3 | 4.44 | 0.55 | 99.4 | 11478 |  |  |
| llamacpp_q8_cpu | none | batch_p512_o32_c4 | UNSTABLE | 2.75 | 46.6 | 35682 | 38833 | 347.3 | 728.6 | 3.29 | 0.196 | 99.2 | 10892 |  |  |

## 6. ZenDNN acceleration (accel=zendnn vs accel=none)

Speedup > 1.0 means ZenDNN is better: throughput metrics are zendnn / baseline, latency metrics are baseline / zendnn. Summary uses the geometric mean over points where both sides are VALID.

| zendnn_config | baseline_config | scenario | points | output_tok_s_speedup_gmean | total_tok_s_speedup_gmean | req_s_speedup_gmean | output_tok_s_per_core_speedup_gmean | per_user_tok_s_p50_speedup_gmean | ttft_p50_ms_speedup_gmean | ttft_p95_ms_speedup_gmean | tpot_p50_ms_speedup_gmean | tpot_p95_ms_speedup_gmean | e2e_p50_ms_speedup_gmean | output_tok_s_speedup_min | output_tok_s_speedup_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llamacpp_q8_zendnn | llamacpp_q8_cpu | online_multi | 1 | 1.54 | 1.54 | 1.54 | 1.54 | 0.996 | 2.57 | 2.56 | 0.996 | 0.998 | 1.54 | 1.54 | 1.54 |
| llamacpp_q8_zendnn | llamacpp_q8_cpu | online_single | 2 | 1.46 | 1.46 | 1.46 | 1.46 | 0.991 | 2.52 | 2.45 | 0.991 | 0.995 | 1.46 | 1.19 | 1.78 |

![accel_llamacpp_q8_zendnn_vs_llamacpp_q8_cpu](img/accel_llamacpp_q8_zendnn_vs_llamacpp_q8_cpu.png)

| zendnn_config | baseline_config | point_id | both_valid | output_tok_s_baseline | output_tok_s_zendnn | output_tok_s_speedup | ttft_p50_ms_baseline | ttft_p50_ms_zendnn | ttft_p50_ms_speedup | tpot_p50_ms_baseline | tpot_p50_ms_zendnn | tpot_p50_ms_speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llamacpp_q8_zendnn | llamacpp_q8_cpu | online_single_p128_o32_c1 | True | 3.36 | 4.01 | 1.19 | 2548.2 | 1003.3 | 2.54 | 224.4 | 225.4 | 0.995 |
| llamacpp_q8_zendnn | llamacpp_q8_cpu | online_single_p1024_o32_c1 | True | 1.17 | 2.09 | 1.78 | 20108 | 8069.2 | 2.49 | 231.5 | 234.3 | 0.988 |
| llamacpp_q8_zendnn | llamacpp_q8_cpu | online_multi_p128_o32_c4 | True | 7.71 | 11.9 | 1.54 | 9631.5 | 3747.8 | 2.57 | 225.1 | 226.0 | 0.996 |
| llamacpp_q8_zendnn | llamacpp_q8_cpu | batch_p512_o32_c4 | **False (!)** | 2.75 | 5.57 | 2.03 | 35682 | 15469 | 2.31 | 347.3 | 243.5 | 1.43 |

## 7. Recommendation

    online_single: geometric mean of per-user tok/s (1/TPOT p50) across valid points.     online_multi:  mean goodput output tok/s across (prompt, output) combinations.     batch:         geometric mean of total tok/s across valid points.

| scenario | precision | engine | accel | server_config | topology | cores | flags | env | headline | valid_points | max_cv |
|---|---|---|---|---|---|---|---|---|---|---|---|
| batch | 8-bit | llamacpp | zendnn | llamacpp_q8_zendnn | 1x14c | 2-15 | -t 14 -tb 14 -np 4 -c 16384 --cont-batching | ZENDNNL_MATMUL_ALGO=1 | total 94.5 tok/s (gmean); 0.40 out tok/s/core | 1 | 0.00831 |
| online_multi | 8-bit | llamacpp | none | llamacpp_q8_cpu | 1x14c | 2-15 | -t 14 -tb 14 -np 4 -c 16384 --cont-batching |  | no SLA-meeting point | 1 | 0.0107 |
| online_multi | 8-bit | llamacpp | zendnn | llamacpp_q8_zendnn | 1x14c | 2-15 | -t 14 -tb 14 -np 4 -c 16384 --cont-batching | ZENDNNL_MATMUL_ALGO=1 | no SLA-meeting point | 1 | 0.00743 |
| online_single | 8-bit | llamacpp | none | llamacpp_q8_cpu | 1x14c | 2-15 | -t 14 -tb 14 -np 4 -c 16384 --cont-batching |  | per-user 4.4 tok/s; TTFT p50 gmean 7158 ms | 2 | 0.0109 |
| online_single | 8-bit | llamacpp | zendnn | llamacpp_q8_zendnn | 1x14c | 2-15 | -t 14 -tb 14 -np 4 -c 16384 --cont-batching | ZENDNNL_MATMUL_ALGO=1 | per-user 4.4 tok/s; TTFT p50 gmean 2845 ms | 2 | 0.0393 |

CSV tables: csv/points.csv, csv/reps.csv, csv/deviation.csv, csv/acceleration.csv, csv/acceleration_summary.csv, csv/recommendations.csv
