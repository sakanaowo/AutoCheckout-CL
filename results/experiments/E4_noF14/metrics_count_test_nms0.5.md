# Counting metrics: /data/runs/E4_noF14 (test, threshold picked on val, class-agnostic NMS at IoU 0.5)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.3400 | 0.7778 | 0.7853 | 0.2737 | 0.0463 | 0.9556 | 0.7868 | 0.3500 |
| 2 | 0.3700 | 0.5363 | 0.5159 | 0.8476 | 0.1141 | 0.9023 | 0.5174 | 0.3900 |
| 3 | 0.3800 | 0.4158 | 0.3843 | 1.3693 | 0.1510 | 0.8777 | 0.3901 | 0.4100 |
| 4 | 0.3700 | 0.3859 | 0.3527 | 1.6633 | 0.1574 | 0.8716 | 0.3550 | 0.3900 |
| 5 | 0.3100 | 0.4358 | 0.3956 | 1.8244 | 0.1481 | 0.8685 | 0.3960 | 0.3000 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 1 | nan | 0.0463 | nan | 0.9937 |
| 2 | 0.0529 | 0.3587 | 1.0033 | 1.0085 |
| 3 | 0.1266 | 0.2730 | 1.0459 | 0.9086 |
| 4 | 0.1475 | 0.2169 | 1.0388 | 0.9178 |
| 5 | 0.1335 | 0.2503 | 1.0075 | 0.9346 |

Wrote /data/runs/E4_noF14/metrics_count_test_nms0.5.json
