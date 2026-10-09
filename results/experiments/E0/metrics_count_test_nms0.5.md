# Counting metrics: /data/runs/E0 (test, threshold picked on val, class-agnostic NMS at IoU 0.5)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.1500 | 0.8649 | 0.8362 | 0.2710 | 0.0224 | 0.9782 | 0.8362 | 0.1500 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 5 | 0.0222 | 0.0234 | 1.0003 | 1.0031 |

Wrote /data/runs/E0/metrics_count_test_nms0.5.json
