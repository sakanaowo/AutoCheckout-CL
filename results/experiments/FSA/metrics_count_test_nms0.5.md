# Counting metrics: /data/runs/FSA (test, threshold picked on val, class-agnostic NMS at IoU 0.5)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.3300 | 0.7651 | 0.7671 | 0.3117 | 0.0528 | 0.9490 | 0.7718 | 0.3100 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 1 | nan | 0.0528 | nan | 0.9825 |

Wrote /data/runs/FSA/metrics_count_test_nms0.5.json
