# Counting metrics: /data/runs/E5 (test, threshold picked on val, class-agnostic NMS at IoU 0.5)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.1100 | 0.0306 | 0.0416 | 5.3996 | 0.9019 | 0.1792 | 0.0416 | 0.1100 |
| 2 | 0.0700 | 0.0153 | 0.0202 | 6.1677 | 0.8192 | 0.2677 | 0.0210 | 0.0800 |
| 3 | 0.0500 | 0.0113 | 0.0125 | 6.7903 | 0.7551 | 0.3364 | 0.0125 | 0.0500 |
| 4 | 0.0500 | 0.0053 | 0.0045 | 7.9450 | 0.7595 | 0.2931 | 0.0045 | 0.0500 |
| 5 | 0.0500 | 0.0013 | 0.0018 | 9.3778 | 0.7672 | 0.2600 | 0.0018 | 0.0500 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 1 | nan | 0.9019 | nan | 0.3326 |
| 2 | 0.8279 | 0.7844 | 0.4406 | 0.5537 |
| 3 | 0.7742 | 0.6592 | 0.5540 | 0.5504 |
| 4 | 0.7584 | 0.7660 | 0.4284 | 0.3653 |
| 5 | 0.7679 | 0.7621 | 0.3276 | 0.3554 |

Wrote /data/runs/E5/metrics_count_test_nms0.5.json
