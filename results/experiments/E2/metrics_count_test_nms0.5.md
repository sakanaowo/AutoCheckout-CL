# Counting metrics: /data/runs/E2 (test, threshold picked on val, class-agnostic NMS at IoU 0.5)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.3200 | 0.8350 | 0.8263 | 0.2149 | 0.0363 | 0.9648 | 0.8271 | 0.3100 |
| 2 | 0.3200 | 0.3919 | 0.3620 | 1.2764 | 0.1731 | 0.8500 | 0.3670 | 0.3400 |
| 3 | 0.2300 | 0.1504 | 0.1376 | 3.1676 | 0.3581 | 0.7002 | 0.1378 | 0.2400 |
| 4 | 0.0900 | 0.0459 | 0.0368 | 7.9040 | 0.7797 | 0.5402 | 0.0368 | 0.0900 |
| 5 | 0.0700 | 0.0166 | 0.0087 | 12.9875 | 1.0579 | 0.3290 | 0.0090 | 0.0800 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 1 | nan | 0.0363 | nan | 0.9931 |
| 2 | 0.1199 | 0.3858 | 0.8969 | 1.2289 |
| 3 | 0.2859 | 0.7196 | 0.7416 | 1.6556 |
| 4 | 0.4641 | 2.6736 | 0.7129 | 3.6582 |
| 5 | 0.6973 | 3.5823 | 0.4579 | 4.5574 |

Wrote /data/runs/E2/metrics_count_test_nms0.5.json
