# Counting metrics: /data/runs/E4 (test, threshold picked on val, class-agnostic NMS at IoU 0.5)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.3500 | 0.7964 | 0.7779 | 0.2804 | 0.0471 | 0.9546 | 0.7784 | 0.3400 |
| 2 | 0.3400 | 0.5190 | 0.4844 | 0.9555 | 0.1286 | 0.8948 | 0.5042 | 0.4000 |
| 3 | 0.3900 | 0.4251 | 0.3858 | 1.2820 | 0.1435 | 0.8768 | 0.3858 | 0.3900 |
| 4 | 0.3500 | 0.3699 | 0.3413 | 1.6182 | 0.1547 | 0.8708 | 0.3433 | 0.3600 |
| 5 | 0.2600 | 0.4644 | 0.4245 | 1.6255 | 0.1338 | 0.8801 | 0.4250 | 0.2700 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 1 | nan | 0.0471 | nan | 0.9887 |
| 2 | 0.0585 | 0.4087 | 1.0067 | 1.0948 |
| 3 | 0.1167 | 0.2775 | 1.0055 | 0.8938 |
| 4 | 0.1440 | 0.2189 | 1.0176 | 0.9786 |
| 5 | 0.1205 | 0.2264 | 0.9878 | 1.0413 |

Wrote /data/runs/E4/metrics_count_test_nms0.5.json
