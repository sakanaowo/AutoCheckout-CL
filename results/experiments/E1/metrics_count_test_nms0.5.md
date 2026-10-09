# Counting metrics: /data/runs/E1 (test, threshold picked on val, class-agnostic NMS at IoU 0.5)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.3600 | 0.8323 | 0.8213 | 0.2219 | 0.0371 | 0.9637 | 0.8266 | 0.3300 |
| 2 | 0.5700 | 0.0213 | 0.0252 | 6.5659 | 0.8725 | 0.1284 | 0.0302 | 0.4400 |
| 3 | 0.3400 | 0.0073 | 0.0087 | 7.8956 | 0.8715 | 0.1374 | 0.0090 | 0.4100 |
| 4 | 0.0700 | 0.0040 | 0.0013 | 13.2221 | 1.3099 | 0.1907 | 0.0015 | 0.0500 |
| 5 | 0.0500 | 0.0007 | 0.0033 | 15.7071 | 1.2623 | 0.2266 | 0.0033 | 0.0500 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 1 | nan | 0.0371 | nan | 0.9836 |
| 2 | 0.9999 | 0.3627 | 0.0001 | 0.6589 |
| 3 | 0.9974 | 0.2423 | 0.0027 | 1.1326 |
| 4 | 0.8441 | 4.1048 | 0.2420 | 5.0921 |
| 5 | 0.8230 | 4.3368 | 0.3336 | 5.3199 |

Wrote /data/runs/E1/metrics_count_test_nms0.5.json
