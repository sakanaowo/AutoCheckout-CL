# Counting metrics: /data/runs/E2 (test, threshold picked on val)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.4300 | 0.7804 | 0.7669 | 0.2967 | 0.0493 | 0.9518 | 0.7688 | 0.4100 |
| 2 | 0.3300 | 0.3679 | 0.3530 | 1.3103 | 0.1775 | 0.8443 | 0.3558 | 0.3400 |
| 3 | 0.2300 | 0.1444 | 0.1259 | 3.2899 | 0.3719 | 0.6948 | 0.1283 | 0.2400 |
| 4 | 0.1500 | 0.0339 | 0.0272 | 7.2729 | 0.7021 | 0.4789 | 0.0280 | 0.1300 |
| 5 | 0.1000 | 0.0106 | 0.0067 | 12.7216 | 1.0308 | 0.3103 | 0.0067 | 0.1000 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 1 | nan | 0.0493 | nan | 0.9836 |
| 2 | 0.1282 | 0.3747 | 0.8923 | 1.2141 |
| 3 | 0.2887 | 0.7880 | 0.7464 | 1.7313 |
| 4 | 0.5270 | 1.7527 | 0.5534 | 2.7357 |
| 5 | 0.7116 | 3.2655 | 0.4189 | 4.2457 |

Wrote /data/runs/E2/metrics_count_test.json
