# Counting metrics: /data/runs/E3 (test, threshold picked on val)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.4000 | 0.7678 | 0.7605 | 0.3037 | 0.0502 | 0.9513 | 0.7618 | 0.4100 |
| 2 | 0.4000 | 0.5675 | 0.5479 | 0.7370 | 0.0992 | 0.9095 | 0.5524 | 0.4300 |
| 3 | 0.4200 | 0.3293 | 0.3072 | 1.7723 | 0.2053 | 0.8251 | 0.3133 | 0.4600 |
| 4 | 0.4000 | 0.2102 | 0.1689 | 3.1879 | 0.3137 | 0.7606 | 0.1709 | 0.4600 |
| 5 | 0.3900 | 0.1391 | 0.1178 | 4.8456 | 0.4007 | 0.6937 | 0.1204 | 0.3700 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 1 | nan | 0.0502 | nan | 0.9917 |
| 2 | 0.0584 | 0.2627 | 1.0068 | 0.9538 |
| 3 | 0.1453 | 0.5053 | 1.0636 | 0.5731 |
| 4 | 0.2753 | 0.5442 | 1.0842 | 0.5749 |
| 5 | 0.3660 | 0.6437 | 1.0236 | 0.6029 |

Wrote /data/runs/E3/metrics_count_test.json
