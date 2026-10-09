# Counting metrics: /data/runs/E4_noF14 (test, threshold picked on val)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.4000 | 0.7345 | 0.7441 | 0.3318 | 0.0551 | 0.9466 | 0.7441 | 0.4000 |
| 2 | 0.3900 | 0.4890 | 0.4721 | 0.9520 | 0.1279 | 0.8871 | 0.4763 | 0.4100 |
| 3 | 0.4500 | 0.3460 | 0.3175 | 1.5396 | 0.1712 | 0.8466 | 0.3175 | 0.4500 |
| 4 | 0.4500 | 0.2468 | 0.2101 | 2.2569 | 0.2136 | 0.8136 | 0.2139 | 0.4400 |
| 5 | 0.4100 | 0.1144 | 0.1000 | 4.4066 | 0.3529 | 0.7325 | 0.1000 | 0.4100 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 1 | nan | 0.0551 | nan | 0.9855 |
| 2 | 0.0694 | 0.3623 | 1.0167 | 0.9429 |
| 3 | 0.1348 | 0.3529 | 1.0106 | 0.7234 |
| 4 | 0.2021 | 0.2832 | 1.0224 | 0.7790 |
| 5 | 0.3539 | 0.3462 | 1.1610 | 0.7828 |

Wrote /data/runs/E4_noF14/metrics_count_test.json
