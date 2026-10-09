# Counting metrics: /data/runs/E4 (test, threshold picked on val)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.4300 | 0.7598 | 0.7341 | 0.3430 | 0.0570 | 0.9443 | 0.7406 | 0.4000 |
| 2 | 0.4000 | 0.4890 | 0.4854 | 0.9337 | 0.1249 | 0.8880 | 0.4854 | 0.4000 |
| 3 | 0.3900 | 0.4032 | 0.3637 | 1.3440 | 0.1503 | 0.8713 | 0.3655 | 0.4000 |
| 4 | 0.3600 | 0.3400 | 0.3175 | 1.6952 | 0.1620 | 0.8635 | 0.3175 | 0.3600 |
| 5 | 0.3200 | 0.3959 | 0.3608 | 1.7538 | 0.1434 | 0.8697 | 0.3632 | 0.3100 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 1 | nan | 0.0570 | nan | 0.9739 |
| 2 | 0.0605 | 0.3825 | 0.9957 | 0.8893 |
| 3 | 0.1237 | 0.2831 | 1.0148 | 0.9042 |
| 4 | 0.1514 | 0.2252 | 1.0220 | 0.9780 |
| 5 | 0.1302 | 0.2359 | 0.9751 | 1.0101 |

Wrote /data/runs/E4/metrics_count_test.json
