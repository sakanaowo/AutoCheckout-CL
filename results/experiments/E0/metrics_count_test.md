# Counting metrics: /data/runs/E0 (test, threshold picked on val)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.3300 | 0.7112 | 0.6803 | 0.4663 | 0.0376 | 0.9632 | 0.6817 | 0.3200 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 5 | 0.0375 | 0.0379 | 0.9959 | 0.9952 |

Wrote /data/runs/E0/metrics_count_test.json
