# Counting metrics: /data/runs/FSA (test, threshold picked on val)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.4000 | 0.6780 | 0.6842 | 0.4351 | 0.0730 | 0.9292 | 0.6905 | 0.3900 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 1 | nan | 0.0730 | nan | 0.9748 |

Wrote /data/runs/FSA/metrics_count_test.json
