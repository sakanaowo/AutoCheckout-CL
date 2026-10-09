# Counting metrics: /data/runs/E5 (test, threshold picked on val)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.1800 | 0.0299 | 0.0378 | 5.6780 | 0.9463 | 0.0863 | 0.0416 | 0.1100 |
| 2 | 0.1400 | 0.0160 | 0.0158 | 6.9034 | 0.9147 | 0.1074 | 0.0208 | 0.0800 |
| 3 | 0.0500 | 0.0093 | 0.0118 | 6.8549 | 0.7617 | 0.3344 | 0.0118 | 0.0500 |
| 4 | 0.0500 | 0.0040 | 0.0040 | 8.0033 | 0.7647 | 0.2915 | 0.0040 | 0.0500 |
| 5 | 0.0500 | 0.0013 | 0.0017 | 9.4298 | 0.7711 | 0.2587 | 0.0017 | 0.0500 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 1 | nan | 0.9463 | nan | 0.1407 |
| 2 | 0.9222 | 0.8847 | 0.1370 | 0.2159 |
| 3 | 0.7811 | 0.6646 | 0.5633 | 0.5590 |
| 4 | 0.7634 | 0.7721 | 0.4358 | 0.3748 |
| 5 | 0.7716 | 0.7673 | 0.3332 | 0.3633 |

Wrote /data/runs/E5/metrics_count_test.json
