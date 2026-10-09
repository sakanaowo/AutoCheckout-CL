# Counting metrics: /data/runs/E1 (test, threshold picked on val)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.3800 | 0.7631 | 0.7583 | 0.3085 | 0.0516 | 0.9500 | 0.7601 | 0.4100 |
| 2 | 0.5700 | 0.0213 | 0.0250 | 6.5680 | 0.8728 | 0.1282 | 0.0298 | 0.4400 |
| 3 | 0.3200 | 0.0073 | 0.0078 | 7.9823 | 0.8816 | 0.1321 | 0.0087 | 0.4100 |
| 4 | 0.0700 | 0.0033 | 0.0010 | 14.0361 | 1.3924 | 0.1886 | 0.0015 | 0.5200 |
| 5 | 0.0600 | 0.0007 | 0.0023 | 16.4171 | 1.3084 | 0.2193 | 0.0023 | 0.0600 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 1 | nan | 0.0516 | nan | 0.9962 |
| 2 | 0.9999 | 0.3642 | 0.0001 | 0.6609 |
| 3 | 0.9960 | 0.3093 | 0.0041 | 1.2230 |
| 4 | 0.8457 | 4.6727 | 0.2522 | 5.6656 |
| 5 | 0.8216 | 4.7161 | 0.3156 | 5.7032 |

Wrote /data/runs/E1/metrics_count_test.json
