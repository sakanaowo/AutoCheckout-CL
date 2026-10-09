# Counting metrics: /data/runs/E3 (test, threshold picked on val, class-agnostic NMS at IoU 0.5)

| stage | threshold | val cAcc | test cAcc | test ACD | test mCCD | test mCIoU | oracle cAcc | oracle threshold |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.3500 | 0.8350 | 0.8219 | 0.2201 | 0.0369 | 0.9640 | 0.8238 | 0.3600 |
| 2 | 0.3900 | 0.6188 | 0.5924 | 0.6462 | 0.0871 | 0.9208 | 0.5944 | 0.4100 |
| 3 | 0.4000 | 0.3826 | 0.3480 | 1.6607 | 0.1933 | 0.8383 | 0.3480 | 0.4000 |
| 4 | 0.3600 | 0.2615 | 0.2134 | 3.0926 | 0.3049 | 0.7712 | 0.2134 | 0.3200 |
| 5 | 0.2700 | 0.2535 | 0.1999 | 4.6517 | 0.3825 | 0.7193 | 0.2009 | 0.2300 |

Old / new classes (test, same threshold; mCCS 1 = counts as many as there are):

| stage | mCCD old | mCCD new | mCCS old | mCCS new |
|---|---|---|---|---|
| 1 | nan | 0.0369 | nan | 0.9865 |
| 2 | 0.0469 | 0.2482 | 0.9945 | 0.9660 |
| 3 | 0.1375 | 0.4718 | 1.0616 | 0.5918 |
| 4 | 0.2701 | 0.5139 | 1.0979 | 0.5025 |
| 5 | 0.3530 | 0.5888 | 1.0848 | 0.4846 |

Wrote /data/runs/E3/metrics_count_test_nms0.5.json
