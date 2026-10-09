import numpy as np
import pytest

from autocheckout.cl_metrics import load_coco
from autocheckout.counting import (
    THRESHOLD_GRID,
    class_group_scores,
    counting_scores,
    dedup_detections,
    gt_counts,
    oracle_by_level,
    pred_counts,
    scores_by_level,
    select_threshold,
)
from autocheckout.predictions import Predictions
from tests.coco_helpers import make_ann, make_gt, make_image, make_preds


# --- rpctool transcription, kept independent of autocheckout.counting for a cross-check ----------
# Copied from rpctool/evaluate_v1/evaluate.py (score1=cAcc, score2=ACD, score4=mCCD, score5=mCIoU).
# rpctool always divides by the fixed K=200 RPC classes, which are guaranteed to appear
# somewhere in the (full, fixed) split it runs on; that assumption holds here too because the
# random matrices below are built with a positive column sum for every class.
def rpctool_score1(pred, gt, n):
    return np.sum(np.all(np.equal(gt, pred), axis=1)) / n


def rpctool_score2(pred, gt, n):
    return np.sum(np.abs(gt - pred)) / n


def rpctool_score4(pred, gt, k):
    class_cd = np.sum(np.abs(gt - pred), axis=0)
    class_gt = np.sum(gt, axis=0)
    return np.sum(np.divide(class_cd, class_gt)) / k


def rpctool_score5(pred, gt, k):
    minimum = np.minimum(pred, gt)
    maximum = np.maximum(pred, gt)
    return np.sum(np.divide(np.sum(minimum, axis=0), np.sum(maximum, axis=0))) / k


def test_counting_scores_match_rpctool_transcription_on_random_matrices():
    rng = np.random.default_rng(0)
    n_images, k = 12, 5
    gt = rng.integers(0, 4, size=(n_images, k))
    gt[0, :] = rng.integers(1, 4, size=k)  # guarantee every class has positive total GT
    pred = rng.integers(0, 4, size=(n_images, k))

    ours = counting_scores(pred, gt)
    assert ours["K_eff"] == k
    assert ours["cAcc"] == pytest.approx(rpctool_score1(pred, gt, n_images))
    assert ours["ACD"] == pytest.approx(rpctool_score2(pred, gt, n_images))
    assert ours["mCCD"] == pytest.approx(rpctool_score4(pred, gt, k))
    assert ours["mCIoU"] == pytest.approx(rpctool_score5(pred, gt, k))


def test_counting_scores_excludes_zero_gt_classes_from_class_averages():
    gt = np.array([[1, 0], [2, 0]])  # class 1 has zero GT everywhere
    pred = np.array([[1, 5], [2, 0]])
    scores = counting_scores(pred, gt)
    assert scores["K"] == 2
    assert scores["K_eff"] == 1
    # Only class 0 counts towards mCCD/mCIoU; it is predicted perfectly (diff 0, min/max 1).
    assert scores["mCCD"] == pytest.approx(0.0)
    assert scores["mCIoU"] == pytest.approx(1.0)
    # cAcc/ACD are unaffected by the zero-GT class: image 0 has extra FPs of class 1 -> mismatch.
    assert scores["cAcc"] == pytest.approx(0.5)
    assert scores["ACD"] == pytest.approx((5 + 0) / 2)


def test_pred_counts_uses_top1_per_query():
    # One query with two candidate classes: only the higher-scoring one should be counted.
    preds = make_preds([
        (1, 0, 2, 0.9, [0, 0, 10, 10]),
        (1, 0, 1, 0.3, [0, 0, 10, 10]),  # same query, weaker class: must not be counted
        (1, 5, 0, 0.5, [0, 0, 10, 10]),
    ])
    counts = pred_counts(preds, image_ids=[1], labels=[0, 1, 2], threshold=0.0)
    assert counts.tolist() == [[1, 0, 1]]


def test_gt_counts_from_annotations():
    gt = make_gt(
        images=[make_image(1), make_image(2)],
        annotations=[
            make_ann(1, 1, 0, [0, 0, 5, 5]),
            make_ann(2, 1, 0, [10, 10, 5, 5]),
            make_ann(3, 2, 1, [0, 0, 5, 5]),
        ],
        num_labels=2,
    )
    coco_gt = load_coco(gt)
    counts = gt_counts(coco_gt, image_ids=[1, 2], labels=[0, 1])
    assert counts.tolist() == [[2, 0], [0, 1]]


def _threshold_val_fixture():
    # 2 images, 1 class. GT count is 1 in both images. A high-score detection (0.8) and a
    # low-score decoy (0.2) sit in image 1; image 2 only has the high-score detection.
    # threshold <= 0.2 -> image 1 predicts count 2 (wrong), image 2 predicts 1 (right): cAcc=0.5
    # 0.2 < threshold <= 0.8 -> both images predict count 1 (right): cAcc=1.0
    # threshold > 0.8 -> both images predict count 0 (wrong): cAcc=0.0
    gt = make_gt(
        images=[make_image(1), make_image(2)],
        annotations=[make_ann(1, 1, 0, [0, 0, 5, 5]), make_ann(2, 2, 0, [0, 0, 5, 5])],
        num_labels=1,
    )
    coco_gt = load_coco(gt)
    preds = make_preds([
        (1, 0, 0, 0.8, [0, 0, 5, 5]),
        (1, 1, 0, 0.2, [10, 10, 15, 15]),
        (2, 0, 0, 0.8, [0, 0, 5, 5]),
    ])
    return coco_gt, preds


def test_select_threshold_picks_the_known_best_threshold():
    coco_gt, preds = _threshold_val_fixture()
    threshold, scores = select_threshold(coco_gt, preds, labels=[0])
    # Smallest threshold in the grid that reaches cAcc=1.0 is 0.21 (grid step 0.01 from 0.05).
    assert threshold == pytest.approx(0.21)
    assert scores["cAcc"] == pytest.approx(1.0)


def test_select_threshold_tie_break_is_the_smallest_threshold():
    # Both 0.3 and 0.5 reach the maximum cAcc; the smallest (0.3) must win.
    coco_gt, preds = _threshold_val_fixture()
    threshold, _ = select_threshold(coco_gt, preds, labels=[0], thresholds=[0.1, 0.3, 0.5, 0.9])
    assert threshold == pytest.approx(0.3)


def test_scores_by_level_and_oracle_by_level():
    gt = make_gt(
        images=[make_image(1, level="easy"), make_image(2, level="hard")],
        annotations=[make_ann(1, 1, 0, [0, 0, 5, 5]), make_ann(2, 2, 0, [0, 0, 5, 5])],
        num_labels=1,
    )
    coco_gt = load_coco(gt)
    preds = make_preds([(1, 0, 0, 0.9, [0, 0, 5, 5]), (2, 0, 0, 0.9, [0, 0, 5, 5])])

    by_level = scores_by_level(coco_gt, preds, labels=[0], threshold=0.5)
    assert by_level["overall"]["cAcc"] == pytest.approx(1.0)
    assert by_level["easy"]["cAcc"] == pytest.approx(1.0)
    assert by_level["hard"]["cAcc"] == pytest.approx(1.0)

    oracle = oracle_by_level(coco_gt, preds, labels=[0])
    assert oracle["overall"]["cAcc"] == pytest.approx(1.0)
    assert oracle["overall"]["threshold"] in THRESHOLD_GRID


def test_count_matrix_matches_a_plain_loop_on_random_rows():
    from autocheckout.counting import count_matrix

    rng = np.random.default_rng(0)
    n = 500
    preds = Predictions(image_id=rng.integers(1, 30, n), query=np.arange(n), label=rng.integers(0, 12, n),
                        score=rng.random(n), boxes=np.zeros((n, 4)))
    image_ids, labels = [3, 1, 7, 29, 100], [0, 5, 2, 11]  # unsorted, with ids/labels absent from rows
    expected = np.zeros((len(image_ids), len(labels)), dtype=np.int64)
    for img, lab, score in zip(preds.image_id, preds.label, preds.score, strict=True):
        if score >= 0.3 and img in image_ids and lab in labels:
            expected[image_ids.index(img), labels.index(lab)] += 1
    np.testing.assert_array_equal(count_matrix(preds, image_ids, labels, 0.3), expected)


def test_dedup_detections_keeps_the_best_row_per_object_across_classes_within_an_image():
    preds = Predictions(
        image_id=[1, 1, 1, 2, 1], query=[0, 1, 2, 0, 3], label=[0, 1, 0, 0, 2],
        score=[0.9, 0.8, 0.7, 0.6, 0.01],
        boxes=[[0, 0, 10, 10], [0, 0, 10, 9], [20, 20, 30, 30], [0, 0, 10, 10], [20, 20, 30, 30]],
    )
    kept = dedup_detections(preds, 0.5)
    # (1, 1): same object as (1, 0) with another class -> dropped; (1, 2): another object; (2, 0): same box in
    # another image; (1, 3): below the lowest threshold of the grid
    assert sorted(zip(kept.image_id.tolist(), kept.query.tolist(), strict=True)) == [(1, 0), (1, 2), (2, 0)]
    assert len(dedup_detections(kept.subset(kept.score > 1), 0.5)) == 0


def test_class_group_scores_mccs_and_mccd_by_hand():
    # class 0: GT 2+2=4, predicted 1+1=2 -> CCS 0.5, CD 2/4; class 1: GT 1, predicted 2 -> CCS 2, CD 1;
    # class 2 has no GT and is excluded.
    gt = np.array([[2, 1, 0], [2, 0, 0]])
    pred = np.array([[1, 1, 3], [1, 1, 0]])
    scores = class_group_scores(pred, gt)
    assert scores["mCCS"] == pytest.approx((0.5 + 2.0) / 2)
    assert scores["mCCD"] == pytest.approx((0.5 + 1.0) / 2)
    assert scores["mCCD"] == pytest.approx(counting_scores(pred, gt)["mCCD"])
    assert (scores["K"], scores["K_eff"]) == (3, 2)
    assert np.isnan(class_group_scores(pred[:, 2:], gt[:, 2:])["mCCS"])
