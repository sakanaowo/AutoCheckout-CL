"""Pseudo-labels for objects of earlier tasks: prototypical pseudo-label generation (PPG), fix F6.

At task t (t >= 2) only objects of task-t classes are annotated. The teacher (the model after task
t-1) proposes objects of earlier classes, and PPG keeps

- high-confidence proposals: score > tau_high;
- medium-confidence proposals (tau_low < score <= tau_high) whose query feature is close to the
  prototype of the proposed class (cosine >= sim_thresh).

Differences with the original `local_trainer.generate_old_class_pseudo_labels` (kept for P1):

- candidates are ranked per query, using the query's best earlier class; classes >= PREV (current
  and future tasks, and the unused last slot) are excluded before ranking. The original took the
  top-5 (query, class) pairs over all classes first and filtered afterwards, which leaves too few
  candidates for images with about 12 objects, and one query could yield several labels;
- only labels < PREV are accepted (the original used <=, which let in the first class of the
  current task);
- a candidate's feature is taken at its query index (the original indexed the feature tensor with
  the candidate's position in the top-k list).

Boxes stay in the normalised (cx, cy, w, h) format of the model outputs and of the targets.
"""

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, nms


def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def select_candidates(logits, boxes, num_old_classes, topk):
    """Top-k queries of each image ranked by their best earlier-class probability.

    logits: [B, Q, C] raw teacher logits; boxes: [B, Q, 4] normalised cxcywh.
    Returns scores [B, k], labels [B, k] (all < num_old_classes), queries [B, k], boxes [B, k, 4].
    """
    best_score, best_label = logits[..., :num_old_classes].sigmoid().max(dim=-1)
    k = min(topk, best_score.shape[1])
    scores, queries = best_score.topk(k, dim=1)
    labels = best_label.gather(1, queries)
    boxes = boxes.gather(1, queries.unsqueeze(-1).expand(-1, -1, boxes.shape[-1]))
    return scores, labels, queries, boxes


def prototype_matrix(class_prototypes, num_old_classes, device):
    """[num_old, D] prototypes of the earlier classes and a mask of the classes that have one."""
    prototypes = torch.stack([class_prototypes[c].to(device) for c in range(num_old_classes)])
    return prototypes, prototypes.norm(dim=1) > 1e-8


def select_pseudo_labels(scores, labels, boxes, features, prototypes, valid, *, mode, tau_high, tau_low,
                         sim_thresh, nearest_prototype=False, gt_boxes=None, gt_iou=0.0, dedup_iou=0.0,
                         return_keep=False, stats=None):
    """Pseudo-labels of one image from its candidates (outputs of select_candidates for that image).

    features: [k, D] teacher query features of the candidates.
    mode 'threshold' keeps score > tau_high only; 'ppg' also keeps prototype-verified
    medium-confidence candidates.
    nearest_prototype (I4): a verified candidate's class must also be its nearest prototype, which
    guards against confusing similar SKUs (same brand, other flavour).
    gt_iou > 0 (I3): drop pseudo-labels overlapping a box of the current task's annotations
    (IoU >= gt_iou), so one object never gets two labels.
    dedup_iou > 0 (F14): at most one pseudo-label per object: class-agnostic NMS at this IoU over the kept
    candidates, the highest score wins. Without it the prototype branch accepts the secondary queries of an
    object that is already labelled (their features match its prototype), and the duplicates compound over
    the tasks (E4 audit, 30/09: 3.7% of the pseudo-labels at task 2, 31% at task 5).
    Returns (boxes [n, 4], labels [n]), plus the boolean mask over the candidates if return_keep (V4).
    stats (a Counter, FULL-SYN F0.6) adds the pseudo-labels selected by the thresholds/prototypes and how many of
    them I3 and F14 drop.
    """
    keep = scores > tau_high
    if mode == 'ppg':
        medium = (scores > tau_low) & ~keep
        if medium.any():
            similarity = F.normalize(features[medium], dim=-1) @ F.normalize(prototypes, dim=-1).T
            candidate_labels = labels[medium]
            own_similarity = similarity.gather(1, candidate_labels[:, None]).squeeze(1)
            verified = valid[candidate_labels] & (own_similarity >= sim_thresh)
            if nearest_prototype:
                similarity[:, ~valid] = -2.0
                verified &= similarity.argmax(dim=1) == candidate_labels
            keep[medium] = verified
    elif mode != 'threshold':
        raise ValueError(f'unknown pseudo-label mode {mode!r}')
    if stats is not None:
        stats['selected'] += int(keep.sum())
    if gt_iou > 0 and gt_boxes is not None and len(gt_boxes) and keep.any():
        before = int(keep.sum())
        overlap = box_iou(cxcywh_to_xyxy(boxes), cxcywh_to_xyxy(gt_boxes.to(boxes.dtype))).max(dim=1).values
        keep &= overlap < gt_iou
        if stats is not None:
            stats['dropped_i3'] += before - int(keep.sum())
    if dedup_iou > 0 and keep.sum() > 1:
        kept = torch.nonzero(keep).squeeze(1)
        survivors = kept[nms(cxcywh_to_xyxy(boxes[kept]).float(), scores[kept].float(), dedup_iou)]
        if stats is not None:
            stats['dropped_f14'] += int(keep.sum()) - len(survivors)
        keep = torch.zeros_like(keep)
        keep[survivors] = True
    if return_keep:
        return boxes[keep], labels[keep], keep
    return boxes[keep], labels[keep]
