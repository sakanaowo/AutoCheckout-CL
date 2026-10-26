"""Quality of the pseudo-labels of earlier classes (V4).

    cd pdp && python ppg_audit.py <model arguments of the experiment> --output_dir <run> --audit_task 2

Runs the teacher of task t (the model after task t-1) and PPG exactly as during training, on train
images of task t whose every object is annotated (train_task_<t>_gt_full.json, never used for
training), and compares the pseudo-labels with the annotations of the earlier classes:
- precision of each branch (high confidence, prototype-verified) and recall of the earlier objects;
- the same per supercategory of the annotated class;
- confusions (pseudo-label of one SKU on an object of another), with the share inside one supercategory;
- how many selected pseudo-labels I3 (current-task box) and F14 (duplicate) drop.
Writes <run>/task_<t>/ppg_audit.json.
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou

import main as pdp_main
from autocheckout.io import save_json
from autocheckout.taskcfg import TaskConfig
from datasets.coco_hug import CocoDetection
from engine import local_trainer
from models.image_processing_deformable_detr import DeformableDetrImageProcessor
from ppg import cxcywh_to_xyxy, prototype_matrix, select_candidates, select_pseudo_labels


def match_pseudo_labels(boxes, labels, scores, gt_boxes, gt_labels, iou_thresh=0.5):
    """Outcome of each pseudo-label against the annotated earlier objects of one image.

    Greedy by score: a pseudo-label takes the unmatched annotated box with the highest IoU >= iou_thresh.
    Returns (outcomes, matched_gt): outcome 'tp' (same class), 'wrong_class' (other class), 'duplicate' (no
    free box, but a box of the same class already taken by a higher-scoring pseudo-label) or 'fp' (no box),
    and the index of the matched (or duplicated) annotation (-1 for 'fp'). Boxes are xyxy in any common frame.
    """
    result = [('fp', -1)] * len(boxes)
    if len(boxes) and len(gt_boxes):
        ious = box_iou(boxes, gt_boxes)
        taken = torch.zeros(len(gt_boxes), dtype=torch.bool)
        for i in torch.argsort(scores, descending=True).tolist():
            candidate = ious[i].masked_fill(taken, -1)
            j = int(candidate.argmax())
            if candidate[j] >= iou_thresh:
                taken[j] = True
                result[i] = ('tp' if int(labels[i]) == int(gt_labels[j]) else 'wrong_class', j)
                continue
            same = ious[i].masked_fill(~(taken & (gt_labels == labels[i])), -1)
            if same.max() >= iou_thresh:
                result[i] = ('duplicate', int(same.argmax()))
    return [outcome for outcome, _ in result], [j for _, j in result]


def audit(trainer, loader, config, device):
    prev = trainer.PREV_INTRODUCED_CLS
    args = trainer.args
    supercategory = {c.label: c.supercategory or 'reserved' for t in config.tasks for c in t.classes}
    branch = defaultdict(Counter)       # branch -> outcome counts
    by_super = defaultdict(Counter)     # supercategory of the annotated class -> tp / missed
    confusions = Counter()              # (pseudo label, true label) of wrong-class pseudo-labels
    old_objects = 0
    filters = Counter()                 # selected pseudo-labels, and those dropped by I3 / F14 (F0.6)
    prototypes, valid = prototype_matrix(trainer.class_prototypes, prev, device)
    for batch in loader:
        pixels, mask = batch['pixel_values'].to(device), batch['pixel_mask'].to(device)
        outputs = trainer.teacher_outputs(pixels, mask)
        scores, cand_labels, queries, boxes = select_candidates(outputs.logits, outputs.pred_boxes, prev,
                                                                args.pseudo_topk)
        for i, target in enumerate(batch['labels']):
            gt_labels, gt_boxes = target['class_labels'], target['boxes']
            old = gt_labels < prev
            current = (gt_labels >= prev) & (gt_labels < trainer.seen_classes)
            features = outputs.last_hidden_state[i, queries[i]]
            kept_boxes, kept_labels, keep = select_pseudo_labels(
                scores[i], cand_labels[i], boxes[i], features, prototypes, valid, mode=args.pseudo,
                tau_high=args.pseudo_thresh_high, tau_low=args.pseudo_thresh_low,
                sim_thresh=args.prototype_sim_thresh, nearest_prototype=bool(args.prototype_nearest),
                gt_boxes=gt_boxes[current].to(device), gt_iou=args.pseudo_gt_iou, dedup_iou=args.pseudo_dedup_iou,
                return_keep=True, stats=filters)
            kept_scores = scores[i][keep]
            is_high = (kept_scores > args.pseudo_thresh_high).cpu()
            outcomes, matched = match_pseudo_labels(
                cxcywh_to_xyxy(kept_boxes).cpu(), kept_labels.cpu(), kept_scores.cpu(),
                cxcywh_to_xyxy(gt_boxes[old]).cpu(), gt_labels[old].cpu())
            old_labels = gt_labels[old].tolist()
            old_objects += len(old_labels)
            found = set()
            for k, (outcome, j) in enumerate(zip(outcomes, matched)):
                branch['high' if is_high[k] else 'prototype'][outcome] += 1
                if outcome == 'tp':
                    found.add(j)
                elif outcome == 'wrong_class':
                    confusions[(int(kept_labels[k]), old_labels[j])] += 1
            for j, label in enumerate(old_labels):
                by_super[supercategory[label]]['found' if j in found else 'missed'] += 1

    def precision(counts):
        total = sum(counts.values())
        return round(counts['tp'] / total, 4) if total else None

    same_super = sum(n for (p, t), n in confusions.items() if supercategory[p] == supercategory[t])
    total_found = sum(c['found'] for c in by_super.values())
    return {
        'task_id': trainer.task_id,
        'images': len(loader.dataset),
        'earlier_objects': old_objects,
        'recall': round(total_found / old_objects, 4) if old_objects else None,
        'branches': {name: {**counts, 'precision': precision(counts)} for name, counts in branch.items()},
        'all': {**sum(branch.values(), Counter()), 'precision': precision(sum(branch.values(), Counter()))},
        'filters': {name: filters[name] for name in ('selected', 'dropped_i3', 'dropped_f14')},
        'by_supercategory': {name: {**counts, 'recall': round(counts['found'] / sum(counts.values()), 4)}
                             for name, counts in sorted(by_super.items())},
        'confusions_top': [{'pseudo': p, 'true': t, 'count': n, 'same_supercategory': supercategory[p] == supercategory[t]}
                           for (p, t), n in confusions.most_common(30)],
        'wrong_class_same_supercategory_share': round(same_super / sum(confusions.values()), 4) if confusions else None,
        'settings': {k: getattr(args, k) for k in ('pseudo', 'pseudo_topk', 'pseudo_thresh_high', 'pseudo_thresh_low',
                                                   'prototype_sim_thresh', 'prototype_nearest', 'pseudo_gt_iou',
                                                   'pseudo_dedup_iou')},
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(parents=[pdp_main.get_args_parser()])
    parser.add_argument('--audit_task', type=int, required=True)
    parser.add_argument('--audit_images', type=int, default=1000)
    return parser.parse_args(argv)


def run(args):
    pdp_main.setup_task_info(args)
    t = args.audit_task
    task_dir = pdp_main.task_dir(args.output_dir, t)
    args.log_file = open(os.devnull, 'w')
    processor = DeformableDetrImageProcessor.from_pretrained(args.repo_name) if args.repo_name \
        else DeformableDetrImageProcessor()
    dataset = CocoDetection(img_folder=args.train_img_dir, processor=processor,
                            ann_file=os.path.join(args.task_ann_dir, f'train_task_{t}_gt_full.json'))
    indices = list(range(len(dataset)))
    random.Random(0).shuffle(indices)
    loader = DataLoader(Subset(dataset, sorted(indices[:args.audit_images])), collate_fn=dataset.collate_fn,
                        batch_size=args.batch_size, num_workers=args.num_workers)

    trainer = local_trainer(train_loader=None, val_loader=None, test_dataset=None, args=args,
                            local_evaluator=argparse.Namespace(), task_id=t)
    if args.use_prompts:
        trainer.model.model.prompts.set_task_id(t - 1)
    trainer.resume(args.prev_ckpt or os.path.join(pdp_main.task_dir(args.output_dir, t - 1), 'task_final.pth'))
    trainer.set_teacher()
    device = torch.device('cuda' if args.accelerator == 'gpu' else 'cpu')
    trainer.teacher.to(device)
    result = audit(trainer, loader, TaskConfig.load(args.task_config), device)
    save_json(os.path.join(task_dir, 'ppg_audit.json'), result, indent=1)
    return result


if __name__ == '__main__':
    print(json.dumps(run(parse_args()), indent=1))
