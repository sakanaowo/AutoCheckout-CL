"""Two-pass inference of the PDP model and export of prediction files (fix F9, task V1).

Inference is the same everywhere (validation during training, predictions after each task,
predict-only runs): a first pass without prompts gives the query, the second pass uses the prompts
of every learned task. Per image, the top-100 (query, class) pairs over the learned classes are kept,
like the post-processing of the original evaluation (which masked unlearned logits and dropped the
unused last slot). Files follow docs/data_preprocessing/formats.md section 5.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.image_transforms import center_to_corners_format

from autocheckout.io import md5_file
from autocheckout.predictions import Predictions, save_predictions
from datasets.coco_hug import CocoDetection

TOPK_PER_IMAGE = 100


@torch.no_grad()
def predict_batch(model, pixel_values, pixel_mask, target_sizes, *, use_prompts, local_query, seen_classes,
                  topk=TOPK_PER_IMAGE):
    """Top-k (query, class) pairs of each image over classes < seen_classes.

    target_sizes: [B, 2] (height, width) the normalised boxes are scaled to.
    Returns one dict per image: scores, labels, queries, boxes (x1, y1, x2, y2 in pixels).
    """
    query = None
    if use_prompts:
        first = model(pixel_values=pixel_values, pixel_mask=pixel_mask, train=False)
        query = first.last_hidden_state if local_query else first.last_hidden_state.mean(dim=1)
    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, query=query, train=False)

    prob = outputs.logits[..., :seen_classes].sigmoid()
    batch_size, num_queries, num_classes = prob.shape
    scores, index = prob.reshape(batch_size, -1).topk(min(topk, num_queries * num_classes), dim=1)
    queries = torch.div(index, num_classes, rounding_mode='floor')
    labels = index % num_classes
    boxes = center_to_corners_format(outputs.pred_boxes).gather(1, queries.unsqueeze(-1).expand(-1, -1, 4))
    height, width = target_sizes.unbind(1)
    boxes = boxes * torch.stack([width, height, width, height], dim=1)[:, None, :].to(boxes.dtype)
    return [{'scores': s, 'labels': l, 'queries': q, 'boxes': b}
            for s, l, q, b in zip(scores, labels, queries, boxes)]


def predict_loader(model, dataloader, device, *, use_prompts, local_query, seen_classes, meta=None):
    """Predictions of the model on every image of a dataloader of CocoDetection batches."""
    model.eval()
    columns = {'image_id': [], 'query': [], 'label': [], 'score': [], 'boxes': []}
    for batch in dataloader:
        targets = batch['labels']
        sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
        results = predict_batch(model, batch['pixel_values'].to(device), batch['pixel_mask'].to(device), sizes,
                                use_prompts=use_prompts, local_query=local_query, seen_classes=seen_classes)
        for target, result in zip(targets, results):
            columns['image_id'].append(np.full(len(result['scores']), int(target['image_id'])))
            columns['query'].append(result['queries'].cpu().numpy())
            columns['label'].append(result['labels'].cpu().numpy())
            columns['score'].append(result['scores'].cpu().numpy())
            columns['boxes'].append(result['boxes'].cpu().numpy())
    arrays = {name: np.concatenate(parts) if parts else np.zeros((0, 4) if name == 'boxes' else 0)
              for name, parts in columns.items()}
    return Predictions(**arrays, meta=dict(meta or {}))


def write_predictions(model, args, *, ann_file, out_file, task_id, seen_classes, split, processor, device):
    """Predict every image of a COCO file (e.g. val_full.json) and save pred_<split>.npz (V1)."""
    dataset = CocoDetection(img_folder=args.test_img_dir, ann_file=ann_file, processor=processor)
    loader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=args.batch_size,
                        num_workers=args.num_workers)
    meta = {'task_id': task_id, 'seen_classes': seen_classes, 'split': split, 'ann_file': str(ann_file),
            'ann_md5': md5_file(ann_file), 'producer': 'pdp', 'topk_per_image': TOPK_PER_IMAGE}
    predictions = predict_loader(model, loader, device, use_prompts=bool(args.use_prompts),
                                 local_query=bool(args.local_query), seen_classes=seen_classes, meta=meta)
    save_predictions(out_file, predictions)
    return predictions
