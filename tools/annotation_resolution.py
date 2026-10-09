"""API adjudication and deterministic mask-based replacement of ambiguous synthetic photos."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tools.ai_annotation_review import build_request

PROMPT_VERSION = 'rpc-annotation-final-resolution-v1'
PROMPT = """You are the final automated reviewer of a synthetic RPC bbox-only training sample.
The operator authorizes a final KEEP or REGENERATE action, without deferring to a human.
Image 1: tray context plus original crop and RED stored bbox / CYAN simplified polygon.
Image 2: larger unannotated native crop. Image 3: magnified unannotated band at the largest
bbox/polygon extremum discrepancy (edge named in metadata). Image 4: original product cutout,
before its unknown placement/rotation/occlusion. Use the native and edge views to verify
any alleged target fragment. Numeric geometry is metadata, not ground-truth ownership.
The generator takes a tight bbox of VISIBLE owner pixels with alpha >127, then separately
simplifies polygon contours and drops tiny contour components. A few-pixel mismatch, a
rotated empty corner, a tiny/disconnected fragment, or an occluded extremum is not by itself
a bbox error. Original final owner maps and transforms were not saved. Do not claim a verified
mask exists, and never guess replacement bbox coordinates on the old photo.
Choose keep_with_justification when the stored bbox is plausible for detection and concrete
visible evidence supports the extremes. Explain the relevant contour limitation. Do not invent
a fragment beyond the cyan outline: point to its location in the native/edge images, and say
when its ownership is uncertain. Look at the original cutout for handles or detached details.
Choose regenerate whenever a meaningful geometry concern OR unresolved fragment ownership
prevents a defensible keep. This is an operational choice to generate a NEW full tray, not a
claim that the original bbox has been proven wrong. Uncertainty is allowed in the evidence,
but the final action must be regenerate, not pending. All instances/SKUs in that tray will be
recomposited from existing cutouts/backgrounds and labels calculated from stored new owner masks.
No lost objects or guessed coordinates. A confident keep also needs pixel evidence: confidence
is self-reported, not calibrated. Return concise Vietnamese observations/reason; preserve IDs.
"""


def _image_content(path: Path, *, jpeg: bool = False, max_side: int | None = None) -> dict[str, str]:
    with Image.open(path) as source:
        im = source.convert('RGB')
        if max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        im.save(buffer, format='JPEG' if jpeg else 'PNG', **({'quality': 95} if jpeg else {}))
    mime = 'image/jpeg' if jpeg else 'image/png'
    return {'type': 'input_image', 'detail': 'high',
            'image_url': 'data:' + mime + ';base64,' + base64.b64encode(buffer.getvalue()).decode()}


def build_adjudication_request(case: dict[str, Any], preview: Path, cutout: Path, *,
                               model: str = 'gpt-6.1-sol', reasoning_effort: str = 'medium') -> dict[str, Any]:
    original = build_request(case, preview, cutout, model=model, reasoning_effort=reasoning_effort)
    properties = {
        'source_task': {'type': 'integer', 'enum': [case['source_task']]},
        'annotation_id': {'type': 'integer', 'enum': [case['annotation_id']]},
        'decision': {'type': 'string', 'enum': ['keep_with_justification', 'regenerate']},
        'geometry_uncertain': {'type': 'boolean'},
        'verified_owner_mask_available': {'type': 'boolean', 'enum': [False]},
        'observations': {'type': 'array', 'items': {'type': 'string'}},
        'reason': {'type': 'string'}, 'supporting_fragment': {'type': 'string'},
        'confidence': {'type': 'number'},
    }
    metadata = {k: case[k] for k in ['source_task','annotation_id','image_id','rpc_category_id','iou','bbox','polygon_bbox']}
    metadata.update({k: case[k] for k in ['largest_gap_edge','crop_box']})
    return {**original, 'instructions': PROMPT, 'max_output_tokens': 4000,
        'input': [{'role': 'user', 'content': [
            {'type':'input_text','text':json.dumps(metadata)},
            _image_content(preview, jpeg=True),
            _image_content(Path(case['native_crop']), max_side=1280),
            _image_content(Path(case['edge_zoom']), max_side=1800),
            original['input'][0]['content'][-1],
        ]}],
        'text': {'format': {'type':'json_schema','name':'rpc_final_annotation_action','strict':True,
            'schema': {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}}}


def parse_adjudication(response: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    if response.get('status') != 'completed':
        raise ValueError('Final review response is incomplete/refused')
    messages = [r for r in response.get('output', []) if r.get('type') == 'message']
    if not messages:
        raise ValueError('No final review message')
    text = ''.join(p['text'] for p in messages[-1].get('content', []) if p.get('type') == 'output_text')
    review = json.loads(text)
    if (review.get('source_task'), review.get('annotation_id')) != (case['source_task'], case['annotation_id']):
        raise ValueError('Final review IDs mismatch')
    if review.get('decision') not in {'keep_with_justification', 'regenerate'}:
        raise ValueError('Final action must keep or regenerate')
    if review.get('verified_owner_mask_available') is not False:
        raise ValueError('Original verified owner mask is unavailable')
    if type(review.get('geometry_uncertain')) is not bool:
        raise ValueError('Review must explicitly assess geometry uncertainty')
    if not isinstance(review.get('reason'), str) or len(review['reason'].strip()) < 30:
        raise ValueError('Final action needs substantive evidence')
    observations = review.get('observations')
    if not isinstance(observations, list) or len(observations) < 2 or not all(isinstance(s, str) and s.strip() for s in observations):
        raise ValueError('Final action needs concrete observations')
    if not isinstance(review.get('supporting_fragment'), str):
        raise ValueError('Fragment assessment missing')
    c = review.get('confidence')
    if type(c) not in [int, float] or not math.isfinite(c) or not 0 <= c <= 1:
        raise ValueError('Invalid confidence')
    return review


def final_action(review: dict[str, Any]) -> str:
    """An ambiguous keep resolves to actual regeneration, without inventing old-image geometry."""
    return 'keep_with_justification' if review['decision'] == 'keep_with_justification' and not review['geometry_uncertain'] else 'regenerate'


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)] if len(xs) else None


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(mask, dtype=np.uint8).ravel(order='F')
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.r_[0, edges, flat.size]).astype(int).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {'size': list(mask.shape), 'counts': counts}


def decode_rle(rle: dict[str, Any]) -> np.ndarray:
    shape, counts = rle['size'], rle['counts']
    if len(shape) != 2 or any(n < 0 for n in counts) or sum(counts) != shape[0]*shape[1]:
        raise ValueError('Invalid RLE size/counts')
    return np.repeat(np.arange(len(counts)) % 2, counts).astype(bool).reshape(shape, order='F')


def _patch(raw_root: Path, annotation: dict[str, Any], rng: np.random.Generator,
           limits: tuple[int, int]) -> tuple[Image.Image, dict[str, Any]]:
    path = raw_root / annotation['cutout']
    with Image.open(path) as source:
        rgba = source.convert('RGBA')
    angle = float(rng.uniform(-180,180))
    rgba = rgba.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    bounds = mask_bbox(np.asarray(rgba.getchannel('A')) > 127)
    if bounds is None:
        raise ValueError('Cutout has no foreground after rotation')
    x,y,w,h = bounds
    rgba = rgba.crop((x,y,x+w,y+h))
    scale = min(limits[0]/rgba.width, limits[1]/rgba.height) * .96
    dims = (max(2,int(rgba.width*scale)),max(2,int(rgba.height*scale)))
    rgba = rgba.resize(dims, Image.Resampling.LANCZOS)
    if not np.any(np.asarray(rgba.getchannel('A')) > 127):
        raise ValueError('Resized cutout has no foreground')
    return rgba, {'cutout':annotation['cutout'],'rotation_degrees':angle,'resize':list(dims)}


def compose_replacement(objects: list[dict[str, Any]], *, raw_root: Path, background: Path,
                        size: tuple[int,int], seed: int, min_visible: float = .70,
                        max_tries: int = 128) -> dict[str, Any]:
    """Recompose all original instances; exact new owner/RLE labels retain detached fragments."""
    if not objects or len(objects) >= 65535 or not 0 < min_visible <= 1:
        raise ValueError('Invalid objects/visibility policy')
    rng = np.random.default_rng(seed)
    width,height = size
    with Image.open(background) as source:
        background_rgb = source.convert('RGB').resize(size,Image.Resampling.BICUBIC)
    for layout in ['template','grid_fallback']:
        canvas = background_rgb.convert('RGBA')
        owner = np.zeros((height,width),dtype=np.uint16)
        placed = []; remaining = []
        columns = max(1,math.ceil(math.sqrt(len(objects)*width/height)))
        rows = math.ceil(len(objects)/columns)
        for index, annotation in enumerate(objects):
            x,y,w,h = annotation['bbox']
            if layout == 'template':
                limits=(max(6,min(int(w),width-4)),max(6,min(int(h),height-4)))
            else:
                limits=(max(6,width//columns-8),max(6,height//rows-8))
            patch, transform = _patch(raw_root,annotation,rng,limits)
            mask = np.asarray(patch.getchannel('A')) > 127
            ph,pw = mask.shape; area=int(mask.sum())
            if pw > width or ph > height:
                raise ValueError('Object cannot fit canvas')
            found = False
            for attempt in range(max_tries if layout=='template' else 1):
                if layout=='grid_fallback':
                    cx=(index%columns+.5)*width/columns;cy=(index//columns+.5)*height/rows
                elif attempt==0:
                    cx=x+w/2;cy=y+h/2
                else:
                    cx,cy=rng.uniform([pw/2,ph/2],[width-pw/2,height-ph/2])
                left=int(np.clip(round(cx-pw/2),0,width-pw))
                top=int(np.clip(round(cy-ph/2),0,height-ph))
                region=owner[top:top+ph,left:left+pw]
                overlap=np.bincount(region[mask],minlength=len(placed)+1)
                if any((remaining[i]-int(overlap[i+1]))/placed[i]['mask_area'] < min_visible for i in range(len(placed))):
                    continue
                for i in range(len(placed)):
                    remaining[i]-=int(overlap[i+1])
                slot=index+1
                region[mask]=slot
                canvas.alpha_composite(patch,dest=(left,top))
                placed.append({'annotation':annotation,'owner_slot':slot,'x':left,'y':top,
                    'w':pw,'h':ph,'mask_area':area,'transform':transform})
                remaining.append(area)
                found=True
                break
            if not found:
                break
        if len(placed)==len(objects):
            break
    if len(placed)!=len(objects):
        raise ValueError('Could not place every instance; no partial scene is accepted')
    annotations=[]
    for p in placed:
        local=owner[p['y']:p['y']+p['h'],p['x']:p['x']+p['w']]==p['owner_slot']
        full=np.zeros(owner.shape,dtype=bool)
        full[p['y']:p['y']+p['h'],p['x']:p['x']+p['w']]=local
        bbox=mask_bbox(full)
        if bbox is None:
            raise ValueError('An original instance lost all visible pixels')
        annotations.append({'source_annotation_id':p['annotation']['id'],
            'category_id':p['annotation']['category_id'],'bbox':bbox,'area':int(local.sum()),
            'segmentation':encode_rle(full),'iscrowd':0,'owner_slot':p['owner_slot'],
            'visible':round(float(local.sum())/p['mask_area'],6),'cutout':p['annotation']['cutout']})
    return {'image':canvas.convert('RGB'),'owner':owner,'annotations':annotations,
            'recipe':{'seed':seed,'size':list(size),'min_visible':min_visible,'alpha_owner_threshold':128,
                      'layout':layout,'background':str(background),'objects':[
                          {k:v for k,v in p.items() if k!='annotation'} | {'source_annotation_id':p['annotation']['id'],
                          'category_id':p['annotation']['category_id']} for p in placed]}}


def _templates(cases: list[dict[str, Any]], raw_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result=[];categories=[]
    for task in sorted({c['source_task'] for c in cases}):
        data=json.loads((raw_root/f'synth/group/task{task}/annotations.json').read_text())
        if not categories:categories=data['categories']
        selected={c['image_id']:c for c in cases if c['source_task']==task}
        images={r['id']:r for r in data['images'] if r['id'] in selected}
        annotations={iid:[] for iid in selected}
        for r in data['annotations']:
            if r['image_id'] in selected:annotations[r['image_id']].append(r)
        for iid,case in selected.items():
            target=[a for a in annotations[iid] if a['id']==case['annotation_id']]
            if len(target)!=1 or target[0]['bbox']!=case['bbox']:
                raise ValueError('Case/source annotation mismatch')
            result.append({'case':case,'image':images[iid],'annotations':annotations[iid]})
    bykey={(t['case']['source_task'],t['case']['annotation_id']):t for t in result}
    return [bykey[c['source_task'],c['annotation_id']] for c in cases],categories


def apply_resolution_dataset(cases: list[dict[str, Any]], receipts: list[dict[str, Any]], *,
                             repo: Path, raw_root: Path, out_dir: Path, backgrounds: list[Path],
                             seed: int, min_visible: float = .70, task_config_path: Path | None = None,
                             templates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Materialize the authorized cases; preserve every SKU/instance and source geometry of keeps."""
    import hashlib
    from collections import Counter
    from tools.ai_annotation_review import file_sha256,write_json
    keys=[(c['source_task'],c['annotation_id']) for c in cases]
    reviews={tuple(r['case_key']):r for r in receipts}
    if len(set(keys))!=len(keys) or set(keys)!=set(reviews) or len(receipts)!=len(cases):
        raise ValueError('Decision coverage does not match resolution scope')
    if not backgrounds:
        raise ValueError('At least one verified background is required')
    if templates is None:
        templates,categories=_templates(cases,raw_root)
    else:
        categories=[{'id':cid,'name':str(cid)} for cid in sorted({a['category_id'] for t in templates for a in t['annotations']})]
    if len(templates)!=len(cases):
        raise ValueError('Template count differs from resolution scope')
    source_photos={c['image_file']:file_sha256(raw_root/c['image_file']) for c in cases}
    cutouts={a['cutout']:file_sha256(raw_root/a['cutout']) for t in templates for a in t['annotations']}
    background_hashes={str(p.resolve()):file_sha256(p) for p in backgrounds}
    task_hash=file_sha256(task_config_path) if task_config_path else None
    parameters={'seed':seed,'min_visible':min_visible,'generator_version':'visible-owner-rle-v1',
        'generator_code_sha256':file_sha256(Path(__file__)),'source_photo_sha256':source_photos,
        'cutout_sha256':cutouts,'background_sha256':background_hashes,'task_config_sha256':task_hash,
        'source_annotations':[t['annotations'] for t in templates],
        'reviews':[{'case_key':k,'fingerprint':reviews[k]['fingerprint'],'review':reviews[k]['review']} for k in keys]}
    fingerprint=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest()
    manifest_path=out_dir/'resolution_manifest.json'
    if manifest_path.exists():
        saved=json.loads(manifest_path.read_text())
        if saved['input_fingerprint']!=fingerprint:
            raise ValueError('Resolution inputs/policy changed; create a new dataset version')
        if file_sha256(out_dir/'annotations.json')!=saved['dataset_sha256']:
            raise ValueError('Resolved annotations changed')
        for path,digest in saved['output_asset_sha256'].items():
            if file_sha256(repo/path)!=digest:raise ValueError('Resolved output asset changed')
        return saved
    out_dir.mkdir(parents=True,exist_ok=True)
    images=[];annotations=[];resolutions=[];assets={};next_ann=10_000_000
    expected_counts=Counter(a['category_id'] for t in templates for a in t['annotations'])
    for index,template in enumerate(templates):
        case=template['case'];key=(case['source_task'],case['annotation_id']);receipt=reviews[key]
        decision=final_action(receipt['review']);image=dict(template['image']);image_id=1_000_000+index
        stem=f'task{key[0]}_ann{key[1]}'
        owner_path=None;recipe_path=None
        if decision=='regenerate':
            case_seed=seed+case['source_task']*1_000_003+case['image_id']
            background=backgrounds[case_seed%len(backgrounds)]
            scene=compose_replacement(template['annotations'],raw_root=raw_root,background=background,
                size=(image['width'],image['height']),seed=case_seed,min_visible=min_visible)
            photo=out_dir/'images'/(stem+'.jpg');photo.parent.mkdir(exist_ok=True)
            scene['image'].save(photo,format='JPEG',quality=95,subsampling=0)
            owner_path=out_dir/'owner_masks'/(stem+'.npz');owner_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(owner_path,owner=scene['owner'])
            recipe_path=out_dir/'recipes'/(stem+'.json');recipe_path.parent.mkdir(exist_ok=True)
            recipe={**scene['recipe'],'background_sha256':file_sha256(background),
                'source_image':case['image_file'],'source_image_sha256':source_photos[case['image_file']],
                'input_fingerprint':fingerprint,'decision_fingerprint':receipt['fingerprint'],
                'cutout_sha256':{a['cutout']:cutouts[a['cutout']] for a in template['annotations']}}
            write_json(recipe_path,recipe)
            new_annotations=scene['annotations']
            status='regenerated'
            for path in [photo,owner_path,recipe_path]:assets[str(path.resolve().relative_to(repo.resolve()))]=file_sha256(path)
        else:
            photo=raw_root/case['image_file']
            new_annotations=[dict(a,source_annotation_id=a['id']) for a in template['annotations']]
            status='kept_by_ai'
        output_photo=str(photo.resolve().relative_to(repo.resolve()))
        image.update(id=image_id,file_name=output_photo,source_task=case['source_task'],
            source_image_id=case['image_id'],source_file_name=case['image_file'],
            train_source='synthetic_resolved_202',resolution_status=status)
        if owner_path:image['owner_mask']=str(owner_path.resolve().relative_to(repo.resolve()))
        images.append(image)
        target_new_id=None
        for ann in new_annotations:
            a={**ann,'id':next_ann,'image_id':image_id,'source_task':case['source_task'],
               'source_image_id':case['image_id'],'resolution_status':status}
            if a['source_annotation_id']==case['annotation_id']:target_new_id=next_ann
            annotations.append(a);next_ann+=1
        if target_new_id is None:raise ValueError('Target instance was not preserved')
        evidence=(f"status={status}; source_task={key[0]}; source_ann={key[1]}; output={output_photo}; "
            f"target_annotation_id={target_new_id}; decision_fingerprint={receipt['fingerprint']}; "
            f"owner_mask={str(owner_path.resolve().relative_to(repo.resolve())) if owner_path else 'original unavailable; AI visual review'}; "
            +receipt['review']['reason']+' Observations: '+' | '.join(receipt['review']['observations']))
        resolutions.append({'case_key':list(key),'decision':decision,'status':status,'output_image':output_photo,
            'output_image_id':image_id,'target_annotation_id':target_new_id,
            'owner_mask':str(owner_path.resolve().relative_to(repo.resolve())) if owner_path else None,
            'recipe':str(recipe_path.resolve().relative_to(repo.resolve())) if recipe_path else None,
            'decision_fingerprint':receipt['fingerprint'],'object_count':len(new_annotations),
            'sku_multiset':dict(Counter(a['category_id'] for a in template['annotations'])),
            'original_image_sha256':source_photos[case['image_file']],'evidence':evidence})
        if (index+1)%20==0:print('Resolved',index+1,'/',len(cases),flush=True)
    coco={'info':{'description':'202-scope RPC synth: AI keep or actual mask-based regeneration',
                 'image_root':str(repo.resolve()),'coordinates':'native_pixels','input_fingerprint':fingerprint},
          'images':images,'annotations':annotations,'categories':categories}
    if Counter(a['category_id'] for a in annotations)!=expected_counts:
        raise ValueError('SKU/instance distribution changed')
    write_json(out_dir/'annotations.json',coco)
    task_files=[]
    if task_config_path:
        from autocheckout.taskcfg import TaskConfig
        from tools.make_task_json import build_task_coco,images_with
        config=TaskConfig.load(task_config_path)
        if [t.num_classes for t in config.data_tasks]!=[100,25,25,25,25]:raise ValueError('Expected locked 5-task protocol')
        for task in config.data_tasks:
            rpc_ids={c.rpc_category_id for c in task.classes}
            image_ids=images_with(coco,rpc_ids)
            for suffix,keep in [('',rpc_ids),('_gt_full',None)]:
                path=out_dir/'tasks'/f'train_task_{task.task_id}{suffix}.json'
                write_json(path,build_task_coco(coco,config,image_ids,keep))
                task_files.append(str(path.resolve().relative_to(repo.resolve())))
        write_json(out_dir/'task_config.json',config.to_dict())
    manifest={'input_fingerprint':fingerprint,'dataset_sha256':file_sha256(out_dir/'annotations.json'),
        'dataset':str((out_dir/'annotations.json').resolve().relative_to(repo.resolve())),
        'image_root':str(repo.resolve()),'resolved_cases':len(cases),
        'kept_images':sum(r['status']=='kept_by_ai' for r in resolutions),
        'regenerated_images':sum(r['status']=='regenerated' for r in resolutions),
        'object_count':len(annotations),'sku_count':len(expected_counts),'sku_multiset':dict(expected_counts),
        'resolutions':resolutions,'output_asset_sha256':assets,'source_photo_sha256':source_photos,
        'source_cutout_sha256':cutouts,'background_sha256':background_hashes,
        'task_files':task_files,'minimum_visible_fraction':min_visible,'seed':seed,
        'raw_modified':False,'original_owner_masks_recovered':False,
        'kept_bbox_evidence':'AI visual adjudication; original final owner masks unavailable',
        'regenerated_bbox_evidence':'exact stored new owner map and lossless COCO RLE'}
    write_json(manifest_path,manifest)
    return manifest


def verify_resolution_dataset(manifest: dict[str, Any], *, repo: Path, raw_root: Path, out_dir: Path) -> dict[str, Any]:
    """Check all new masks against RLE/bbox/area, all sources, instance multisets, IDs and task labels."""
    from collections import Counter
    from tools.ai_annotation_review import file_sha256
    coco=json.loads((out_dir/'annotations.json').read_text())
    if file_sha256(out_dir/'annotations.json')!=manifest['dataset_sha256']:raise ValueError('Dataset checksum changed')
    images={r['id']:r for r in coco['images']}
    if len(images)!=manifest['resolved_cases'] or len({a['id'] for a in coco['annotations']})!=len(coco['annotations']):
        raise ValueError('Dataset ID/count mismatch')
    byimage={iid:[] for iid in images}
    for ann in coco['annotations']:byimage[ann['image_id']].append(ann)
    if Counter(a['category_id'] for a in coco['annotations'])!=Counter({int(k):v for k,v in manifest['sku_multiset'].items()}):
        raise ValueError('Dataset SKU multiset changed')
    mask_objects=0
    for resolution in manifest['resolutions']:
        image=images[resolution['output_image_id']];anns=byimage[image['id']]
        if Counter(a['category_id'] for a in anns)!=Counter({int(k):v for k,v in resolution['sku_multiset'].items()}):
            raise ValueError('Image instance/SKU multiset changed')
        with Image.open(repo/image['file_name']) as photo:
            photo.load()
            if photo.size!=(image['width'],image['height']):raise ValueError('Image dimensions changed')
        owner=None
        if resolution['owner_mask']:
            with np.load(repo/resolution['owner_mask']) as stored:owner=stored['owner']
            if owner.shape!=(image['height'],image['width']):raise ValueError('Owner dimensions mismatch')
        for ann in anns:
            x,y,w,h=ann['bbox']
            if not (x>=0 and y>=0 and w>0 and h>0 and x+w<=image['width'] and y+h<=image['height']):
                raise ValueError('BBox out of native bounds')
            if not 0<ann['area']<=w*h:raise ValueError('Invalid annotation area')
            if owner is not None:
                mask=decode_rle(ann['segmentation'])
                if not np.array_equal(mask,owner==ann['owner_slot']):raise ValueError('RLE differs from verified owner map')
                if mask_bbox(mask)!=ann['bbox'] or int(mask.sum())!=ann['area']:raise ValueError('BBox/area differ from owner pixels')
                if ann['visible']+1e-6<manifest['minimum_visible_fraction']:raise ValueError('Instance visibility below policy')
                mask_objects+=1
        if resolution['status']=='kept_by_ai' and file_sha256(repo/image['file_name'])!=resolution['original_image_sha256']:
            raise ValueError('Kept source photo changed')
    for path,digest in manifest['output_asset_sha256'].items():
        if file_sha256(repo/path)!=digest:raise ValueError('Derived asset changed')
    for path,digest in manifest['source_photo_sha256'].items():
        if file_sha256(raw_root/path)!=digest:raise ValueError('Raw photo changed')
    for path,digest in manifest['source_cutout_sha256'].items():
        if file_sha256(raw_root/path)!=digest:raise ValueError('Source cutout changed')
    if manifest['task_files']:
        from autocheckout.taskcfg import TaskConfig
        config=TaskConfig.load(out_dir/'task_config.json')
        for path in manifest['task_files']:
            p=repo/path;payload=json.loads(p.read_text())
            task_id=int(p.stem.split('_')[2])
            allowed=set(range(config.seen_classes(5))) if 'gt_full' in p.stem else set(config.task(task_id).labels)
            if not {a['category_id'] for a in payload['annotations']}<=allowed:raise ValueError('Incorrect task labels')
    return {'passed':True,'images':len(images),'objects':len(coco['annotations']),
            'regenerated_mask_objects_verified':mask_objects,'native_geometry':True,
            'sku_multisets_preserved':True,'sources_unchanged':True,'task_mapping_checked':bool(manifest['task_files'])}


def update_review_decisions(path: Path, resolutions: list[dict[str, Any]], *, reviewer: str) -> None:
    """Apply only scope decisions with actual output evidence; preserve unrelated/manual rows."""
    import csv
    with path.open(newline='') as source:
        reader=csv.DictReader(source);fields=reader.fieldnames;rows=list(reader)
    updates={tuple(r['case_key']):r for r in resolutions}
    if len(updates)!=len(resolutions) or not reviewer:raise ValueError('Duplicate/missing resolution reviewer')
    existing={(int(r['source_task']),int(r['annotation_id'])):r for r in rows}
    if not set(updates)<=set(existing):raise ValueError('Resolution key absent from review CSV')
    for key,item in updates.items():
        row=existing[key]
        if row['decision']!='pending' and row['reviewer']!=reviewer:
            raise ValueError('Existing manual decision must be preserved')
        if item['decision'] not in ['keep_with_justification','regenerate'] or item['status'] not in ['kept_by_ai','regenerated']:
            raise ValueError('Resolution not applied')
        if len(item['evidence'].strip())<30:raise ValueError('Resolution evidence missing')
        row.update(decision=item['decision'],evidence=item['evidence'],reviewer=reviewer)
    temporary=path.with_suffix('.csv.tmp')
    with temporary.open('w',newline='') as target:
        writer=csv.DictWriter(target,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temporary.replace(path)
