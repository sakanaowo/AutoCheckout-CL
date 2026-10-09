import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools import annotation_resolution as resolution
from tools.ai_annotation_review import file_sha256


def review_files(tmp_path):
    for name in ['card','native','edge']:
        Image.new('RGB',(120,80),'gray').save(tmp_path/(name+'.png'))
    Image.new('RGBA',(40,70),(255,0,0,255)).save(tmp_path/'cutout.png')
    return {'source_task':2,'annotation_id':42,'image_id':4,'rpc_category_id':10,'iou':.84,
        'bbox':[5,5,100,60],'polygon_bbox':[15,5,90,60], 'source_sha256':'source',
        'native_crop':str(tmp_path/'native.png'),'edge_zoom':str(tmp_path/'edge.png'),
        'largest_gap_edge':'left','crop_box':[0,0,120,80]}


def verdict(case, decision='keep_with_justification', uncertain=False):
    return {'source_task':case['source_task'],'annotation_id':case['annotation_id'],
        'decision':decision,'geometry_uncertain':uncertain,'verified_owner_mask_available':False,
        'observations':['The left neck fragment is near the bounding extreme.',
                        'The body and bottom are enclosed without clipping.'],
        'reason':'The visible fragments plausibly support the bbox; polygon simplification can omit tiny contours.',
        'supporting_fragment':'Visible neck at the top-left side, body at the right/bottom sides.',
        'confidence':.9}


def reply(value):
    return {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(value)}]}]}


def test_resolution_sends_native_edge_cutout_context_and_requires_an_operational_action(tmp_path):
    case=review_files(tmp_path)
    p=resolution.build_adjudication_request(case,tmp_path/'card.png',tmp_path/'cutout.png',model='gpt-6.1-sol',reasoning_effort='medium')
    assert p['reasoning']['effort']=='medium'
    assert len([r for r in p['input'][0]['content'] if r['type']=='input_image'])==4
    choices=p['text']['format']['schema']['properties']['decision']['enum']
    assert set(choices)=={'keep_with_justification','regenerate'} and 'pending' not in choices


def test_adjudication_rejects_invented_masks_wrong_ids_and_incomplete_responses(tmp_path):
    case=review_files(tmp_path)
    for change in [{'verified_owner_mask_available':True},{'annotation_id':999},{'decision':'pending'}]:
        with pytest.raises(ValueError):resolution.parse_adjudication(reply({**verdict(case),**change}),case)
    with pytest.raises(ValueError):resolution.parse_adjudication({'status':'incomplete'},case)


def test_uncertain_keep_is_resolved_by_regeneration_instead_of_remaining_pending(tmp_path):
    case=review_files(tmp_path)
    r=resolution.parse_adjudication(reply(verdict(case,uncertain=True)),case)
    assert resolution.final_action(r)=='regenerate'
    assert resolution.final_action(verdict(case))=='keep_with_justification'


def test_uncompressed_coco_rle_roundtrips_disconnected_pixels_and_exact_bbox():
    mask=np.zeros((8,11),dtype=bool);mask[2:6,3:7]=True;mask[0,10]=True
    encoded=resolution.encode_rle(mask)
    assert encoded['size']==[8,11]
    decoded=resolution.decode_rle(encoded)
    assert np.array_equal(mask,decoded)
    assert resolution.mask_bbox(decoded)==[3,0,8,6]


def scene_files(tmp_path):
    background=tmp_path/'background.jpg';Image.new('RGB',(100,90),'gray').save(background)
    for name,color in [('red',(250,0,0,255)),('green',(0,250,0,255))]:
        canvas=Image.new('RGBA',(30,25),(0,0,0,0))
        canvas.paste(color,(3,4,27,22));canvas.save(tmp_path/(name+'.png'))
    objects=[{'id':10,'category_id':1,'bbox':[15,15,35,30],'cutout':'red.png'},
             {'id':20,'category_id':2,'bbox':[30,25,35,30],'cutout':'green.png'}]
    return background,objects


def test_regeneration_preserves_skus_stores_exact_owner_masks_and_is_reproducible(tmp_path):
    background,objects=scene_files(tmp_path)
    hashes={p.name:file_sha256(p) for p in tmp_path.glob('*.png')}
    first=resolution.compose_replacement(objects,raw_root=tmp_path,background=background,size=(100,90),seed=3,min_visible=.7)
    second=resolution.compose_replacement(objects,raw_root=tmp_path,background=background,size=(100,90),seed=3,min_visible=.7)
    assert np.array_equal(np.asarray(first['image']),np.asarray(second['image']))
    assert np.array_equal(first['owner'],second['owner'])
    assert sorted(r['category_id'] for r in first['annotations'])==[1,2]
    for r in first['annotations']:
        mask=resolution.decode_rle(r['segmentation'])
        assert np.array_equal(mask,first['owner']==r['owner_slot'])
        assert r['bbox']==resolution.mask_bbox(mask) and r['area']==int(mask.sum())
        assert r['visible']>=.7
    assert hashes=={p.name:file_sha256(p) for p in tmp_path.glob('*.png')}


def test_scene_keeps_every_instance_even_with_coincident_source_boxes(tmp_path):
    background,objects=scene_files(tmp_path)
    objects=[{**objects[i%2],'id':i,'bbox':[25,25,35,30]} for i in range(9)]
    scene=resolution.compose_replacement(objects,raw_root=tmp_path,background=background,size=(100,90),seed=9,min_visible=.85)
    assert len(scene['annotations'])==9
    assert all(r['area']>0 and r['visible']>=.85 for r in scene['annotations'])


def test_coco_rle_uses_fortran_order_with_an_initial_zero_run():
    mask=np.array([[True,False],[True,True]])
    assert resolution.encode_rle(mask)=={'size':[2,2],'counts':[0,2,1,1]}


def test_scope_dataset_actually_regenerates_and_keeps_other_photo_geometry(tmp_path):
    background,objects=scene_files(tmp_path)
    for iid in range(2):Image.new('RGB',(100,90),'gray').save(tmp_path/f'photo{iid}.jpg')
    records=[{'source_task':1,'annotation_id':10,'image_id':0,'image_file':'photo0.jpg','bbox':objects[0]['bbox'],'source_sha256':'s'},
             {'source_task':1,'annotation_id':20,'image_id':1,'image_file':'photo1.jpg','bbox':objects[1]['bbox'],'source_sha256':'s'}]
    templates=[]
    for c in records:
        templates.append({'case':c,'image':{'id':c['image_id'],'file_name':c['image_file'],'width':100,'height':90},
            'annotations':[{**a,'image_id':c['image_id'],'area':400,'segmentation':[]} for a in objects]})
    decisions=[{'case_key':[1,10],'review':verdict(records[0]),'fingerprint':'one'},
               {'case_key':[1,20],'review':verdict(records[1],decision='regenerate',uncertain=True),'fingerprint':'two'}]
    first=resolution.apply_resolution_dataset(records,decisions,repo=tmp_path,raw_root=tmp_path,
        out_dir=tmp_path/'processed',backgrounds=[background],seed=1,templates=templates)
    assert first['resolved_cases']==2 and first['kept_images']==1 and first['regenerated_images']==1
    coco=json.loads((tmp_path/'processed/annotations.json').read_text())
    assert len(coco['images'])==2 and len(coco['annotations'])==4
    kept=[a for a in coco['annotations'] if a['image_id']==coco['images'][0]['id']]
    assert [a['bbox'] for a in kept]==[a['bbox'] for a in objects]
    assert Path(tmp_path/coco['images'][1]['file_name']).exists()
    resolution.verify_resolution_dataset(first,repo=tmp_path,raw_root=tmp_path,out_dir=tmp_path/'processed')
    second=resolution.apply_resolution_dataset(records,decisions,repo=tmp_path,raw_root=tmp_path,
        out_dir=tmp_path/'processed',backgrounds=[background],seed=1,templates=templates)
    assert first['dataset_sha256']==second['dataset_sha256']


def test_csv_resolution_only_closes_authorized_rows_and_preserves_manual_work(tmp_path):
    import csv
    path=tmp_path/'review.csv'
    rows=[{'source_task':1,'annotation_id':10,'decision':'pending','evidence':'','reviewer':''},
          {'source_task':1,'annotation_id':20,'decision':'keep_with_justification','evidence':'Already reviewed by teammate.','reviewer':'teammate'}]
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    items=[{'case_key':[1,10],'decision':'regenerate','status':'regenerated',
            'evidence':'Actual replacement photo and owner mask were saved in derived artifacts.'}]
    resolution.update_review_decisions(path,items,reviewer='OpenAI/test')
    updated=list(csv.DictReader(path.open()));assert updated[0]['decision']=='regenerate' and updated[1]['reviewer']=='teammate'
    before=path.read_bytes()
    with pytest.raises(ValueError,match='manual'):
        resolution.update_review_decisions(path,[{**items[0],'case_key':[1,20]}],reviewer='OpenAI/test')
    assert path.read_bytes()==before


def test_composited_pixels_show_each_owned_instance_not_just_a_correct_mask(tmp_path):
    background,objects=scene_files(tmp_path)
    scene=resolution.compose_replacement(objects,raw_root=tmp_path,background=background,size=(100,90),seed=7,min_visible=.85)
    rgb=np.asarray(scene['image']).astype(float)
    for slot,channel in [(1,0),(2,1)]:
        values=rgb[scene['owner']==slot].mean(axis=0)
        assert values[channel]>max(values[j] for j in range(3) if j!=channel)+40
