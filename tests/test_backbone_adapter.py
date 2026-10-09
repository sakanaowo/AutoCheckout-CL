"""Exercise the production encoder contract with lightweight feature-provider doubles.

No ML packages are installed by these tests. Real timm/CUDA checks are notebook gates.
"""
import ast
from contextlib import nullcontext
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


class Mask:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def __getitem__(self, key):
        return self

    def float(self):
        return self

    def to(self, dtype):
        return self


def encoder_class():
    source = ast.parse((ROOT / 'pdp/models/modeling_deformable_detr.py').read_text())
    definition = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'DeformableDetrConvEncoder')
    namespace = {
        'nn': SimpleNamespace(Module=object, functional=SimpleNamespace(
            interpolate=lambda data, size: Mask(size))),
        'torch': SimpleNamespace(Tensor=object, bool=bool, no_grad=nullcontext),
        'requires_backends': lambda *a: None, 'replace_batch_norm': lambda m: None,
    }
    helper = ROOT / 'pdp/models/backbones.py'
    if helper.exists():
        namespace.update(runpy.run_path(str(helper)))
    exec(compile(ast.Module(body=[definition], type_ignores=[]), 'production encoder', 'exec'), namespace)
    return namespace['DeformableDetrConvEncoder'], namespace


def provider(metadata, calls):
    def create(name, **kwargs):
        calls.append(kwargs)
        indices = kwargs.get('out_indices', tuple(range(len(metadata))))
        # Real timm rejects ResNet's index 4 for four-stage ConvNeXt.
        chosen = [metadata[i] for i in indices]
        info = SimpleNamespace(channels=lambda: [r['num_chs'] for r in chosen],
            get_dicts=lambda: chosen, out_indices=indices)
        class Features:
            feature_info = info
            def modules(self):
                return [self]
            def named_parameters(self):
                return []
            def __call__(self, resolution):
                return [SimpleNamespace(shape=(1, r['num_chs'], resolution//r['reduction'],
                    resolution//r['reduction'])) for r in chosen]
        return Features()
    return create


def config(name, levels=4):
    return SimpleNamespace(use_timm_backbone=True, backbone=name, dilation=False,
        use_pretrained_backbone=False, num_channels=3, num_feature_levels=levels)


@pytest.mark.parametrize('resolution', [640, 800])
def test_convnext_stages_channels_and_masks_follow_metadata(resolution):
    cls, namespace = encoder_class()
    calls = []
    rows = [dict(num_chs=c, reduction=r, module=f'stage{i}')
            for i,(c,r) in enumerate(zip([128,256,512,1024],[4,8,16,32]))]
    namespace['create_model'] = provider(rows, calls)
    model = cls(config('convnextv2_base.fcmae_ft_in22k_in1k'))
    assert model.intermediate_channel_sizes == [256,512,1024]
    outputs = model.forward(resolution, Mask((resolution,resolution)))
    assert [f.shape[1] for f,m in outputs] == [256,512,1024]
    assert [f.shape[-1] for f,m in outputs] == [resolution//8,resolution//16,resolution//32]
    assert all(mask.shape == feature.shape[-2:] for feature,mask in outputs)
    assert calls[0]['pretrained'] is False


def test_resnet_keeps_existing_stride_and_channel_contract():
    cls, namespace = encoder_class()
    calls = []
    rows = [dict(num_chs=c, reduction=r, module=f'layer{i}')
            for i,(c,r) in enumerate(zip([64,256,512,1024,2048],[2,4,8,16,32]))]
    namespace['create_model'] = provider(rows, calls)
    model = cls(config('resnet50'))
    assert model.intermediate_channel_sizes == [512,1024,2048]
    assert [f.shape[-1] for f,m in model.forward(640, Mask((640,640)))] == [80,40,20]


def test_single_feature_level_uses_last_available_stage():
    cls, namespace = encoder_class()
    namespace['create_model'] = provider([dict(num_chs=128*(2**i),reduction=4*(2**i),module=str(i)) for i in range(4)], [])
    model = cls(config('convnextv2_base', levels=1))
    assert model.intermediate_channel_sizes == [1024]
    assert len(model.forward(640, Mask((640,640)))) == 1


def test_checkpoint_feature_metadata_cannot_silently_change():
    cls, namespace = encoder_class()
    namespace['create_model'] = provider([dict(num_chs=128*(2**i),reduction=4*(2**i),module=str(i)) for i in range(4)], [])
    cfg = config('convnextv2_base')
    cfg.backbone_feature_info = []
    with pytest.raises(ValueError, match='Checkpoint'):
        cls(cfg)


def transfer_plan(source, target):
    helpers = runpy.run_path(str(ROOT/'pdp/models/backbones.py'))
    return helpers['plan_detector_transfer'](source, target)


def test_detector_transfer_keeps_transformer_and_resets_backbone_projections_classifier():
    source = {'model.encoder.weight':(256,256), 'model.decoder.weight':(256,256),
              'model.backbone.old.weight':(64,3,7,7), 'model.input_proj.0.0.weight':(256,512,1,1),
              'class_embed.0.weight':(91,256)}
    target = {'model.encoder.weight':(256,256), 'model.decoder.weight':(256,256),
              'model.backbone.new.weight':(128,3,4,4), 'model.input_proj.0.0.weight':(256,256,1,1),
              'class_embed.0.weight':(225,256),'model.prompts.shared_p_0':(100,256)}
    plan = transfer_plan(source,target)
    assert set(plan['load_keys'])=={'model.encoder.weight','model.decoder.weight'}
    assert set(plan['new_target_keys'])==set(target)-set(plan['load_keys'])
    assert set(plan['skipped_source_keys'])==set(source)-set(plan['load_keys'])


@pytest.mark.parametrize('case', ['missing','shape','unexpected'])
def test_detector_transfer_rejects_incompatible_transformer(case):
    source = {'model.encoder.weight':(256,256),'model.decoder.weight':(256,256)}
    target = dict(source)
    if case=='missing':del source['model.encoder.weight']
    elif case=='shape':source['model.decoder.weight']=(512,256)
    else:source['model.decoder.unknown']=(256,)
    with pytest.raises(ValueError,match='core'):
        transfer_plan(source,target)


def test_local_pretrained_backbone_file_is_passed_to_provider_without_hub_lookup(tmp_path):
    cls, namespace = encoder_class()
    calls=[]
    namespace['create_model']=provider([dict(num_chs=128*(2**i),reduction=4*(2**i),module=str(i)) for i in range(4)],calls)
    cfg=config('convnextv2_base.fcmae_ft_in22k_in1k')
    cfg.use_pretrained_backbone=True
    cfg.backbone_pretrained_file=str(tmp_path/'model.safetensors')
    cls(cfg)
    overlay=calls[0]['pretrained_cfg_overlay']
    assert overlay['file']==cfg.backbone_pretrained_file
    assert overlay['hf_hub_id'] is None and not overlay['url']


def test_pretrained_provider_is_protected_from_detector_post_init():
    helpers=runpy.run_path(str(ROOT/'pdp/models/backbones.py'))
    cfg=config('convnextv2_base')
    cfg.use_pretrained_backbone=True
    model,indices,channels=helpers['build_feature_backbone'](cfg,provider([
        dict(num_chs=128*(2**i),reduction=4*(2**i),module=str(i)) for i in range(4)],[]))
    assert all(getattr(m,'_is_hf_initialized',False) for m in model.modules())
