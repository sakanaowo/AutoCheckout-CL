"""Feature metadata shared by the detector adapter (no ML imports)."""


def build_feature_backbone(config, create_model):
    """Discover provider outputs, then keep up to three final stages for Deformable DETR.

    ResNet uses strides 8/16/32; ConvNeXt-V2 uses the matching final three of its
    four native stages. The detector adds a fourth projection level itself.
    """
    if config.num_feature_levels < 1:
        raise ValueError('num_feature_levels must be positive')
    kwargs = {'output_stride': 16} if config.dilation else {}
    local_weights = getattr(config, 'backbone_pretrained_file', None)
    if local_weights and config.use_pretrained_backbone:
        kwargs['pretrained_cfg_overlay'] = {'file': str(local_weights), 'hf_hub_id': None, 'url': ''}
    backbone = create_model(config.backbone, pretrained=config.use_pretrained_backbone,
        features_only=True, in_chans=config.num_channels, **kwargs)
    # Transformers 4.37 post_init recursively initializes Conv/Linear layers.
    # A backbone already loaded by timm must retain its pretrained tensors.
    if config.use_pretrained_backbone:
        for module in backbone.modules():
            module._is_hf_initialized = True
    rows = backbone.feature_info.get_dicts()
    if not rows or any(int(row['num_chs']) <= 0 or int(row['reduction']) <= 0 for row in rows):
        raise ValueError('Invalid backbone feature channels/strides')
    strides = [int(row['reduction']) for row in rows]
    if strides != sorted(strides):
        raise ValueError('Backbone outputs must be ordered by stride')
    count = min(3, config.num_feature_levels, len(rows))
    positions = tuple(range(len(rows)-count, len(rows)))
    selected = [{'output_index': i, 'module': str(rows[i].get('module', i)),
                 'channels': int(rows[i]['num_chs']), 'stride': int(rows[i]['reduction'])}
                for i in positions]
    previous = getattr(config, 'backbone_feature_info', None)
    if previous is not None and previous != selected:
        raise ValueError('Checkpoint backbone feature metadata differs from live provider')
    config.backbone_feature_info = selected
    return backbone, positions, [r['channels'] for r in selected]


def plan_detector_transfer(source_shapes, target_shapes):
    """Keep compatible detector core; new backbone/projections/classes/prompts stay fresh.

    This is an intentional cross-backbone warm start, distinct from exact resume.
    All source/target shapes are compared before any tensor is loaded.
    """
    def fresh(name):
        return (name.startswith(('model.backbone.', 'model.input_proj.', 'class_embed.',
                                 'model.decoder.class_embed.')) or '.prompts.' in name)
    load = []
    new = []
    for name, shape in target_shapes.items():
        if fresh(name):
            new.append(name)
        elif name not in source_shapes or tuple(source_shapes[name]) != tuple(shape):
            raise ValueError(f'Incompatible detector core tensor: {name}')
        else:
            load.append(name)
    unexpected = [name for name in source_shapes if name not in target_shapes and not fresh(name)]
    if unexpected or not any(n.startswith('model.encoder.') for n in load) or not any(
            n.startswith('model.decoder.') for n in load):
        raise ValueError(f'Incompatible detector core coverage: {unexpected}')
    return {'load_keys': sorted(load), 'new_target_keys': sorted(new),
            'skipped_source_keys': sorted(set(source_shapes)-set(load))}
