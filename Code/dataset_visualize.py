# dataset settings
dataset_type = 'CocoDataset'
classes = ('car', 'hov', 'person', 'motorcycle')

project_root = ''
data_root = 'Dataset/'

img_scale = (832, 832)  # height, width

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, center_ratio_range=(1, 1), prob=1),
    dict(
        type='RandomAffine',
        max_rotate_degree=1,
        max_translate_ratio=0,
        scaling_ratio_range=(1, 1.1),
        ),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(1.5, 2),
        pad_val=114.0,
        prob=0.3),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(1664, 1664), keep_ratio=True),  # multi-scale
    dict(type='SmallObjectAugmentation',all_objects=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'SuperResolution_Training_Sliced_coco.json',
            img_prefix=data_root + 'SuperResolution_Training_Sliced',
            classes=classes,
            pipeline=[dict(type='LoadImageFromFile'),
                      dict(type='LoadAnnotations', with_bbox=True)],
            filter_empty_gt=False, ),
        pipeline=train_pipeline))
