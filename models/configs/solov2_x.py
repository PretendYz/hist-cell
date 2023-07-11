_base_ = '../configs/solov2/solov2_r50_fpn_ms-3x_coco.py'

# model settings
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')),
    mask_head=dict(
        mask_feature_head=dict(conv_cfg=dict(type='DCNv2')),
        num_classes=5,
        dcn_cfg=dict(type='DCNv2'),
        dcn_apply_to_all_conv=True))

# 修改数据集相关配置
data_root = 'data/hist-cell/'
metainfo = {
    'classes': ('1', '2', '3', '4', '5')
}
# train_pipeline = [
#     dict(type='Resize', scale=(256, 256), keep_ratio=True)
# ]
# test_pipeline = [
#     dict(type='Resize', scale=(256, 256), keep_ratio=True)
# ]
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train_coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/valid_coco.json',
        data_prefix=dict(img='val/')))
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test_coco.json',
        data_prefix=dict(img='test/')))

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/valid_coco.json')
test_evaluator = dict(ann_file=data_root + 'annotations/test_coco.json')

# 使用预训练的模型权重来做初始化，可以提高模型性能
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_x101_dcn_fpn_3x_coco/solov2_x101_dcn_fpn_3x_coco_20220513_214337-aef41095.pth'