# 新配置继承了基本配置，并做了必要的修改
_base_ = '../configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=5), mask_head=dict(num_classes=5)))

# 修改数据集相关配置
data_root = 'data/hist-cell/'
metainfo = {
    'classes': ('1', '2', '3', '4', '5')
}
train_pipeline = [
    dict(type='Resize', scale=(256, 256), keep_ratio=True)
]
test_pipeline = [
    dict(type='Resize', scale=(256, 256), keep_ratio=True)
]
train_dataloader = dict(
    batch_size=1,
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

# 设置超参数
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.05))
auto_scale_lr = dict(enable=True, base_batch_size=16)

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/valid_coco.json')
test_evaluator = dict(ann_file=data_root + 'annotations/test_coco.json')

# 使用预训练的 Mask R-CNN 模型权重来做初始化，可以提高模型性能
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
