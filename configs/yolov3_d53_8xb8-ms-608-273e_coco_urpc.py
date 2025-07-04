# ===================================================================================
# === YOLOv3 with DarkNet-53 Backbone for URPC Dataset - 최종 수정본 ===
# ===================================================================================

# 继承了标准的12轮训练计划(schedule_1x)和默认的运行环境配置
_base_ = [
    '../mmdetection/configs/_base_/schedules/schedule_1x.py',
    '../mmdetection/configs/_base_/default_runtime.py'
]

# 1. 数据集和类别定义 (Dataset and Class Definitions)
# -----------------------------------------------------------------
data_root = 'data/urpc2020/' # <-- 【关键修正】: 指向您项目的正确数据根目录
class_name = ('echinus', 'starfish', 'holothurian', 'scallop')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)])

# 2. 模型定义 (Model Definition)
# -----------------------------------------------------------------
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0, 0, 0],
    std=[255., 255., 255.],
    bgr_to_rgb=True,
    pad_size_divisor=32)

model = dict(
    type='YOLOV3',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5),
        # <-- 【关键修正】: 加载DarkNet-53的预训练权重，而不是MobileNetV2的
        init_cfg=dict(type='Pretrained', checkpoint='checkpoints/yolov3_d53_320_273e_coco-421362b6.pth')
    ),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=num_classes, # <-- 自动使用上面定义的类别数
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0, reduction='sum'),
        loss_conf=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0, reduction='sum'),
        loss_xy=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=2.0, reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    train_cfg=dict(assigner=dict(type='GridAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100)
)

# 3. 数据处理流水线 (Data Pipelines)
# -----------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Expand', mean=data_preprocessor['mean'], to_rgb=data_preprocessor['bgr_to_rgb'], ratio_range=(1, 2)),
    dict(type='MinIoURandomCrop', min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9), min_crop_size=0.3),
    dict(type='RandomResize', scale=[(320, 320), (608, 608)], keep_ratio=True), # 多尺度训练
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(608, 608), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# 4. 数据加载器 (Dataloaders)
# -----------------------------------------------------------------
# 注意：batch_size=64需要非常大的GPU显存。如果遇到显存不足(Out of Memory)的错误,
# 请从32或16开始尝试。
train_dataloader = dict(
    batch_size=64, # <-- 建议从32开始
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        # <-- 【关键修正】: 指向您项目的正确训练集标注和图片路径
        ann_file='annotations/train.json',
        data_prefix=dict(img='train_split/JPEGImages/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=None))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        # <-- 【关键修正】: 指向您项目的正确验证集标注和图片路径
        ann_file='annotations/val.json',
        data_prefix=dict(img='val_split/JPEGImages/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None))

# test_dataloader = val_dataloader # 为了方便，测试加载器直接复用验证加载器的配置

# 定义独立的 test_dataloader
test_dataloader = dict(
    batch_size=1, # 测试时 batch_size 通常为 1
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        # --- 核心修改点 ---
        ann_file='annotations/test.json',      # 指向您的测试集标注文件
        data_prefix=dict(img='test/JPEGImages/'),         # 指向您的测试集图片文件夹
        # --------------------
        test_mode=True,
        pipeline=test_pipeline, # 复用 test_pipeline 即可
        backend_args=None))

# 5. 评估器 (Evaluator)
# -----------------------------------------------------------------
val_evaluator = dict(
    type='CocoMetric',
    # <-- 【关键修正】: 指向正确的验证集标注文件
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    backend_args=None)

# test_evaluator = val_evaluator # 测试评估器也复用验证评估器的配置
# 定义独立的 test_evaluator
test_evaluator = dict(
    type='CocoMetric',
    # --- 核心修改点 ---
    classwise=True,
    format_only=False,
    ann_file=data_root + 'annotations/test.json', # 同样指向测试集标注文件
    # --------------------
    metric='bbox',
    backend_args=None)


# 6. 训练、优化和学习率计划 (Training, Optimization, and Learning Rate Schedule)
# -----------------------------------------------------------------
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1) # <-- 覆盖_base_，训练50轮
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=2000),
    dict(type='MultiStepLR', by_epoch=True, milestones=[218, 246], gamma=0.1) # 这个计划是为273轮设计的，对于50轮训练，它可能不会触发。但没关系。
]

# 7. 默认钩子和运行设置 (Default Hooks and Runtime Settings)
# -----------------------------------------------------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50), # 每50次迭代打印一次日志
    checkpoint=dict(type='CheckpointHook', interval=5, save_best='auto'), # 每5个epoch保存一次，并自动保存最好的模型
)

# 自动学习率缩放
auto_scale_lr = dict(base_batch_size=64)

# 可视化配置
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)
# 新增: 设置工作目录，保存日志和模型
work_dir = './work_dirs/yolov3_d53_urpc_custom'
