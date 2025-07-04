# 您的原始代码部分，原封不动
_base_ = '../mmdetection/configs/yolo/yolov3_mobilenetv2_8xb24-ms-416-300e_coco.py'

# yapf:disable
model = dict(
    # 新增下面这部分，来覆盖掉_base_文件中的下载行为
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./checkpoints/mobilenet_v2_batch256_imagenet-ff34753d.pth'  # <-- 指向本地文件
        )
    ),
    bbox_head=dict(
        anchor_generator=dict(
            base_sizes=[[(220, 125), (128, 222), (264, 266)],
                        [(35, 87), (102, 96), (60, 170)],
                        [(10, 15), (24, 36), (72, 42)]]),
        # 新增：将模型的类别数改为我们项目的4类
        num_classes=4
    )
)
# yapf:enable

# 您的原始代码部分，原封不动
input_size = (320, 320)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=[123.675, 116.28, 103.53],
        to_rgb=True,
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', scale=input_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=input_size, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


# ----- 以下是为让您的配置能运行而新增的必要部分 -----

# 新增：定义我们项目的数据集信息
data_root = 'data/urpc2020/' 
class_name = ('echinus', 'starfish', 'holothurian', 'scallop')
metainfo = dict(classes=class_name)

# 新增：补全数据加载器的配置，应用您的数据处理流程，并指向正确的数据路径
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(  # 这是第一层，对应 MultiImageMixDataset
        dataset=dict(  # 这是第二层，对应 CocoDataset，我们的修改在这里！
            data_root=data_root,
            metainfo=metainfo,
            ann_file='annotations/train.json',
            data_prefix=dict(img='train_split/JPEGImages/'),
            # 注意：这里的 pipeline 是针对单张图片的，_base_文件会用它
            
        )
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val_split/JPEGImages/'),
        pipeline=test_pipeline) # 使用您定义的test_pipeline
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/JPEGImages/'),
        pipeline=test_pipeline) # 使用您定义的test_pipeline
)


# 新增：定义评估器
val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = dict(ann_file=data_root + 'annotations/test.json')

# 新增：定义训练轮数、预训练模型和工作目录
max_epochs = 50 
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

# 新增：加载与您选择的_base_文件匹配的预训练权重
load_from = './checkpoints/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
# 新增：设置工作目录，保存日志和模型
work_dir = './work_dirs/yolov3_my_custom'

# 新增：优化检查点保存策略
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=5, save_best='auto'),
)
