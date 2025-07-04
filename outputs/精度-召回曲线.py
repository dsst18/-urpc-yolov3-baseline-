# 计算PR曲线数据
from mmdetection/mmdet.evaluation import eval_map
import numpy as np

pr_results = eval_map(
    results=detection_results,  # 模型输出
    annotations=gt_annotations,  # 真实标注
    iou_thr=0.5,  # IoU阈值
    scale_ranges=None,  # 可设置尺度范围
    eval_mode='area')  # 评估模式

# 提取PR数据
precision = pr_results[0]['precision'][:, :, 0, -1].mean(axis=1)
recall = pr_results[0]['recall'][:, 0, -1]

# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, 'b-', linewidth=2)
plt.fill_between(recall, precision, alpha=0.2, color='b')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title(f'Precision-Recall Curve (AP={ap:.3f})', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim([0, 1.0])
plt.ylim([0, 1.05])
plt.savefig('pr_curve.png', dpi=300)