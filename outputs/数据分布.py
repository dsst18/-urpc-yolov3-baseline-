import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

# 数据集配置 - 直接指定三个文件夹路径
train_dir = "data/urpc2020/train_split/Annotations"  # 替换为您的训练集路径
val_dir = "data/urpc2020/val_split/Annotations"     # 替换为您的验证集路径
test_dir = "data/urpc2020/test/Annotations"   # 替换为您的测试集路径

# 类别映射（根据您的实际类别调整）
category_mapping = {
    "holothurian": "holothurian",
    "echinus": "echinus",
    "scallop": "scallop",
    "starfish": "starfish"
    # 添加更多类别...
}

def parse_annotation(xml_path):
    """解析单个XML标注文件（适配您的格式）"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取文件名标识（frame标签）
        frame_elem = root.find('frame')
        img_name = frame_elem.text if frame_elem is not None else "unknown"
        
        # 初始化对象列表
        objects = []
        
        # 遍历所有object标签
        for obj in root.findall('object'):
            # 获取类别名称
            name_elem = obj.find('name')
            if name_elem is None:
                continue
            cls_name = name_elem.text
            
            # 获取边界框
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
                
            # 提取坐标值
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # 计算目标尺寸
            width = xmax - xmin
            height = ymax - ymin
            area = width * height
            
            # 由于XML中没有图像尺寸信息，我们无法计算面积占比
            # 但可以统计其他信息
            objects.append({
                "class": cls_name,
                "class_zh": category_mapping.get(cls_name, cls_name),
                "width": width,
                "height": height,
                "area": area,
                "area_ratio": 0  # 设为0或None
            })
        
        return objects
    except Exception as e:
        print(f"解析 {xml_path} 时出错: {e}")
        return []

def analyze_dataset(folder_path):
    """分析指定文件夹的数据集"""
    # 获取所有XML文件
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
    
    # 验证数据
    if not xml_files:
        print(f"警告: 在 {folder_path} 中没有找到XML文件!")
        return {
            "image_count": 0,
            "object_count": 0,
            "class_count": {},
            "size_distribution": {},
            "per_image_counts": [],
            "area_ratios": []
        }
    
    # 初始化统计容器
    stats = {
        "class_count": defaultdict(int),
        "size_distribution": defaultdict(int),
        "image_count": len(xml_files),
        "object_count": 0,
        "per_image_counts": [],
        "area_values": []
    }
    
    for xml_file in xml_files:
        xml_path = os.path.join(folder_path, xml_file)
        objects = parse_annotation(xml_path)

        if not objects:
            continue
            
        stats['per_image_counts'].append(len(objects))
        stats['object_count'] += len(objects)
        
        for obj in objects:
            # 类别统计
            stats['class_count'][obj['class_zh']] += 1
            
            # 尺寸分类（使用绝对尺寸而非相对比例）
            max_dim = max(obj['width'], obj['height'])
            if max_dim < 32:
                size_category = "small targets(<32px)"
            elif max_dim <= 256:
                size_category = "medium targets(32-256px)"
            else:
                size_category = "large targets(>256px)"
                
            stats['size_distribution'][size_category] += 1
            
            # 使用面积代替面积占比（因为没有图像尺寸）
            stats['area_values'].append(obj['area'])
        
    
    # 计算衍生统计量
    if stats['object_count'] > 0:
        stats['class_distribution'] = {
            cls: count / stats['object_count'] 
            for cls, count in stats['class_count'].items()
        }
    else:
        stats['class_distribution'] = {}
    stats['avg_objects_per_image'] = np.mean(stats['per_image_counts']) if stats['per_image_counts'] else 0
    stats['median_area'] = np.median(stats['area_values']) if stats['area_values'] else 0
    
    return stats   
   

def visualize_results(stats_dict):
    """可视化三个数据集的结果"""
    sets = ['training set', 'validation set', 'test set']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    plt.figure(figsize=(18, 12))
    
    # 1. 类别分布对比
    all_classes = set()
    for stats in stats_dict.values():
        all_classes.update(stats['class_count'].keys())
    all_classes = sorted(all_classes)
    
    plt.subplot(2, 2, 1)
    bar_width = 0.25
    x = np.arange(len(all_classes))
    
    for i, set_name in enumerate(sets):
        counts = [stats_dict[set_name]['class_count'].get(cls, 0) for cls in all_classes]
        plt.bar(x + i * bar_width, counts, width=bar_width, label=set_name, color=colors[i])
    
    plt.xlabel('Class')
    plt.ylabel('the number of objects')
    plt.title('Distribution of Classes Across Datasets')
    plt.xticks(x + bar_width, all_classes, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. 尺寸分布对比
    size_categories = ["small targets(<32px)", "medium targets(32-256px)", "large targets(>256px)"]
    
    plt.subplot(2, 2, 2)
    for i, set_name in enumerate(sets):
        sizes = [stats_dict[set_name]['size_distribution'].get(cat, 0) for cat in size_categories]
        plt.plot(size_categories, sizes, 'o-', label=set_name, color=colors[i], linewidth=2)
    
    plt.xlabel('Object Size Category')
    plt.ylabel('the number of objects')
    plt.title('Comparison of Object Size Distribution')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 数据集组成饼图
    plt.subplot(2, 2, 3)
    image_counts = [stats_dict[set_name]['image_count'] for set_name in sets]
    
    # 验证数据 - 添加这一部分
    if any(not isinstance(count, (int, float)) for count in image_counts) or any(count <= 0 for count in image_counts):
        print(f"无效的图像数量数据: {image_counts}")
        # 跳过饼图绘制
        plt.title('图像数量分布 - 数据无效')
    else:
        plt.pie(image_counts, labels=sets, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('图像数量分布')
    
    # 4. 每张图像目标数量分布
    plt.subplot(2, 2, 4)
    box_data = [stats_dict[set_name]['per_image_counts'] for set_name in sets]
    plt.boxplot(box_data, labels=sets, patch_artist=True,
                boxprops=dict(facecolor='skyblue', color='blue'),
                medianprops=dict(color='red'))
    plt.ylabel('Objects per Image')
    plt.title('Object Density Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('dataset_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 目标面积占比分布
    plt.figure(figsize=(10, 6))
    for i, set_name in enumerate(sets):
        plt.hist(stats_dict[set_name]['area_values'], bins=50, alpha=0.6, 
                 label=set_name, color=colors[i])
    
    plt.xlabel('Object Area (pixels)')  # 修改标签
    plt.ylabel('Frequency')
    plt.title('Distribution of Object Area')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('object_area_distribution.png', dpi=300)
    plt.show()



# 主分析流程
if __name__ == "__main__":
    # 分析三个数据集
    print("开始分析训练集...")
    train_stats = analyze_dataset(train_dir)
    
    print("开始分析验证集...")
    val_stats = analyze_dataset(val_dir)
    
    print("开始分析测试集...")
    test_stats = analyze_dataset(test_dir)
    
    # 组织结果
    stats_dict = {
        "training set": train_stats,
        "validation set": val_stats,
        "test set": test_stats
    }
    

    visualize_results(stats_dict)
    
    print("\n分析完成!结果已保存到:")
    
    print("- dataset_distribution_analysis.png (分布可视化)")
    print("- object_area_distribution.png (面积分布图)")