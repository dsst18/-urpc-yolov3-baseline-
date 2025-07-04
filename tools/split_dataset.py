import os
import random
import shutil
from tqdm import tqdm
import argparse

def split_dataset(base_dir, val_split_ratio=0.1):
    """
    根据“文件名对文件名”的规则，将数据集划分为训练集和验证集。
    (此版本适用于 image_name.jpg 对应 image_name.xml 的情况)
    """
    image_dir = os.path.join(base_dir, 'JPEGImages')
    ann_dir = os.path.join(base_dir, 'Annotations')

    if not os.path.isdir(image_dir) or not os.path.isdir(ann_dir):
        print(f"错误: 在 {base_dir} 中找不到 JPEGImages 或 Annotations 文件夹")
        return

    # --- 核心修正：直接以图片文件名为准进行划分 ---
    all_image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    
    # 找出那些既有图片又有对应XML的文件
    valid_basenames = []
    for img_file in all_image_files:
        basename = os.path.splitext(img_file)[0]
        if os.path.exists(os.path.join(ann_dir, f"{basename}.xml")):
            valid_basenames.append(basename)
        else:
            print(f"警告: 找到图片 {img_file} 但缺少对应的XML文件，将忽略此文件。")

    print(f"共找到 {len(valid_basenames)} 个有效的图片-标注对。")

    random.seed(42)
    random.shuffle(valid_basenames)

    split_index = int(len(valid_basenames) * (1 - val_split_ratio))
    train_basenames = valid_basenames[:split_index]
    val_basenames = valid_basenames[split_index:]

    print(f"训练集数量: {len(train_basenames)}")
    print(f"验证集数量: {len(val_basenames)}")

    parent_dir = os.path.dirname(os.path.abspath(base_dir))
    train_output_dir = os.path.join(parent_dir, 'train_split')
    val_output_dir = os.path.join(parent_dir, 'val_split')
    
    def copy_files(basename_list, dest_dir):
        dest_img_dir = os.path.join(dest_dir, 'JPEGImages')
        dest_ann_dir = os.path.join(dest_dir, 'Annotations')
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_ann_dir, exist_ok=True)

        for basename in tqdm(basename_list, desc=f"正在复制到 {os.path.basename(dest_dir)}"):
            # --- 核心修正：使用basename直接拼接jpg和xml文件名 ---
            shutil.copy(os.path.join(image_dir, f"{basename}.jpg"), os.path.join(dest_img_dir, f"{basename}.jpg"))
            shutil.copy(os.path.join(ann_dir, f"{basename}.xml"), os.path.join(dest_ann_dir, f"{basename}.xml"))

    copy_files(train_basenames, train_output_dir)
    copy_files(val_basenames, val_output_dir)

    print("数据集划分完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将数据集划分为训练集和验证集。')
    parser.add_argument('--base_dir', type=str, required=True, help='包含JPEGImages和Annotations的源文件夹路径。')
    parser.add_argument('--ratio', type=float, default=0.1, help='用于验证集的数据比例 (例如, 0.1 代表10%)。')
    
    args = parser.parse_args()
    split_dataset(args.base_dir, args.ratio)
