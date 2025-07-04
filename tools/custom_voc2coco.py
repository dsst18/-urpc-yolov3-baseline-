import os
import argparse
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image

CLASSES = ("echinus", "starfish", "holothurian", "scallop")

def get_image_info_and_annos(xml_path, image_dir, img_id, ann_id_start, class_to_id):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"错误: 解析 {xml_path} 失败: {e}")
        return None, [], ann_id_start

    xml_basename = os.path.splitext(os.path.basename(xml_path))[0]
    image_filename = f"{xml_basename}.jpg"
    image_path = os.path.join(image_dir, image_filename)

    if not os.path.exists(image_path):
        image_filename_upper = f"{xml_basename}.JPG"
        image_path_upper = os.path.join(image_dir, image_filename_upper)
        if os.path.exists(image_path_upper):
            image_path = image_path_upper
            image_filename = image_filename_upper
        else:
            print(f"警告: 找不到图片 {image_filename} 或 {image_filename_upper}。跳过标注 {xml_path}。")
            return None, [], ann_id_start
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"错误: 打开图片 {image_path} 失败: {e}。跳过。")
        return None, [], ann_id_start

    image_info = {'file_name': image_filename, 'height': height, 'width': width, 'id': img_id}
    annotations = []
    ann_id = ann_id_start
    for obj in root.findall('object'):
        class_name = obj.findtext('name')
        if class_name not in class_to_id:
            print(f"警告: 在 {xml_path} 中发现未知类别 '{class_name}'。跳过。")
            continue
        category_id = class_to_id[class_name]
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.findtext('xmin')))
        ymin = int(float(bndbox.findtext('ymin')))
        xmax = int(float(bndbox.findtext('xmax')))
        ymax = int(float(bndbox.findtext('ymax')))
        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        area = (xmax - xmin) * (ymax - ymin)
        annotation_info = {
            'id': ann_id, 'image_id': img_id, 'category_id': category_id,
            'bbox': bbox, 'area': area, 'iscrowd': 0, 'segmentation': []
        }
        annotations.append(annotation_info)
        ann_id += 1
        
    return image_info, annotations, ann_id

def convert_to_coco(ann_dir, image_dir, save_path):
    class_to_id = {cls: i for i, cls in enumerate(CLASSES)}
    coco_format = {"images": [], "annotations": [], "categories": []}
    for i, cls in enumerate(CLASSES):
        coco_format['categories'].append({'id': i, 'name': cls, 'supercategory': 'underwater'})
    xml_files = [f for f in os.listdir(ann_dir) if f.endswith('.xml')]
    img_id_counter, ann_id_counter = 0, 0
    print(f"开始转换 {ann_dir} 中的XML文件...")
    for xml_file in tqdm(xml_files, desc="Processing XMLs"):
        xml_path = os.path.join(ann_dir, xml_file)
        image_info, annotations, next_ann_id = get_image_info_and_annos(
            xml_path, image_dir, img_id_counter, ann_id_counter, class_to_id
        )
        if image_info:
            coco_format['images'].append(image_info)
            coco_format['annotations'].extend(annotations)
            img_id_counter += 1
            ann_id_counter = next_ann_id
    with open(save_path, 'w') as f:
        json.dump(coco_format, f, indent=4)
    print(f"转换完成！总共处理了 {img_id_counter} 张图片和 {len(coco_format['annotations'])} 个标注。")
    print(f"COCO格式的JSON文件已保存到: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert custom VOC format to COCO format.')
    parser.add_argument('--ann_dir', required=True, help='Annotations directory path.')
    parser.add_argument('--image_dir', required=True, help='JPEGImages directory path.')
    # --- 这里是修正的地方 ---
    parser.add_argument('--save_path', required=True, help='Output COCO json file path.')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    convert_to_coco(args.ann_dir, args.image_dir, args.save_path)
