from pathlib import Path
import random
import cv2
import shutil

# 设置随机数种子
random.seed(0)

# 数据集路径
ground_truth_dir = Path('NWPU VHR-10 dataset/ground truth')
# 图像路径
positive_image_set = Path('NWPU VHR-10 dataset/positive image set')

# 创建一个data目录，用于保存数据
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

# 创建图像与标签目录
images_dir = data_dir / 'images'
images_dir.mkdir(exist_ok=True)

labels_dir = data_dir / 'labels'
labels_dir.mkdir(exist_ok=True)

# 创建训练集与验证集目录
for set_name in ['train', 'val']:
    (images_dir / set_name).mkdir(exist_ok=True)
    (labels_dir / set_name).mkdir(exist_ok=True)

# 所有图像
image_paths = list(positive_image_set.glob('*.jpg'))
# 所有标签
labels = []
for image_path in image_paths:
    # 读取图像
    image = cv2.imread(str(image_path))
    ground_truth = ground_truth_dir / (image_path.stem + '.txt')
    # 获取图像宽高
    im_h, im_w = image.shape[:2]

    # 读取标签
    lines = []
    for line in ground_truth.open('r').readlines():
        line = line.replace('\n', '')
        if line:
            (x1, y1), (x2, y2), c = eval(line)
            # 计算中心点和宽高
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = abs(x2 - x1)
            h = abs(y2 - y1)

            # 归一化
            x = x / im_w
            y = y / im_h
            w = w / im_w
            h = h / im_h

            # 把序号变为索引
            c -= 1
            lines.append(f'{c} {x} {y} {w} {h}\n')

    # 保存标签
    labels.append(lines)

# 合并图像路径与标签
data_pairs = list(zip(image_paths, labels))
# 打乱数据
random.shuffle(data_pairs)

# 取80%的数据为训练集，剩下的为测试集
train_size = int(len(data_pairs) * 0.8)
train_set = data_pairs[:train_size]
val_set = data_pairs[train_size:]

# 遍历训练集与测试集
for set_name, dataset in zip(['train', 'val'], [train_set, val_set]):
    # 遍历每张图像与标签
    for image_path, label in dataset:
        # 复制图像到新的文件夹
        shutil.copy2(image_path, images_dir / set_name / image_path.name)
        # 生成标签到新的文件夹
        with open(labels_dir / set_name / (image_path.stem + '.txt'), 'w') as f:
            f.writelines(label)