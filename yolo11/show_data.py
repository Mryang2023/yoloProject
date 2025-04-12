from pathlib import Path
import cv2
import numpy as np
import matplotlib.cm as cm


def generate_colors(n, colormap_name='hsv'):
    colormap = cm.get_cmap(colormap_name, n)
    colors = colormap(np.linspace(0, 1, n))
    colors = (colors[:, :3] * 255).astype(np.uint8)  # 将颜色值转换为0到255的整数
    return colors


# 数据集路径
dataset = Path('NWPU VHR-10 dataset')
# 图像路径
images_dir = dataset / 'positive image set'
# 标签路径
ground_truth_dir = dataset / 'ground truth'

# 标签框的颜色
colors = generate_colors(10)[:, ::-1].tolist()
# 标签名
names = ['airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court',
         'ground track field', 'harbor', 'bridge', 'vehicle']

for image_path in images_dir.glob('*.jpg'):
    image = cv2.imread(str(image_path))
    ground_truth = ground_truth_dir / (image_path.stem + '.txt')

    # 读取标签
    labels = []
    for line in ground_truth.open('r').readlines():
        line = line.replace('\n', '')
        if line:
            labels.append(eval(line))

    # 遍历所有标签
    for (x1, y1), (x2, y2), cls in labels:
        cv2.rectangle(image, (x1, y1), (x2, y2), tuple(colors[cls - 1]), 2)
        cv2.putText(image, names[cls - 1], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[cls - 1], 2)

    cv2.imshow('image', image)
    cv2.waitKey(0)