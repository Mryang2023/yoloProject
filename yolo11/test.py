from ultralytics import YOLO
import cv2
import os

# 加载训练好的模型
model = YOLO('runs/detect/train2/weights/best.pt')

# 进行预测
results = model.predict(source='data/test', device='cuda')  # 你可以根据需要选择device='cuda'

# 确保结果保存的文件夹存在
output_dir = 'data/test/result'
os.makedirs(output_dir, exist_ok=True)

# 获取类别名称
class_names = model.names

# 处理预测结果
for result in results:
    # 读取原始图像
    image_path = result.path
    image = cv2.imread(image_path)

    # 获取边界框信息
    boxes = result.boxes.cpu().numpy()
    cls = boxes.cls.astype(int)
    conf = boxes.conf

    # 遍历每个边界框并绘制
    for box, c, conf in zip(boxes, cls, conf):
        if conf > 0.50:  # 只绘制置信度大于0.50的边界框
            r = box.xyxy[0].astype(int)
            cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)

            # 获取类别名称和置信度
            label = f'{class_names[c]} {conf:.2f}'

            # 计算文本位置
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = r[0]
            text_y = r[1] - 10 if r[1] - 10 > 10 else r[1] + 10

            # 绘制背景矩形
            cv2.rectangle(image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0, 255, 0), -1)

            # 绘制文本
            cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 获取图像文件名
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, image_name)

    # 保存绘制了边界框的图像
    cv2.imwrite(output_path, image)
    print(f'Saved result to {output_path}')
