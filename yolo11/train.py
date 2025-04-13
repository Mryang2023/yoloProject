from ultralytics import YOLO

if __name__ == '__main__':
    yolo = YOLO('yolo11n.yaml')
    yolo.train(data='dataset.yaml', epochs=100, batch=3, device='cuda')
