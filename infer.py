import cv2
import numpy as np
import argparse
from ultralytics import YOLO

def draw_custom_keypoints(frame, box, color):
    x1, y1, x2, y2 = box
    kp1 = (int(x1), int(y1))
    kp2 = (int(x2), int(y1))
    kp3 = (int(x2), int(y2))
    kp4 = (int(x1), int(y2))
    keypoints = [kp1, kp2, kp3, kp4]
    for i in range(4):
        pt1 = keypoints[i]
        pt2 = keypoints[(i + 1) % 4]
        cv2.line(frame, pt1, pt2, color, 2)
    for i, pt in enumerate(keypoints):
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)
        cv2.circle(frame, pt, 4, color, -1)
        cv2.putText(frame, str(i + 1), (pt[0] + 5, pt[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    parser = argparse.ArgumentParser(description='YOLO 装甲板推理脚本')
    parser.add_argument('--weights', type=str, default='best.pt')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='result.mp4')
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    print("正在加载模型...")
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"无法打开视频文件: {args.input}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"视频信息: {width}x{height}, FPS={fps}")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        if not out.isOpened():
            print("视频写入器初始化失败")
            cap.release()
            return

    colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 255, 255), 3: (255, 0, 255)}
    default_color = (0, 0, 255)

    frame_count = 0
    print(f"开始推理，保存至: {args.output}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=args.conf, device=args.device, verbose=False)
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                cls = boxes.cls
                conf = boxes.conf
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                for i in range(len(boxes)):
                    box_coords = xyxy[i]
                    target_cls = int(cls[i].item())
                    target_conf = float(conf[i].item())
                    color = colors.get(target_cls, default_color)
                    cv2.rectangle(frame, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), color, 2)
                    label = f"Armor {target_cls}: {target_conf:.2f}"
                    cv2.putText(frame, label, (box_coords[0], box_coords[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    draw_custom_keypoints(frame, box_coords, color)
        out.write(frame)
        frame_count += 1
        if not args.no_show:
            cv2.imshow("Inference Result", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"推理完成，共处理 {frame_count} 帧。")

if __name__ == "__main__":
    main()
