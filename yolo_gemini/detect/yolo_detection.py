import cv2
from ultralytics import YOLO
from openapi_its import yolo_its

model = YOLO("yolo11n.pt")

def run_yolo_detection_once():
    cap = cv2.VideoCapture(yolo_its.get_its())
    success, frame = cap.read()
    cap.release()

    if not success:
        return None, [], None

    results = model(frame)
    result = results[0]
    boxes = result.boxes
    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()
    bboxes = boxes.xyxy.tolist()
    names = result.names

    detections = []
    for cls_id, conf, bbox in zip(class_ids, confidences, bboxes):
        detections.append({
            "class_id": int(cls_id),
            "class_name": names[int(cls_id)],
            "confidence": float(conf),
            "bbox": [round(v, 2) for v in bbox],
        })

    return frame, detections, result.plot()
