from ultralytics import YOLO
from config import cfg
import cv2
import yaml
import os


class Inference:
    def __init__(self):
        self.model = YOLO(cfg.detector.weight_file)
        self.names = self.model.names
        self.COLORS = cfg.general.COLORS
        
    def filter_overlapping_bboxes(self, boxes, confidences, class_ids, iou_threshold=cfg.filter.NMS_THRESHOLD):
        """
        Remove bounding boxes that overlap significantly with other boxes of a different class.
        """
        to_remove = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                # Only check for boxes belonging to different classes
                if class_ids[i] != class_ids[j]:
                    # Compute IoU between boxes i and j
                    iou = self.compute_iou(boxes[i], boxes[j])
                    if iou > iou_threshold:
                        # Remove the lower-confidence box
                        if confidences[i] < confidences[j]:
                            to_remove.append(i)
                        else:
                            to_remove.append(j)

        # Return filtered boxes, confidence, and class IDs
        filtered_boxes = [box for i, box in enumerate(boxes) if i not in to_remove]
        filtered_confidences = [conf for i, conf in enumerate(confidences) if i not in to_remove]
        filtered_class_ids = [class_id for i, class_id in enumerate(class_ids) if i not in to_remove]
        
        return filtered_boxes, filtered_confidences, filtered_class_ids
    
    def filter_bboxes_by_area(self, frame, boxes, confidences, class_ids):
        """
        Remove bounding boxes that have an area smaller than the defined threshold.
        """
        img_h, img_w, c = frame.shape
        min_bbox_area = (img_h*img_w/cfg.filter.image_size_factor) + cfg.filter.min_box_area_adjust # 835
        filtered_boxes = []
        filtered_confidences = []
        filtered_class_ids = []

        for i, box in enumerate(boxes):
            _, _, w, h = box
            area = w * h
            if area >= min_bbox_area:
                filtered_boxes.append(box)
                filtered_confidences.append(confidences[i])
                filtered_class_ids.append(class_ids[i])
        
        return filtered_boxes, filtered_confidences, filtered_class_ids
    
    def compute_iou(self, box1, box2):
        """
        Compute the Intersection over Union (IoU) of two bounding boxes.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection area
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # Calculate union area
        union_area = w1 * h1 + w2 * h2 - inter_area
        iou = inter_area / union_area
        return iou

    def infer(self, frame):
        # Run the model
        if cfg.general.frame_rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        processed_boxes = []
        processed_confidence = []
        processed_class_id = []
        names = []
        results = self.model.predict(task= 'detect',
                                     source=frame, conf=cfg.detector.OBJECTNESS_CONFIDANCE,
                                     iou=cfg.detector.NMS_THRESHOLD,
                                     classes=cfg.detector.classes,
                                     device=cfg.detector.device,
                                     verbose=cfg.detector.verbose)
        for result in results:
            boxes = result.boxes.xywh.tolist()  # box with xywh format, (N, 4)
            class_ids = result.boxes.cls.tolist()  # cls, (N, 1)
            confidences = result.boxes.conf.tolist()  # confidence score, (N, 1)

            # Filter overlapping boxes between different classes
            boxes, confidences, class_ids = self.filter_overlapping_bboxes(boxes, confidences, class_ids)
            # Filter boxes by area 
            boxes, confidences, class_ids = self.filter_bboxes_by_area(frame, boxes, confidences, class_ids)
            
            for i, box in enumerate(boxes):
                class_id = int(class_ids[i])
                names.append(self.names[class_id])
                confidence = confidences[i]
                w = box[2]
                h = box[3]
                x = box[0] - w / 2
                y = box[1] - h / 2
                p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
                line_width = 2 or max(round(sum(frame.shape) / 2 * 0.003), 2)  # line width
                color = self.COLORS[list(self.COLORS)[int(class_id) % len(self.COLORS)]]
                if cfg.flags.render_detections:
                    cv2.rectangle(frame, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)
                if cfg.flags.render_labels:
                    label = "{}: {:.4f}".format(self.names[class_id], confidence)
                    cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, line_width)

                processed_boxes.append([x, y, w, h])
                processed_confidence.append(confidence)
                processed_class_id.append(class_id)

        return frame, processed_boxes, processed_confidence, processed_class_id


if __name__ == '__main__':
    inference = Inference()
    source_dir = 'test_data/'
    destination = 'results/'
    os.makedirs(destination, exist_ok=True)
    for file in os.listdir(source_dir):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".bmp"):
            im_path = f'{source_dir}{file}'
            out_path = f'{destination}out_{file}'
            im = cv2.imread(im_path)
            frame_out, boxes_out, _, _ = inference.infer(im)
            cv2.imwrite(out_path, frame_out)
