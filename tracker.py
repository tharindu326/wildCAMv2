import time
import logging
from collections import deque
import cv2
import numpy as np
from config import cfg

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_tracker(cfg):
    tracker_type = cfg.tracker.type.lower()
    if tracker_type == 'bytetrack':
        from trackers.bytetrack.byte_tracker import BYTETracker
        return BYTETracker(
            track_thresh=cfg.bytetrack.track_thresh,
            match_thresh=cfg.bytetrack.match_thresh,
            track_buffer=cfg.bytetrack.track_buffer,
            frame_rate=cfg.bytetrack.frame_rate
        )
    elif tracker_type == 'strongsort':
        from trackers.strongsort.strong_sort import StrongSORT
        return StrongSORT(
            model_weights=cfg.tracker.reid_weights,
            device=cfg.general.device,
            fp16=False,
            max_dist=cfg.strongsort.max_dist,
            max_iou_dist=cfg.strongsort.max_iou_dist,
            max_age=cfg.strongsort.max_age,
            max_unmatched_preds=cfg.strongsort.max_unmatched_preds,
            n_init=cfg.strongsort.n_init,
            nn_budget=cfg.strongsort.nn_budget,
            mc_lambda=cfg.strongsort.mc_lambda,
            ema_alpha=cfg.strongsort.ema_alpha,
        )
    elif tracker_type == 'ocsort':
        from trackers.ocsort.ocsort import OCSort
        return OCSort(
            det_thresh=cfg.ocsort.det_thresh,
            max_age=cfg.ocsort.max_age,
            min_hits=cfg.ocsort.min_hits,
            iou_threshold=cfg.ocsort.iou_thresh,
            delta_t=cfg.ocsort.delta_t,
            asso_func=cfg.ocsort.asso_func,
            inertia=cfg.ocsort.inertia,
            use_byte=cfg.ocsort.use_byte,
        )
    elif tracker_type == 'boosttrack':
        from trackers.boosttrack.boost_track import BoostTrack
        return BoostTrack(
            det_thresh=cfg.boosttrack.det_thresh,
            lambda_iou=cfg.boosttrack.lambda_iou,
            lambda_mhd=cfg.boosttrack.lambda_mhd,
            lambda_shape=cfg.boosttrack.lambda_shape,
            dlo_boost_coef=cfg.boosttrack.dlo_boost_coef,
            use_dlo_boost=cfg.boosttrack.use_dlo_boost,
            use_duo_boost=cfg.boosttrack.use_duo_boost,
            max_age=cfg.boosttrack.max_age
        )
    else:
        raise ValueError(f"Undefined Tracker. Supported types: bytetrack, strongsort, ocsort, boosttrack")


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.frame_id = 0
        self.avg_fps = deque(maxlen=100)
        self.tracker = initialize_tracker(cfg)

    def process_detections(self, detections: np.ndarray) -> np.ndarray:
        if detections.size == 0:
            return np.empty((0, 6))
        if len(self.cfg.tracker.classes) == 0:
            return detections
        mask = np.isin(detections[:, 5].astype(int), self.cfg.tracker.classes)
        return detections[mask]

    def track(self, frame: np.ndarray, detections: np.ndarray):
        filtered_detections = self.process_detections(detections)
        return self.tracker.update(filtered_detections, frame)

    def update(self, frame: np.ndarray, boxes: list, confidences: list, class_ids: list):
        """
        Update tracker with detection output from inference.py

        Args:
            frame: Current video frame
            boxes: List of [x, y, w, h] bounding boxes (top-left format)
            confidences: List of confidence scores
            class_ids: List of class IDs

        Returns:
            List of tracked objects: [[x1, y1, x2, y2, track_id, class_id, score], ...]
        """
        start_time = time.monotonic()

        if len(boxes) == 0:
            detections = np.empty((0, 6))
        else:
            # Convert [x, y, w, h] to [x1, y1, x2, y2, conf, class_id]
            detections = []
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x, y, w, h = box
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                detections.append([x1, y1, x2, y2, conf, cls_id])
            detections = np.array(detections)

        targets = self.track(frame, detections)

        elapsed = time.monotonic() - start_time
        if elapsed > 0:
            self.avg_fps.append(1 / elapsed)

        self.frame_id += 1
        return targets

    def process_frame(self, frame: np.ndarray, results):
        detections = results[0].boxes.data.cpu().numpy()
        start_time = time.monotonic()
        targets = self.track(frame, detections)
        elapsed = time.monotonic() - start_time
        if elapsed > 0:
            self.avg_fps.append(1 / elapsed)
        return frame, targets


