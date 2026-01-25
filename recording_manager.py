#!/usr/bin/env python3
"""
Recording Manager for wildlife camera system
Handles video recording with track-based candidate management
"""

import cv2
import os
import time
import threading
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from config import cfg

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrackCandidate:
    """Represents a tracked object that may become a recording candidate"""
    track_id: int
    class_id: int
    first_seen_frame: int
    last_seen_frame: int
    frames_seen: int = 0
    frames_missing: int = 0
    is_confirmed: bool = False
    is_recording_candidate: bool = False
    recording_start_frame: int = -1
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2
    last_confidence: float = 0.0


@dataclass
class RecordingSession:
    """Represents an active recording session"""
    session_id: str
    camera_id: int
    start_time: float
    start_frame: int
    video_path: str
    video_writer: cv2.VideoWriter = None
    candidate_track_ids: set = field(default_factory=set)
    frame_count: int = 0
    is_active: bool = True


class FrameBuffer:
    """Circular buffer for frames"""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.frame_ids = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add(self, frame: np.ndarray, frame_id: int):
        """Add a frame to the buffer"""
        with self.lock:
            self.buffer.append(frame.copy())
            self.frame_ids.append(frame_id)

    def get_frames_from(self, start_frame_id: int) -> List[Tuple[np.ndarray, int]]:
        """Get all frames from a specific frame ID onwards"""
        with self.lock:
            result = []
            for frame, fid in zip(self.buffer, self.frame_ids):
                if fid >= start_frame_id:
                    result.append((frame.copy(), fid))
            return result

    def get_frames_range(self, start_frame_id: int, end_frame_id: int) -> List[Tuple[np.ndarray, int]]:
        """Get frames within a specific range"""
        with self.lock:
            result = []
            for frame, fid in zip(self.buffer, self.frame_ids):
                if start_frame_id <= fid <= end_frame_id:
                    result.append((frame.copy(), fid))
            return result

    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, int]]:
        """Get the most recent frame"""
        with self.lock:
            if self.buffer:
                return self.buffer[-1].copy(), self.frame_ids[-1]
            return None

    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
            self.frame_ids.clear()


class RecordingManager:
    """Manages recording for a single camera with track-based candidate system"""

    def __init__(self, camera_id: int, s3_manager=None):
        """
        Initialize recording manager

        Args:
            camera_id: Camera identifier
            s3_manager: Optional S3Manager for uploads
        """
        self.camera_id = camera_id
        self.s3_manager = s3_manager

        # Configuration
        self.confirmation_frames = cfg.recording.confirmation_frames
        self.disappear_frames = cfg.recording.disappear_frames
        self.min_bbox_ratio = cfg.recording.min_bbox_ratio
        self.photo_interval = cfg.recording.photo_interval
        self.max_video_duration = cfg.recording.max_video_duration
        self.buffer_size = cfg.recording.frame_buffer_size
        self.output_path = cfg.video.output_path
        self.video_fps = cfg.video.video_writer_fps
        self.fourcc = cv2.VideoWriter_fourcc(*cfg.video.FOURCC)

        # State
        self.tracks: Dict[int, TrackCandidate] = {}
        self.recording_session: Optional[RecordingSession] = None
        self.frame_buffer = FrameBuffer(self.buffer_size)
        self.current_frame_id = 0
        self.frame_size = None

        # Photo capture state
        self.last_photo_time: Dict[int, float] = {}

        # Track segment info for spawning videos
        self.track_segments: Dict[int, dict] = {}  # track_id -> {start_frame, frames: []}

        # Upload queue
        self.upload_queue: List[str] = []
        self.upload_lock = threading.Lock()
        self.upload_thread = None
        self.upload_running = False

        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'photos'), exist_ok=True)

        logger.info(f"RecordingManager initialized for camera {camera_id}")

    def _generate_filename(self, prefix: str = "video", extension: str = "mp4") -> str:
        """Generate a unique filename with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{prefix}_cam{self.camera_id}_{timestamp}.{extension}"

    def _calculate_bbox_ratio(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate bbox area ratio to frame size"""
        if self.frame_size is None:
            return 0.0
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = self.frame_size[0] * self.frame_size[1]
        return bbox_area / frame_area if frame_area > 0 else 0.0

    def _draw_candidate_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                             track_id: int, is_confirmed: bool) -> np.ndarray:
        """Draw red bounding box for recording candidates"""
        x1, y1, x2, y2 = map(int, bbox)
        color = cfg.general.COLORS['candidate_red']  # Red in BGR
        thickness = 3 if is_confirmed else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Add track ID label
        label = f"ID:{track_id}"
        font_scale = 0.6
        font_thickness = 2
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Draw label background
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thickness)

        return frame

    def _start_recording_session(self, frame: np.ndarray, track_ids: set):
        """Start a new recording session"""
        if self.recording_session is not None:
            return

        filename = self._generate_filename("wildlife")
        video_path = os.path.join(self.output_path, filename)

        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(video_path, self.fourcc, self.video_fps, (w, h))

        if not writer.isOpened():
            logger.error(f"Failed to create video writer: {video_path}")
            return

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_session = RecordingSession(
            session_id=session_id,
            camera_id=self.camera_id,
            start_time=time.time(),
            start_frame=self.current_frame_id,
            video_path=video_path,
            video_writer=writer,
            candidate_track_ids=track_ids.copy()
        )

        # Initialize track segments for all candidates
        for track_id in track_ids:
            self.track_segments[track_id] = {
                'start_frame': self.current_frame_id,
                'frames': []
            }

        logger.info(f"Camera {self.camera_id}: Started recording session {session_id} with tracks {track_ids}")

    def _write_frame_to_session(self, frame: np.ndarray):
        """Write a frame to the active recording session"""
        if self.recording_session is None or not self.recording_session.is_active:
            return

        self.recording_session.video_writer.write(frame)
        self.recording_session.frame_count += 1

        # Store frame for all active track segments
        for track_id in self.recording_session.candidate_track_ids:
            if track_id in self.track_segments:
                self.track_segments[track_id]['frames'].append(frame.copy())

        # Check max duration
        elapsed = time.time() - self.recording_session.start_time
        if elapsed >= self.max_video_duration:
            logger.info(f"Camera {self.camera_id}: Max recording duration reached")
            self._finalize_recording_session()

    def _spawn_track_video(self, track_id: int):
        """Spawn a separate video for a track that has ended"""
        if track_id not in self.track_segments:
            return

        segment = self.track_segments[track_id]
        frames = segment.get('frames', [])

        if len(frames) < 10:  # Minimum frames for a valid video
            logger.info(f"Track {track_id}: Too few frames ({len(frames)}), skipping video spawn")
            del self.track_segments[track_id]
            return

        # Create video from buffered frames
        filename = self._generate_filename(f"track_{track_id}")
        video_path = os.path.join(self.output_path, filename)

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(video_path, self.fourcc, self.video_fps, (w, h))

        if not writer.isOpened():
            logger.error(f"Failed to create track video: {video_path}")
            del self.track_segments[track_id]
            return

        for frame in frames:
            writer.write(frame)

        writer.release()
        logger.info(f"Camera {self.camera_id}: Spawned video for track {track_id}: {video_path} ({len(frames)} frames)")

        # Queue for upload
        self._queue_upload(video_path)

        # Clean up segment
        del self.track_segments[track_id]

    def _finalize_recording_session(self):
        """Finalize and close the current recording session"""
        if self.recording_session is None:
            return

        session = self.recording_session
        session.is_active = False

        if session.video_writer is not None:
            session.video_writer.release()

        duration = time.time() - session.start_time
        logger.info(f"Camera {self.camera_id}: Finalized recording session {session.session_id} "
                    f"({session.frame_count} frames, {duration:.1f}s)")

        # Queue main video for upload
        self._queue_upload(session.video_path)

        # Clean up remaining track segments
        for track_id in list(self.track_segments.keys()):
            if track_id in session.candidate_track_ids:
                self._spawn_track_video(track_id)

        self.recording_session = None

    def _capture_photo(self, frame: np.ndarray, track_id: int, bbox: Tuple[int, int, int, int]):
        """Capture a photo for a track with small bbox"""
        current_time = time.time()
        last_time = self.last_photo_time.get(track_id, 0)

        if current_time - last_time < self.photo_interval:
            return

        # Draw bbox on frame
        annotated_frame = frame.copy()
        annotated_frame = self._draw_candidate_bbox(annotated_frame, bbox, track_id, True)

        # Save photo
        filename = self._generate_filename(f"photo_track{track_id}", "jpg")
        photo_path = os.path.join(self.output_path, 'photos', filename)

        cv2.imwrite(photo_path, annotated_frame)
        self.last_photo_time[track_id] = current_time

        logger.debug(f"Camera {self.camera_id}: Captured photo for track {track_id}")

        # Queue for upload
        self._queue_upload(photo_path)

    def _queue_upload(self, file_path: str):
        """Add file to upload queue"""
        with self.upload_lock:
            self.upload_queue.append(file_path)

        # Start upload thread if not running
        if not self.upload_running:
            self._start_upload_thread()

    def _start_upload_thread(self):
        """Start background upload thread"""
        if self.upload_thread is not None and self.upload_thread.is_alive():
            return

        self.upload_running = True
        self.upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.upload_thread.start()

    def _upload_worker(self):
        """Background worker for S3 uploads"""
        while self.upload_running or self.upload_queue:
            file_path = None
            with self.upload_lock:
                if self.upload_queue:
                    file_path = self.upload_queue.pop(0)

            if file_path is None:
                time.sleep(0.1)
                continue

            if self.s3_manager is None:
                logger.warning(f"S3 manager not available, skipping upload: {file_path}")
                continue

            try:
                filename = os.path.basename(file_path)
                s3_key = f"recordings/cam{self.camera_id}/{filename}"

                success = self.s3_manager.upload_file(s3_key, file_path)
                if success:
                    logger.info(f"Uploaded to S3: {s3_key}")
                    if cfg.s3.delete_local_after_upload:
                        os.remove(file_path)
                        logger.debug(f"Deleted local file: {file_path}")
                else:
                    logger.error(f"Failed to upload: {file_path}")
            except Exception as e:
                logger.error(f"Upload error for {file_path}: {e}")

        self.upload_running = False

    def process_frame(self, frame: np.ndarray, tracked_objects: List) -> np.ndarray:
        """
        Process a frame with tracked objects

        Args:
            frame: Current video frame
            tracked_objects: List of tracked objects [[x1, y1, x2, y2, track_id, class_id, score], ...]

        Returns:
            Annotated frame
        """
        self.current_frame_id += 1

        # Store frame size on first frame
        if self.frame_size is None:
            self.frame_size = (frame.shape[1], frame.shape[0])  # width, height

        # Add frame to buffer
        self.frame_buffer.add(frame, self.current_frame_id)

        # Get current track IDs
        current_track_ids = set()
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj[0:4]
            track_id = int(obj[4])
            class_id = int(obj[5])
            score = float(obj[6])

            current_track_ids.add(track_id)
            bbox = (x1, y1, x2, y2)

            # Update or create track
            if track_id in self.tracks:
                track = self.tracks[track_id]
                track.last_seen_frame = self.current_frame_id
                track.frames_seen += 1
                track.frames_missing = 0
                track.last_bbox = bbox
                track.last_confidence = score
            else:
                self.tracks[track_id] = TrackCandidate(
                    track_id=track_id,
                    class_id=class_id,
                    first_seen_frame=self.current_frame_id,
                    last_seen_frame=self.current_frame_id,
                    frames_seen=1,
                    last_bbox=bbox,
                    last_confidence=score
                )

            # Check for confirmation (FP filter)
            track = self.tracks[track_id]
            if not track.is_confirmed and track.frames_seen >= self.confirmation_frames:
                track.is_confirmed = True
                logger.info(f"Camera {self.camera_id}: Track {track_id} confirmed after {track.frames_seen} frames")

        # Update missing frames for tracks not seen
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in current_track_ids:
                track.frames_missing += 1

                # Check if track should be removed
                if track.frames_missing >= self.disappear_frames:
                    tracks_to_remove.append(track_id)

                    # If this was a recording candidate, spawn its video
                    if track.is_recording_candidate and self.recording_session is not None:
                        if track_id in self.recording_session.candidate_track_ids:
                            self._spawn_track_video(track_id)
                            self.recording_session.candidate_track_ids.discard(track_id)

        # Remove disappeared tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            self.last_photo_time.pop(track_id, None)
            logger.info(f"Camera {self.camera_id}: Track {track_id} removed (disappeared for {self.disappear_frames} frames)")

        # Determine recording candidates (confirmed tracks with sufficient bbox size)
        recording_candidates = {}
        photo_candidates = {}

        for track_id, track in self.tracks.items():
            if not track.is_confirmed:
                continue

            bbox_ratio = self._calculate_bbox_ratio(track.last_bbox)

            if bbox_ratio >= self.min_bbox_ratio:
                recording_candidates[track_id] = track
                track.is_recording_candidate = True
            else:
                photo_candidates[track_id] = track

        # Handle recording
        annotated_frame = frame.copy()

        if recording_candidates:
            candidate_ids = set(recording_candidates.keys())

            # Start recording if not already
            if self.recording_session is None:
                self._start_recording_session(annotated_frame, candidate_ids)
            else:
                # Add new candidates to existing session
                new_candidates = candidate_ids - self.recording_session.candidate_track_ids
                for track_id in new_candidates:
                    self.recording_session.candidate_track_ids.add(track_id)
                    self.track_segments[track_id] = {
                        'start_frame': self.current_frame_id,
                        'frames': []
                    }
                    logger.info(f"Camera {self.camera_id}: Added track {track_id} to recording session")

            # Draw red bboxes for all recording candidates
            for track_id, track in recording_candidates.items():
                annotated_frame = self._draw_candidate_bbox(
                    annotated_frame, track.last_bbox, track_id, True
                )

            # Write frame to recording
            self._write_frame_to_session(annotated_frame)

        elif self.recording_session is not None:
            # No more candidates, finalize recording
            self._finalize_recording_session()

        # Handle photo candidates
        for track_id, track in photo_candidates.items():
            self._capture_photo(frame, track_id, track.last_bbox)
            # Draw bbox for photo candidates too (different style)
            annotated_frame = self._draw_candidate_bbox(
                annotated_frame, track.last_bbox, track_id, False
            )

        return annotated_frame

    def get_status(self) -> dict:
        """Get current recording status"""
        confirmed_tracks = [t for t in self.tracks.values() if t.is_confirmed]
        recording_tracks = [t for t in self.tracks.values() if t.is_recording_candidate]

        status = {
            'camera_id': self.camera_id,
            'current_frame': self.current_frame_id,
            'total_tracks': len(self.tracks),
            'confirmed_tracks': len(confirmed_tracks),
            'recording_candidates': len(recording_tracks),
            'is_recording': self.recording_session is not None and self.recording_session.is_active,
            'pending_uploads': len(self.upload_queue)
        }

        if self.recording_session:
            status['recording_duration'] = time.time() - self.recording_session.start_time
            status['recording_frames'] = self.recording_session.frame_count

        return status

    def stop(self):
        """Stop the recording manager and clean up"""
        # Finalize any active recording
        if self.recording_session is not None:
            self._finalize_recording_session()

        # Stop upload thread
        self.upload_running = False
        if self.upload_thread is not None:
            self.upload_thread.join(timeout=5.0)

        logger.info(f"Camera {self.camera_id}: Recording manager stopped")


