"""Webcam / video pose source built on OpenCV + MediaPipe Pose (CPU only)."""
import os
import time

import numpy as np

from .common import LM_MP_IDS


class PoseSourceError(RuntimeError):
    pass


class PoseSource:
    """read() -> (landmarks (12,4) [x,y,z,visibility] or None, bgr_frame or None, ok_flag)."""

    def __init__(self, source="camera", camera_index=0, video_path="", loop_video=True, model_complexity=1):
        try:
            import cv2
        except ImportError as e:
            raise PoseSourceError("OpenCV (cv2) is not installed in this Python environment. Activate the venv: "
                                  "`source ~/robot_cloning/.venv/bin/activate` or re-run ./install_dependencies.sh") from e
        try:
            import mediapipe as mp
        except ImportError as e:
            raise PoseSourceError("MediaPipe is not installed in this Python environment. Activate the venv: "
                                  "`source ~/robot_cloning/.venv/bin/activate` or re-run ./install_dependencies.sh") from e
        self.cv2 = cv2
        self.loop = loop_video
        self.is_video = source == "video"
        if self.is_video:
            if not video_path or not os.path.exists(video_path):
                raise PoseSourceError(f"Video file not found: '{video_path}'. Pass a valid --video path.")
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(int(camera_index))
        if not self.cap.isOpened():
            raise PoseSourceError(
                f"Could not open {'video ' + video_path if self.is_video else 'webcam index ' + str(camera_index)}. "
                "Check: `ls /dev/video*`, that no other app uses the camera, and that your user is in the "
                "'video' group (`sudo usermod -aG video $USER`, then re-login). "
                "No webcam? Use a pre-recorded video: --video path/to/video.mp4")
        self.pose = mp.solutions.pose.Pose(model_complexity=model_complexity, min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            if self.is_video and self.loop:
                self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                return None, None, False
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if res.pose_landmarks is None:
            return None, frame, True
        h, w = frame.shape[:2]
        img_lm = res.pose_landmarks.landmark
        world = res.pose_world_landmarks.landmark if res.pose_world_landmarks else None
        out = np.zeros((len(LM_MP_IDS), 4))
        for k, i in enumerate(LM_MP_IDS):
            if world is not None:                       # metric hip-centred coordinates (preferred)
                out[k, :3] = (world[i].x, world[i].y, world[i].z)
            else:                                       # fallback: image coords with aspect correction
                out[k, :3] = (img_lm[i].x * w / h, img_lm[i].y, img_lm[i].z * w / h)
            out[k, 3] = img_lm[i].visibility
        return out, frame, True

    def draw(self, frame, text=""):
        if frame is not None and text:
            self.cv2.putText(frame, text, (10, 30), self.cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def close(self):
        try:
            self.cap.release()
            self.pose.close()
        except Exception:
            pass
