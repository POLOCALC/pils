import cv2
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import glob
import os
from datetime import timedelta

from ..utils.tools import get_logpath_from_datapath, read_log_time


class Camera:
    def __init__(self, path, logpath=None, time_index=None):
        """
        path: either a video file or a directory containing images
        time_index: optional dict mapping image filenames → timestamps
                    Example: {"img_0001.jpg": datetime, ...}
        """
        self.path = path
        self.logpath = (
            logpath if logpath is not None else get_logpath_from_datapath(self.path)
        )

        # Video attributes
        self.capture = None
        self.fps = None
        self.tstart = None

        # Image-sequence attributes
        self.is_image_sequence = False
        self.images = []  # list of filepaths
        self.time_index = time_index  # optional timestamps for images

    def load_data(self):
        # ------------------------------
        # Case 1: VIDEO FILE
        # ------------------------------
        if self.path.lower().endswith((".mp4", ".avi", ".mov")):
            self.capture = cv2.VideoCapture(self.path)
            self.tstart, _ = read_log_time(
                "INFO:Camera Sony starts recording", self.logpath
            )

            frame_count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = self.capture.get(cv2.CAP_PROP_FPS)

            if fps > 0:
                self.fps = fps
            else:
                # fallback
                duration_sec = frame_count / fps if fps != 0 else 1
                self.fps = frame_count / duration_sec

        # ------------------------------
        # Case 2: IMAGE SEQUENCE
        # ------------------------------
        else:
            self.is_image_sequence = True
            # get all images in folder / glob pattern
            self.images = sorted(
                glob.glob(os.path.join(self.path, "*.*")),
                key=lambda x: os.path.basename(x),
            )

            if len(self.images) == 0:
                raise FileNotFoundError(f"No images found in {self.path}")

            # Default FPS estimation for images (if timestamps not provided)
            if self.time_index is None:
                self.fps = None  # unknown
            else:
                # infer fps from provided timestamps
                times = list(self.time_index.values())
                if len(times) >= 2:
                    dt = (times[1] - times[0]).total_seconds()
                    self.fps = 1.0 / dt if dt > 0 else None

            # Optional: parse "tstart" from first timestamp if available
            if self.time_index is not None:
                first_image = os.path.basename(self.images[0])
                self.tstart = self.time_index.get(first_image)

    def get_frame(self, frame_number) -> np.ndarray:
        # -------------------
        # VIDEO
        # -------------------
        if not self.is_image_sequence:
            if self.capture is None:
                raise ValueError(
                    "Video capture not initialized. Call load_data() first."
                )
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.capture.read()
            if not ret or frame is None:
                raise ValueError(f"Failed to read frame {frame_number}")
            return frame

        # -------------------
        # IMAGE SEQUENCE
        # -------------------
        if frame_number < 0 or frame_number >= len(self.images):
            raise IndexError("Frame index out of range for image sequence")

        frame = cv2.imread(self.images[frame_number])
        if frame is None:
            raise ValueError(f"Failed to read image {self.images[frame_number]}")
        return frame

    def get_timestamp(self, frame_number):
        """
        Returns the timestamp associated with a frame.
        For videos → tstart + frame_number / fps
        For image sequences → from time_index if available, else None
        """
        if not self.is_image_sequence:
            if self.tstart is None or self.fps is None:
                return None
            return self.tstart + timedelta(seconds=frame_number / self.fps)

        # Image sequence case
        if self.time_index is not None:
            fname = os.path.basename(self.images[frame_number])
            return self.time_index.get(fname, None)
        else:
            return None

    def plot_frame(self, frame_number, color="rgb"):
        frame = self.get_frame(frame_number)

        if color == "rgb":
            img: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif color == "hsv":
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif color == "gray":
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif color == "bgr":
            img = frame.copy()
        else:
            raise KeyError(f"{color} is not known")

        plt.figure()
        plt.imshow(img)
        plt.title(f"Frame {frame_number} — Time: {self.get_timestamp(frame_number)}")
        plt.axis("off")
        plt.show()
