from datetime import timedelta
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..utils.tools import get_logpath_from_datapath, read_log_time


class Camera:
    """Camera sensor for video files and image sequences.

    Attributes
    ----------
    path : Union[str, Path]
        Path to video file or image directory.
    logpath : Optional[Union[str, Path]]
        Path to log file for timestamp extraction.
    capture : Optional[Any]
        OpenCV VideoCapture object (for videos).
    fps : Optional[float]
        Frames per second.
    tstart : Optional[Any]
        Start timestamp from log file.
    is_image_sequence : bool
        Whether path is an image sequence directory.
    images : list
        List of image file paths (for image sequences).
    time_index : Optional[Dict[str, Any]]
        Optional timestamp mapping for images.
    """

    def __init__(
        self,
        path: str | Path,
        logpath: str | Path | None = None,
        time_index: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Camera sensor.

        Parameters
        ----------
        path : Union[str, Path]
            Path to video file or directory containing images.
        logpath : Optional[Union[str, Path]], optional
            Optional path to log file. If None, will be inferred.
        time_index : Optional[Dict[str, Any]], optional
            Optional dict mapping image filenames to timestamps.
            Example: {"img_0001.jpg": datetime, ...}
        """
        self.path = path
        self.logpath = (
            logpath if logpath is not None else get_logpath_from_datapath(self.path)
        )

        # Video attributes
        self.capture: Any | None = None
        self.fps: float | None = None
        self.tstart: Any | None = None

        # Image-sequence attributes
        self.is_image_sequence: bool = False
        self.images: list = []  # list of filepaths
        self.time_index: dict[str, Any] | None = (
            time_index  # optional timestamps for images
        )

    def load_data(self) -> None:
        """Load camera data from video file or image sequence.

        For video files (.mp4, .avi, .mov):
            - Initializes cv2.VideoCapture
            - Extracts FPS and frame count
            - Reads start timestamp from log file

        For image sequences:
            - Lists all images in directory
            - Estimates FPS from time_index if provided
            - Sorts images by filename

        Raises
        ------
        FileNotFoundError
            If image directory is empty.
        """
        # ------------------------------
        # Case 1: VIDEO FILE
        # ------------------------------
        path_str = str(self.path)
        if path_str.lower().endswith((".mp4", ".avi", ".mov")):
            self.capture = cv2.VideoCapture(path_str)
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
            # get all images in folder using pathlib
            path_obj = Path(self.path)
            image_paths = sorted(
                path_obj.glob("*.*"),
                key=lambda x: x.name,
            )
            self.images = [str(p) for p in image_paths]

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
                first_image = Path(self.images[0]).name
                self.tstart = self.time_index.get(first_image)

    def get_frame(self, frame_number: int) -> np.ndarray:
        """Get frame at specified index.

        Parameters
        ----------
        frame_number : int
            Frame index to retrieve.

        Returns
        -------
        np.ndarray
            Frame as numpy array (BGR format for OpenCV).

        Raises
        ------
        ValueError
            If video capture not initialized or frame read fails.
        IndexError
            If frame_number is out of range for image sequence.
        """
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

    def get_timestamp(self, frame_number: int) -> Any | None:
        """Get timestamp for specified frame.

        For videos:
            Returns tstart + frame_number / fps

        For image sequences:
            Returns timestamp from time_index if available, else None.

        Parameters
        ----------
        frame_number : int
            Frame index.

        Returns
        -------
        Optional[Any]
            Timestamp (datetime) or None if not available.
        """
        if not self.is_image_sequence:
            if self.tstart is None or self.fps is None:
                return None
            return self.tstart + timedelta(seconds=frame_number / self.fps)

        # Image sequence case
        if self.time_index is not None:
            fname = Path(self.images[frame_number]).name
            return self.time_index.get(fname, None)
        else:
            return None

    def plot_frame(self, frame_number: int, color: str = "rgb") -> None:
        """Plot frame at specified index.

        Parameters
        ----------
        frame_number : int
            Frame index to plot.
        color : str, optional
            Color space for display ("rgb", "hsv", "gray", or "bgr").

        Raises
        ------
        KeyError
            If color space not recognized.
        """
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
