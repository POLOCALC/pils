import cv2
import numpy as np
import pandas as pd
import glob
import os
from queue import Queue
from threading import Thread

from tools import get_logpath_from_datapath, read_log_time


class Camera:
    def __init__(self, path, logpath=None, time_index=None, buffer_size=128):
        self.path = path
        self.logpath = logpath if logpath is not None else get_logpath_from_datapath(self.path)

        # Video attributes
        self.capture = None
        self.fps = None
        self.tstart = None

        # Image sequence attributes
        self.is_image_sequence = False
        self.images = []
        self.time_index = time_index

        # Streaming system
        self.frame_queue = Queue(maxsize=buffer_size)
        self.reader_thread = None
        self.stopped = False

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------
    def load_data(self):

        # ---------------- VIDEO ----------------
        if self.path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            self.capture = cv2.VideoCapture(self.path)
            if not self.capture.isOpened():
                raise IOError(f"Cannot open video {self.path}")

            self.tstart = read_log_time("INFO:Camera Sony starts recording", self.logpath)

            frame_count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps > 0 else None

            # Start background decoding thread
            self.reader_thread = Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()

        # ---------------- IMAGE SEQUENCE ----------------
        else:
            self.is_image_sequence = True
            self.images = sorted(
                glob.glob(os.path.join(self.path, "*.*")),
                key=lambda x: os.path.basename(x)
            )

            if len(self.images) == 0:
                raise FileNotFoundError(f"No images found in {self.path}")

            if self.time_index is not None:
                first_image = os.path.basename(self.images[0])
                self.tstart = self.time_index[first_image]

                times = pd.to_datetime(list(self.time_index.values()))
                dt = (times.iloc[1] - times.iloc[0]).total_seconds()
                self.fps = 1.0 / dt if dt > 0 else None

    # --------------------------------------------------
    # VIDEO READER THREAD
    # --------------------------------------------------
    def _reader_loop(self):
        frame_idx = 0
        while not self.stopped:
            ret, frame = self.capture.read()
            if not ret:
                self.frame_queue.put(None)  # End signal
                break
            self.frame_queue.put((frame_idx, frame))
            frame_idx += 1

    # --------------------------------------------------
    # GET NEXT FRAME (STREAMING)
    # --------------------------------------------------
    def get_next_frame(self):
        """
        Returns:
            (frame_index, frame) or None if video ended
        """
        if self.is_image_sequence:
            raise RuntimeError("Use get_frame() for image sequences.")

        item = self.frame_queue.get()
        return item

    # --------------------------------------------------
    # RANDOM ACCESS FOR IMAGE SEQUENCES ONLY
    # --------------------------------------------------
    def get_frame(self, frame_number):
        if not self.is_image_sequence:
            raise RuntimeError("Random access disabled for videos (streaming mode).")

        if frame_number < 0 or frame_number >= len(self.images):
            raise IndexError("Frame index out of range.")

        return cv2.imread(self.images[frame_number])

    # --------------------------------------------------
    # TIMESTAMPS
    # --------------------------------------------------
    def get_timestamp(self, frame_number):
        if not self.is_image_sequence:
            if self.tstart is None or self.fps is None:
                return None
            return self.tstart + pd.Timedelta(seconds=frame_number / self.fps)

        if self.time_index is not None:
            fname = os.path.basename(self.images[frame_number])
            return self.time_index.get(fname, None)

        return None

    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------
    def release(self):
        self.stopped = True
        if self.capture is not None:
            self.capture.release()


    def plot_frame(self, frame=None, frame_number=None, color="rgb", save_path=None):
        import matplotlib.pyplot as plt
        # import matplotlib
        # matplotlib.use("Agg")

        if frame is None:
            if frame_number is None:
                raise ValueError("Either frame or frame_number must be provided")
            frame = self.get_frame(frame_number)

        if color == "rgb":
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif color == "hsv":
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif color == "gray":
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif color == "bgr":
            img = frame.copy()
        else:
            raise KeyError(f"{color} is not known")

        # cv2.imshow(f"Frame {frame_number}", frame)
        # cv2.waitKey(0)  # Wait until a key is pressed
        # cv2.destroyAllWindows()


        plt.figure()
        plt.imshow(img)
        title = f"Frame {frame_number if frame_number is not None else 'streamed'} — Time: {self.get_timestamp(frame_number) if frame_number is not None else 'unknown'}"
        plt.title(title)
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()  # Will work only if environment supports GUI


