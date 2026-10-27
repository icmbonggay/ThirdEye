"""Threaded frame reader.

Live sources (RTSP, HTTP, webcam): only the most recent frame is kept, and the
connection is retried with exponential backoff if it drops.

Video files: every frame is delivered in order.
"""
import logging
import os
import queue
import threading
import time

import cv2

log = logging.getLogger(__name__)

_END = object()


def is_live(source):
    s = str(source)
    return s.isdigit() or s.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))


class FrameReader:
    def __init__(self, source, max_retries=4, retry_delay=5.0, rtsp_transport="tcp"):
        self.source = int(source) if str(source).isdigit() else source
        self.live = is_live(source)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        if rtsp_transport:
            # RTSP transport (tcp/udp). Must be set before any VideoCapture is created.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{rtsp_transport}")
        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.failed = False
        self.fps = None

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def read(self, timeout=None):
        """Return (frame, timestamp_seconds), or (None, None) when the stream ends."""
        item = self._queue.get(timeout=timeout)
        if item is _END:
            self._queue.put(_END)  # keep returning "ended" on later calls
            return None, None
        return item

    # --- internals -------------------------------------------------------

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        if self.live:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap if cap.isOpened() else None

    def _put(self, item):
        if self.live:
            # Replace any unread frame with the newest one.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put(item)
        else:
            while not self._stop.is_set():
                try:
                    self._queue.put(item, timeout=0.5)
                    return
                except queue.Full:
                    continue

    def _connect(self):
        """Open the source, retrying with growing delays. Returns a capture or None."""
        for attempt in range(self.max_retries + 1):
            if self._stop.is_set():
                return None
            cap = self._open()
            if cap is not None:
                if attempt:
                    log.info("Reconnected to video source")
                return cap
            if not self.live or attempt == self.max_retries:
                break
            delay = self.retry_delay * (2 ** attempt)
            log.warning("Could not open video source (attempt %d/%d), retrying in %.0fs",
                        attempt + 1, self.max_retries, delay)
            self._stop.wait(delay)
        return None

    def _run(self):
        cap = self._connect()
        start_wall = time.time()
        while cap is not None and not self._stop.is_set():
            self.fps = cap.get(cv2.CAP_PROP_FPS) or None
            ok, frame = cap.read()
            if ok:
                if self.live:
                    ts = time.time()
                else:
                    # Timestamp from the video's position, not the wall clock.
                    ts = start_wall + cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                self._put((frame, ts))
                continue
            cap.release()
            if not self.live:
                cap = None
                break
            log.warning("Lost the video stream, reconnecting...")
            cap = self._connect()
        if cap is None and self.live and not self._stop.is_set():
            self.failed = True
            log.error("Lost connection to the camera.")
        if cap is not None:
            cap.release()
        self._put(_END)
