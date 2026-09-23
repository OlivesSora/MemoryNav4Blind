"""Record every HTTP camera callback and its entire Android IMU batch."""
import argparse
import json
import queue
import signal
import threading
import time
from pathlib import Path

import cv2

CLOCK = 'android_elapsed_realtime_ns'


def timestamp(value, name):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(f'{name} must be an integer nanosecond timestamp')
    return value


class AndroidRecorder:
    def __init__(self, output, preview=None):
        self.output = Path(output).resolve()
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(self.output)
        (self.output / 'dataset/cam0').mkdir(parents=True)
        self.preview = Path(preview) if preview else None
        if self.preview:
            self.preview.parent.mkdir(parents=True, exist_ok=True)
        self.session = None
        self.frames = 0
        self.samples = 0
        self.frame_times = set()

    def append(self, name, row):
        with (self.output / name).open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def write(self, image, metadata):
        if metadata.get('clock_domain') != CLOCK:
            raise ValueError('Expected Android elapsed realtime clock; arrival-time fallback is forbidden')
        session = metadata.get('session_id')
        if not isinstance(session, str) or not session:
            raise ValueError('Missing session_id')
        if self.session is not None and session != self.session:
            raise ValueError('App session changed: start a new recording')
        self.session = session
        t = timestamp(metadata.get('frame_timestamp_ns'), 'frame_timestamp_ns')
        if t in self.frame_times:
            raise ValueError('Repeated frame timestamp')
        samples = metadata.get('imu_samples')
        if not isinstance(samples, list):
            raise ValueError('Missing imu_samples array')
        rows = []
        for sample in samples:
            timestamp(sample.get('timestamp_ns'), 'IMU timestamp_ns')
            timestamp(sample.get('accel_timestamp_ns'), 'accel_timestamp_ns')
            for key in ('accel', 'gyro'):
                import math
                vector = sample.get(key)
                if not isinstance(vector, list) or len(vector) != 3 or not all(
                        type(v) in (int, float) and math.isfinite(v) for v in vector):
                    raise ValueError(f'Invalid {key}')
            rows.append(dict(sample, session_id=session, clock_domain=CLOCK,
                             batch_frame_timestamp_ns=t))
        rel = f'dataset/cam0/{t}.png'
        # The callback returns BGR: no RGB conversion here.
        if not cv2.imwrite(str(self.output / rel), image):
            raise OSError('Image write failed')
        for row in rows:
            self.append('imu.jsonl', row)
        frame = {k: v for k, v in metadata.items() if k != 'imu_samples'}
        frame.update(filename=rel, width=image.shape[1], height=image.shape[0],
                     source_frame=self.frames, index=self.frames,
                     timestamp_semantics='first_row_exposure_start')
        self.append('frames.jsonl', frame)
        self.frames += 1
        self.samples += len(rows)
        self.frame_times.add(t)
        if self.preview:
            temporary = self.preview.with_name('.tmp.jpg.android-writing.jpg')
            try:
                if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    raise OSError('Preview write failed')
                temporary.replace(self.preview)
            finally:
                temporary.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--duration', type=float, default=120)
    a = p.parse_args(argv)
    if not 0 < a.duration < 86400:
        p.error('duration must be between 0 and 86400 seconds')
    recorder = AndroidRecorder(a.output, Path(__file__).resolve().parents[2] / 'calibration_data/tmp.jpg')
    pending = queue.Queue(maxsize=256)
    overflow = threading.Event()
    stop = threading.Event()

    def callback(image, metadata):
        try:
            pending.put_nowait((image.copy(), metadata))
        except queue.Full:
            overflow.set()

    from utils.glasses_camera import Camera
    camera = None
    error = None
    handlers = {s: signal.signal(s, lambda *_: stop.set()) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        camera = Camera(save_pic=False, use_blur_filter=False, frame_callback=callback)
        end = time.monotonic() + a.duration
        while not stop.is_set() and time.monotonic() < end:
            if overflow.is_set():
                raise ValueError('Recorder queue overflow: incomplete session')
            try:
                image, metadata = pending.get(timeout=0.1)
            except queue.Empty:
                continue
            recorder.write(image, metadata)
    except Exception as exc:
        error = str(exc)
    finally:
        try:
            if camera:
                camera.release()
            if error is None:
                while not pending.empty():
                    recorder.write(*pending.get_nowait())
        except Exception as exc:
            error = str(exc)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    diagnostics = camera.get_diagnostics() if camera else {}
    if overflow.is_set() or any(diagnostics.get(k, 0) for k in ('queue_full', 'decode_errors', 'callback_errors', 'processing_errors')):
        error = error or 'Camera/recorder dropped data; see collection_report.json'
    if recorder.frames < 2 or recorder.samples < 2:
        error = error or 'Insufficient camera/IMU data'
    report = dict(status='failed' if error else 'complete', error=error,
                  source='android_http', session_id=recorder.session, clock_domain=CLOCK,
                  frames=recorder.frames, raw_imu_samples=recorder.samples,
                  recorder_queue_overflow=overflow.is_set(), camera=diagnostics)
    (recorder.output / 'collection_report.json').write_text(json.dumps(report, indent=2) + '\n')
    if error:
        raise RuntimeError(error)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    main()
