import threading
import unittest
from unittest.mock import patch
import numpy as np
from utils.glasses_camera import Camera

class CameraMetadataTests(unittest.TestCase):
    def make_camera(self):
        with patch.object(threading.Thread,'start'), patch('utils.glasses_camera.time.sleep'):
            return Camera()

    def test_frame_and_metadata_are_one_atomic_snapshot(self):
        camera=self.make_camera()
        with camera.frame_lock:
            camera.frame=np.full((10,20,3),7,dtype=np.uint8)
            camera.frame_count=7
            camera.frame_metadata=dict(frame_count=7,monotonic_ns=123456,width=20,height=10,
                                       camera_timestamp_ns=987654321,source_frame_id=71)
        image,metadata=camera.capture_frame_with_metadata()
        self.assertEqual(int(image[0,0,0]),metadata['frame_count'])
        self.assertEqual(metadata['camera_timestamp_ns'],987654321)
        with camera.frame_lock:
            camera.frame[:]=8
            camera.frame_metadata['source_frame_id']=72
        self.assertEqual(metadata['source_frame_id'],71)
        self.assertTrue(np.all(image==7))
        _,count=camera.capture_frame()
        self.assertEqual(count,7)

    def test_latest_only_replaces_backlogged_upload(self):
        from utils.glasses_camera import ImageRequest
        with patch.object(threading.Thread, 'start'), patch('utils.glasses_camera.time.sleep'):
            camera = Camera(latest_only=True)
        first = ImageRequest(image_base64="x", frame_timestamp_ns=1)
        second = ImageRequest(image_base64="x", frame_timestamp_ns=2)
        camera._enqueue_upload(b"first", first)
        camera._enqueue_upload(b"second", second)
        self.assertEqual(camera.data_queue.qsize(), 1)
        self.assertEqual(camera.data_queue.get_nowait()[0], b"second")
        self.assertEqual(camera.dropped_old_frames, 1)

    def test_latest_only_preserves_queued_one_shot_capture(self):
        import queue
        from utils.glasses_camera import ImageRequest
        with patch.object(threading.Thread, 'start'), patch('utils.glasses_camera.time.sleep'):
            camera = Camera(latest_only=True)
        with camera._capture_condition:
            camera._waiting_capture = (1, (1280, 720))
        requested = ImageRequest(
            image_base64="x", capture_width=1280, capture_height=720
        )
        regular = ImageRequest(
            image_base64="x", capture_width=640, capture_height=480
        )
        camera._enqueue_upload(b"requested", requested)
        with self.assertRaises(queue.Full):
            camera._enqueue_upload(b"regular", regular)
        self.assertEqual(camera.data_queue.get_nowait()[0], b"requested")

    def test_one_shot_preserves_its_own_metadata(self):
        camera=self.make_camera();result=[]
        worker=threading.Thread(target=lambda: result.append(camera.capture_frame_with_metadata(20,10)))
        worker.start()
        import time
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            with camera._capture_condition:
                if camera._waiting_capture:
                    request_id,_=camera._waiting_capture
                    camera._requested_frame=np.zeros((10,20,3),np.uint8)
                    camera._requested_frame_metadata=dict(frame_count=4,monotonic_ns=999,camera_timestamp_ns=123)
                    camera._completed_capture_request_id=request_id
                    camera._capture_condition.notify_all()
                    break
            time.sleep(.001)
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result[0][1]['camera_timestamp_ns'],123)

if __name__=='__main__':unittest.main()
