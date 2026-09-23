import asyncio
import base64
import json
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from memory_nav.calibration.android_capture import AndroidRecorder, CLOCK
from memory_nav.calibration.android_prepare import prepare, rows
from memory_nav.calibration.Basalt.convert_to_bag import convert
from utils.glasses_camera import Camera, ImageRequest


class AndroidCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = 100_000_000_000_000_001
        self.image = np.zeros((8, 12, 3), np.uint8)
        self.image[:, :, 2] = 255

    def metadata(self, n=0):
        return dict(frame_timestamp_ns=self.base+n*1_000_000_000,
                    session_id='test-app', clock_domain=CLOCK,
                    received_at_unix_ns=1750000000000000000+n*123456789)

    def record(self):
        rec = AndroidRecorder(self.root/'raw', self.root/'tmp.jpg')
        for n in range(24):
            samples = []
            for k in range(n*100, (n+1)*100):
                # accel is delayed 2 ms, with a known linear signal.
                t = self.base+k*10_000_000
                samples.append(dict(timestamp_ns=t, accel_timestamp_ns=t-2_000_000,
                                    accel=[k*.01-.002, 0., 9.81], gyro=[.1, .2, .3], mag=[1.,2.,3.]))
            meta = self.metadata(n)
            meta['imu_samples'] = samples
            rec.write(self.image, meta)
        (rec.output/'collection_report.json').write_text(json.dumps(dict(
            status='complete',source='android_http',session_id=rec.session,
            frames=rec.frames, raw_imu_samples=rec.samples)))
        return rec

    def test_bag_roundtrip_preserves_android_nanoseconds_and_all_batches(self):
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_typestore
        rec = self.record()
        self.assertEqual(rec.samples, 2400)
        saved = cv2.imread(str(rec.output/rows(rec.output/'frames.jsonl')[0]['filename']))
        np.testing.assert_array_equal(saved, self.image)
        report = prepare(rec.output, self.root/'aligned')
        self.assertEqual(report['timestamp_offset_ns'], 0)
        self.assertEqual(report['imu_samples'], 2399)
        aligned = rows(self.root/'aligned/imu.jsonl')
        for row in aligned:
            self.assertEqual(row['calibrated_ns'], row['timestamp_ns'])
            self.assertAlmostEqual(row['accel_m_s2'][0], (row['timestamp_ns']-self.base)*1e-9)
        convert(self.root/'aligned', self.root/'test.bag', 'camera-imu')
        store = get_typestore(Stores.ROS1_NOETIC)
        counts = {}
        with Reader(self.root/'test.bag') as reader:
            for connection, t, data in reader.messages():
                msg = store.deserialize_ros1(data, connection.msgtype)
                self.assertEqual(t, msg.header.stamp.sec*10**9+msg.header.stamp.nanosec)
                counts[connection.topic] = counts.get(connection.topic, 0)+1
                if connection.topic == '/imu0':
                    self.assertAlmostEqual(msg.angular_velocity.y, .2)
                    self.assertAlmostEqual(msg.linear_acceleration.x, (t-self.base)*1e-9)
        self.assertEqual(counts, {'/imu0':2399, '/cam0/image_raw':24})

    def test_reject_session_change_and_missing_sensor_times(self):
        rec = AndroidRecorder(self.root/'raw')
        m = self.metadata(); m['imu_samples'] = []
        rec.write(self.image, m)
        m = self.metadata(1); m.update(session_id='restart', imu_samples=[])
        with self.assertRaisesRegex(ValueError, 'session changed'): rec.write(self.image, m)
        m['session_id']='test-app'; m['imu_samples']=[dict(timestamp_ns=42, accel=[0,0,0],gyro=[0,0,0])]
        with self.assertRaisesRegex(ValueError, 'accel_timestamp'): rec.write(self.image,m)
        m['clock_domain']='unix'
        with self.assertRaisesRegex(ValueError, 'fallback'): rec.write(self.image,m)

    def test_refuse_imu_gap(self):
        rec = self.record()
        data=rows(rec.output/'imu.jsonl'); del data[100:110]
        (rec.output/'imu.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
        p=rec.output/'collection_report.json'; report=json.loads(p.read_text());report['raw_imu_samples']=len(data);p.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'gap too large'):prepare(rec.output,self.root/'aligned')
        self.assertFalse((self.root/'aligned').exists())

    def test_refuse_failed_collection(self):
        rec=self.record();p=rec.output/'collection_report.json';report=json.loads(p.read_text());report['status']='failed';p.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'successfully completed'):prepare(rec.output,self.root/'aligned')

    def camera(self, callback):
        with patch('threading.Thread.start'), patch('utils.glasses_camera.time.sleep'):
            return Camera(frame_callback=callback)

    def test_camera_drains_after_stop_and_preserves_batch(self):
        got=[];camera=self.camera(lambda im,meta:got.append((im,meta)))
        jpg=cv2.imencode('.jpg',self.image)[1].tobytes()
        batch=[dict(timestamp_ns=1),dict(timestamp_ns=2)]
        camera.data_queue.put((jpg,12,8,self.base,batch,CLOCK,'test-app',987))
        camera.is_recording=False
        camera._capture_frames()
        self.assertEqual(got[0][1]['imu_samples'],batch)
        self.assertEqual(got[0][1]['frame_timestamp_ns'],self.base)
        self.assertGreater(got[0][0][0,0,2],250)
        self.assertEqual(sum(camera.get_diagnostics().values()),0)

    def test_camera_reports_corrupt_jpeg_and_callback_error(self):
        def fail(*a):raise RuntimeError('test callback failure')
        camera=self.camera(fail)
        for jpg in (b'badjpeg',cv2.imencode('.jpg',self.image)[1].tobytes()):
            camera.data_queue.put((jpg,12,8,self.base,[],CLOCK,'test-app',987))
        camera.is_recording=False;camera._capture_frames()
        self.assertEqual(camera.get_diagnostics()['decode_errors'],1)
        self.assertEqual(camera.get_diagnostics()['callback_errors'],1)

    def test_default_collector_dispatches_to_android(self):
        from memory_nav.calibration.collect_calibration import main
        with patch('memory_nav.calibration.android_capture.main', return_value=0) as run:
            self.assertEqual(main(['--output', 'example', '--duration', '120']), 0)
            run.assert_called_once_with(['--output', 'example', '--duration', '120'])

    def test_http_queue_full_is_reported(self):
        camera=self.camera(None)
        for _ in range(10):camera.data_queue.put(None)
        endpoint=next(r.endpoint for r in camera._create_app().routes if r.path=='/upload_image')
        request=ImageRequest(image_base64=base64.b64encode(b'jpeg').decode())
        result=asyncio.run(endpoint(request))
        self.assertEqual(result['status'],'queue_full')
        self.assertEqual(camera.get_diagnostics()['queue_full'],1)

if __name__=='__main__':unittest.main()
