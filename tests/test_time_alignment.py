import base64
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from fastapi.testclient import TestClient
from utils.glasses_camera import Camera
from memory_nav.calibration.align_timestamps import align_session
from memory_nav.calibration.collect_calibration import CalibrationCollector


def write_rows(path,rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def synthetic_session(root):
    root.mkdir()
    co=10**17;ho=2*10**16;io=3*10**16
    frames=[];imus=[];sync=[]
    for n in range(32):
        t=co+n*10**9;h=ho+round(n*10**9*1.00008)
        path=root/f'{n}.png';cv2.imwrite(str(path),np.full((24,32),n,np.uint8))
        frames.append(dict(filename=path.name,width=32,height=24,source_frame=n,
                           source_frame_id=n,camera_clock_id='sensor-boot-1',
                           camera_timestamp_ns=t,timestamp_semantics='exposure_midpoint',
                           host_receive_ns=h+((n%5)+1)*40_000_000,
                           monotonic_ns=h+((n%5)+1)*40_000_000+9_000_000))
        rtt=40_000_000 if n%7==3 else 1_000_000
        sync.append(dict(clock_id='sensor-boot-1',client_send_ns=t-rtt//2,
                         client_receive_ns=t+rtt//2,server_receive_ns=h,server_send_ns=h))
    for n in range(3501):
        device=io-10**9+n*10_000_000
        h=ho+round((device-io)*.99995)
        latency=700_000+(20_000_000 if n%19==7 else 0)
        imus.append(dict(device_timestamp_us=device//1000,device_time_s=device/1e9,
                         monotonic_ns=h+latency,gyro_rad_s=[0.,0.,.1],accel_m_s2=[0.,0.,9.81]))
    for a,b in zip(imus,imus[1:]):b['monotonic_ns']=max(b['monotonic_ns'],a['monotonic_ns']+1000)
    diag=[dict(imu=dict(imu_sample_buffer_dropped=0,dropped_frames=0,frame_errors={}),
               camera=dict(queue_full=0,decode_errors=0,clock_sync_samples_dropped=0))]*2
    for name,rows in [('frames.jsonl',frames),('imu.jsonl',imus),('clock_sync.jsonl',sync),('diagnostics.jsonl',diag)]:
        write_rows(root/name,rows)
    return frames,imus,sync,ho


class TimeAlignmentTests(unittest.TestCase):
    def test_default_receipt_without_sender_metadata_or_sync(self):
        with tempfile.TemporaryDirectory() as td:
            raw=Path(td)/'raw';frames,*_=synthetic_session(raw)
            for row in frames:
                for key in ('camera_timestamp_ns','camera_clock_id','source_frame_id','timestamp_semantics'):
                    row.pop(key)
            write_rows(raw/'frames.jsonl',frames)
            (raw/'clock_sync.jsonl').unlink()
            out=Path(td)/'aligned';report=align_session(raw,out)
            aligned=[json.loads(x) for x in (out/'frames.jsonl').read_text().splitlines()]
            self.assertEqual([r['calibrated_ns'] for r in aligned],[r['host_receive_ns'] for r in frames])
            self.assertEqual(report['camera_time_source'],'host-receive')
            self.assertTrue(report['camera_clock']['network_latency_unknown'])
            del frames[0]['host_receive_ns'];write_rows(raw/'frames.jsonl',frames)
            with self.assertRaisesRegex(ValueError,'host_receive_ns'):
                align_session(raw,Path(td)/'bad')

    def test_affine_clocks_ignore_variable_camera_upload_latency(self):
        with tempfile.TemporaryDirectory() as td:
            raw=Path(td)/'raw';frames,imus,sync,host=synthetic_session(raw)
            out=Path(td)/'aligned';report=align_session(raw,out,camera_time_source='device')
            aligned=[json.loads(x) for x in (out/'frames.jsonl').read_text().splitlines()]
            self.assertEqual(len(aligned),32)
            for n,row in enumerate(aligned):
                self.assertLess(abs(row['calibrated_ns']-(host+round(n*1e9*1.00008))),100)
                self.assertEqual(row['monotonic_ns'],frames[n]['monotonic_ns'])
                self.assertTrue((out/row['filename']).is_file())
            self.assertAlmostEqual(report['camera_clock']['drift_ppm'],80,places=4)
            self.assertAlmostEqual(report['imu_clock']['drift_ppm'],-50,places=4)
            self.assertTrue(report['imu_clock']['constant_latency_unknown'])
            self.assertFalse(report['hardware_synchronization_verified'])
            self.assertEqual(len((out/'imu.jsonl').read_text().splitlines()),len(imus))

    def test_missing_timestamp_never_falls_back_to_arrival(self):
        with tempfile.TemporaryDirectory() as td:
            raw=Path(td)/'raw';frames,*_=synthetic_session(raw)
            del frames[0]['camera_timestamp_ns'];write_rows(raw/'frames.jsonl',frames)
            with self.assertRaisesRegex(ValueError,'camera_timestamp_ns'):align_session(raw,Path(td)/'out',camera_time_source='device')
            self.assertFalse((Path(td)/'out').exists())

    def test_rejects_reset_bad_sync_sample_loss_gaps_and_clock_changes(self):
        for defect in ('reset','rtt','loss','gap','id'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                raw=Path(td)/'raw';frames,imus,sync,_=synthetic_session(raw)
                if defect=='reset':
                    imus[100]['device_timestamp_us']=1;write_rows(raw/'imu.jsonl',imus)
                elif defect=='rtt':
                    for s in sync:s['client_receive_ns']=s['client_send_ns']+100_000_000
                    write_rows(raw/'clock_sync.jsonl',sync)
                elif defect=='loss':
                    rows=[json.loads(x) for x in (raw/'diagnostics.jsonl').read_text().splitlines()]
                    rows[-1]['imu']['imu_sample_buffer_dropped']=1;write_rows(raw/'diagnostics.jsonl',rows)
                elif defect=='gap':write_rows(raw/'clock_sync.jsonl',sync[:8]+sync[24:])
                else:
                    frames[10]['camera_clock_id']='restarted';write_rows(raw/'frames.jsonl',frames)
                with self.assertRaises(ValueError):align_session(raw,Path(td)/'out',camera_time_source='device')

    def test_sequence_warning_requires_continuous_actual_imu(self):
        for defect in ('none','missing','reset','crc','overflow','counter_reset'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                raw=Path(td)/'raw';_,imus,*_=synthetic_session(raw)
                diagnostics=[json.loads(x) for x in (raw/'diagnostics.jsonl').read_text().splitlines()]
                diagnostics[-1]['imu']['dropped_frames']=14592
                if defect=='missing':del imus[100]
                if defect=='reset':imus[100]['device_timestamp_us']=1
                if defect=='crc':diagnostics[-1]['imu']['frame_errors']={'payload_crc':1}
                if defect=='overflow':diagnostics[-1]['imu']['imu_sample_buffer_dropped']=1
                if defect=='counter_reset':diagnostics[0]['imu']['dropped_frames']=15000
                write_rows(raw/'imu.jsonl',imus);write_rows(raw/'diagnostics.jsonl',diagnostics)
                if defect=='none':
                    report=align_session(raw,Path(td)/'aligned')
                    self.assertEqual(report['diagnostics']['serial_sequence_counter_delta'],14592)
                    self.assertTrue(report['diagnostics']['warnings'])
                    self.assertEqual(report['imu_continuity']['observed_rate_hz'],100)
                else:
                    with self.assertRaises(ValueError):align_session(raw,Path(td)/'aligned')
                    self.assertFalse((Path(td)/'aligned').exists())

    def test_endpoint_handshake_and_metadata(self):
        with patch.object(threading.Thread,'start'),patch('utils.glasses_camera.time.sleep'):camera=Camera()
        with TestClient(camera._create_app()) as client:
            r=client.post('/time_sync',json=dict(clock_id='boot-1',client_send_ns=1_000_000))
            self.assertEqual(r.status_code,200);token=r.json()['token']
            self.assertEqual(client.post('/time_sync/sample',json=dict(token=token,client_receive_ns=11_000_000)).status_code,200)
            self.assertEqual(client.post('/time_sync/sample',json=dict(token=token,client_receive_ns=12_000_000)).status_code,422)
            ok,png=cv2.imencode('.png',np.zeros((10,20,3),np.uint8));self.assertTrue(ok)
            body=dict(image_base64=base64.b64encode(png).decode(),camera_timestamp_ns=1000,
                      camera_clock_id='boot-1',source_frame_id=1,timestamp_semantics='exposure_start',exposure_time_ns=200)
            self.assertEqual(client.post('/upload_image',json=body).status_code,200)
            queued=camera.data_queue.get_nowait()
            self.assertEqual(queued[3]['camera_timestamp_ns'],1000)
            self.assertGreater(queued[3]['host_receive_ns'],0)
            del body['exposure_time_ns'];self.assertEqual(client.post('/upload_image',json=body).status_code,422)
            self.assertEqual(client.post('/upload_image',json=dict(image_base64=body['image_base64'])).status_code,200)
            self.assertNotIn('camera_timestamp_ns',camera.data_queue.get_nowait()[3])

    def test_collector_preserves_metadata_and_strictly_rejects_legacy(self):
        class CameraFake:
            def capture_frame_with_metadata(self,**kw):
                return np.zeros((10,20,3),np.uint8),dict(frame_count=1,monotonic_ns=123,
                  camera_timestamp_ns=12,camera_clock_id='boot',source_frame_id=3,timestamp_semantics='exposure_midpoint')
            def drain_clock_sync_samples(self):return [dict(clock_id='boot')]
        class IMUFake:
            def drain_imu_samples(self):
                return [dict(monotonic_ns=10,device_time_s=1.,device_timestamp_us=1_000_000,
                             gyro_rad_s=[1.,2.,3.],accel_m_s2=[0.,0.,9.81])]
        with tempfile.TemporaryDirectory() as td:
            c=CalibrationCollector(td,CameraFake(),IMUFake(),True)
            c.capture_frame(5);c.capture_pending_imu();c.capture_sync()
            self.assertEqual(json.loads((Path(td)/'frames.jsonl').read_text())['camera_timestamp_ns'],12)
            self.assertEqual(json.loads((Path(td)/'imu.jsonl').read_text())['device_timestamp_us'],1_000_000)
            c.camera.capture_frame_with_metadata=lambda **kw:(np.zeros((10,20,3),np.uint8),dict(frame_count=2,monotonic_ns=124))
            with self.assertRaisesRegex(ValueError,'sender lacks'):c.capture_frame(6)


if __name__=='__main__':unittest.main()
