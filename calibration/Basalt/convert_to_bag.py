#!/usr/bin/env python3
"""Convert blind-nav CalibrationCollector JSONL + PNG to a single-camera ROS1 bag.

Install: python3 -m pip install rosbags==0.11.5 numpy opencv-python-headless
This converter preserves timestamps; it does not synchronize clocks.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore


def load_rows(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def check_times(rows, key, label):
    times = [r.get(key) for r in rows]
    if not times or any(type(t) is not int or not 0 <= t < (2**32)*10**9 for t in times):
        raise ValueError(f'{label}: missing/invalid integer nanoseconds in {key}')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f'{label}: duplicate or decreasing timestamps; fix acquisition, do not fabricate times')
    return times


def convert(session, output, mode, timestamp_key=None, experimental=False):
    session = Path(session).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if timestamp_key is None:
        timestamp_key = 'monotonic_ns' if (mode == 'intrinsics' and not (session/'alignment_report.json').exists()) or experimental else 'calibrated_ns'
    if mode == 'camera-imu' and timestamp_key == 'monotonic_ns' and not experimental:
        raise ValueError('Current host timestamps are not exposure timestamps. Use verified common-clock '
                         'calibrated_ns, or --experimental-host-time for pipeline experiments only.')
    frames = load_rows(session/'frames.jsonl')
    ft = check_times(frames, timestamp_key, 'camera')
    if len(frames) < 2:
        raise ValueError('At least two images are required')
    imus = load_rows(session/'imu.jsonl') if mode == 'camera-imu' else []
    if mode == 'camera-imu' and not experimental:
        if timestamp_key != 'calibrated_ns':
            raise ValueError('Production camera-IMU bags must use calibrated_ns from align_timestamps')
        report_path = session/'alignment_report.json'
        if not report_path.is_file():
            raise ValueError('Missing alignment_report.json; run memory_nav.calibration.align_timestamps first')
        report = json.loads(report_path.read_text())
        alignment_id = report.get('alignment_id')
        if (not alignment_id or report.get('status') not in ('software_aligned_requires_basalt_time_offset', 'host_receive_aligned_requires_basalt_time_offset', 'android_shared_clock')
                or any(r.get('alignment_id') != alignment_id for r in frames+imus)):
            raise ValueError('Alignment report and records do not match')
    it = check_times(imus, timestamp_key, 'imu') if imus else []
    if mode == 'camera-imu' and not imus:
        raise ValueError('No IMU samples')
    if it and (it[0] > ft[0] or it[-1] < ft[-1]):
        raise ValueError('IMU must cover the complete camera interval; explicitly crop the session first')
    resolution = None
    seen_sources = set()
    paths = []
    for row in frames:
        p = (session/row['filename']).resolve()
        if not p.is_relative_to(session):
            raise ValueError('Image path must be inside session directory')
        image = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Unreadable image: {p}')
        if resolution is None:
            resolution = image.shape
        if image.shape != resolution:
            raise ValueError('Mixed image resolutions')
        if (row.get('height'), row.get('width')) != image.shape:
            raise ValueError('Image and metadata resolutions disagree')
        source = row.get('source_frame')
        if source is not None:
            if source in seen_sources:
                raise ValueError('Repeated source_frame; recorder metadata may be inconsistent')
            seen_sources.add(source)
        paths.append(p)
    for row in imus:
        for key in ['gyro_rad_s','accel_m_s2']:
            v = row.get(key)
            if not isinstance(v,list) or len(v)!=3 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in v):
                raise ValueError(f'Invalid IMU vector: {key}')
    store = get_typestore(Stores.ROS1_NOETIC)
    types = store.types
    Time = types['builtin_interfaces/msg/Time']
    Header = types['std_msgs/msg/Header']
    Image = types['sensor_msgs/msg/Image']
    Imu = types['sensor_msgs/msg/Imu']
    V3 = types['geometry_msgs/msg/Vector3']
    Quat = types['geometry_msgs/msg/Quaternion']
    def header(t, seq, frame_id):
        sec, ns = divmod(t, 10**9)
        return Header(seq=seq, stamp=Time(sec=sec, nanosec=ns), frame_id=frame_id)
    events = sorted([(t,0,n) for n,t in enumerate(ft)] + [(t,1,n) for n,t in enumerate(it)])
    output.parent.mkdir(parents=True, exist_ok=True)
    with Writer(output) as writer:
        cam = writer.add_connection('/cam0/image_raw','sensor_msgs/msg/Image',typestore=store)
        imu = writer.add_connection('/imu0','sensor_msgs/msg/Imu',typestore=store) if it else None
        for t, kind, n in events:
            if kind == 0:
                gray = cv2.imread(str(paths[n]),cv2.IMREAD_GRAYSCALE)
                h,w = gray.shape
                msg = Image(header=header(t,n,'cam0'),height=h,width=w,encoding='mono8',
                            is_bigendian=0,step=w,data=gray.reshape(-1))
                writer.write(cam,t,store.serialize_ros1(msg,'sensor_msgs/msg/Image'))
            else:
                row = imus[n]
                covariance = np.zeros(9,dtype=np.float64)
                orientation_cov = covariance.copy(); orientation_cov[0] = -1
                msg = Imu(header=header(t,n,'imu0'),orientation=Quat(x=0.,y=0.,z=0.,w=1.),
                          orientation_covariance=orientation_cov,
                          angular_velocity=V3(*[float(v) for v in row['gyro_rad_s']]),
                          angular_velocity_covariance=covariance,
                          linear_acceleration=V3(*[float(v) for v in row['accel_m_s2']]),
                          linear_acceleration_covariance=covariance)
                writer.write(imu,t,store.serialize_ros1(msg,'sensor_msgs/msg/Imu'))
    print(json.dumps(dict(bag=str(output),images=len(ft),imu_samples=len(it),
                         resolution_wh=list(reversed(resolution)),timestamp_key=timestamp_key,
                         experimental_host_time=experimental),ensure_ascii=False,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--mode',choices=['intrinsics','camera-imu'],required=True)
    p.add_argument('--timestamp-key',default=None,
                   help='Default: calibrated_ns for camera-IMU; monotonic_ns for intrinsics/experiments')
    p.add_argument('--experimental-host-time',action='store_true')
    a = p.parse_args()
    convert(a.session,a.output,a.mode,a.timestamp_key,a.experimental_host_time)


if __name__ == '__main__':
    main()
