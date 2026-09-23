"""Prepare same-clock Android data for the existing Basalt bag converter."""
import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from .android_capture import CLOCK, timestamp


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def prepare(source, output, max_gap_ms=50.0):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError('Output must be a new directory outside the raw session')
    collection = json.loads((source / 'collection_report.json').read_text())
    if collection.get('status') != 'complete' or collection.get('source') != 'android_http':
        raise ValueError('A successfully completed Android collection is required')
    frames, samples = rows(source / 'frames.jsonl'), rows(source / 'imu.jsonl')
    if len(frames) != collection['frames'] or len(samples) != collection['raw_imu_samples']:
        raise ValueError('Raw data counts do not match completed collection report')
    session = collection['session_id']
    for row in frames + samples:
        if row.get('session_id') != session or row.get('clock_domain') != CLOCK:
            raise ValueError('Mixed session_id or clock_domain')
    frames.sort(key=lambda x: timestamp(x.get('frame_timestamp_ns'), 'frame_timestamp_ns'))
    if len({f['frame_timestamp_ns'] for f in frames}) != len(frames):
        raise ValueError('Repeated camera timestamp')
    gyros, accelerations = {}, {}
    for row in samples:
        t = timestamp(row.get('timestamp_ns'), 'timestamp_ns')
        at = timestamp(row.get('accel_timestamp_ns'), 'accel_timestamp_ns')
        for key in ('gyro', 'accel'):
            v = np.asarray(row[key], dtype=float)
            if v.shape != (3,) or not np.isfinite(v).all():
                raise ValueError('Invalid IMU vector')
        if t in gyros and any(gyros[t][k] != row[k] for k in ('gyro', 'accel', 'accel_timestamp_ns')):
            raise ValueError('Conflicting duplicate gyro sample')
        gyros[t] = row
        if at in accelerations and accelerations[at] != row['accel']:
            raise ValueError('Conflicting acceleration values at same timestamp')
        accelerations[at] = row['accel']
    gt, at = sorted(gyros), sorted(accelerations)
    for times, label in ((gt, 'gyro'), (at, 'accel')):
        if len(times) < 2:
            raise ValueError(f'Insufficient distinct {label} samples')
        delta = np.diff(times)
        if delta.max() > min(max_gap_ms * 1e6, 5 * float(np.median(delta))):
            raise ValueError(f'{label} sampling gap too large; check missing batches or low sensor rate')
    # Preserve gyro timestamps exactly. Interpolate only the independent accel
    # stream, using centered integer differences to avoid large-epoch rounding.
    imu = []
    j = 0
    for t in gt:
        if t < at[0] or t > at[-1]:
            continue
        while j + 1 < len(at) and at[j + 1] < t:
            j += 1
        if t == at[j]:
            accel = accelerations[at[j]]
        else:
            lo, hi = at[j], at[j + 1]
            alpha = (t - lo) / (hi - lo)
            accel = ((1 - alpha) * np.array(accelerations[lo]) + alpha * np.array(accelerations[hi])).tolist()
        row = gyros[t]
        imu.append(dict(row, calibrated_ns=t, gyro_rad_s=row['gyro'], accel_m_s2=accel,
                        acceleration_method='linear_at_gyro_timestamp'))
    if len(imu) < 2:
        raise ValueError('No overlap of gyro and acceleration streams')
    kept = [f for f in frames if imu[0]['calibrated_ns'] <= f['frame_timestamp_ns'] <= imu[-1]['calibrated_ns']]
    if len(kept) < 2 or kept[-1]['frame_timestamp_ns'] - kept[0]['frame_timestamp_ns'] < 20_000_000_000:
        raise ValueError('Need at least 20 seconds of camera/IMU overlap')
    alignment_id = uuid.uuid4().hex
    temp = output.with_name(output.name + '.partial-' + alignment_id)
    try:
        (temp / 'dataset/cam0').mkdir(parents=True)
        exported = []
        for frame in kept:
            src = (source / frame['filename']).resolve()
            if not src.is_relative_to(source) or not src.is_file():
                raise ValueError('Missing/invalid image path')
            t = frame['frame_timestamp_ns']
            filename = f'dataset/cam0/{t}.png'
            shutil.copy2(src, temp / filename)
            exported.append(dict(frame, filename=filename, calibrated_ns=t, alignment_id=alignment_id))
        imu = [dict(row, alignment_id=alignment_id) for row in imu]
        for name, data in [('frames.jsonl', exported), ('imu.jsonl', imu)]:
            (temp / name).write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in data))
        report = dict(schema_version=2, status='android_shared_clock', alignment_id=alignment_id,
                      source=str(source), clock_domain=CLOCK, session_id=session,
                      camera_time_source='android', timestamp_offset_ns=0,
                      camera_timestamp_semantics='first_row_exposure_start',
                      raw_frames=len(frames), exported_frames=len(kept), trimmed_frames=len(frames)-len(kept),
                      raw_imu_samples=len(samples), unique_gyro_samples=len(gt), imu_samples=len(imu),
                      duplicate_gyro_samples=len(samples)-len(gt), unique_accel_samples=len(at),
                      gyro_boundary_samples_trimmed=len(gt)-len(imu),
                      acceleration_method='linear interpolation at unchanged gyro timestamps; no extrapolation',
                      max_gyro_gap_ms=float(np.diff(gt).max()/1e6), max_accel_gap_ms=float(np.diff(at).max()/1e6),
                      note='Shared Android clock per sender contract; HTTP arrival times unused. Rolling shutter and timestamp accuracy require sensor validation.')
        (temp / 'alignment_report.json').write_text(json.dumps(report, indent=2)+'\n')
        if output.exists():
            raise FileExistsError(output)
        temp.rename(output)
    except BaseException:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    return report
