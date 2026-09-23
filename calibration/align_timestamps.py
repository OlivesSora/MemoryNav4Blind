"""Prepare Android same-clock data by default, preserving camera/gyro nanoseconds.
Legacy N100 alignment requires explicit --camera-time-source host-receive or device.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import uuid
from pathlib import Path

import numpy as np


def read_rows(path):
    if not path.is_file():
        raise ValueError(f"missing {path.name}; record data with the updated collector first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def integer(value, label):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(f"{label}: expected nonnegative integer timestamp/counter")
    return value


def ordered(values, label):
    if len(values) < 2 or any(b <= a for a,b in zip(values,values[1:])):
        raise ValueError(f"{label}: fewer than two samples, repeated timestamp or clock reset")


def fit_clock(xs, ys, max_drift_ppm=500.0, min_span_s=20.0):
    """Median pairwise slope on separated anchors; centered arithmetic avoids
    loss of integer nanoseconds when clocks have large absolute epochs.
    """
    ordered(xs, "clock anchors")
    x0, y0 = xs[0], ys[0]
    x = np.array([v-x0 for v in xs], dtype=float) / 1e9
    y = np.array([v-y0 for v in ys], dtype=float) / 1e9
    if len(x) < 6 or x[-1] < min_span_s:
        raise ValueError(f"clock mapping requires >=6 anchors across >= {min_span_s:g} seconds")
    # At most 600 anchors for the slope estimate; evaluate all anchors below.
    indices = np.unique(np.linspace(0,len(x)-1,min(600,len(x)),dtype=int))
    slopes=[]
    for i in indices:
        later=indices[x[indices]-x[i] >= x[-1]/3]
        slopes.extend(((y[later]-y[i])/(x[later]-x[i])).tolist())
    slope=float(np.median(slopes))
    if not math.isfinite(slope) or abs(slope-1)*1e6 > max_drift_ppm:
        raise ValueError("clock drift exceeds limit; check units, resets and clock domains")
    offset=float(np.median(y-slope*x))
    residual_ns=(y-(slope*x+offset))*1e9
    model=dict(device_origin_ns=x0,host_origin_ns=y0+round(offset*1e9),
               scale=slope,drift_ppm=(slope-1)*1e6,span_s=float(x[-1]),anchors=len(xs),
               anchor_max_residual_ns=float(np.max(np.abs(residual_ns))))
    return model,residual_ns


def mapped(model, timestamp):
    return model['host_origin_ns']+round(model['scale']*(timestamp-model['device_origin_ns']))


def camera_model(samples, clock_id, max_error_ns, max_drift_ppm):
    rows=[r for r in samples if r.get('clock_id')==clock_id]
    if not rows:
        raise ValueError("no time-sync exchanges for the camera_clock_id")
    anchors=[]
    for r in rows:
        t1,t2,t3,t4=[integer(r.get(k),k) for k in
                     ('client_send_ns','server_receive_ns','server_send_ns','client_receive_ns')]
        rtt=(t4-t1)-(t3-t2)
        if t4<=t1 or t3<t2 or rtt<0:
            raise ValueError("invalid four-timestamp exchange")
        anchors.append(((t1+t4)//2,(t2+t3)//2,rtt))
    # Completion order may differ for concurrent HTTP requests.
    anchors.sort()
    ordered([r[0] for r in anchors], 'camera sync clock')
    bins={}
    for a in anchors:
        bucket=(a[0]-anchors[0][0])//1_000_000_000
        if bucket not in bins or a[2]<bins[bucket][2]:bins[bucket]=a
    selected=[a for a in bins.values() if a[2] <= 2*max_error_ns]
    if len(selected)<6:
        raise ValueError("insufficient low-RTT camera exchanges; improve network or inspect the client clock")
    model,residuals=fit_clock([a[0] for a in selected],[a[1] for a in selected],max_drift_ppm)
    # Includes exchange asymmetry bound and departures from the affine model.
    bound=max(a[2]/2+abs(float(r)) for a,r in zip(selected,residuals))
    if bound>max_error_ns:
        raise ValueError(f"camera clock uncertainty {bound/1e6:.3f} ms exceeds {max_error_ns/1e6:g} ms")
    if max(b[0]-a[0] for a,b in zip(selected,selected[1:]))>10_000_000_000:
        raise ValueError("camera sync gap exceeds 10 seconds; send clock exchanges throughout recording")
    model.update(clock_id=clock_id,uncertainty_bound_ns=math.ceil(bound),
                 first_anchor_ns=selected[0][0],last_anchor_ns=selected[-1][0],
                 rejected_high_rtt=len(bins)-len(selected),method='four_timestamp_affine')
    return model


def imu_model(imus,max_error_ns,max_drift_ppm):
    xs=[integer(r.get('device_timestamp_us'),'device_timestamp_us')*1000 for r in imus]
    ys=[integer(r.get('monotonic_ns'),'IMU receive timestamp') for r in imus]
    ordered(xs,'IMU device clock');ordered(ys,'IMU host receive clock')
    # Compare samples within each 1-second bin; lowest delay is least affected
    # by serial queuing. A fixed serial latency cannot be identified here.
    bins={}
    for x,y in zip(xs,ys):
        bucket=(x-xs[0])//1_000_000_000
        if bucket not in bins or y-x<bins[bucket][1]-bins[bucket][0]:bins[bucket]=(x,y)
    anchors=list(bins.values())
    model,residuals=fit_clock([a[0] for a in anchors],[a[1] for a in anchors],max_drift_ppm)
    if max(abs(residuals))>max_error_ns:
        raise ValueError("IMU low-delay clock anchors are unstable; inspect serial stalls/clock changes")
    model.update(method='serial_low_delay_affine',constant_latency_unknown=True,
                 requires_basalt_time_offset=True)
    return model,xs


def validate_imu_continuity(imus):
    """Check the actual TYPE_IMU stream, independent of mixed-packet sequence IDs.

    Assumes a fixed sample rate during one recording. Median cadence detects
    isolated missing samples, but cannot prove the configured absolute rate.
    """
    ts=[integer(r.get('device_timestamp_us'),'device_timestamp_us') for r in imus]
    ordered(ts,'IMU device clock')
    intervals=np.array([b-a for a,b in zip(ts,ts[1:])],dtype=np.int64)
    period=float(np.median(intervals))
    bad=np.flatnonzero((intervals < period*0.5) | (intervals > period*1.5))
    if len(bad):
        i=int(bad[0])
        raise ValueError(f"IMU sample interval discontinuity at index {i}: "
                         f"{int(intervals[i])} us, median {period:g} us; "
                         "possible missing samples or rate change")
    return dict(method='device_timestamp_cadence',samples=len(ts),
                span_s=(ts[-1]-ts[0])/1e6,median_interval_us=period,
                min_interval_us=int(intervals.min()),max_interval_us=int(intervals.max()),
                observed_rate_hz=1e6/period,interval_ratio_limits=[0.5,1.5],
                configured_sample_rate_verified=False)


def validate_diagnostics(rows):
    if len(rows)<2:
        raise ValueError("need start/end diagnostics from the updated collector")
    paths=[('imu','imu_sample_buffer_dropped'),
           ('camera','queue_full'),('camera','decode_errors'),('camera','clock_sync_samples_dropped')]
    for device,key in paths:
        if any(key not in r.get(device,{}) for r in rows):
            raise ValueError(f"missing diagnostic {device}.{key}")
        values=[r[device][key] for r in rows]
        if max(values)!=min(values):
            raise ValueError(f"{device}.{key} changed during recording; inspect dropped/corrupt data")
    for r in rows:
        errors=r.get('imu',{}).get('frame_errors')
        if not isinstance(errors,dict):raise ValueError("missing IMU frame_errors diagnostics")
        if errors!=rows[0]['imu']['frame_errors']:
            raise ValueError("IMU parser errors changed during recording")

    # This counter spans all serial packet types. Without recorded sequence IDs
    # it cannot identify lost TYPE_IMU samples; retain it as an explicit warning.
    counters=[integer(r.get('imu',{}).get('dropped_frames'),'imu.dropped_frames') for r in rows]
    if any(b<a for a,b in zip(counters,counters[1:])):
        raise ValueError("imu.dropped_frames reset during recording")
    delta=counters[-1]-counters[0]
    return dict(serial_sequence_counter_start=counters[0],
                serial_sequence_counter_end=counters[-1],serial_sequence_counter_delta=delta,
                warnings=(["Serial sequence counter increased; this is not a TYPE_IMU loss count. "
                           "Export accepted only after device timestamp cadence, parser and buffer checks. "
                           "Packet sequence cause remains unverified."] if delta else []))


def align_session(source,output,max_camera_error_ms=5.0,max_imu_error_ms=3.0,
                  max_drift_ppm=500.0,trim_to_overlap=False,camera_time_source="host-receive"):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(f"output already exists: {output}")
    if output.is_relative_to(source):raise ValueError("aligned output must be outside raw session")
    for value in (max_camera_error_ms,max_imu_error_ms,max_drift_ppm):
        if not math.isfinite(value) or value<=0:raise ValueError("quality thresholds must be finite and positive")
    frames=read_rows(source/'frames.jsonl');imus=read_rows(source/'imu.jsonl')
    for row in imus:
        for key in ('gyro_rad_s','accel_m_s2'):
            vector=row.get(key)
            if not isinstance(vector,list) or len(vector)!=3 or not all(type(v) in (int,float) and math.isfinite(v) for v in vector):
                raise ValueError(f"invalid IMU vector {key}")
    continuity=validate_imu_continuity(imus)
    diagnostics=validate_diagnostics(read_rows(source/'diagnostics.jsonl'))
    if camera_time_source == 'host-receive':
        camera_times=[integer(r.get('host_receive_ns'),'host_receive_ns') for r in frames]
        ordered(camera_times,'camera host receive time')
        cam=dict(method='host_receive',device_origin_ns=0,host_origin_ns=0,scale=1.0,
                 exposure_time_verified=False,network_latency_unknown=True)
    elif camera_time_source == 'device':
        sync=read_rows(source/'clock_sync.jsonl')
        clock_ids={r.get('camera_clock_id') for r in frames}
        if len(clock_ids)!=1 or None in clock_ids or '' in clock_ids:
            raise ValueError("missing or mixed camera clock IDs; cannot align host-only frames")
        clock_id=next(iter(clock_ids))
        camera_times=[]
        for row in frames:
            t=integer(row.get('camera_timestamp_ns'),'camera_timestamp_ns')
            semantics=row.get('timestamp_semantics')
            if semantics=='exposure_start':
                t+=integer(row.get('exposure_time_ns'),'exposure_time_ns')//2
            elif semantics!='exposure_midpoint':raise ValueError("unknown exposure timestamp semantics")
            camera_times.append(t)
        ordered(camera_times,'camera exposure time')
        ordered([integer(r.get('source_frame_id'),'source_frame_id') for r in frames],'camera frame IDs')
        cam=camera_model(sync,clock_id,round(max_camera_error_ms*1e6),max_drift_ppm)
        if camera_times[0]<cam['first_anchor_ns']-2_000_000_000 or camera_times[-1]>cam['last_anchor_ns']+2_000_000_000:
            raise ValueError("camera exposures extend beyond sync coverage by more than 2 seconds")
    else:
        raise ValueError("camera_time_source must be host-receive or device")
    imu,imu_times=imu_model(imus,round(max_imu_error_ms*1e6),max_drift_ppm)
    ft=camera_times if camera_time_source == 'host-receive' else [mapped(cam,t) for t in camera_times];it=[mapped(imu,t) for t in imu_times]
    ordered(ft,'aligned camera');ordered(it,'aligned IMU')
    keep=[n for n,t in enumerate(ft) if it[0]<=t<=it[-1]]
    if len(keep)!=len(ft) and not trim_to_overlap:
        raise ValueError("IMU does not cover all camera times; inspect data or use --trim-to-overlap")
    if len(keep)<2 or ft[keep[-1]]-ft[keep[0]]<20_000_000_000:
        raise ValueError("aligned overlap must contain >=2 images and >=20 seconds")
    paths=[]
    for n in keep:
        p=(source/frames[n]['filename']).resolve()
        if not p.is_relative_to(source) or not p.is_file():raise ValueError("missing/invalid source image path")
        paths.append(p)
    alignment_id=uuid.uuid4().hex
    report=dict(schema_version=1,alignment_id=alignment_id,
                status=('host_receive_aligned_requires_basalt_time_offset' if camera_time_source == 'host-receive'
                        else 'software_aligned_requires_basalt_time_offset'),
                camera_time_source=camera_time_source,
                source=str(source),camera_clock=cam,imu_clock=imu,
                imu_continuity=continuity,diagnostics=diagnostics,
                raw_frames=len(frames),exported_frames=len(keep),trimmed_frames=len(frames)-len(keep),
                imu_samples=len(imus),max_camera_error_ms=max_camera_error_ms,
                max_imu_error_ms=max_imu_error_ms,max_drift_ppm=max_drift_ppm,
                hardware_synchronization_verified=False,
                note='Host-receive camera time includes unknown variable upload latency.' if camera_time_source == 'host-receive' else 'Exposure clock alignment only; refine constant IMU offset in Basalt.')
    # Write to a separate sibling temporary directory, then publish atomically.
    temp=output.with_name(output.name+'.partial-'+alignment_id)
    try:
        (temp/'dataset/cam0').mkdir(parents=True)
        exported=[]
        for n,p in zip(keep,paths):
            rel=f'dataset/cam0/{ft[n]}{p.suffix}'
            shutil.copy2(p,temp/rel)
            exported.append(dict(frames[n],filename=rel,calibrated_ns=ft[n],alignment_id=alignment_id))
        imu_export=[dict(r,calibrated_ns=t,alignment_id=alignment_id) for r,t in zip(imus,it)]
        for name,rows in [('frames.jsonl',exported),('imu.jsonl',imu_export)]:
            (temp/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        with (temp/'dataset/imu0.csv').open('w') as f:
            f.write('timestamp,omega_x,omega_y,omega_z,alpha_x,alpha_y,alpha_z\n')
            for r in imu_export:
                f.write(','.join(map(str,[r['calibrated_ns'],*r['gyro_rad_s'],*r['accel_m_s2']]))+'\n')
        (temp/'alignment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        if output.exists():raise FileExistsError(output)
        temp.rename(output)
    except BaseException:
        if temp.exists():shutil.rmtree(temp)
        raise
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--camera-time-source',choices=['android','host-receive','device'],default='android')
    p.add_argument('--max-camera-error-ms',type=float,default=5.)
    p.add_argument('--max-imu-error-ms',type=float,default=3.)
    p.add_argument('--max-drift-ppm',type=float,default=500.)
    p.add_argument('--trim-to-overlap',action='store_true')
    a=p.parse_args(argv)
    if a.camera_time_source == 'android':
        from .android_prepare import prepare
        print(json.dumps(prepare(a.source, a.output), ensure_ascii=False, indent=2))
        return
    report=align_session(a.source,a.output,a.max_camera_error_ms,a.max_imu_error_ms,a.max_drift_ppm,a.trim_to_overlap,a.camera_time_source)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
