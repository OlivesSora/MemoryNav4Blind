"""Reference client for integration into the CAMERA sender, not the Orin.

read_camera_clock_ns must read the SAME running clock as camera_timestamp_ns.
Do not replace a camera sensor timestamp with an upload/callback timestamp.
An Android sender should port the two HTTP exchanges using its verified clock.
"""
import requests


class CameraSenderSync:
    def __init__(self, server_url, clock_id, read_camera_clock_ns):
        if not clock_id or not callable(read_camera_clock_ns):
            raise ValueError("a per-boot camera clock ID and its clock reader are required")
        self.url = server_url.rstrip('/')
        self.clock_id = clock_id
        self.read_clock = read_camera_clock_ns
        self.http = requests.Session()

    def synchronize_once(self):
        sent = self.read_clock()
        response = self.http.post(self.url+'/time_sync',
                                  json=dict(clock_id=self.clock_id,client_send_ns=sent),timeout=5)
        received = self.read_clock()
        response.raise_for_status()
        token = response.json()['token']
        result = self.http.post(self.url+'/time_sync/sample',
                                json=dict(token=token,client_receive_ns=received),timeout=5)
        result.raise_for_status()
        return result.json()

    def metadata(self, source_frame_id, sensor_timestamp_ns, exposure_time_ns=None,
                 timestamp_semantics='exposure_start'):
        """Merge this dictionary into the existing /upload_image JSON body."""
        if type(sensor_timestamp_ns) is not int or type(source_frame_id) is not int:
            raise ValueError("sensor timestamps/frame IDs must be integers")
        if timestamp_semantics not in ('exposure_start','exposure_midpoint'):
            raise ValueError("unsupported timestamp semantics")
        if timestamp_semantics=='exposure_start' and type(exposure_time_ns) is not int:
            raise ValueError("exposure_start requires exposure duration from the camera")
        return dict(camera_clock_id=self.clock_id,source_frame_id=source_frame_id,
                    camera_timestamp_ns=sensor_timestamp_ns,
                    exposure_time_ns=exposure_time_ns,timestamp_semantics=timestamp_semantics)

    def close(self):
        self.http.close()
