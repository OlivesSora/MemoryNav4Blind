"""Optional ROS1 serialization integration; run in Basalt/.venv-bag."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from memory_nav.calibration.align_timestamps import align_session
from memory_nav.tests.test_time_alignment import synthetic_session

HAS_ROSBAGS=importlib.util.find_spec('rosbags') is not None


@unittest.skipUnless(HAS_ROSBAGS,'install Basalt/requirements-bag.txt to run bag integration')
class AlignedBagTests(unittest.TestCase):
    def test_aligned_session_round_trips_and_unaligned_session_is_rejected(self):
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import get_typestore,Stores
        script=Path(__file__).parents[1]/'calibration/Basalt/convert_to_bag.py'
        spec=importlib.util.spec_from_file_location('basalt_bag',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);raw=root/'raw';synthetic_session(raw)
            aligned=root/'aligned';align_session(raw,aligned)
            with self.assertRaises(ValueError):
                module.convert(raw,root/'invalid.bag','camera-imu')
            bag=root/'valid.bag';module.convert(aligned,bag,'camera-imu')
            expected={}
            for name,topic in [('frames.jsonl','/cam0/image_raw'),('imu.jsonl','/imu0')]:
                expected[topic]=[json.loads(line)['calibrated_ns'] for line in (aligned/name).read_text().splitlines()]
            store=get_typestore(Stores.ROS1_NOETIC);seen={topic:[] for topic in expected}
            with Reader(bag) as reader:
                for con,t,data in reader.messages():
                    msg=store.deserialize_ros1(data,con.msgtype)
                    self.assertEqual(t,msg.header.stamp.sec*10**9+msg.header.stamp.nanosec)
                    seen[con.topic].append(t)
            self.assertEqual(seen,expected)
            report=json.loads((aligned/'alignment_report.json').read_text())
            report['alignment_id']='wrong';(aligned/'alignment_report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'do not match'):
                module.convert(aligned,root/'wrong.bag','camera-imu')


if __name__=='__main__':unittest.main()
