import unittest

from memory_nav.models import ReferencePoint
from memory_nav.replay.deviation import DeviationMonitor, NavigationState
from memory_nav.replay.matcher import RouteMatcher, project_to_segment


def point(index, east, north, s, heading=90):
    return ReferencePoint(index, s, east, north, 113.0, 23.0, heading)


class MatcherTests(unittest.TestCase):
    def setUp(self):
        self.route = [point(0, 0, 0, 0), point(1, 10, 0, 10), point(2, 20, 0, 20)]

    def test_signed_cross_track_left_and_right(self):
        left = project_to_segment(5, 2, self.route[0], self.route[1])
        right = project_to_segment(5, -2, self.route[0], self.route[1])
        self.assertAlmostEqual(left[2], 2)
        self.assertAlmostEqual(right[2], -2)

    def test_match_progress_is_monotonic(self):
        matcher = RouteMatcher(self.route)
        first = matcher.match(8, 0.5, 90)
        second = matcher.match(7, 0.2, 90)
        self.assertEqual(first.match_quality, "good")
        self.assertGreaterEqual(matcher.last_s_m, first.matched_s_m)
        self.assertLess(second.matched_s_m, first.matched_s_m)

    def test_heading_disambiguates_parallel_return(self):
        route = [point(0, 0, 0, 0, 90), point(1, 10, 0, 10, 90), point(2, 10, 1, 11, 270), point(3, 0, 1, 21, 270)]
        matcher = RouteMatcher(route, max_heading_error_deg=60)
        result = matcher.match(5, 0.5, 90)
        self.assertEqual(result.segment_index, 0)

    def test_route_completion_uses_progress_tolerance(self):
        matcher = RouteMatcher(self.route)
        matcher.match(10, 0, 90)
        matcher.match(19, 0, 90)
        self.assertTrue(matcher.is_complete(2))
        self.assertFalse(matcher.is_complete(0.5))

    def test_initial_match_uses_nearest_segment_regardless_of_offset(self):
        long_route = [point(index, float(index * 10), 0, float(index * 10)) for index in range(11)]
        matcher = RouteMatcher(long_route, initial_search_m=15, lost_distance_m=8)
        result = matcher.match(95, 0, 90)
        self.assertEqual(result.match_quality, "good")
        self.assertAlmostEqual(result.matched_s_m, 95)
        self.assertEqual(matcher.last_s_m, result.matched_s_m)


class DeviationTests(unittest.TestCase):
    def test_hysteresis_and_recovery(self):
        monitor = DeviationMonitor(3, 5, enter_samples=2, recover_samples=2)
        self.assertEqual(monitor.update(4, "good").state, NavigationState.DEGRADED)
        self.assertEqual(monitor.update(4, "good").state, NavigationState.DEVIATED)
        self.assertEqual(monitor.update(1, "good").state, NavigationState.RECOVERING)
        self.assertEqual(monitor.update(1, "good").state, NavigationState.NORMAL)

    def test_lost_is_not_user_deviation(self):
        monitor = DeviationMonitor()
        update = monitor.update(None, "lost")
        self.assertEqual(update.state, NavigationState.LOST)
        self.assertTrue(update.pause_progress)

    def test_completed_state_pauses_progress(self):
        monitor = DeviationMonitor()
        update = monitor.complete()
        self.assertEqual(update.state, NavigationState.COMPLETED)
        self.assertTrue(update.pause_progress)


if __name__ == "__main__":
    unittest.main()
