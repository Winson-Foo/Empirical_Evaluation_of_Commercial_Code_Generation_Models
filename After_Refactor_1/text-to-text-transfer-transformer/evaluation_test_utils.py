from absl.testing import absltest


class BaseMetricsTest(absltest.TestCase):
    """Base class for metrics tests."""

    def assert_dict_close(self, expected, actual, delta=None, places=None):
        """Asserts that two dictionaries are approximately equal."""
        self.assertCountEqual(expected.keys(), actual.keys())
        for key in expected:
            self.assertAlmostEqual(expected[key], actual[key], delta=delta, places=places, msg=f"Failed for key '{key}'")