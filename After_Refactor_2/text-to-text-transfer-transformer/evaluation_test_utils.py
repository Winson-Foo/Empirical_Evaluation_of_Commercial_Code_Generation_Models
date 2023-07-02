from absl.testing import absltest


class BaseMetricsTest(absltest.TestCase):

    def assert_dict_close(self, dict1, dict2, delta=None, places=None):
        """Asserts that two dictionaries are close.

        Args:
            dict1 (dict): The first dictionary.
            dict2 (dict): The second dictionary.
            delta (float): The maximum difference between values.
            places (int): The number of decimal places to consider.

        Raises:
            AssertionError: If the dictionaries are not close.
        """
        self.assertCountEqual(dict1.keys(), dict2.keys())

        for key in dict1:
            try:
                self.assertAlmostEqual(dict1[key], dict2[key], delta=delta, places=places)
            except AssertionError as e:
                raise AssertionError(f"{str(e)} for key '{key}'")