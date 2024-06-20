import tensorflow as tf


class SquareTest:

    def test_square(self):
        """
        Tests the square function in TensorFlow.
        """
        with tf.Session() as sess:
            x = tf.square([2, 3])
            result = sess.run(x)
            expected_result = [4, 9]
            assert result == expected_result, f"Expected: {expected_result}, Actual: {result}"


if __name__ == '__main__':
    SquareTest().test_square()