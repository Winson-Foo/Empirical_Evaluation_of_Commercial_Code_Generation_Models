import tensorflow as tf


class SquareTest(tf.test.TestCase):
    def test_square(self):
        with self.test_session():
            data = [2, 3]
            squared_data = tf.square(data)
            expected_output = [4, 9]
            self.assertAllEqual(squared_data.eval(), expected_output)


if __name__ == '__main__':
    tf.test.main()