import tensorflow as tf

class SquareTest(tf.test.TestCase):
    def test_square(self):
        with self.test_session() as sess:
            x = tf.square([2, 3])
            self.assertAllEqual(sess.run(x), [4, 9])

if __name__ == '__main__':
    tf.test.main()