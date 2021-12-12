"TEST MLP SONNEN"


import sonnet as snt
from examples import simple_mnist
from sonnet.src import test_utils
import tensorflow as tf


class SimpleMnistTest(test_utils.TestCase):

  def setUp(self):
    self.ENTER_PRIMARY_DEVICE = False  # pylint: disable=invalid-name
    super().setUp()

  def test_train_epoch(self):
    model = snt.Sequential([
        snt.Flatten(),
        snt.Linear(10),
    ]
    )

    optimizer = snt.optimizers.SGD(0.1)

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.random.normal([2, 8, 8, 1]),
         tf.ones([2], dtype=tf.int64))).batch(2).repeat(4)

    for _ in range(3):
      loss = simple_mnist.train_epoch(model, optimizer, dataset)
    self.assertEqual(loss.shape, [])
    self.assertEqual(loss.dtype, tf.float32)

  def test_test_accuracy(self):
    model = snt.Sequential([
        snt.Flatten(),
        snt.Linear(10),
    ]
    )
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.random.normal([2, 8, 8, 1]),
         tf.ones([2], dtype=tf.int64))).batch(2).repeat(4)

    outputs = simple_mnist.test_accuracy(model, dataset)
    self.assertEqual(len(outputs), 2)


if __name__ == "__main__":
  tf.test.main()