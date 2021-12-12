# ============================================================================

from absl import app
from absl import logging
import sonnet as snt
import tensorflow as tf
import tensorflow_datasets as tfds

fn = snt.functional


def main(unused_argv):
  del unused_argv

  with fn.variables():
    net = snt.nets.MLP([1000, 100, 10])

  def loss_fn(images, labels):
    images = snt.flatten(images)
    logits = net(images)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                       logits=logits))
    return loss

  loss_fn = fn.transform(loss_fn)

  def preprocess(images, labels):
    images = tf.image.convert_image_dtype(images, tf.float32)
    return images, labels

  dataset = tfds.load("mnist", split="train", as_supervised=True)
  dataset = dataset.map(preprocess)
  dataset = dataset.cache()
  dataset = dataset.shuffle(batch_size * 8)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch()

  # Как и прежде, мы хотим распаковать наш loss_fn в init и подать заявку.
  optimizer = fn.adam(0.01)

  # Чтобы получить наше начальное состояние, нам нужно извлечь запись из нашего набора данных и передать
  # это к нашей функции инициализации. Мы также обязательно будем использовать `device_put` таким образом, чтобы
  # параметры находились на ускорителе.
  images, labels = next(iter(dataset))
  params = fn.device_put(loss_fn.init(images, labels))
  opt_state = fn.device_put(optimizer.init(params))

  # Наш цикл обучения состоит в том, чтобы повторить 10 эпох набора данных поезда, и
  # используйте sgd после каждого мини-матча, чтобы обновить наши параметры в соответствии с
  # градиентом из нашей функции потерь.
  grad_apply_fn = fn.jit(fn.value_and_grad(loss_fn.apply))
  apply_opt_fn = fn.jit(optimizer.apply)

  for epoch in range(10):
    for images, labels in dataset:
      loss, grads = grad_apply_fn(params, images, labels)
      params, opt_state = apply_opt_fn(opt_state, grads, params)
    logging.info("[Epoch %s] loss=%s", epoch, loss.numpy())

  def accuracy_fn(images, labels):
    images = snt.flatten(images)
    predictions = tf.argmax(net(images), axis=1)
    correct = tf.math.count_nonzero(tf.equal(predictions, labels))
    total = tf.shape(labels)[0]
    return correct, total

  accuracy_fn = fn.transform(accuracy_fn)

  dataset = tfds.load("mnist", split="test", as_supervised=True)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.map(preprocess)

  # Обратите внимание, что, хотя мы все еще разархивируем нашу функцию точности, мы можем игнорировать
  # init_fn здесь, так как у нас уже есть все состояние, необходимое для нашего обучения
  # функция.
  apply_fn = fn.jit(accuracy_fn.apply)

  # Compute top-1 accuracy.
  num_correct = num_total = 0
  for images, labels in dataset:
    correct, total = apply_fn(params, images, labels)
    num_correct += correct
    num_total += total
  accuracy = (int(num_correct) / int(num_total)) * 100
  logging.info("Accuracy %.5f%%", accuracy)

if "__name__" == "__main__":
    app.run(main)
