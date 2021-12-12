class MyLinear(snt.Module):

  def __init__(self, output_size, name=None):
    super(MyLinear, self).__init__(name=name)
    self.output_size = output_size


  def _initialize(self, x):
    initial_w = tf.random.normal([x.shape[1], self.output_size])
    self.w = tf.Variable(initial_w, name="w")
    self.b = tf.Variable(tf.zeros([self.output_size]), name="b")

  def __call__(self, x):
    self._initialize(x)
    return tf.matmul(x, self.w) + self.b

  mod = MyLinear(32)
  mod(tf.ones([batch_size, input_size]))
>> > print(repr(mod)
MyLinear(output_size=10)
>>> mod.variables
(<tf.Variable 'my_linear/b:0' shape=(10, 1) ...)>
(<tf.Variable 'my_linear/w:0' shape=(1, 10) ...)>

checkpoint_root = "/tmp/checkpoints"
checkpoint_name = "example"
save_prefix = os.path.join(checkpoint_root, checkpoint_name)
