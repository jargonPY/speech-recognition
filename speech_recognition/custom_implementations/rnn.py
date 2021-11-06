import tensorflow as tf

"""
  All higher level APIs inherit from tf.Module, this class has some basic utility methods (saving model etc.)
  that are often necessary when interacting with other objects (losses, gradients etc.). Keras Layer inhertis 
  from tf.Module and adds on more functionality (support for training=True, build method etc.).
"""

class RNN(tf.keras.layers.Layer):

  """
    A 'layer' encapsulates a state (weights) and a computation (transformation from input to
    output, a 'call', the layer's forward pass)
  """

  def __init__(self, units, activation=None, initializer="random_normal", **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.initializer = initializer
    self.activation = tf.keras.activations.get(activation)

  def build(self, input_shape):
    # keras provides this as an extra lifecycle step that allows for more flexibility
    # automatically run first time __call__ is called (defined in keras.layers.Layer)
    # add_weight is a helper function that initializes tf.Variable
    self.w_h = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    initializer=self.initializer,
                    trainable=True,
                  )

    self.w_x = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    initializer=self.initializer,
                    trainable=True,
                  )

    self.w_y = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    initializer=self.initializer,
                    trainable=True,
                  )

    self.b_h = self.add_weight(
                    shape=(self.units,),
                    initializer=self.initializer,
                    trainable=True,
                  )

    self.b_y = self.add_weight(
                    shape=(self.units,),
                    initializer=self.initializer,
                    trainable=True,
                  )

    # initialize the hidden layer
    self.h = self.add_weight(
                    shape=(self.units,),
                    initializer=self.initializer
                  )

    # tells Keras that the layer is built and sets self.built=True
    super().build(input_shape)

  def call(self, x, training=None): 
    """
      makes the layer a "callable", can be called like a function while still being an object and maintaining 
      internal state
    """
    # x --> (batch_size, timesteps, features)
    for index in range(x.shape[1]):
      self.step(x[:, index, :])

  def step(self, x): # equivalent to RNNCell
    # x --> (batch, features)
    linear = tf.matmul(self.h, self.w_h) + tf.matmul(x, self.w_x) + self.b_h
    self.h = tf.math.tanh(linear)
    y = self.activation(tf.matmul(self.h, self.w_y) + self.b_y)

    # if self.return_sequences:
    #   return y, self.h
    return y

class CustomModel(tf.keras.Model):

  """
    Abstractly a Model encapsulates variables that can be updated in response to training, and a function 
    that computes on tensors (a forward pass). It also includes other utility methods that are necessary in
    the model development cycle. 

    Define a Model object (class) by passing in inputs and outputs (using functional API), this allows the class
    to build an internal representation of the model (what to compute during the forward pass).

    1. Model object exposes several predefined methods (i.e. fit, evaluate, etc.)
    2. Many of the methods can be overriden to get custom behaviours while maintaing the other predefined methods
    3. Model() --> Model.compile(intialize metric and optimizer and loss objects) --> Model.fit(start training loop)
    4. If init, call are not overriden the CustomModel can be used in functional API

    Metrics are 'callable' objects. The metric is updated by calling the new value with the callable and 
    the result (usually an average of the batch) can be retrieved from utility functions. There is also a clear
    method.
    
    The role of the Optimizers is to update the model's weights based on the loss function. These are classes
    that are used to change weights and learning rate. All optimizations require gradients (thus its computed by
    the Model object), but how to apply these gradients may vary (thus its left to the Optimizer object).
    
    Losses are functions.
  """

  loss_tracker = tf.keras.metrics.Mean(name="loss")
  mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")

  def train_step(self, data):
    # data depends on what is passed to fit
    x, y = data

    with tf.GradientTape() as tape:
      # Model(data) --> prediction, training=True allows for different behaviour in different stages
      y_pred = self(x, training=True)
      # the loss function is configured in compile()
      loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

    # compute gradients
    gradients = tape.gradient(loss, self.trainable_variables)

    # update weights
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    # update state of metrics that were passed to compile()
    #self.compiled_metrics.update_state(y, y_pred)

    # query results from self.metrics to retrieve their current value
    #return {metric.name: metric.result() for metric in self.metrics}

    # compute our own metrics
    CustomModel.loss_tracker.update_state(loss)
    CustomModel.mae_metric.update_state(y, y_pred)
    return {"loss": CustomModel.loss_tracker.result(), "mae": CustomModel.mae_metric.result()}

  def test_step(self, data): # override the model.evaluate() method
    pass

  @property
  def metrics(self):
    # list "Metric" objects here so that reset_states() can be called
    return [CustomModel.loss_tracker, CustomModel.mae_metric]

# Construct and compile an instance of CustomModel
inputs = tf.keras.Input(shape=(32,))
outputs = RNN(32)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])