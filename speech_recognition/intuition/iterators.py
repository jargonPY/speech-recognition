import tensorflow as tf
import numpy as np

"""
  - Dataset object is a python iterable
    - think of it as a lazy list of tuples of tensors

  - upon initializing the "op" returns a representation of the data object
    - so when called/initialized it doesn't produce a computation but rather
      produces a generator which computes once iterated upon
    - when map is executed the user defined function is traced and stored in the TF runtime
      and a handle to it is passed

  Iterators:
  An object that can be iterated upon. Will return one element at a time. Must follow
  the iterator protocol (__iter__, __next__).

  Generators are an easy way to create iterators (i.e. dont need to worry about
  state intialization or raising exceptions).
"""

# ----------------------------- Iterators ---------------------------------------
class Iter():

  def __init__(self, max = 0):
    self.max = max

  def __iter__(self):
    # used to initilize the iterator/state and return the iterator object
    self.n = 0
    return self

  def __next__(self):
    # returns the next element and raise exception when all elements are done
    if self.n <= self.max:
      result = self.n
      self.n += 1
      return result
    else:
      raise StopIteration




# ----------------------------- Lazy evaluation ---------------------------------------
class Person:
  def __init__(self):
    self._relatives = None
        
  @property
  def relatives(self):
    if self._relatives is None:
      self._relatives = ... # Get all relatives
    return self._relatives


def lazy_property(fn):
  """
  Decorator that makes a property lazy-evaluated.
  """
  attr_name = "_lazy_" + fn.__name__

  @property
  def _lazy_property(self):
    if not hasattr(self, attr_name):
      setattr(self, attr_name, fn(self))
    return getattr(self, attr_name)
  return _lazy_property

class Person:
  def __init__(self):
    pass
  
  @lazy_property
  def relatives(self):
    # Get all relatives
    relatives = ...
    return relatives





# ----------------------------- tf.data.Dataset ---------------------------------------
def random_generator():
  print("Enter Generator")
  for i in range(1000):
    print("Run loop")
    x = np.random.randint(1,10, size=1)
    if i % 2 == 0:
      yield [x]
    else:
      yield [x, x]

dataset = tf.data.Dataset.from_generator(random_generator, tf.int16)
dataset = dataset.padded_batch(2, padded_shapes=(2, None))
for batch in dataset:
  print("Batch: ", batch)




def square(x):
  print("Square: ", x)
  return x ** 2

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])
dataset = dataset.map(lambda x: square(x))
dataset = dataset.batch(2).map(lambda x: square(x))
for element in dataset:
  print(element)

def our_generator():
  print("Enter Generator")
  for i in range(1000):
    print("Run loop")
    x = np.random.rand(28,28)
    y = np.random.randint(1,10, size=1)
    yield x,y

def single_mapping(x, y):
  print("Single mapping")
  return x, y

def batch_mapping(x, y):
  print("Batch mapping")
  return x, y

dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))
dataset = dataset.map(lambda x, y: single_mapping(x, y))
dataset = dataset.batch(10)
dataset = dataset.map(lambda x, y: batch_mapping(x, y))
for batch in dataset:
  print("Batch")