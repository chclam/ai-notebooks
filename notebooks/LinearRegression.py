import random
import numpy as np
from tqdm import trange

class LinearRegression:
  def __init__(self, lr):
    self.lr = lr
    self.weights = None
    self.bias = None

  def fit(self, X, y, N=1000):
    if len(X) != len(y):
      raise Exception("Inputs X and y not the same length.")
    self.weights = np.random.rand(X.shape[1])
    self.bias = random.random()
    for _ in (t := trange(N)):
      p = self.predict(X)
      loss = np.mean((p - y) ** 2)
      t.set_description(f"Loss: {loss}, Progress")
      self.weights -= self.lr * (np.sum(X * (2 * (p - y)), axis=0) / len(X))
      self.bias -= self.lr * np.mean(2 * (p - y))

  def predict(self, X):
    return np.expand_dims(np.dot(self.weights, X.T) + self.bias, 1)

def format_input(a):
  ret = np.array(a)
  if len(ret.shape) == 1:
    ret = np.expand_dims(ret, 1)
  assert(len(ret.shape) == 2)
  return ret

if __name__ == "__main__":
  X_train_1 = [1, 2, 3, 4, 5, 6, 7]
  X_train_2 = [[1, 2], [2, 2], [3, 3], [6, 5], [6, 6], [7, 7], [9, 8]]
  X_test_1 = [1, 8, 15, 120]
  X_test_2 = [[8, 9], [10, 7], [1, 3]]
  y_train = [10, 20, 30, 50, 60, 70, 80]

  X_train_1 = format_input(X_train_1)
  X_train_2 = format_input(X_train_2)
  X_test_1 = format_input(X_test_1)
  X_test_2 = format_input(X_test_2)
  y_train = format_input(y_train)

  model = LinearRegression(lr=0.01)
  model.fit(X_train_1, y_train)
  print(model.predict(X_test_1))
  model.fit(X_train_2, y_train)
  print(model.predict(X_test_2))
