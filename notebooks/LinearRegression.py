import numpy as np
import random
from tqdm import trange

class LinearRegression:
  def __init__(self, lr):
    self.lr = lr
    self.weights = None
    self.bias = None

  def fit(self, X, y, N=1000):
    if len(X) != len(y):
      raise Exception("Inputs X and y not the same length.")
    if self.weights is None:
      self.weights = np.random.rand(X.shape[1])
      self.bias = random.random()
    for _ in (t := trange(N)):
      p = self.predict(X)
      loss = np.mean((p - y) ** 2)
      t.set_description(f"Loss: {loss}, Progress:")
      self.weights -= self.lr * np.sum((self.weights * (2 * (p - y)))) / len(X)
      self.bias -= self.lr * np.sum(2 * self.bias) / len(X)

  def predict(self, X):
    return self.weights * X + self.bias

def format_input(a):
  ret = np.array(a)
  return np.expand_dims(ret, 1)

if __name__ == "__main__":
  X_train = [1, 2, 3, 5, 6, 7, 8]
  y_train = [10, 20, 30, 50, 60, 70, 80]
  X_test = [1, 8, 11]

  X_train = format_input(X_train)
  y_train = format_input(y_train)
  X_test = format_input(X_test)

  model = LinearRegression(lr=0.01)
  model.fit(X_train, y_train)
  p = model.predict(X_test)
  print(p)

