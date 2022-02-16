import numpy as np

class Go:
  def __init__(self):
    self.board = -1 * np.ones((19, 19))
    self.turn = 1 # 0 = white, 1 = black

  def move(self, x, y):
    if self.board[x][y] > 0: 
      raise Exception(f"Invalid move. {x},{y} is already played.")
    self.board[x][y] = self.turn
    self.turn = (self.turn + 1) % 2

  def get_current_turn(self):
    if self.turn == 0:
      return "WHITE"
    return "BLACK"

