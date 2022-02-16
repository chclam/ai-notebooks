#!/usr/bin/env python3
from go import Go

def parse_move(s):
  a = s.split(",")
  a = [int(s.strip()) for s in a]
  if len(a) != 2:
    raise ValueError("Invalid input: {}.".format(s))
  for v in a:
    if not (v >= 0 and v <= 10):
      raise ValueError("Invalid input: {}".format(s))
  return a[0], a[1]
  
if __name__ == "__main__":
  game = Go()

  while 1:
    try:
      move = input(f"{game.get_current_turn()} player to move: ")
      x, y = parse_move(move)
      game.move(x, y)
      print(game.board)
    except Exception as e:
      print(e)
  
