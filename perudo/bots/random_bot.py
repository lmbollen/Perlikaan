"""
The simplest possible bot: picks a uniformly random legal action.

This demonstrates the absolute minimum needed to write a Perudo bot -- it
never has to know a single rule of the game.
"""

import random


class RandomBot:
    name = "Random Bot"
    count = 0

    def on_game_start(self, info):
        pass

    def play(self, state):
        return random.choice(state.legal_actions)

    def observe(self, event):
        pass

    def on_game_end(self, result):
        pass
