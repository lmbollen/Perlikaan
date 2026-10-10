"""
The simplest possible bot: picks a uniformly random legal action.

This demonstrates the absolute minimum needed to write a Perudo bot -- it
never has to know a single rule of the game.
"""

import random


class RandomBot:
    name = "Random Bot"
    count = 0
    def play(self, state):
        return random.choice(state.legal_actions)
