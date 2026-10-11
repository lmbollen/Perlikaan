"""Every test starts without learned memory (see perudo.memory.SharedMemory)."""

import pytest

from perudo.bots.bluffo_boyyo import BluffoBoyyo
from perudo.bots.carlos import CarlosBot
from perudo.bots.pokerface import PokerfaceBot
from perudo.bots.the_don import TheDonBot


@pytest.fixture(autouse=True)
def _fresh_memory():
    for cls in (BluffoBoyyo, CarlosBot, PokerfaceBot, TheDonBot):
        cls.reset_memory()
    yield
