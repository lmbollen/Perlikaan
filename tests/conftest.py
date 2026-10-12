"""Every test starts without learned memory (see perudo.memory.SharedMemory)."""

import pytest

from perudo.bots.carlos import CarlosBot
from perudo.bots.carlos_dos import CarlosDosBot
from perudo.bots.pokerface import PokerfaceBot
from perudo.bots.the_don import TheDonBot


@pytest.fixture(autouse=True)
def _fresh_memory():
    for cls in (CarlosBot, CarlosDosBot, PokerfaceBot, TheDonBot):
        cls.reset_memory()
    yield
