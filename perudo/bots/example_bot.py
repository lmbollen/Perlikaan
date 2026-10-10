"""
DIT IS IN PRINCIPE DE ENIGE FILE DIE JE ALS DEELNEMER HOEFT AAN TE PASSEN.

Kopieer dit bestand, geef je bot een nieuwe class-naam en naam, en pas
play() (en optioneel observe()) aan met je eigen strategie.

Je hoeft GEEN Perudo-regels te implementeren: de engine berekent voor je
welke acties legaal zijn (state.legal_actions) en regelt zelf de
beurtvolgorde, Palifico, Calza, eliminaties en geheime dobbelstenen.

Beschikbaar op `state` tijdens jouw beurt:
    state.my_dice          - jouw eigen dobbelstenen, bv. (1, 3, 3, 5, 6)
    state.my_dice_count    - aantal dobbelstenen dat je nog hebt
    state.players          - publieke info over ALLE spelers (dus ook hoeveel
                              spelers er zijn, wie er nog meedoet en met hoeveel
                              dobbelstenen -- ook spelers die al uitgeschakeld
                              zijn staan erin, met alive=False)
    state.current_bid      - het huidige bod (Bid) of None als je opent
    state.palifico         - True/False: is dit een Palifico-ronde
    state.round_number     - het huidige ronde-nummer
    state.history           - alle acties die deze ronde al gedaan zijn, elk
                              met .player (de NAAM van de bot die het deed)
                              en .action (Bid/Dudo/Calza)
    state.legal_actions     - alle acties die je nu mag teruggeven

Je bot heeft een `name`. Je mag die net als hieronder in __init__ zetten
(instance variable), of gewoon als class-attribute (`name = "..."`) zoals
RandomBot dat doet -- beide werken. De naam is je stabiele identiteit: hij
duikt overal op (state.players, state.history, observe()-events), ook in
simulation mode over honderden games heen, dus je kunt tegenstanders
herkennen en van ze leren ook al wisselt elke game de zitpositie.
"""


# ---------------------------------------------------------------------------
# Game types
#
# Bots may not import the framework, so this file carries its own copies of
# the game's types (same names and fields as the engine's). The
# @engine_types decorator on the bot class translates at the border: play()
# and observe() get these local types, and play() hands the engine back its
# own action object from state.legal_actions.
# ---------------------------------------------------------------------------

PACO = 1
NORMAL_FACES = (2, 3, 4, 5, 6)


class _Value:
    """Immutable value: the fields in _FIELDS, equal and hashed by value."""

    _FIELDS = ()

    def __init__(self, *args, **kwargs):
        values = dict(zip(self._FIELDS, args), **kwargs)
        for field in self._FIELDS:
            object.__setattr__(self, field, values[field])

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def _key(self):
        return tuple(getattr(self, f) for f in self._FIELDS)

    def __eq__(self, other):
        return type(other) is type(self) and other._key() == self._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(repr(v) for v in self._key())})"


class Bid(_Value):
    _FIELDS = ("quantity", "face")

    def __repr__(self):
        face = "Paco" if self.face == PACO else str(self.face)
        return f"Bid({self.quantity}x {face})"


class Dudo(_Value):
    pass


class Calza(_Value):
    pass


class ActionRecord(_Value):
    _FIELDS = ("player", "action")


class PublicPlayer(_Value):
    _FIELDS = ("id", "name", "dice_count", "alive")


class Observation(_Value):
    _FIELDS = ("my_id", "my_dice", "my_dice_count", "players", "current_bid", "palifico",
               "round_number", "history", "legal_actions")


class GameStartInfo(_Value):
    _FIELDS = ("players", "my_id")


class GameResult(_Value):
    _FIELDS = ("winner", "standings", "rounds_played")


class BidEvent(_Value):
    _FIELDS = ("player", "bid")


class DudoEvent(_Value):
    _FIELDS = ("player",)


class CalzaEvent(_Value):
    _FIELDS = ("player",)


class RevealEvent(_Value):
    _FIELDS = ("dice", "current_bid", "actual_count", "challenge_type", "challenger", "bidder", "result")

    def dice_by_player(self):
        return dict(self.dice)


class DieLostEvent(_Value):
    _FIELDS = ("player", "dice_count")


class DieGainedEvent(_Value):
    _FIELDS = ("player", "dice_count")


class PlayerEliminatedEvent(_Value):
    _FIELDS = ("player",)


class RoundEndedEvent(_Value):
    _FIELDS = ("next_starting_player", "palifico_next")


class GameEndedEvent(_Value):
    _FIELDS = ("winner", "standings")


_LOCAL_TYPES = {cls.__name__: cls for cls in (
    Bid, Dudo, Calza, ActionRecord, PublicPlayer, Observation, GameStartInfo, GameResult,
    BidEvent, DudoEvent, CalzaEvent, RevealEvent, DieLostEvent, DieGainedEvent,
    PlayerEliminatedEvent, RoundEndedEvent, GameEndedEvent)}


def _to_local(x):
    """An engine object as the local type of the same name (nested values too)."""
    cls = _LOCAL_TYPES.get(type(x).__name__)
    if cls is not None and type(x) is not cls:
        return cls(**{f: _to_local(getattr(x, f)) for f in cls._FIELDS})
    if type(x) is tuple:
        return tuple(_to_local(v) for v in x)
    return x


def _to_engine(state, action):
    """The engine's own object in state.legal_actions for a local action."""
    for legal in state.legal_actions:
        if _to_local(legal) == action:
            return legal
    return action


def engine_types(cls):
    """Class decorator: the bot sees local types, the engine gets its own back."""
    play, observe = cls.play, cls.observe
    game_start, game_end = cls.on_game_start, cls.on_game_end

    def _play(self, state):
        return _to_engine(state, play(self, _to_local(state)))

    def _observe(self, event):
        return observe(self, _to_local(event))

    def _on_game_start(self, info):
        return game_start(self, _to_local(info))

    def _on_game_end(self, result):
        return game_end(self, _to_local(result))

    cls.play, cls.observe = _play, _observe
    cls.on_game_start, cls.on_game_end = _on_game_start, _on_game_end
    return cls


@engine_types
class ExampleBot:
    def __init__(self):
        self.name = "Example Bot"
        self.count = 0

        # Gewone instance variables. In simulation mode blijft dit bot-object
        # bestaan over meerdere games heen, dus je kunt hier dingen over
        # tegenstanders onthouden (bijv. hoe vaak iemand bluft) -- gekoppeld
        # aan hun naam, niet aan hun (elke game willekeurige) zitpositie.
        self.opponent_bid_counts = {}

    def on_game_start(self, info):
        # Optioneel: wordt één keer aangeroepen aan het begin van elk potje.
        pass

    def play(self, state):
        # 1) Kijk naar je eigen dobbelstenen.
        my_dice = state.my_dice  # bv. (1, 2, 2, 4, 6)

        # 2) Hoeveel spelers zitten er in totaal aan tafel, en hoeveel
        #    dobbelstenen zijn er in totaal nog in het spel? Dat heb je
        #    nodig om te bepalen of een bod "hoog" is -- 8 vieren is
        #    doodnormaal bij 6 spelers, maar bijna zeker gebluft bij 2.
        num_players = len(state.players)
        total_dice_in_play = sum(p.dice_count for p in state.players)

        # 3) Kijk wat er deze ronde al gezegd is, en door wie.
        for record in state.history:
            pass  # record.player (naam) en record.action (Bid/Dudo/Calza)

        # 4) Kijk naar het huidige bod en of dit een Palifico-ronde is.
        current_bid = state.current_bid  # None als jij de ronde opent
        is_palifico = state.palifico

        # 5) Splits de legale acties op in categorieën.
        possible_bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        can_say_dudo = any(isinstance(a, Dudo) for a in state.legal_actions)
        can_say_calza = any(isinstance(a, Calza) for a in state.legal_actions)

        # Simpele voorbeeldstrategie: roep Dudo zodra het bod méér dan de
        # helft van alle dobbelstenen op tafel claimt (relatief aan het
        # aantal spelers/dobbelstenen, niet een vast getal).
        if current_bid is not None and can_say_dudo:
            if current_bid.quantity > total_dice_in_play / 2:
                return Dudo()

        # Anders: als je nog mag bieden, doe het goedkoopste legale bod.
        if possible_bids:
            bid = min(possible_bids, key=lambda b: (b.quantity, b.face))
            return bid

        # Geen bids meer mogelijk (zeldzaam) -> val terug op Calza of Dudo.
        if can_say_calza:
            return Calza()
        return Dudo()

    def observe(self, event):
        # Optioneel: wordt aangeroepen voor elke publieke gebeurtenis, ook
        # als jij niet aan zet bent. Zo kun je tegenstanders leren kennen --
        # event.player is steeds hun stabiele naam.
        if isinstance(event, BidEvent):
            self.opponent_bid_counts[event.player] = (
                self.opponent_bid_counts.get(event.player, 0) + 1
            )

    def on_game_end(self, result):
        # Optioneel: wordt één keer aangeroepen als het potje afgelopen is.
        pass
