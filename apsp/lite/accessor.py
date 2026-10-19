"""Paid-information API. Future/inactive/pruned/terminal scores stay outside actor views."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ActorView:
    j: int
    alive: tuple
    bought_scores: object
    history: tuple


class PaidAccessor:
    def __init__(self, rows, final_scores, J, bank_id=0, active="ir"):
        if active not in ("ir", "pickscore_raw"):
            raise ValueError("unsupported active verifier")
        self.__rows = {(int(r["j"]), (int(r["bank_id"]), int(r["seed_idx"]))): dict(r) for r in rows}
        self.__final = {(int(r["bank_id"]), int(r["seed_idx"])): r[active] for r in final_scores}
        self.J, self.active = J, active
        self.j, self.alive = 0, tuple((bank_id, s) for s in range(8))
        self.__bought, self.__history = {}, []
        self.__last_prune = None

    def continue_to_next_checkpoint(self):
        if self.j >= self.J:
            raise ValueError("already terminal")
        self.j += 1
        self.__bought = {}

    def buy_verify(self):
        if not 1 <= self.j < self.J:
            raise ValueError("VERIFY only at a nonterminal checkpoint")
        # Environment must charge the ledger before calling this API.
        self.__bought = {cid: self.__rows[self.j, cid][self.active] for cid in self.alive}
        self.__history.append((self.j, tuple(self.__bought.items())))

    def prune(self, survivors):
        survivors = tuple(survivors)
        if self.__last_prune == self.j or not self.__bought or len(survivors) != len(self.alive) // 2 or len(self.alive) <= 2:
            raise ValueError("fresh verification and 8->4/4->2 required")
        if len(set(survivors)) != len(survivors) or not set(survivors).issubset(self.alive):
            raise ValueError("invalid survivor ids")
        self.alive = tuple(c for c in self.alive if c in survivors)
        self.__last_prune = self.j
        self.__bought = {c: self.__bought[c] for c in self.alive}
        # History observed before pruning may stay in memory, but pruned candidates
        # are removed from every future actor view.

    def view(self):
        alive = set(self.alive)
        history = tuple((j, tuple((c, v) for c, v in scores if c in alive)) for j, scores in self.__history)
        return ActorView(self.j, self.alive, MappingProxyType(dict(self.__bought)), history)

    def buy_final(self):
        if self.j != self.J:
            raise ValueError("terminal scores unavailable before FINAL")
        return {c: self.__final[c] for c in self.alive}
