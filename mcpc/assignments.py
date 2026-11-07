"""Incremental Boolean enumeration: conflicts persist when constraints relax."""

from .limits import check_memory


class Assignments:
    """Enumerate untried assignments with permanent nogoods and forced values.

    Clauses use signed, one-based variable indices. Watched literals, learned
    clauses and backjumping avoid restarting an exponential search after each
    new ambiguity. Decisions follow variable order, with False before True.
    """

    def __init__(self, variables):
        self.variables = list(variables)
        self.index = {key: i + 1 for i, key in enumerate(self.variables)}
        self.clauses, self.lookup, self.watches, self.units = [], {}, {}, []
        self.forced, self.assumption_levels = {}, 0
        self.value = [-1] * len(self.variables)
        self.reason = [None] * len(self.variables)
        self.level = [0] * len(self.variables)
        self.trail, self.starts = [], []
        self.head = 0
        self.dirty, self.exhausted, self.impossible = True, False, False
        self.blocking = None

    def force(self, values):
        self.forced = dict(values)
        self.dirty, self.exhausted = True, False

    def forbid(self, values):
        self._add([-self.index[key] if value else self.index[key]
                   for key, value in sorted(values.items())])
        self.dirty, self.exhausted = True, False

    def _add(self, clause):
        key = frozenset(clause)
        if key in self.lookup:
            return self.lookup[key]
        if not clause:
            self.impossible = True
            return None
        index = len(self.clauses)
        self.lookup[key] = index
        self.clauses.append(list(clause))
        if len(clause) == 1:
            self.units.append(index)
        else:
            for literal in clause[:2]:
                self.watches.setdefault(literal, []).append(index)
        return index

    def _truth(self, literal):
        value = self.value[abs(literal) - 1]
        return -1 if value < 0 else int(value == (literal > 0))

    def _assign(self, literal, reason=None):
        variable = abs(literal) - 1
        self.value[variable] = int(literal > 0)
        self.reason[variable] = reason
        self.level[variable] = len(self.starts)
        self.trail.append(literal)

    def _backtrack(self, level):
        if len(self.starts) <= level:
            return
        start = self.starts[level]
        for literal in self.trail[start:]:
            self.value[abs(literal) - 1] = -1
            self.reason[abs(literal) - 1] = None
        del self.trail[start:]
        del self.starts[level:]
        self.head = min(self.head, len(self.trail))

    def _propagate(self):
        while self.head < len(self.trail):
            false = -self.trail[self.head]
            self.head += 1
            watched = self.watches.get(false, [])
            position = 0
            while position < len(watched):
                index = watched[position]
                clause = self.clauses[index]
                if clause[0] == false:
                    clause[0], clause[1] = clause[1], clause[0]
                other = clause[0]
                if self._truth(other) == 1:
                    position += 1
                    continue
                replacement = next((k for k in range(2, len(clause))
                                    if self._truth(clause[k]) != 0), None)
                if replacement is not None:
                    clause[1], clause[replacement] = clause[replacement], clause[1]
                    self.watches.setdefault(clause[1], []).append(index)
                    watched[position] = watched[-1]
                    watched.pop()
                elif self._truth(other) == 0:
                    return index
                else:
                    self._assign(other, index)
                    position += 1
        return None

    def _conflict(self, index):
        learned = set(self.clauses[index])
        current = max(self.level[abs(lit) - 1] for lit in learned)
        if current == 0:
            self.impossible = True
            return False
        while sum(self.level[abs(lit) - 1] == current for lit in learned) > 1:
            pivot = next(lit for lit in reversed(self.trail)
                         if -lit in learned and self.level[abs(lit) - 1] == current)
            reason = self.reason[abs(pivot) - 1]
            if reason is None:
                raise RuntimeError("invalid implication graph")
            learned.remove(-pivot)
            learned.update(lit for lit in self.clauses[reason] if lit != pivot)
        clause = sorted(learned, key=lambda lit: (-self.level[abs(lit) - 1], abs(lit)))
        index = self._add(clause)
        if current <= self.assumption_levels:
            return False
        jump = max((self.level[abs(lit) - 1] for lit in clause[1:]), default=0)
        self._backtrack(max(jump, self.assumption_levels))
        self._assign(clause[0], index)
        return True

    def _reset(self):
        self.value[:] = [-1] * len(self.variables)
        self.reason[:] = [None] * len(self.variables)
        self.level[:] = [0] * len(self.variables)
        self.trail.clear()
        self.starts.clear()
        self.head = self.assumption_levels = 0
        for index in self.units:
            literal = self.clauses[index][0]
            if self._truth(literal) == 0:
                self.impossible = True
                return False
            if self._truth(literal) < 0:
                self._assign(literal, index)
        if self._propagate() is not None:
            self.impossible = True
            return False
        for key, value in sorted(self.forced.items()):
            literal = self.index[key] * (1 if value else -1)
            if self._truth(literal) == 0:
                return False
            if self._truth(literal) == 1:
                continue
            self.starts.append(len(self.trail))
            self.assumption_levels += 1
            self._assign(literal)
            conflict = self._propagate()
            if conflict is not None:
                self._conflict(conflict)
                return False
        return True

    def next(self):
        check_memory()
        if self.impossible:
            return None
        if self.dirty:
            self.dirty = False
            if not self._reset():
                self.exhausted = True
                return None
        elif self.exhausted:
            return None
        elif self.blocking is None or not self._conflict(self.blocking):
            self.exhausted = True
            return None
        while True:
            conflict = self._propagate()
            if conflict is not None:
                if not self._conflict(conflict):
                    self.exhausted = True
                    return None
                continue
            variable = next((i for i, value in enumerate(self.value) if value < 0), None)
            if variable is None:
                result = dict(zip(self.variables, map(bool, self.value)))
                clause = [-(i + 1) if value else i + 1 for i, value in enumerate(self.value)]
                clause.sort(key=lambda lit: -self.level[abs(lit) - 1])
                self.blocking = self._add(clause)
                return result
            self.starts.append(len(self.trail))
            self._assign(-(variable + 1))
