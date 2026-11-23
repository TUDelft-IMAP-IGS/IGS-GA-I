"""IGS-GA-I: the foundational Inter-Generational Solver (Genetic Algorithm backbone).

This package is the GA-I instantiation of the IMAP (Integrative Maximisation of
Aggregated Preferences) Inter-Generational Solver (IGS) framework. It integrates IMAP
preference aggregation directly into standard GA survival selection, exposing the
``Preferendus`` decorator as the user-facing interface.
"""

from igs_ga.optimization import Preferendus, a_fine_aggregator

__all__ = ["Preferendus", "a_fine_aggregator"]
