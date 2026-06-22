# Export the IGS-GA-I (Preferendus) optimisation interface
from igs_ga.optimization.modo import Preferendus
from igs_ga.optimization.optimization import a_fine_aggregator, run_optimization

__all__ = ["Preferendus", "run_optimization", "a_fine_aggregator"]
