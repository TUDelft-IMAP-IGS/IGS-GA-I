import functools
import inspect
import logging
import warnings
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from igs_ga.optimization.optimization import (
    normalize_constraints,
    run_imap_optimization,
)

N_PREFIX_ITEMS = 2  # simulation returns (env, objects, *performance_values)


class Preferendus:
    """Optimization Engine decorator for simulations following the IMAP method.

    This decorator wraps around a simulation function to automatically optimise decision variables
    by maximising the aggregated preference score across all stakeholders. It uses a Genetic
    Algorithm (GA) to find the optimal decision variables that balance stakeholder preferences.

    Parameters
    ----------
    preference_functions : Dict[int, Dict[str, Callable]]
        A nested dictionary mapping stakeholder IDs to their respective objective preference functions.
        Each preference function should accept a single numeric argument (the objective value) and
        return a numeric preference score (0-100, higher is better). Every (stakeholder, objective)
        preference function is aggregated as its own row in the a-fine aggregator.
        The simulation must return the objective values in the order in which the objectives first
        appear in this dictionary.

        Example: {1: {"cost": lambda c: max(0, 100-c/10), "time": lambda t: max(0, 100-t/0.24)}

    stakeholder_weights : List[float]
        A list of weights representing the relative importance of each stakeholder, in the same
        order as the stakeholders in `preference_functions`.
        These weights will be normalized internally to sum to 1.

        Example: [0.6, 0.4] for a scenario with two stakeholders where the first stakeholder has 60% influence and the second stakeholder has 40%.

    objective_weights : Dict[int, Dict[str, float]], optional
        A nested dictionary mapping stakeholder IDs to their respective objective weights.
        If not provided, objectives are weighted equally for each stakeholder.

        Example: {1: {"cost": 0.7, "time": 0.3}, 2: {"cost": 0.2, "time": 0.8}}

    optimization_options : Dict, optional
        A dictionary specifying Genetic Algorithm parameters:
        - population_size (int): Number of individuals in the GA population (default: 100)
        - generations (int): Maximum number of generations to run the GA (default: 10)
        - crossover_rate (float): Probability that a float variable is crossed over (default: 0.7)
        - mutation_rate (float): Probability that a child is mutated; the per-gene mutation rate
          is self-adaptive (default: 1.0)
        - tournament_size (int): Number of individuals competing in tournament selection (default: 3)
        - hall_of_fame_size (int): Number of all-time-best designs re-injected into every generation;
          the hall of fame itself keeps every generation's best design (default: 10)
        - max_stall (int): Stop after this many generations without a new best solution (default: 100)
        - strict (bool or None): Rerun filter: True keeps scores above 70, False above 40, and
          re-scores them among themselves; None disables the filter (default: None)
        - suppress_warnings (bool): Whether to suppress warnings during optimization (default: True)
        - suppress_logging (bool): Whether to suppress logging below ERROR level during optimization (default: True)

    constraints : List, optional
        A list of constraints. Each is either a function that takes the decision variable list `x`
        and returns True if the solution is feasible, or a tuple ('ineq', f) with f(x) <= 0
        feasible, or ('eq', f) with f(x) == 0 feasible.

        Example: [lambda x: x[0] >= 1, ('ineq', lambda x: sum(x) - 100)]

    cons_handler : str, optional
        'simple' (default) ranks infeasible solutions by the number of violated constraints;
        'CND' (Coello non-dominance) ranks them by that number, then by the violation magnitude.

    variable_types : List[Union[type, str]], optional
        A list specifying the type (`int` or `float`) for each decision variable. This allows
        mixed-type optimization where some variables are continuous and others are discrete.

        Example: [float, int, float, int] for a 4-variable problem with alternating types.

    value_range : List[tuple], required
        A list of `(min, max)` tuples specifying bounds for each decision variable individually.
        The length must match the number of decision variables.

        Example: [(0.1, 5.0), (1, 10), (0.0, 1.0), (0, 100)] for a 4-variable problem.

    Returns
    -------
    Callable
        A decorated function that optimizes the simulation and returns results as a dictionary.

    Examples
    --------
    >>> stakeholder_preferences = {
    ...     1: {"cost": lambda c: max(0, 100-c/100), "time": lambda t: max(0, 100-t/0.48)},
    ...     2: {"cost": lambda c: max(0, 100-c/50), "time": lambda t: max(0, 100-t/0.24)}
    ... }
    >>> @Preferendus(
    ...     preference_functions=stakeholder_preferences,
    ...     stakeholder_weights=[0.7, 0.3],
    ...     variable_types=[float, int, float],
    ...     value_range=[(0.1, 5.0), (1, 10), (0.5, 2.0)]
    ... )
    >>> def run_simulation(x):
    ...     # Simulation logic here
    ...     return env, objects, cost, time
    ...
    >>> result = run_simulation([1.0, 5, 1.0])  # Initial guess or can be omitted
    >>> print(f"Optimal variables: {result['x_opt']}, Utility: {result['utility']}")

    """

    def __init__(
            self,
            preference_functions: Dict[int, Dict[str, Callable]],
            stakeholder_weights: List[float],
            objective_weights: Dict[int, Dict[str, float]] = None,
            optimization_options: Dict = None,
            constraints: List[Callable[[list], bool]] = None,
            variable_types: Optional[List[Union[type, str]]] = None,
            value_range: List[tuple] = None,
            cons_handler: str = "simple",
    ):
        self.preference_functions = preference_functions

        if len(stakeholder_weights) != len(preference_functions):
            raise ValueError(
                f"{len(stakeholder_weights)} stakeholder weights given for "
                f"{len(preference_functions)} stakeholders in preference_functions."
            )

        # Normalize stakeholder weights to sum to 1
        total_weight = sum(stakeholder_weights)
        self.stakeholder_weights = [w / total_weight for w in stakeholder_weights]

        # Initialize objective weights (default: equal weights)
        self.objective_weights = {}
        if objective_weights is None:
            for stakeholder_id in self.preference_functions:
                objectives = list(self.preference_functions[stakeholder_id].keys())
                weight_per_obj = 1.0 / len(objectives)
                self.objective_weights[stakeholder_id] = dict.fromkeys(objectives, weight_per_obj)
        else:
            for stakeholder_id, weights in objective_weights.items():
                total = sum(weights.values())
                self.objective_weights[stakeholder_id] = {
                    obj: w / total for obj, w in weights.items()
                }

        # Default optimization options
        self.optimization_options = {
            'population_size': 100,
            'generations': 10,
            'crossover_rate': 0.7,
            'mutation_rate': 1.0,
            'tournament_size': 3,
            # Generation bests are never removed from the hall of fame; this many are re-injected each generation
            'hall_of_fame_size': 10,
            'max_stall': 100,
            'strict': None,
            'suppress_warnings': True,
            'suppress_logging': True
        }
        if optimization_options:
            self.optimization_options.update(optimization_options)
        self.constraints = constraints if constraints is not None else []
        normalize_constraints(self.constraints)
        if cons_handler not in ("simple", "CND"):
            raise ValueError("cons_handler must be 'simple' or 'CND'.")
        self.cons_handler = cons_handler
        self.variable_types = variable_types
        if value_range is None:
            raise ValueError("value_range must be provided as a list of tuples.")
        if self.variable_types and len(value_range) != len(variable_types):
            raise ValueError("Length of value_range must match length of variable_types.")
        self.value_range = value_range

    def __call__(self, simulation_fn: Callable):
        """Wraps the simulation function to perform optimization.

        Parameters
        ----------
        simulation_fn : Callable
            The simulation function to optimize. It must accept a list of decision variables `x`
            and return a tuple containing at least:
            1. The simulation environment
            2. A list of simulation objects
            3. One or more objective values that correspond to the objectives in preference_functions

        Returns
        -------
        Callable
            A wrapped function that, when called, runs the optimization and returns the results.
            The returned function accepts the same parameters as the original simulation function.

            When called, the wrapped function returns a dictionary containing:
            - 'x_opt': List of optimal decision variables found by the GA
            - 'utility': Aggregated IMAP preference score of the optimal solution
            - 'simulation_results': Complete results from running the simulation with optimal variables

        Notes
        -----
        - The first call to the wrapped function runs a full optimization.
        - The optimization process involves multiple simulation runs to evaluate different solutions.
        - The GA maximises the aggregated preference score by balancing all stakeholder preferences.
        - If constraints are provided, infeasible solutions are penalized.
        - Warnings and logging below ERROR level can be automatically suppressed during optimization.

        """
        sig = inspect.signature(simulation_fn)

        @functools.wraps(simulation_fn)
        def wrapper(*args, **kwargs):
            # Configure warning and logging suppression
            suppress_warnings = self.optimization_options.get('suppress_warnings', True)
            suppress_logging = self.optimization_options.get('suppress_logging', True)

            # Use a context manager to properly restore settings after completion
            with warnings.catch_warnings():
                if suppress_warnings:
                    warnings.simplefilter("ignore")

                # Save original logging level and restore it after optimization
                root_logger = logging.getLogger()
                old_level = root_logger.level
                if suppress_logging:
                    root_logger.setLevel(logging.ERROR)

                try:
                    # Objectives in the order in which they first appear; the simulation returns them in this order
                    objectives = list(dict.fromkeys(
                        obj for prefs in self.preference_functions.values() for obj in prefs
                    ))

                    # One aggregation row per (stakeholder, objective) preference function: w'_k,i = w_k * w_k,i
                    rows = []
                    for s_weight, (s_id, prefs) in zip(self.stakeholder_weights, self.preference_functions.items()):
                        obj_weights = self.objective_weights[s_id]
                        for obj, preference_function in prefs.items():
                            weight = s_weight * obj_weights.get(obj, 1.0)
                            if weight > 0:
                                rows.append((obj, preference_function, weight))
                    total_weight = sum(weight for _, _, weight in rows)
                    if total_weight <= 0:
                        raise ValueError("At least one stakeholder objective must have a positive weight.")
                    row_weights = [weight / total_weight for _, _, weight in rows]

                    # Initial guess: first positional argument, else a sequence default of the simulation function
                    initial_x = None
                    if args and hasattr(args[0], '__len__'):
                        initial_x = list(args[0])
                    else:
                        for param in sig.parameters.values():
                            if param.default is not inspect.Parameter.empty and hasattr(param.default, '__len__'):
                                initial_x = list(param.default)
                                break

                    def evaluate(x):
                        sim_results = simulation_fn(x)
                        performance_values = sim_results[N_PREFIX_ITEMS:]
                        if len(performance_values) < len(objectives):
                            raise ValueError(
                                f"The simulation returned {len(performance_values)} objective values, "
                                f"expected {len(objectives)} for {objectives}."
                            )
                        performance = dict(zip(objectives, performance_values))
                        return [float(np.squeeze(pf(performance[obj]))) for obj, pf, _ in rows]

                    result = run_imap_optimization(
                        evaluate=evaluate,
                        weights=row_weights,
                        value_range=self.value_range,
                        variable_types=self.variable_types,
                        constraints=self.constraints,
                        cons_handler=self.cons_handler,
                        options=self.optimization_options,
                        start_point=initial_x,
                    )

                    best_x = result['best_solution']
                    final_results = simulation_fn(best_x)
                    print(f"Final IMAP utility (within the hall of fame): {result['utility']}")

                    return {
                        'x_opt': best_x,
                        'utility': result['utility'],
                        'simulation_results': final_results
                    }

                finally:
                    # Restore original logging level regardless of success or failure
                    if suppress_logging:
                        root_logger.setLevel(old_level)

        return wrapper
