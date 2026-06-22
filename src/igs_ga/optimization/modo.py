import functools
import inspect
import logging
import warnings
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from igs_ga.optimization.optimization import a_fine_aggregator, run_optimization


class Preferendus:
    """Optimization Engine decorator for simulations following the IMAP method.

    This decorator wraps around a simulation function to automatically optimize decision variables
    by maximizing stakeholder utility across multiple objectives. It uses a Genetic Algorithm (GA)
    to find the optimal decision variables that balance stakeholder preferences.

    Parameters
    ----------
    preference_functions : Dict[int, Dict[str, Callable]]
        A nested dictionary mapping stakeholder IDs to their respective objective preference functions.
        Each preference function should accept a single numeric argument (the objective value) and
        return a numeric preference score (typically 0-1, higher is better).

        Example: {1: {"cost": lambda c: max(0, 1-c/1000), "time": lambda t: max(0, 1-t/24)}

    stakeholder_weights : List[float]
        A list of weights representing the relative importance of each stakeholder.
        These weights will be normalized internally to sum to 1.

        Example: [0.6, 0.4] for a scenario with two stakeholders where the first stakeholder has 60% influence and the second stakeholder has 40%.

    objective_weights : Dict[int, Dict[str, float]], optional
        A nested dictionary mapping stakeholder IDs to their respective objective weights.
        If not provided, objectives are weighted equally for each stakeholder.

        Example: {1: {"cost": 0.7, "time": 0.3}, 2: {"cost": 0.2, "time": 0.8}}

    optimization_options : Dict, optional
        A dictionary specifying Genetic Algorithm parameters:
        - population_size (int): Number of individuals in the GA population (default: 20)
        - generations (int): Number of generations to run the GA (default: 10)
        - crossover_rate (float): Probability of crossover operation (default: 0.7)
        - mutation_rate (float): Probability of mutation operation (default: 0.2)
        - tournament_size (int): Number of individuals competing in tournament selection (default: 3)
        - suppress_warnings (bool): Whether to suppress warnings during optimization (default: True)
        - suppress_logging (bool): Whether to suppress logging below ERROR level during optimization (default: True)

    constraints : List[Callable[[list], bool]], optional
        A list of constraint functions. Each function takes a decision variable list `x` and returns
        True if the solution is feasible, False otherwise. Infeasible solutions receive a large
        penalty in the fitness function.

        Example: [lambda x: x[0] >= 1, lambda x: sum(x) <= 100]

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
    ...     1: {"cost": lambda c: max(0, 1-c/10000), "time": lambda t: max(0, 1-t/48)},
    ...     2: {"cost": lambda c: max(0, 1-c/5000), "time": lambda t: max(0, 1-t/24)}
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
    ):
        self.preference_functions = preference_functions

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
            'population_size': 20,
            'generations': 10,
            'crossover_rate': 0.7,
            'mutation_rate': 0.2,
            'tournament_size': 3,
            # GA-backbone elitism setting (general GA concept, not an IGS-method term):
            'hall_of_fame_size': 10,
            'suppress_warnings': True,
            'suppress_logging': True
        }
        if optimization_options:
            self.optimization_options.update(optimization_options)
        self.constraints = constraints if constraints is not None else []
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
            - 'utility': Overall utility score achieved with the optimal solution
            - 'simulation_results': Complete results from running the simulation with optimal variables

        Notes
        -----
        - The first call to the wrapped function runs a full optimization.
        - The optimization process involves multiple simulation runs to evaluate different solutions.
        - The GA attempts to maximize the global dby balancing all stakeholder preferences.
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
                    # Collect all objectives
                    all_objectives = set()
                    for prefs in self.preference_functions.values():
                        all_objectives.update(prefs.keys())
                    all_objectives = sorted(all_objectives)

                    # Determine number of decision variables
                    initial_x = [1] * 5
                    if args and hasattr(args[0], '__len__'):
                        n_decision_vars = len(args[0])
                        initial_x = args[0]
                    else:
                        for param in sig.parameters.values():
                            if param.default is not inspect.Parameter.empty and hasattr(param.default, '__len__'):
                                n_decision_vars = len(param.default)
                                initial_x = param.default
                                break
                        else:
                            n_decision_vars = 5

                    # Run simulation once to validate output structure
                    simulation_fn(initial_x)
                    n_prefix_items = 2  # env, objects

                    # Create population-wide data structures to store objective scores for a_fine_aggregator
                    population_scores = {}  # Will store scores for each objective across the population
                    population_raw_data = {}  # Will store raw simulation data for debugging
                    population_counter = 0  # Count evaluated individuals

                    # Track the objective weights properly across all stakeholders
                    objective_total_weights = dict.fromkeys(all_objectives, 0.0)

                    # Pre-compute total weights for each objective across all stakeholders
                    for s_idx, s_id in enumerate(sorted(self.preference_functions.keys())):
                        s_weight = self.stakeholder_weights[s_idx]
                        obj_weights = self.objective_weights[s_id]
                        for obj in all_objectives:
                            if obj in self.preference_functions[s_id]:
                                obj_weight = obj_weights.get(obj, 1.0)
                                objective_total_weights[obj] += s_weight * obj_weight

                    # Normalize objective weights to sum to 1
                    total_weight = sum(objective_total_weights.values())
                    norm_objective_weights = {obj: w / total_weight for obj, w in objective_total_weights.items()}

                    # Create list of normalized weights in same order as all_objectives
                    weights_list = [norm_objective_weights[obj] for obj in all_objectives]

                    def fitness_function(x):
                        nonlocal population_counter

                        # Convert x to list if needed
                        if hasattr(x, 'tolist'):
                            x = x.reshape(-1).tolist()
                        elif isinstance(x, np.ndarray):
                            x = x.flatten().tolist()

                        # Check constraints
                        for constraint in self.constraints:
                            if not constraint(x):
                                return float('inf')  # Penalize infeasible solutions

                        # Run simulation
                        sim_results = simulation_fn(x)
                        objective_values = sim_results[n_prefix_items:]

                        # Map objectives to names
                        objective_dict = {}
                        for i, name in enumerate(all_objectives):
                            if i < len(objective_values):
                                objective_dict[name] = objective_values[i]
                            else:
                                raise ValueError(f"Not enough objective values returned. Expected '{name}'")

                        # Store raw data for debugging
                        solution_id = population_counter
                        population_raw_data[solution_id] = {
                            'x': x.copy() if hasattr(x, 'copy') else x[:],
                            'objectives': objective_dict.copy()
                        }

                        # Calculate preference scores for each objective
                        objective_scores = dict.fromkeys(all_objectives, 0.0)

                        # For each stakeholder, calculate preference scores
                        for s_idx, s_id in enumerate(sorted(self.preference_functions.keys())):
                            s_prefs = self.preference_functions[s_id]
                            s_weight = self.stakeholder_weights[s_idx]
                            obj_weights = self.objective_weights[s_id]

                            for obj in all_objectives:
                                if obj in s_prefs:
                                    # Calculate weighted preference score
                                    pref_score = s_prefs[obj](objective_dict[obj])
                                    obj_weight = obj_weights.get(obj, 1.0)
                                    weight = s_weight * obj_weight

                                    # Add to the objective score (weighted by stakeholder)
                                    objective_scores[obj] += weight * pref_score

                        # Now normalize each objective score by dividing by total weight for that objective
                        for obj in all_objectives:
                            if objective_total_weights[obj] > 0:
                                objective_scores[obj] /= objective_total_weights[obj]

                        # Store preference scores for this solution in the population
                        scores_list = [objective_scores[obj] for obj in all_objectives]
                        if solution_id not in population_scores:
                            population_scores[solution_id] = scores_list

                        # Increment population counter
                        population_counter += 1

                        # If we have fewer than 2 solutions, just return a temporary score using weighted sum
                        if len(population_scores) < 2:
                            # Without enough population members, can't use a_fine_aggregator yet
                            # Use weighted sum as a fallback
                            weighted_sum = sum(w * s for w, s in zip(weights_list, scores_list))
                            return -weighted_sum  # Negative for minimization

                        # Use a_fine_aggregator with the entire population
                        try:
                            # Convert population_scores to 2D array:
                            # rows = objectives, columns = population members
                            p_array = np.zeros((len(all_objectives), len(population_scores)))

                            for j, solution_id in enumerate(sorted(population_scores.keys())):
                                for i, _obj in enumerate(all_objectives):
                                    p_array[i, j] = population_scores[solution_id][i]

                            # Get aggregated scores for all solutions
                            # This evaluates the entire population at once!
                            aggregated_scores = a_fine_aggregator(weights_list, p_array)

                            # WHAT HAPPENS NEXT:
                            # 1. We extract just the score for the current solution being evaluated
                            # 2. We return the negative of this score (because we're minimizing)
                            # 3. This score is used by the GA to rank this individual
                            return -aggregated_scores[solution_id % len(aggregated_scores)]

                        except Exception as e:
                            # Fallback to weighted sum
                            weighted_sum = sum(w * s for w, s in zip(weights_list, scores_list))
                            print(f"Aggregation failed: {str(e)}, using weighted sum: {weighted_sum}")
                            return -weighted_sum

                    value_range = self.value_range
                    individual_shape = n_decision_vars
                    options = dict(self.optimization_options)
                    if self.variable_types is not None:
                        options['variable_types'] = self.variable_types
                    if isinstance(value_range, list) and len(value_range) == n_decision_vars:
                        options['per_variable_value_range'] = value_range
                        value_range_for_optimizer = value_range
                    else:
                        value_range_for_optimizer = value_range

                    # Run optimization
                    result = run_optimization(
                        fitness_function=fitness_function,
                        individual_shape=individual_shape,
                        value_range=value_range_for_optimizer,
                        options=options
                    )

                    if result is None:
                        print("Optimization failed")
                        return None

                    best_x = result['best_solution'].reshape(-1).tolist()
                    final_results = simulation_fn(best_x)

                    # Calculate final utility for the best solution for clearer reporting
                    # Re-run the simulation with best solution to get objective values
                    objective_values = final_results[n_prefix_items:]
                    objective_dict = {}
                    for i, name in enumerate(all_objectives):
                        if i < len(objective_values):
                            objective_dict[name] = objective_values[i]

                    # Calculate preference scores for final solution
                    final_scores = []
                    for obj in all_objectives:
                        obj_score = 0
                        obj_weight_total = 0

                        for s_idx, s_id in enumerate(sorted(self.preference_functions.keys())):
                            if obj in self.preference_functions[s_id]:
                                s_weight = self.stakeholder_weights[s_idx]
                                obj_weight = self.objective_weights[s_id].get(obj, 1.0)
                                pref_score = self.preference_functions[s_id][obj](objective_dict[obj])

                                obj_score += s_weight * obj_weight * pref_score
                                obj_weight_total += s_weight * obj_weight

                        if obj_weight_total > 0:
                            final_scores.append(obj_score / obj_weight_total)

                    # Calculate weighted sum for final utility
                    final_utility = sum(w * s for w, s in zip(weights_list, final_scores))

                    print(f"Final objective values: {objective_dict}")
                    print(f"Final preference scores: {final_scores}")
                    print(f"Final weights: {weights_list}")
                    print(f"Final utility (weighted sum): {final_utility}")

                    # Display final population data for verification (optional)
                    print(f"Total solutions evaluated: {population_counter}")

                    return {
                        'x_opt': best_x,
                        'utility': final_utility,  # Use positive calculated utility instead of -best_fitness
                        'simulation_results': final_results
                    }

                finally:
                    # Restore original logging level regardless of success or failure
                    if suppress_logging:
                        root_logger.setLevel(old_level)

        return wrapper
