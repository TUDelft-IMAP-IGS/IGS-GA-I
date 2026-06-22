import random
from typing import Union

import numpy as np
from deap import base, creator, tools


def a_fine_aggregator(
        w: Union[list[float], np.ndarray],
        p: Union[list[Union[float, int]], np.ndarray],
) -> np.ndarray:
    """Function for aggregating scores in affine spaces, by means of the least square
    distance minimization.

    :param w: weights of the different objectives
    :param p: 2d-array with the scores of the objectives. n-by-m, where n is the number
        of objectives and m the population size
    :return: ndarray with the aggregated scores
    """
    assert len(w) == len(p), (
        f"The number of weights ({len(w)}) is not equal to the number of objectives "
        f"({len(p)})."
    )

    assert (
            round(sum(w), 4) == 1
    ), f"The sum of the weights ({round(sum(w), 4)}) is not equal to 1."

    # transpose the array to make further calculations easier
    p_transposed = np.array(p).transpose()

    # calculate the standard deviation per criteria. If std == 0, a value << 1 is
    # inserted to prevent divide by zero error
    std = np.std(p_transposed, axis=0)
    std[std == 0] = 1e-6

    # calculate the z-score normalized scores per criteria
    z = (p_transposed - np.mean(p_transposed, axis=0)) / std

    # calculate representative preference scores (P_i^*)
    p_star = np.sum(w * z, axis=1)

    if len(np.unique(p_star, axis=0)) == 1:
        # if there is only one unique member in p_star
        return np.full(len(p_transposed), 50.0, dtype=float)
    else:
        # return min-max normalized results, so everything is on the scale [0-100]
        return 1 * (p_star - min(p_star)) / (max(p_star) - min(p_star)) * 100


def create_individual_default(shape, value_range, options=None):
    """Create an individual with random values within specified bounds for genetic algorithm optimization.

    This function generates an individual (solution candidate) for GA optimization with mixed variable types
    (integers and floats). It supports both uniform bounds for all variables and per-variable bounds.

    Parameters
    ----------
    shape : int, tuple, or list
        Defines the number of decision variables for the individual.
        - If int: Creates an individual with that many decision variables
        - If tuple or list: Uses the first element as the number of variables

    value_range : list of tuples
        List of tuples specifying (min, max) bounds for each decision variable.
        Each tuple corresponds to one decision variable, defining its valid range.
        Example: [(0.5, 5.0), (1, 10), (0.0, 1.0)] for a 3-variable problem.

    options : dict, optional
        Additional options for individual creation:
        - variable_types: List specifying the type (`int` or `float`) for each variable.
          Must match the length of decision variables.
          Example: [float, int, float] for a 3-variable problem.
        - per_variable_value_range: Alternative way to specify per-variable bounds.
          Will be used if provided, otherwise value_range is used.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the generated individual with values within
        the specified bounds and of the specified types.

    Raises
    ------
    ValueError
        If value_range is not a list of tuples matching the number of variables.
        If variable_types length doesn't match the number of variables.
        If an unsupported variable type is specified.

    Notes
    -----
    - For integer variables, random values are generated using randint (inclusive of bounds)
    - For float variables, random values are generated using uniform distribution
    - All variables are generated within their respective bounds
    - The function ensures type correctness for each variable

    Examples
    --------
    >>> # Create an individual with 3 variables: [float, int, float]
    >>> options = {'variable_types': [float, int, float]}
    >>> value_range = [(0.5, 5.0), (1, 10), (0.0, 1.0)]
    >>> ind = create_individual_default(3, value_range, options)
    >>> print(ind)  # Example output: [2.71, 7, 0.35]

    """
    # Only unpack if value_range is a tuple, not a list
    if isinstance(value_range, tuple):
        min_value, max_value = value_range

    # Support both 1D and 2D shapes
    if isinstance(shape, int):
        size = shape
    elif isinstance(shape, tuple) and len(shape) == 1:
        size = shape[0]
    else:
        size = shape

    variable_types = None
    per_variable_value_range = None
    if options:
        if 'variable_types' in options and options['variable_types'] is not None:
            variable_types = options['variable_types']
            assert len(variable_types) == size, "Length of variable_types must match number of variables"
        if 'per_variable_value_range' in options and options['per_variable_value_range'] is not None:
            per_variable_value_range = options['per_variable_value_range']
            assert len(
                per_variable_value_range) == size, "Length of per_variable_value_range must match number of variables"

    if not isinstance(value_range, list) or len(value_range) != size:
        raise ValueError("value_range must be a list of tuples matching the number of variables.")
    individual = []
    for i in range(size):
        vtype = variable_types[i] if variable_types else int
        min_value, max_value = value_range[i]
        if vtype in [int, 'int']:
            val = random.randint(int(min_value), int(max_value))
        elif vtype in [float, 'float']:
            val = random.uniform(float(min_value), float(max_value))
        else:
            raise ValueError(f"Unsupported variable type: {vtype}")
        individual.append(val)
    return np.array(individual)


def crossover_individuals_default(ind1, ind2, options=None):
    """Perform single-point crossover between two individuals for genetic algorithm optimization.

    This function implements a standard single-point crossover operation where genetic material
    is exchanged between two parent individuals at a randomly selected point. It supports both
    1D arrays (vectors) and 2D arrays (matrices) as individuals, applying appropriate crossover
    strategy based on the dimensionality.

    Parameters
    ----------
    ind1 : numpy.ndarray
        First parent individual. Can be a 1D or 2D array.

    ind2 : numpy.ndarray
        Second parent individual. Must have the same shape as ind1.

    options : dict, optional
        Additional options for crossover operation (unused in default implementation
        but maintained for interface consistency).

    Returns
    -------
    tuple
        A tuple of two individuals (numpy.ndarray) after crossover.
        The children inherit genetic material from both parents.

    Notes
    -----
    - For 1D arrays: Selects a random point and swaps all elements after that point
    - For 2D arrays: Selects a random row and swaps all rows after that point
    - Creates deep copies to avoid modifying the original individuals
    - If individuals have fewer than 2 elements/rows, they are returned unchanged

    Examples
    --------
    >>> # 1D array example
    >>> ind1 = np.array([1, 2, 3, 4, 5])
    >>> ind2 = np.array([6, 7, 8, 9, 10])
    >>> child1, child2 = crossover_individuals_default(ind1, ind2)
    >>> # Possible result: child1 = [1, 2, 8, 9, 10], child2 = [6, 7, 3, 4, 5]

    >>> # 2D array example
    >>> ind1 = np.array([[1, 2], [3, 4], [5, 6]])
    >>> ind2 = np.array([[7, 8], [9, 10], [11, 12]])
    >>> child1, child2 = crossover_individuals_default(ind1, ind2)
    >>> # Possible result: child1 = [[1, 2], [9, 10], [11, 12]],
    >>> #                  child2 = [[7, 8], [3, 4], [5, 6]]

    """
    # Support both 1D and 2D individuals
    ind1_cp = ind1.copy()
    ind2_cp = ind2.copy()

    if ind1_cp.ndim == 1:
        size = ind1_cp.shape[0]
        if size < 2:
            return ind1_cp, ind2_cp
        point = random.randint(1, size - 1)
        ind1_cp[point:], ind2_cp[point:] = ind2_cp[point:].copy(), ind1_cp[point:].copy()
    else:
        num_rows = ind1_cp.shape[0]
        if num_rows < 2:
            return ind1_cp, ind2_cp
        crossover_point = random.randint(1, num_rows - 1)
        ind1_cp[crossover_point:], ind2_cp[crossover_point:] = \
            ind2_cp[crossover_point:].copy(), ind1_cp[crossover_point:].copy()

    return ind1_cp, ind2_cp


def mutate_individual_default(individual, mutation_rate, value_range, options=None):
    """Perform mutation on an individual by randomly changing values based on mutation rate.

    This function implements a standard mutation operation for genetic algorithms where each
    variable in an individual has a chance (determined by mutation_rate) to be replaced with
    a new random value within the specified bounds. It supports both integer and floating-point
    variables, and respects the variable types and bounds for each variable.

    Parameters
    ----------
    individual : numpy.ndarray
        The individual to mutate. Typically a 1D array of decision variables.

    mutation_rate : float
        Probability (0.0 to 1.0) that each element in the individual will be mutated.
        A higher value leads to more exploration but less stability.

    value_range : list of tuples
        List of (min, max) tuples specifying the valid range for each variable.
        Each tuple corresponds to one decision variable in the individual.

    options : dict, optional
        Additional options for mutation:
        - variable_types: List specifying the type (int or float) for each variable.
          Must match the length of the individual.
        - per_variable_value_range: Alternative way to specify per-variable bounds.
          Will be used if provided, otherwise value_range is used.

    Returns
    -------
    tuple
        A tuple containing the mutated individual as a numpy.ndarray.
        Returned as tuple to match DEAP's expectation for mutation operators.

    Raises
    ------
    ValueError
        If value_range is not a list of tuples matching the individual length.
        If variable_types length doesn't match the individual length.
        If an unsupported variable type is specified.

    Notes
    -----
    - Creates a deep copy of the individual to avoid modifying the original
    - For integer variables, generates new random integers within bounds
    - For floating-point variables, generates new random floats within bounds
    - Only values selected for mutation (based on mutation_rate) are changed

    Examples
    --------
    >>> # Mutate an individual with mixed variable types
    >>> individual = np.array([2.5, 7, 0.3])
    >>> options = {'variable_types': [float, int, float]}
    >>> value_range = [(0.5, 5.0), (1, 10), (0.0, 1.0)]
    >>> mutated, = mutate_individual_default(individual, 0.3, value_range, options)
    >>> print(mutated)  # Example output: [4.1, 3, 0.3]  (only some values mutated)

    """
    # Create a copy to avoid modifying the original
    mutated = individual.copy()

    if isinstance(value_range, tuple):
        min_value, max_value = value_range

    variable_types = None
    per_variable_value_range = None
    if options:
        if 'variable_types' in options and options['variable_types'] is not None:
            variable_types = options['variable_types']
            assert len(variable_types) == mutated.shape[0], "Length of variable_types must match number of variables"
        if 'per_variable_value_range' in options and options['per_variable_value_range'] is not None:
            per_variable_value_range = options['per_variable_value_range']
            assert len(per_variable_value_range) == mutated.shape[
                0], "Length of per_variable_value_range must match number of variables"

    if not isinstance(value_range, list) or len(value_range) != mutated.shape[0]:
        raise ValueError("value_range must be a list of tuples matching the number of variables.")
    for i in range(mutated.shape[0]):
        if random.random() < mutation_rate:
            vtype = variable_types[i] if variable_types else int
            min_value, max_value = value_range[i]
            if vtype in [int, 'int']:
                mutated[i] = random.randint(int(min_value), int(max_value))
            elif vtype in [float, 'float']:
                mutated[i] = random.uniform(float(min_value), float(max_value))
            else:
                raise ValueError(f"Unsupported variable type: {vtype}")
    return (mutated,)


def repair_individual_default(individual, value_range, repair_rules=None):
    """Repair an individual by ensuring all values are within bounds and of the correct type.

    This function ensures that an individual from a genetic algorithm remains valid by:
    1. Clamping all values to their allowed ranges (min/max)
    2. Converting values to the appropriate types (int or float)
    3. Applying optional custom repair rules

    Parameters
    ----------
    individual : numpy.ndarray
        The individual to repair. Typically a 1D array of decision variables.

    value_range : list of tuples
        List of (min, max) tuples specifying the valid range for each variable.
        Each tuple corresponds to one decision variable in the individual.

    repair_rules : callable or dict, optional
        Can be either:
        - A callable that takes an individual and returns a repaired version
        - A dict with additional repair options:
            - 'variable_types': List of types for each variable
            - 'per_variable_value_range': Per-variable bounds

    Returns
    -------
    numpy.ndarray
        A repaired copy of the individual with all values within bounds and correctly typed.

    Raises
    ------
    ValueError
        If value_range is not a list of tuples matching the individual length.

    Notes
    -----
    - Creates a deep copy of the individual to avoid modifying the original
    - Handles both int and float variable types
    - For integer variables, rounds to the nearest integer
    - Custom repair_rules are applied after basic repairs if provided

    Examples
    --------
    >>> # Repair an individual with both ints and floats
    >>> individual = np.array([-1.5, 12, 0.3])
    >>> repair_rules = {'variable_types': [float, int, float]}
    >>> value_range = [(0.0, 5.0), (1, 10), (0.0, 1.0)]
    >>> repaired = repair_individual_default(individual, value_range, repair_rules)
    >>> print(repaired)  # Example output: [0.0, 10, 0.3]

    >>> # Using a custom repair function
    >>> def my_repair(ind):
    ...     # Ensure the first two variables sum to at most 10
    ...     if ind[0] + ind[1] > 10:
    ...         ind[0] = max(0, 10 - ind[1])
    ...     return ind
    >>> individual = np.array([7.5, 8, 0.3])
    >>> repaired = repair_individual_default(individual, value_range, my_repair)
    >>> print(repaired)  # Example output: [2.0, 8, 0.3]

    """
    # Create a copy to avoid modifying the original
    repaired = individual.copy()

    if isinstance(value_range, tuple):
        min_value, max_value = value_range

    variable_types = None
    per_variable_value_range = None
    if repair_rules and isinstance(repair_rules, dict):
        if 'variable_types' in repair_rules:
            variable_types = repair_rules['variable_types']
        if 'per_variable_value_range' in repair_rules:
            per_variable_value_range = repair_rules['per_variable_value_range']

    if not isinstance(value_range, list) or len(value_range) != repaired.shape[0]:
        raise ValueError("value_range must be a list of tuples matching the number of variables.")
    for i in range(repaired.shape[0]):
        min_value, max_value = value_range[i]
        repaired[i] = max(min(repaired[i], max_value), min_value)
        vtype = variable_types[i] if variable_types else None
        if vtype in [int, 'int']:
            repaired[i] = int(round(repaired[i]))
        elif vtype in [float, 'float']:
            repaired[i] = float(repaired[i])
    if repair_rules and callable(repair_rules):
        repaired = repair_rules(repaired)

    return repaired


def custom_varOr(population, toolbox, lambda_, cxpb, mutpb):
    """Variation of DEAP's varOr with enhanced safety checks for fitness values.

    This function is a safer version of DEAP's standard varOr function, producing offspring
    through crossover, mutation or reproduction (copy). The key enhancement is robust
    handling of fitness attributes to prevent errors when individuals have improperly
    initialized fitness values.

    Parameters
    ----------
    population : list
        A list of individuals to select from

    toolbox : deap.base.Toolbox
        A toolbox containing the evolutionary operators:
        - toolbox.clone: to clone individuals
        - toolbox.mate: for crossover
        - toolbox.mutate: for mutation

    lambda_ : int
        The number of offspring to produce

    cxpb : float
        The probability of applying crossover (0.0 to 1.0)

    mutpb : float
        The probability of applying mutation (0.0 to 1.0)

    Returns
    -------
    list
        A list of lambda_ offspring

    Notes
    -----
    - The sum of cxpb and mutpb should be in [0,1]. If it's below 1, the remaining
      probability is used for reproduction (cloning).
    - Crossover produces two offspring, but only one is returned.
    - If an individual's fitness attribute is not properly initialized or is corrupted,
      this function safely resets it rather than raising an error.
    - This is a key component for making the genetic algorithm more robust against
      edge cases that might occur during optimization.

    See Also
    --------
    deap.algorithms.varOr : The standard DEAP function this is based on

    """
    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:  # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)

            # Safely delete fitness values
            if hasattr(ind1, 'fitness'):
                if isinstance(ind1.fitness, base.Fitness):
                    if hasattr(ind1.fitness, 'values'):
                        del ind1.fitness.values
                else:
                    # If fitness is not a proper instance, create a new one
                    ind1.fitness = creator.FitnessMin()

            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)

            # Safely delete fitness values
            if hasattr(ind, 'fitness'):
                if isinstance(ind.fitness, base.Fitness):
                    if hasattr(ind.fitness, 'values'):
                        del ind.fitness.values
                else:
                    # If fitness is not a proper instance, create a new one
                    ind.fitness = creator.FitnessMin()

            offspring.append(ind)
        else:  # Apply reproduction
            offspring.append(toolbox.clone(random.choice(population)))

    return offspring

def custom_eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None, halloffame=None,
                          verbose=__debug__):
    """Robust implementation of the (μ+λ) evolution algorithm with error handling.

    This function is a safer version of DEAP's eaMuPlusLambda algorithm that includes
    additional error handling and uses the custom_varOr function for generating offspring.
    In the (μ+λ) strategy, μ parents are selected from the population, λ offspring are
    produced by variation, and the next generation is formed by selecting μ individuals
    from both parents and offspring.

    Parameters
    ----------
    population : list
        A list of individuals to evolve

    toolbox : deap.base.Toolbox
        A DEAP toolbox containing the evolution operators:
        - toolbox.map: for parallel or sequential evaluation
        - toolbox.evaluate: for fitness evaluation
        - toolbox.select: for selection of next generation

    mu : int
        The number of individuals to select for the next generation

    lambda_ : int
        The number of offspring to produce at each generation

    cxpb : float
        The probability of mating two individuals (0.0 to 1.0)

    mutpb : float
        The probability of mutating an individual (0.0 to 1.0)

    ngen : int
        The number of generations to evolve

    stats : deap.tools.Statistics, optional
        A statistics object that is updated with the statistics of the
        evolved population

    halloffame : deap.tools.HallOfFame, optional
        A hall of fame object that retains the best individuals across generations.
        NOTE: the Hall of Fame is a general Genetic Algorithm (DEAP) mechanism for
        elitism / best tracking. It is part of the GA backbone and is *not* a foundational
        IGS term; the IGS method itself is the IMAP aggregation (see a_fine_aggregator).
        The Hall of Fame here only carries elite individuals forward — every preference
        ordering it relies on is produced by the IMAP aggregator.

    verbose : bool, optional
        Whether to print the statistics of each generation

    Returns
    -------
    tuple
        A tuple containing the final population and the logbook with statistics

    Notes
    -----
    - This algorithm uses the (μ+λ) evolutionary algorithm strategy
    - At each generation, λ offspring are created through crossover and mutation
    - The next generation is selected from both the current population and offspring
    - The algorithm includes safeguards against fitness calculation errors
    - The implementation uses our custom_varOr function with enhanced error handling

    See Also
    --------
    deap.algorithms.eaMuPlusLambda : The standard DEAP function this is based on
    custom_varOr : The customized offspring generation function used by this algorithm

    """
    logbook = tools.Logbook()
    # Evaluate initial population all at once (relative fitness)
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        if not hasattr(ind, 'fitness') or not isinstance(ind.fitness, base.Fitness):
            ind.fitness = creator.FitnessMin()
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    len_hof = len(halloffame) if halloffame is not None else 0

    for gen in range(1, ngen + 1):
        # Generate offspring
        offspring = custom_varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not hasattr(ind, 'fitness') or not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            if not hasattr(ind, 'fitness') or not isinstance(ind.fitness, base.Fitness):
                ind.fitness = creator.FitnessMin()
            ind.fitness.values = fit

        # Combine offspring and population with elite individuals
        elite_hof = [toolbox.clone(ind) for ind in halloffame[:len_hof]]
        halloffame_candidates = offspring + elite_hof + population

        # Re-evaluate all candidates together (relative)
        fitnesses_hof = list(map(toolbox.evaluate, halloffame_candidates))
        for ind, fit in zip(halloffame_candidates, fitnesses_hof):
            ind.fitness.values = fit

        # Update Hall of Fame
        halloffame.clear()  # start fresh
        halloffame.update(halloffame_candidates)

        # Select next generation from best individuals
        population[:] = toolbox.select(population + offspring + halloffame.items, mu)

        # Re-evaluate sorted HOF (needed for convergence check)
        fitnesses_hof_sorted = list(map(toolbox.evaluate, halloffame))

        percent_diff = abs(fitnesses_hof_sorted[0][0] - fitnesses_hof_sorted[-1][0])
        if percent_diff < 0.1:
            print(f"Stopping early at generation {gen} due to convergence.")
            break
        else:
            print(
                f"Generation {gen}: Difference in relative utility score between best and worst individual is {percent_diff:.2f}% of the GA Hall of Fame.")

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(offspring), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def run_optimization(fitness_function, individual_shape, value_range, options=None, custom_functions=None):
    """Run a genetic algorithm optimization to find decision variables that minimize a fitness function.

    This function configures and executes a robust genetic algorithm optimization process
    using DEAP's evolutionary framework with enhanced error handling. It supports flexible
    individual creation, mixed variable types (integers and floats), custom genetic operators,
    and repairs for constraint handling.

    Parameters
    ----------
    fitness_function : callable
        Function to evaluate individuals (lower is better). Should accept a numpy array or list
        representing decision variables and return a numeric value or tuple with the first
        value being the objective to minimize.

    individual_shape : int or tuple
        Shape of individuals to optimize:
        - If int: Number of decision variables per individual
        - If tuple: Shape of the individual array (only first element is used)

    value_range : list of tuples
        List of (min_value, max_value) tuples defining the bounds for each decision variable.
        Must have the same length as the number of variables in individuals.
        Example: [(0.0, 10.0), (1, 5), (0.0, 1.0)] for a 3-variable problem.

    options : dict, optional
        Dictionary of algorithm parameters:
        - population_size (int): Size of population (default: 50)
        - generations (int): Number of generations to run (default: 20)
        - crossover_rate (float): Probability of crossover (default: 0.5)
        - mutation_rate (float): Probability of mutation (default: 0.2)
        - tournament_size (int): Size of tournament selection (default: 3)
        - repair_rules (callable or dict): Rules for repairing invalid individuals
        - variable_types (list): Types for each decision variable (int or float)
        - per_variable_value_range (list): Alternative way to specify variable bounds

    custom_functions : dict, optional
        Dictionary of custom genetic algorithm functions:
        - create_individual (callable): Function to create individuals
        - crossover (callable): Function for crossover operations
        - mutate (callable): Function for mutation operations
        - repair (callable): Function for repairing invalid individuals

    Returns
    -------
    dict or None
        If successful, a dictionary containing:
        - 'best_solution': The best individual found (numpy.ndarray)
        - 'best_fitness': The fitness value of the best solution
        - 'final_population': The final population of individuals
        - 'logbook': DEAP's logbook with statistics from the evolution

        Returns None if an error occurs during optimization.

    Notes
    -----
    - The function uses mu+lambda evolution strategy with tournament selection
    - All individuals are repaired after crossover and mutation to ensure validity
    - Error handling is robust, with detailed traceback for debugging
    - The fitness function should handle any domain-specific constraints

    Examples
    --------
    >>> # Simple optimization example with mixed types
    >>> def sphere_function(x):
    ...     return sum(xi**2 for xi in x)
    >>>
    >>> options = {
    ...     'population_size': 50,
    ...     'generations': 20,
    ...     'variable_types': [float, int, float]
    ... }
    >>> value_range = [(0.0, 5.0), (1, 10), (0.0, 5.0)]
    >>>
    >>> result = run_optimization(
    ...     fitness_function=sphere_function,
    ...     individual_shape=3,
    ...     value_range=value_range,
    ...     options=options
    ... )
    >>>
    >>> if result:
    ...     print(f"Best solution: {result['best_solution']}")
    ...     print(f"Best fitness: {result['best_fitness']}")

    >>> # Example with constraints through repair rules
    >>> def constraint_repair(ind):
    ...     # Ensure the sum of all variables is at most 10
    ...     if sum(ind) > 10:
    ...         scale = 10 / sum(ind)
    ...         ind = ind * scale
    ...     return ind
    >>>
    >>> options['repair_rules'] = constraint_repair
    >>> result = run_optimization(sphere_function, 3, value_range, options)

    """
    if options is None:
        options = {}

    if custom_functions is None:
        custom_functions = {}

    # Algorithm parameters
    pop_size = options.get('population_size', 50)
    ngen = options.get('generations', 20)
    cxpb = options.get('crossover_rate', 0.5)
    mutpb = options.get('mutation_rate', 0.2)
    tournsize = options.get('tournament_size', 3)
    repair_rules = options.get('repair_rules', None)

    if options.get('hall_of_fame_size') is None:
        hall_of_fame_size = pop_size
    else:
        # Use provided hall of fame size or default to 10
        hall_of_fame_size = options.get('hall_of_fame_size')

    # COMPLETELY REMOVE previous definitions if they exist
    if hasattr(creator, 'FitnessMin'):
        delattr(creator, 'FitnessMin')
    if hasattr(creator, 'Individual'):
        delattr(creator, 'Individual')

    # Create fresh definitions
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    # Create toolbox
    toolbox = base.Toolbox()

    # Register create function (either custom or default)
    create_func = custom_functions.get('create_individual', create_individual_default)

    # Use a direct function that returns a properly formed individual
    def create_individual_wrapper():
        ind = create_func(individual_shape, value_range, options)
        # Flatten to 1D if needed for compatibility
        if isinstance(ind, np.ndarray) and ind.ndim > 1 and ind.shape[0] == 1:
            ind = ind.flatten()
        # Convert to Individual class and initialize fitness
        individual = creator.Individual(ind)
        return individual

    # Register individual creation and population
    toolbox.register("individual", create_individual_wrapper)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the evaluation function with explicit error handling
    def safe_evaluate(individual):
        try:
            # Flatten if needed
            x = individual
            if isinstance(x, np.ndarray) and x.ndim > 1 and x.shape[0] == 1:
                x = x.flatten()
            result = fitness_function(x)
            # Make sure result is a tuple
            if not isinstance(result, tuple):
                result = (result,)
            return result
        except Exception:
            # print(f"Error in fitness function: {e}")
            import traceback
            traceback.print_exc()
            return (float('inf'),)

    toolbox.register("evaluate", safe_evaluate)

    # Register genetic operators (either custom or default)
    crossover_func = custom_functions.get('crossover', crossover_individuals_default)
    mutate_func = custom_functions.get('mutate', mutate_individual_default)
    repair_func = custom_functions.get('repair', repair_individual_default)

    # Create wrapped functions that incorporate repair if needed
    def wrapped_crossover(ind1, ind2):
        ind1, ind2 = crossover_func(ind1, ind2, options)
        ind1 = repair_func(ind1, value_range, repair_rules)
        ind2 = repair_func(ind2, value_range, repair_rules)
        return ind1, ind2

    def wrapped_mutate(ind):
        ind, = mutate_func(ind, mutpb, value_range, options)
        ind = repair_func(ind, value_range, repair_rules)
        return ind,

    def numpy_array_equal(ind1, ind2):
        """Properly compare NumPy arrays for Hall of Fame."""
        return np.array_equal(ind1, ind2)

    # Register the wrapped functions
    toolbox.register("mate", wrapped_crossover)
    toolbox.register("mutate", wrapped_mutate)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)

    # Create initial population
    population = toolbox.population(n=pop_size)

    # Debug information
    print("Created initial population")

    # Explicitly evaluate all individuals
    print("Evaluating initial population...")
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    print("Initial population evaluated")

    # Run the algorithm
    try:
        # Create Hall of Fame to store best individuals.
        # NOTE: Hall of Fame is a general GA-backbone mechanism (DEAP) for elitism /
        # best tracking; it is not part of the IGS (IMAP) method itself.
        if hall_of_fame_size == 0:
            hof = None
        else:
            hof = tools.HallOfFame(maxsize=hall_of_fame_size, similar=numpy_array_equal)

        print("Starting evolutionary algorithm...")
        # Use custom algorithm instead of the default
        result, logbook = custom_eaMuPlusLambda(
            population, toolbox,
            mu=pop_size, lambda_=2 * pop_size,
            cxpb=cxpb, mutpb=mutpb,
            ngen=ngen, verbose=True,
            halloffame=hof
        )

        print("Evolution completed")

        # Get best individual
        if result:
            # best_ind = tools.selBest(result, 1)[0]
            # Get best from Hall of Fame rather than final population
            if hof is None:
                hof = []

            best_ind = hof[0] if len(hof) > 0 else tools.selBest(result, 1)[0]

            # Defensive check for fitness
            if hasattr(best_ind, 'fitness') and hasattr(best_ind.fitness, 'values') and best_ind.fitness.values:
                best_fitness = best_ind.fitness.values[0]
            else:
                print("Warning: Best individual missing fitness values")
                best_fitness = None

            return {
                'best_solution': best_ind,
                'best_fitness': best_fitness,
                'final_population': result,
                'logbook': logbook
            }
        else:
            print("Empty result from optimization")
            return None

    except Exception:
        # print(f"Error in optimization: {e}")
        import traceback
        traceback.print_exc()
        return None
