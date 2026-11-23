"""IMAP engine of the IGS-GA, restoring the oth-preferendus (baseline) behaviour on a DEAP (mu + lambda) backbone.

Per generation the whole candidate pool (offspring + hall of fame + population) is scored in one
a-fine aggregation, followed by the baseline rerun filter. The hall of fame is the baseline's
all-time-best archive: every generation's best design is added, nothing is ever removed, and the
whole archive is re-aggregated jointly every generation. Its top members are re-injected into the pool.
"""

import copy
import math
import random
import traceback
from operator import attrgetter
from typing import Callable, Optional, Sequence, Union

import numpy as np
from deap import base, creator, tools

STRICT_THRESHOLD = 70.0
LENIENT_THRESHOLD = 40.0
EQ_TOLERANCE = 1e-4

DEFAULT_OPTIONS = {
    "population_size": 100,
    "generations": 10,
    "crossover_rate": 0.7,
    "mutation_rate": 1.0,
    "tournament_size": 3,
    "hall_of_fame_size": 10,
    "max_stall": 100,
    "strict": None,
    "n_bits": 12,
    "mutation_rate_order": 2,
    "verbose": True,
}


def a_fine_aggregator(
        w: Union[list[float], np.ndarray],
        p: Union[list[Union[float, int]], np.ndarray],
) -> np.ndarray:
    """Aggregate preference scores in affine space by least-squares distance minimisation.

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


def imap_scores(weights: Sequence[float], p: np.ndarray, strict: Optional[bool] = None) -> np.ndarray:
    """Aggregate one pool with the a-fine aggregator, optionally followed by the baseline rerun filter.

    With the filter, alternatives scoring above the threshold (70 if ``strict``, 40 if ``strict`` is
    False) are re-aggregated among themselves and all other alternatives are dismissed with score 0.

    :param weights: one weight per preference row, summing to 1
    :param p: preference scores, rows = preference functions, columns = alternatives
    :param strict: True (threshold 70), False (threshold 40) or None (no rerun filter)
    :return: aggregated preference scores on [0, 100], higher is better
    """
    p = np.asarray(p, dtype=float)
    scores = np.asarray(a_fine_aggregator(weights, p), dtype=float)
    if strict is None:
        return scores
    keep = scores > (STRICT_THRESHOLD if strict else LENIENT_THRESHOLD)
    if keep.sum() > 1:
        rescored = np.zeros_like(scores)
        rescored[keep] = a_fine_aggregator(weights, p[:, keep])
        return rescored
    return scores


def normalize_constraints(constraints) -> list:
    """Convert constraints to ``(kind, func)`` pairs.

    Accepted formats: a callable returning True when feasible, or the baseline tuples
    ``('ineq', f)`` (feasible when ``f(x) <= 0``) and ``('eq', f)`` (feasible when ``f(x) == 0``).
    """
    normalized = []
    for constraint in constraints or []:
        if callable(constraint):
            normalized.append(("bool", constraint))
        elif (
            isinstance(constraint, (tuple, list))
            and len(constraint) == 2
            and constraint[0] in ("ineq", "eq")
            and callable(constraint[1])
        ):
            normalized.append((constraint[0], constraint[1]))
        else:
            raise ValueError(
                "Constraints must be callables returning True when feasible, or tuples "
                "('ineq', f) with f(x) <= 0 feasible, or ('eq', f) with f(x) == 0 feasible."
            )
    return normalized


def constraint_violation(x: list, constraints: list) -> tuple:
    """Return the number of violated constraints and the summed violation magnitude.

    A violated boolean constraint contributes a magnitude of 1.
    """
    count, magnitude = 0, 0.0
    for kind, func in constraints:
        value = func(x)
        if kind == "bool":
            violated, amount = not bool(value), 1.0
        else:
            value = float(np.squeeze(value))
            if kind == "ineq":
                violated, amount = value > 0, max(value, 0.0)
            else:
                violated, amount = abs(value) > EQ_TOLERANCE, abs(value)
        if violated:
            count += 1
            magnitude += amount
    return count, magnitude


def _to_bits(value: float, low: float, high: float, n_bits: int) -> list:
    levels = (1 << n_bits) - 1
    index = 0 if high == low else int(round((value - low) / (high - low) * levels))
    index = min(max(index, 0), levels)
    return [(index >> (n_bits - 1 - b)) & 1 for b in range(n_bits)]


def _from_bits(bits: list, low: float, high: float) -> float:
    index = 0
    for bit in bits:
        index = (index << 1) | bit
    return low + (high - low) * index / ((1 << len(bits)) - 1)


def crossover(parent_1, parent_2, crossover_rate, variable_types, value_range, n_bits):
    """Baseline per-variable crossover.

    Integers are swapped with probability 0.5; floats undergo, with probability ``crossover_rate``,
    a single-point crossover of their bit encodings, which creates new values between the parents.
    """
    child_1, child_2 = list(parent_1), list(parent_2)
    for i, vtype in enumerate(variable_types):
        if vtype is int:
            if random.random() < 0.5:
                child_1[i], child_2[i] = parent_2[i], parent_1[i]
        elif random.random() < crossover_rate:
            low, high = value_range[i]
            bits_1 = _to_bits(parent_1[i], low, high, n_bits)
            bits_2 = _to_bits(parent_2[i], low, high, n_bits)
            point = random.randint(1, n_bits - 1)
            child_1[i] = _from_bits(bits_1[:point] + bits_2[point:], low, high)
            child_2[i] = _from_bits(bits_2[:point] + bits_1[point:], low, high)
    return child_1, child_2


def mutate(individual, rate, variable_types, value_range, n_bits):
    """Baseline mutation with per-gene rate ``rate``: integers are redrawn, float bits are flipped."""
    mutated = list(individual)
    for i, vtype in enumerate(variable_types):
        low, high = value_range[i]
        if vtype is int:
            if random.random() < rate:
                mutated[i] = random.randint(int(low), int(high))
        else:
            bits = _to_bits(mutated[i], low, high, n_bits)
            flipped = [1 - b if random.random() < rate else b for b in bits]
            if flipped != bits:
                mutated[i] = _from_bits(flipped, low, high)
    return mutated


def _normalize_types(variable_types, n_vars):
    if variable_types is None:
        return [float] * n_vars
    if len(variable_types) != n_vars:
        raise ValueError("Length of variable_types must match length of value_range.")
    types = []
    for vtype in variable_types:
        if vtype in (int, "int"):
            types.append(int)
        elif vtype in (float, "float"):
            types.append(float)
        else:
            raise ValueError(f"Unsupported variable type: {vtype}")
    return types


def _repair(values, variable_types, value_range):
    repaired = []
    for value, vtype, (low, high) in zip(values, variable_types, value_range):
        value = min(max(float(value), low), high)
        repaired.append(int(round(value)) if vtype is int else value)
    return repaired


def _random_values(variable_types, value_range):
    return [
        random.randint(int(low), int(high)) if vtype is int else random.uniform(low, high)
        for vtype, (low, high) in zip(variable_types, value_range)
    ]


class _Scorer:
    """Caches constraint checks and preference rows per design, and scores pools in one batch."""

    def __init__(self, evaluate, weights, constraints, cons_handler, strict):
        self.evaluate = evaluate
        self.weights = weights
        self.constraints = constraints
        self.use_magnitude = cons_handler == "CND"
        self.strict = strict
        self.status = {}  # key -> (n_violated, magnitude)
        self.rows = {}  # key -> preference scores, feasible designs only

    def _check(self, key):
        if key in self.status:
            return self.status[key]
        x = list(key)
        try:
            status = constraint_violation(x, self.constraints)
            if status[0] == 0:
                rows = np.asarray(self.evaluate(x), dtype=float).reshape(-1)
                if len(rows) != len(self.weights) or not np.all(np.isfinite(rows)):
                    raise ValueError(f"Expected {len(self.weights)} finite preference scores, got {rows}")
                self.rows[key] = rows
        except Exception:  # noqa: BLE001 - a failing design must not stop the search
            traceback.print_exc()
            status = (math.inf, math.inf)
        self.status[key] = status
        return status

    def score(self, individuals):
        """Set fitness ``(n_violated, violation, -preference_score)`` relative to ``individuals``."""
        keys = [tuple(ind) for ind in individuals]
        unique = list(dict.fromkeys(keys))
        feasible = [key for key in unique if self._check(key)[0] == 0]
        scores = {}
        if feasible:
            p = np.array([self.rows[key] for key in feasible]).T
            scores = dict(zip(feasible, imap_scores(self.weights, p, self.strict)))
        for ind, key in zip(individuals, keys):
            n_violated, magnitude = self.status[key]
            if n_violated == 0:
                ind.fitness.values = (0.0, 0.0, -float(scores[key]))
            else:
                ind.fitness.values = (float(n_violated), magnitude if self.use_magnitude else 0.0, 0.0)


def _make_classes():
    for name in ("FitnessIMAP", "IndividualIMAP"):
        if hasattr(creator, name):
            delattr(creator, name)
    # Lexicographic minimisation: feasibility first, then violation magnitude, then IMAP preference
    creator.create("FitnessIMAP", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("IndividualIMAP", list, fitness=creator.FitnessIMAP)
    return creator.IndividualIMAP


def update_archive(archive: list, pool: list, scorer: _Scorer) -> list:
    """Add the best design of ``pool`` to the all-time-best archive and re-aggregate the whole archive.

    Designs are never removed, so an earlier winner is always re-judged against later ones.

    :return: the archive, best design first
    """
    best = max(pool, key=attrgetter("fitness"))
    candidates = list({tuple(ind): ind for ind in archive + [copy.deepcopy(best)]}.values())
    scorer.score(candidates)
    return sorted(candidates, key=attrgetter("fitness"), reverse=True)


def run_imap_optimization(
    evaluate: Callable[[list], Sequence[float]],
    weights: Sequence[float],
    value_range: list,
    variable_types: Optional[list] = None,
    constraints: Optional[list] = None,
    cons_handler: str = "simple",
    options: Optional[dict] = None,
    start_point: Optional[list] = None,
) -> dict:
    """Run the IMAP inter-generational GA.

    :param evaluate: maps a design ``x`` (list) to its preference scores, one per weight
    :param weights: weight per preference row (w'_k,i), summing to 1
    :param value_range: ``(min, max)`` per decision variable
    :param variable_types: ``int``/``float`` per variable (default: float)
    :param constraints: see :func:`normalize_constraints`
    :param cons_handler: ``'simple'`` (rank infeasible designs by number of violations) or ``'CND'``
        (by number of violations, then violation magnitude)
    :param options: GA options, see ``DEFAULT_OPTIONS``
    :param start_point: initial guess, clipped to the bounds and added to the initial population
    :return: dict with ``best_solution``, ``utility``, ``archive`` and ``generations``
    """
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    if cons_handler not in ("simple", "CND"):
        raise ValueError("cons_handler must be 'simple' or 'CND'.")
    if opts["n_bits"] < 2:
        raise ValueError("n_bits must be at least 2.")
    value_range = [tuple(bounds) for bounds in value_range]
    types = _normalize_types(variable_types, len(value_range))
    mu = max(2, int(opts["population_size"]))
    lambda_ = 2 * mu
    hof_size = int(opts["hall_of_fame_size"] or 0)
    max_stall = int(opts["max_stall"])
    verbose = opts["verbose"]

    individual = _make_classes()
    scorer = _Scorer(evaluate, list(weights), normalize_constraints(constraints), cons_handler, opts["strict"])

    def vary(population, mutation_rate_per_gene):
        offspring = []
        while len(offspring) < lambda_:
            parent_1, parent_2 = random.sample(population, 2)
            for child in crossover(parent_1, parent_2, opts["crossover_rate"], types, value_range, opts["n_bits"]):
                if random.random() < opts["mutation_rate"]:
                    child = mutate(child, mutation_rate_per_gene, types, value_range, opts["n_bits"])
                offspring.append(individual(_repair(child, types, value_range)))
        return offspring[:lambda_]

    population = [individual(_random_values(types, value_range)) for _ in range(mu)]
    if start_point is not None:
        if len(start_point) != len(value_range):
            raise ValueError(
                f"The initial guess has {len(start_point)} values, but value_range defines "
                f"{len(value_range)} decision variables."
            )
        population[0] = individual(_repair(start_point, types, value_range))

    scorer.score(population)
    archive = update_archive([], population, scorer)

    # Self-adaptive per-gene mutation rate (Eiben & Smith), as in the baseline
    gene_count = mu * sum(opts["n_bits"] if vtype is float else 1 for vtype in types)
    tau = 1 / gene_count ** (1 / opts["mutation_rate_order"])
    mutation_rate_per_gene = tau

    if verbose:
        print(f"{'Generation':<12}{'Best score':<12}{'Infeasible':<12}{'Stall':<8}{'Gene mutation rate':<12}")
    best_key, stall, generation = tuple(archive[0]), 0, 0
    for generation in range(1, int(opts["generations"]) + 1):
        mutation_rate_per_gene *= math.exp(tau * random.gauss(0, 1))
        offspring = vary(population, mutation_rate_per_gene)
        elites = [copy.deepcopy(ind) for ind in archive[:hof_size]]
        pool = offspring + elites + population
        scorer.score(pool)
        population = tools.selTournament(pool, mu, tournsize=opts["tournament_size"])
        archive = update_archive(archive, pool, scorer)

        if tuple(archive[0]) == best_key:
            stall += 1
        else:
            best_key, stall = tuple(archive[0]), 0
        if verbose:
            infeasible = sum(ind.fitness.values[0] > 0 for ind in pool)
            print(f"{generation:<12}{-archive[0].fitness.values[2]:<12.2f}{infeasible:<12}{stall:<8}"
                  f"{mutation_rate_per_gene:<12.4g}")
        if stall >= max_stall:
            if verbose:
                print(f"Stopped at generation {generation}: no new best solution for {max_stall} generations.")
            break

    if stall < max_stall:
        print("The iteration is stopped since the max number of generations is reached. "
              "The results might be incorrect! Please be cautious.")
    elif generation < 1.1 * max_stall:
        print("The number of generations is terribly close to the number of max stall iterations. "
              "This suggests a too fast convergence and wrong results.")

    if archive[0].fitness.values[0] > 0:
        raise ValueError(
            "No feasible solutions could be found: every evaluated design violated a constraint "
            "or failed to evaluate."
        )

    return {
        "best_solution": list(archive[0]),
        "utility": -archive[0].fitness.values[2],
        "archive": [list(ind) for ind in archive],
        "generations": generation,
    }
