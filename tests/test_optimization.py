"""Test module for optimization functions."""

import random

import numpy as np
import pytest
from deap import base, creator, tools

from igs_ga.optimization.optimization import a_fine_aggregator, custom_eaMuPlusLambda, run_optimization


def test_a_fine_aggregator():
    """Test the a_fine_aggregator function for preference aggregation."""
    # Create a simple test case with 3 objectives and 4 solutions
    weights = [0.2, 0.3, 0.5]  # Weights sum to 1

    # Preference scores for each objective across different solutions
    # Each row is an objective, each column is a solution
    p = np.array([
        [80, 60, 90, 70],  # Objective 1 scores
        [70, 80, 60, 90],  # Objective 2 scores
        [60, 70, 80, 60]   # Objective 3 scores
    ])

    # Run the aggregator
    result = a_fine_aggregator(weights, p)

    # Check the output shape
    assert len(result) == 4  # Should return a score for each solution

    # Check the results are normalized to the 0-100 range
    assert 0 <= min(result) <= max(result) <= 100

    # Check that the result is not all equal values
    assert len(np.unique(result)) > 1


def test_a_fine_aggregator_validation():
    """Test the validation in a_fine_aggregator function."""
    # Test with weights that don't sum to 1
    weights = [0.2, 0.3, 0.6]  # Sum is 1.1
    p = np.array([
        [80, 60],
        [70, 80],
        [60, 70]
    ])

    # Should raise an assertion error
    with pytest.raises(AssertionError):
        a_fine_aggregator(weights, p)

    # Test with mismatched dimensions
    weights = [0.2, 0.8]  # Only 2 weights
    p = np.array([
        [80, 60],
        [70, 80],
        [60, 70]
    ])  # But 3 objectives

    with pytest.raises(AssertionError):
        a_fine_aggregator(weights, p)


def test_run_optimization_simple():
    """Test the run_optimization function with a simple fitness function."""
    # Seed the RNG so this stochastic GA test is deterministic/reproducible.
    random.seed(42)
    np.random.seed(42)

    def simple_fitness(x):
        """Minimize the sum of squares."""
        return sum(xi**2 for xi in x),

    # Run optimization with a budget large enough to reliably converge.
    result = run_optimization(
        fitness_function=simple_fitness,
        individual_shape=2,
        value_range=[(-5.0, 5.0), (-5.0, 5.0)],
        options={
            'population_size': 40,
            'generations': 20,
            'variable_types': [float, float]
        }
    )

    # Check the result structure
    assert isinstance(result, dict)
    assert 'best_solution' in result
    assert 'best_fitness' in result
    assert 'final_population' in result
    assert 'logbook' in result

    # Check that the solution is approximately near zero
    assert abs(result['best_solution'][0]) < 1.0
    assert abs(result['best_solution'][1]) < 1.0


def test_optimization_with_mixed_types():
    """Test optimization with mixed variable types (float and int)."""
    # Seed the RNG so this stochastic GA test is deterministic/reproducible.
    random.seed(123)
    np.random.seed(123)

    def mixed_types_fitness(x):
        """Fitness function for mixed types problem."""
        return (x[0]**2 + (x[1] - 5)**2),

    # Run optimization with mixed types
    result = run_optimization(
        fitness_function=mixed_types_fitness,
        individual_shape=2,
        value_range=[(-5.0, 5.0), (0, 10)],
        options={
            'population_size': 50,
            'generations': 20,
            'variable_types': [float, int]
        }
    )

    # Check the result has correct types
    assert isinstance(result['best_solution'][0], (float, np.float32, np.float64))
    assert (result['best_solution'][1]%1) == 0  # Ensure second variable is an integer

    # The optimal solution should be near [0, 5]
    assert abs(result['best_solution'][0]) < 1.0
    # Integer variable should be close to 5.
    assert abs(result['best_solution'][1] - 5) <= 1


def test_custom_eaMuPlusLambda_convergence_uses_cached_hof_fitness():
    """Convergence check should not re-evaluate Hall of Fame members one-by-one."""
    # Use dedicated DEAP classes for this test to avoid collisions with other tests.
    if hasattr(creator, 'FitnessMinDebug'):
        delattr(creator, 'FitnessMinDebug')
    if hasattr(creator, 'IndividualDebug'):
        delattr(creator, 'IndividualDebug')

    creator.create("FitnessMinDebug", base.Fitness, weights=(-1.0,))
    creator.create("IndividualDebug", list, fitness=creator.FitnessMinDebug)

    toolbox = base.Toolbox()
    toolbox.register("clone", lambda ind: creator.IndividualDebug(ind))
    toolbox.register(
        "mate",
        lambda a, b: (creator.IndividualDebug(a), creator.IndividualDebug(b)),
    )
    toolbox.register("mutate", lambda ind: (creator.IndividualDebug(ind),))
    toolbox.register("select", tools.selBest)

    calls = {"count": 0}

    def evaluate(ind):
        calls["count"] += 1
        return (float(sum(ind)),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("map", map)

    population = [creator.IndividualDebug([i]) for i in range(4)]
    halloffame = tools.HallOfFame(maxsize=2, similar=lambda a, b: a == b)

    custom_eaMuPlusLambda(
        population=population,
        toolbox=toolbox,
        mu=4,
        lambda_=4,
        cxpb=1.0,
        mutpb=0.0,
        ngen=1,
        halloffame=halloffame,
        verbose=False,
    )

    # Expected calls without extra Hall-of-Fame re-evaluation during convergence check:
    # initial pop (4) + offspring invalid (4) + candidate pool (offspring 4 + elites 2 + pop 4) = 18.
    assert calls["count"] == 18


def test_custom_eaMuPlusLambda_does_not_stop_on_gen1_spread_tie():
    """Spread-based convergence should not be allowed to stop at generation 1."""
    if hasattr(creator, 'FitnessMinDebug2'):
        delattr(creator, 'FitnessMinDebug2')
    if hasattr(creator, 'IndividualDebug2'):
        delattr(creator, 'IndividualDebug2')

    creator.create("FitnessMinDebug2", base.Fitness, weights=(-1.0,))
    creator.create("IndividualDebug2", list, fitness=creator.FitnessMinDebug2)

    toolbox = base.Toolbox()
    toolbox.register("clone", lambda ind: creator.IndividualDebug2(ind))
    toolbox.register(
        "mate",
        lambda a, b: (creator.IndividualDebug2(a), creator.IndividualDebug2(b)),
    )
    toolbox.register("mutate", lambda ind: (creator.IndividualDebug2(ind),))
    toolbox.register("select", tools.selBest)
    toolbox.register("evaluate", lambda ind: (0.0,))
    toolbox.register("map", map)

    population = [creator.IndividualDebug2([i]) for i in range(4)]
    halloffame = tools.HallOfFame(maxsize=2, similar=lambda a, b: a == b)

    _result, logbook = custom_eaMuPlusLambda(
        population=population,
        toolbox=toolbox,
        mu=4,
        lambda_=4,
        cxpb=1.0,
        mutpb=0.0,
        ngen=2,
        halloffame=halloffame,
        verbose=False,
    )

    # The algorithm should not terminate at generation 1 due only to spread ties.
    assert logbook[-1]["gen"] == 2


if __name__ == "__main__":
    # Run tests manually if this file is executed directly
    test_a_fine_aggregator()
    test_a_fine_aggregator_validation()
    test_run_optimization_simple()
    test_optimization_with_mixed_types()
    print("All tests passed!")
