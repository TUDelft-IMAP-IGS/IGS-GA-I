"""Test module for optimization functions."""

import random

import numpy as np
import pytest

from igs_ga.optimization.optimization import a_fine_aggregator, run_optimization


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


if __name__ == "__main__":
    # Run tests manually if this file is executed directly
    test_a_fine_aggregator()
    test_a_fine_aggregator_validation()
    test_run_optimization_simple()
    test_optimization_with_mixed_types()
    print("All tests passed!")
