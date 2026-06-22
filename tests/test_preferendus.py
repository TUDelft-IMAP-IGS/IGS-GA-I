"""Test module for the Preferendus class."""

import random

import numpy as np
import pytest

from igs_ga.optimization.modo import Preferendus


def test_preferendus_initialization():
    """Test the initialization of the Preferendus class."""
    # Define test preference functions
    preference_functions = {
        1: {"cost": lambda c: max(0, 1 - c / 1000), "time": lambda t: max(0, 1 - t / 24)},
        2: {"cost": lambda c: max(0, 1 - c / 500), "time": lambda t: max(0, 1 - t / 12)},
    }

    # Test with minimal required arguments
    modo = Preferendus(
        preference_functions=preference_functions,
        stakeholder_weights=[0.7, 0.3],
        value_range=[(0.1, 5.0), (1, 10)],
    )

    # Check that preference functions were stored correctly
    assert modo.preference_functions == preference_functions

    # Check that weights were normalized
    assert sum(modo.stakeholder_weights) == pytest.approx(1.0)
    assert modo.stakeholder_weights[0] / modo.stakeholder_weights[1] == pytest.approx(
        0.7 / 0.3
    )

    # Check default optimization options
    assert modo.optimization_options.get("population_size") == 20
    assert modo.optimization_options.get("generations") == 10


def test_preferendus_with_options():
    """Test the initialization of Preferendus with custom options."""
    preference_functions = {
        1: {"cost": lambda c: max(0, 1 - c / 1000)},
    }

    objective_weights = {1: {"cost": 1.0}}

    optimization_options = {
        "population_size": 50,
        "generations": 10,
        "mutation_rate": 0.5,
    }

    modo = Preferendus(
        preference_functions=preference_functions,
        stakeholder_weights=[1.0],
        objective_weights=objective_weights,
        optimization_options=optimization_options,
        variable_types=[float, int],
        value_range=[(0.1, 5.0), (1, 10)],
    )

    # Check that options were stored correctly
    assert modo.optimization_options.get("population_size") == 50
    assert modo.optimization_options.get("generations") == 10
    assert modo.optimization_options.get("mutation_rate") == 0.5
    assert modo.variable_types == [float, int]


def test_default_objective_weights_are_normalized():
    """Default objective weights are created and normalized per stakeholder."""
    preference_functions = {
        1: {"cost": lambda c: c, "time": lambda t: t},
    }

    modo = Preferendus(
        preference_functions=preference_functions,
        stakeholder_weights=[1.0],
        value_range=[(0.0, 1.0), (0.0, 1.0)],
    )

    # Two objectives -> equal weights of 0.5 each, summing to 1
    obj_weights = modo.objective_weights[1]
    assert obj_weights == {"cost": 0.5, "time": 0.5}
    assert sum(obj_weights.values()) == pytest.approx(1.0)


def test_invalid_initialization():
    """Test invalid initializations raise appropriate errors."""
    preference_functions = {
        1: {"cost": lambda c: max(0, 1 - c / 1000)},
    }

    # Test missing required argument (stakeholder_weights)
    with pytest.raises(TypeError):
        Preferendus(preference_functions=preference_functions)

    # Test invalid value_range
    with pytest.raises(ValueError):
        Preferendus(
            preference_functions=preference_functions,
            stakeholder_weights=[1.0],
            value_range=None,
        )

    # Test inconsistent variable_types and value_range
    with pytest.raises(ValueError):
        Preferendus(
            preference_functions=preference_functions,
            stakeholder_weights=[1.0],
            variable_types=[float, int, float],  # 3 types
            value_range=[(0.1, 5.0), (1, 10)],  # but only 2 ranges
        )


def test_simulation_decorator():
    """Test that the Preferendus decorator wraps simulation functions correctly."""
    # Seed the RNG so this stochastic GA test is deterministic/reproducible.
    random.seed(7)
    np.random.seed(7)

    preference_functions = {
        1: {"result": lambda r: r / 100},
    }

    @Preferendus(
        preference_functions=preference_functions,
        stakeholder_weights=[1.0],
        variable_types=[float],
        value_range=[(0, 100)],
        # Small GA settings keep this structural test fast.
        optimization_options={"population_size": 8, "generations": 3},
    )
    def simple_simulation(x):
        """Return the input as the simulation objective."""
        return None, None, x[0]

    # Check that the wrapper returns a dictionary with expected keys
    result = simple_simulation([50])
    assert isinstance(result, dict)
    assert "x_opt" in result
    assert "utility" in result
    assert "simulation_results" in result
    # x_opt should respect the declared decision-variable bounds
    assert 0 <= result["x_opt"][0] <= 100


if __name__ == "__main__":
    # Run tests manually if this file is executed directly
    test_preferendus_initialization()
    test_preferendus_with_options()
    test_default_objective_weights_are_normalized()
    test_invalid_initialization()
    test_simulation_decorator()
    print("All tests passed!")