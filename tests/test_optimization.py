"""Tests for the IGS-GA optimisation engine (IMAP aggregation and GA)."""

import random

import numpy as np
import pytest

import igs_ga.optimization.optimization as imap
from igs_ga.optimization.modo import Preferendus
from igs_ga.optimization.optimization import (
    a_fine_aggregator,
    constraint_violation,
    crossover,
    imap_scores,
    mutate,
    normalize_constraints,
    run_imap_optimization,
)


@pytest.fixture(autouse=True)
def _seed():
    random.seed(11)
    np.random.seed(11)


@pytest.fixture
def aggregator_spy(monkeypatch):
    """Record the weights and matrix shape of every a_fine_aggregator call."""
    calls = []
    real = imap.a_fine_aggregator

    def spy(w, p):
        calls.append((list(np.round(w, 6)), np.asarray(p).shape))
        return real(w, p)

    monkeypatch.setattr(imap, "a_fine_aggregator", spy)
    return calls


def _lin(v, lo, hi):
    return float(np.clip(100 * (v - lo) / (hi - lo), 0, 100))


TWO_ACTORS = {
    1: {"cost": lambda c: 100 - _lin(c, 150, 700), "time": lambda t: _lin(t, 10, 30)},
    2: {"cost": lambda c: 100 - _lin(c, 150, 700), "time": lambda t: 100 - _lin(t, 10, 30)},
}
TWO_ACTOR_WEIGHTS = {1: {"cost": 0.9, "time": 0.1}, 2: {"cost": 0.2, "time": 0.8}}


def _perf(x):
    t, a = float(x[0]), float(x[1])
    return 100 + 2000 / t + 80 * (a - 1.5) ** 2 + 40 / a, t * (0.8 + 0.2 * a)


# ----------------------------------------------------------------------------- scoring

def test_a_fine_aggregator():
    """The aggregator returns one score per solution on the [0, 100] scale."""
    weights = [0.2, 0.3, 0.5]
    p = np.array([
        [80, 60, 90, 70],
        [70, 80, 60, 90],
        [60, 70, 80, 60],
    ])
    result = a_fine_aggregator(weights, p)
    assert len(result) == 4
    assert 0 <= min(result) <= max(result) <= 100
    assert len(np.unique(result)) > 1


def test_a_fine_aggregator_validation():
    """Weights must sum to 1 and match the number of objectives."""
    p = np.array([[80, 60], [70, 80], [60, 70]])
    with pytest.raises(AssertionError):
        a_fine_aggregator([0.2, 0.3, 0.6], p)
    with pytest.raises(AssertionError):
        a_fine_aggregator([0.2, 0.8], p)


def test_imap_scores_strict_filter_rescores_top_and_dismisses_rest():
    """Strict rerun: only scores above 70 are re-aggregated, the rest become 0."""
    p = np.array([[0.0, 50.0, 80.0, 100.0]])
    np.testing.assert_allclose(imap_scores([1.0], p, strict=True), [0, 0, 0, 100])


def test_imap_scores_lenient_filter_uses_40_threshold():
    """Non-strict rerun uses the 40 threshold."""
    p = np.array([[0.0, 50.0, 80.0, 100.0]])
    np.testing.assert_allclose(imap_scores([1.0], p, strict=False), [0, 0, 60, 100])


def test_imap_scores_without_filter():
    """strict=None skips the rerun filter."""
    p = np.array([[0.0, 50.0, 80.0, 100.0]])
    np.testing.assert_allclose(imap_scores([1.0], p, strict=None), [0, 50, 80, 100])


def test_filter_is_off_by_default():
    """Both the engine and the Preferendus decorator default to no rerun filter."""
    modo = Preferendus(preference_functions={1: {"o": lambda v: v}}, stakeholder_weights=[1.0],
                       value_range=[(0.0, 1.0)])
    assert imap.DEFAULT_OPTIONS["strict"] is None
    assert modo.optimization_options["strict"] is None


def test_aggregator_gets_one_row_per_stakeholder_objective(aggregator_spy):
    """Each (stakeholder, objective) curve is its own row with weight w_k * w_k,i."""
    @Preferendus(preference_functions=TWO_ACTORS, stakeholder_weights=[0.5, 0.5],
                 objective_weights=TWO_ACTOR_WEIGHTS, variable_types=[float, float],
                 value_range=[(10.0, 30.0), (0.5, 3.0)],
                 optimization_options={"population_size": 10, "generations": 2})
    def sim(x):
        return (None, None, *_perf(x))

    sim([20.0, 1.0])
    assert aggregator_spy[0][0] == [0.45, 0.05, 0.1, 0.4]


def test_zero_weight_rows_are_not_aggregated(aggregator_spy):
    """Preference functions with zero weight are left out of the aggregation."""
    @Preferendus(preference_functions=TWO_ACTORS, stakeholder_weights=[0.5, 0.5],
                 objective_weights={1: {"cost": 1, "time": 0}, 2: {"cost": 0, "time": 1}},
                 variable_types=[float, float], value_range=[(10.0, 30.0), (0.5, 3.0)],
                 optimization_options={"population_size": 10, "generations": 2})
    def sim(x):
        return (None, None, *_perf(x))

    sim([20.0, 1.0])
    assert aggregator_spy[0][0] == [0.5, 0.5]


def test_each_pool_is_scored_in_one_batch_not_against_history(aggregator_spy):
    """The aggregator sees one generation's pool, never the full evaluation history."""
    pop, hof, ngen = 10, 5, 4

    @Preferendus(preference_functions=TWO_ACTORS, stakeholder_weights=[0.5, 0.5],
                 objective_weights=TWO_ACTOR_WEIGHTS, variable_types=[float, float],
                 value_range=[(10.0, 30.0), (0.5, 3.0)],
                 optimization_options={"population_size": pop, "generations": ngen, "hall_of_fame_size": hof})
    def sim(x):
        return (None, None, *_perf(x))

    sim([20.0, 1.0])
    largest_pool = 3 * pop + hof
    assert max(shape[1] for _, shape in aggregator_spy) <= largest_pool
    # pool + archive, each optionally followed by a filter rerun
    assert len(aggregator_spy) <= 4 * (ngen + 1)


def test_each_design_is_simulated_once():
    """Re-scoring uses cached preference scores instead of re-running the simulation."""
    seen = []

    @Preferendus(preference_functions=TWO_ACTORS, stakeholder_weights=[0.5, 0.5],
                 objective_weights=TWO_ACTOR_WEIGHTS, variable_types=[float, float],
                 value_range=[(10.0, 30.0), (0.5, 3.0)],
                 optimization_options={"population_size": 10, "generations": 4})
    def sim(x):
        seen.append(tuple(x))
        return (None, None, *_perf(x))

    sim([20.0, 1.0])
    optimisation_calls = seen[:-1]  # the last call re-runs the optimum for 'simulation_results'
    assert len(optimisation_calls) == len(set(optimisation_calls))


# ----------------------------------------------------------------------------- mapping

def test_objective_values_follow_declared_order():
    """Returned values map to objectives in declaration order, not alphabetically."""
    received = {"time": set(), "cost": set()}
    prefs = {1: {"time": lambda t: received["time"].add(t) or 50.0,
                 "cost": lambda c: received["cost"].add(c) or 50.0}}

    @Preferendus(preference_functions=prefs, stakeholder_weights=[1.0], variable_types=[float],
                 value_range=[(0.0, 1.0)], optimization_options={"population_size": 4, "generations": 1})
    def sim(x):
        return None, None, 7.0, 1_000_000.0

    sim([0.5])
    assert received == {"time": {7.0}, "cost": {1_000_000.0}}


def test_stakeholder_weights_follow_dict_order(aggregator_spy):
    """Stakeholder weights map to stakeholders in dict order, not sorted IDs."""
    prefs = {"contractor": {"o": lambda v: v}, "client": {"o": lambda v: 100 - v}}

    @Preferendus(preference_functions=prefs, stakeholder_weights=[0.8, 0.2], variable_types=[float],
                 value_range=[(0.0, 100.0)], optimization_options={"population_size": 4, "generations": 1})
    def sim(x):
        return None, None, x[0]

    sim([1.0])
    assert aggregator_spy[0][0] == [0.8, 0.2]


def test_initial_guess_is_seeded_and_clipped_to_bounds():
    """The initial guess joins the first population, clipped to value_range."""
    seen = []

    @Preferendus(preference_functions={1: {"o": lambda v: v}}, stakeholder_weights=[1.0],
                 variable_types=[float, int], value_range=[(0.0, 100.0), (1, 5)],
                 optimization_options={"population_size": 6, "generations": 1})
    def sim(x):
        seen.append(tuple(x))
        return None, None, x[0]

    sim([150.0, 3])
    assert (100.0, 3) in seen


def test_number_of_variables_comes_from_value_range():
    """Without an initial guess the number of variables follows value_range."""
    @Preferendus(preference_functions={1: {"o": lambda v: v}}, stakeholder_weights=[1.0],
                 variable_types=[float, float], value_range=[(0.0, 1.0), (0.0, 1.0)],
                 optimization_options={"population_size": 4, "generations": 1})
    def sim(x):
        assert len(x) == 2
        return None, None, x[0] + x[1]

    assert len(sim()["x_opt"]) == 2


# ----------------------------------------------------------------------------- constraints

def test_constraint_formats_and_violation_magnitude():
    """Boolean, 'ineq' and 'eq' constraints report count and magnitude of violations."""
    cons = normalize_constraints([lambda x: x[0] < 2, ("ineq", lambda x: x[0] - 1), ("eq", lambda x: x[0] - 3)])
    assert constraint_violation([3.0], cons) == (2, 3.0)
    assert constraint_violation([1.0], cons) == (1, 2.0)


def test_invalid_constraint_format_is_rejected():
    """Unknown constraint formats raise a clear error."""
    with pytest.raises(ValueError):
        normalize_constraints([("between", lambda x: 0)])


@pytest.mark.parametrize("constraint", [lambda x: x[0] >= 50, ("ineq", lambda x: 50 - x[0])])
@pytest.mark.parametrize("handler", ["simple", "CND"])
def test_constraints_are_respected(constraint, handler):
    """Both constraint formats and both handlers drive the search to the feasible region."""
    @Preferendus(preference_functions={1: {"o": lambda v: 100 - v}}, stakeholder_weights=[1.0],
                 variable_types=[float], value_range=[(0.0, 100.0)], constraints=[constraint],
                 cons_handler=handler, optimization_options={"population_size": 20, "generations": 15})
    def sim(x):
        return None, None, x[0]

    x_opt = sim([90.0])["x_opt"]
    assert 50 <= x_opt[0] < 60


def test_no_feasible_solution_raises():
    """An optimisation without any feasible design raises instead of returning a result."""
    @Preferendus(preference_functions={1: {"o": lambda v: v}}, stakeholder_weights=[1.0], variable_types=[float],
                 value_range=[(0.0, 1.0)], constraints=[lambda x: False],
                 optimization_options={"population_size": 4, "generations": 2})
    def sim(x):
        return None, None, x[0]

    with pytest.raises(ValueError, match="No feasible"):
        sim([0.5])


def test_cnd_prefers_smaller_violation_among_infeasible():
    """CND uses the violation magnitude to steer towards a narrow feasible region."""
    result = run_imap_optimization(
        evaluate=lambda x: [100.0 - x[0]], weights=[1.0], value_range=[(0.0, 100.0)],
        variable_types=[float], constraints=[("ineq", lambda x: 99.0 - x[0])], cons_handler="CND",
        options={"population_size": 20, "generations": 30, "verbose": False},
    )
    assert result["best_solution"][0] >= 99.0


# ----------------------------------------------------------------------------- GA mechanics

def test_archive_adds_each_generation_best_and_never_evicts():
    """The hall of fame stores every generation best and keeps earlier ones."""
    individual = imap._make_classes()
    scorer = imap._Scorer(lambda x: [x[0], 100 - x[1]], [0.5, 0.5], [], "simple", True)
    archive, generation_bests = [], []
    for best in ([90.0, 50.0], [20.0, 5.0], [30.0, 6.0], [40.0, 7.0], [50.0, 8.0]):
        pool = [individual(best)] + [individual([best[0] - d, best[1] + d]) for d in (5.0, 10.0)]
        scorer.score(pool)
        archive = imap.update_archive(archive, pool, scorer)
        generation_bests.append(best)
    assert sorted(list(a) for a in archive) == sorted(generation_bests)


def test_bridge_example_finds_lowest_air_draft():
    """Regression: the Bridge example keeps [5, 3] instead of drifting along the x1 = 5 edge."""
    def bridge(x):
        cost = 3 * (4 * x[0] + 7 * x[1])
        time = 100 - 1.2 * (1.7 * x[0] + 1.9 * x[1])
        return [min(100.0, max(0.0, (225 - cost) / 145 * 100)), min(100.0, max(0.0, (120 - time) / 60 * 100))]

    air_drafts = []
    for seed in range(1, 11):
        random.seed(seed)
        np.random.seed(seed)
        result = run_imap_optimization(
            evaluate=bridge, weights=[0.5, 0.5], value_range=[(1.0, 5.0), (3.0, 8.0)],
            variable_types=[float, float], start_point=[1, 3],
            options={"population_size": 100, "generations": 15, "mutation_rate": 0.8,
                     "hall_of_fame_size": 100, "verbose": False},
        )
        air_drafts.append(result["best_solution"][1])
    assert max(air_drafts) < 3.8, air_drafts
def test_archive_best_is_returned_and_scored_jointly(aggregator_spy):
    """The returned solution is the best of the jointly aggregated hall of fame."""
    result = run_imap_optimization(
        evaluate=lambda x: [100 - _lin(_perf(x)[0], 150, 700), _lin(_perf(x)[1], 10, 30)],
        weights=[0.6, 0.4], value_range=[(10.0, 30.0), (0.5, 3.0)], variable_types=[float, float],
        options={"population_size": 10, "generations": 3, "hall_of_fame_size": 4, "verbose": False},
    )
    assert result["best_solution"] == result["archive"][0]
    assert 2 <= len(result["archive"]) <= 4  # one generation best per generation, incl. generation 0
    assert result["utility"] == pytest.approx(100.0)


def test_hall_of_fame_size_zero_still_runs():
    """hall_of_fame_size=0 disables re-injection but still returns a result."""
    result = run_imap_optimization(
        evaluate=lambda x: [x[0]], weights=[1.0], value_range=[(0.0, 100.0)], variable_types=[float],
        options={"population_size": 6, "generations": 2, "hall_of_fame_size": 0, "verbose": False},
    )
    assert 0 <= result["best_solution"][0] <= 100


def test_max_stall_stops_early():
    """The run stops after max_stall generations without a new best solution."""
    result = run_imap_optimization(
        evaluate=lambda x: [100.0 if x[0] == 1 else 0.0], weights=[1.0], value_range=[(0, 1)],
        variable_types=[int], start_point=[1],
        options={"population_size": 6, "generations": 50, "max_stall": 3, "verbose": False},
    )
    assert result["generations"] == 3


def test_float_crossover_creates_new_values_within_bounds():
    """Bit crossover creates new float values; integers are only swapped."""
    children = [crossover([0.0, 1], [10.0, 4], 1.0, [float, int], [(0.0, 10.0), (1, 4)], 12) for _ in range(50)]
    floats = {c[0] for pair in children for c in pair}
    ints = {c[1] for pair in children for c in pair}
    assert any(0.0 < f < 10.0 for f in floats)
    assert all(0.0 <= f <= 10.0 for f in floats)
    assert ints <= {1, 4}


def test_mutation_respects_bounds_and_types():
    """Mutated variables stay in bounds and keep their type."""
    for _ in range(50):
        child = mutate([5.0, 2], 0.5, [float, int], [(0.0, 10.0), (1, 4)], 12)
        assert 0.0 <= child[0] <= 10.0
        assert isinstance(child[1], int) and 1 <= child[1] <= 4


def test_mutation_rate_zero_means_no_mutation_in_offspring():
    """mutation_rate is the probability that a child is mutated at all."""
    seen = []

    def evaluate(x):
        seen.append(tuple(x))
        return [x[0]]

    run_imap_optimization(
        evaluate=evaluate, weights=[1.0], value_range=[(0, 9)], variable_types=[int],
        options={"population_size": 4, "generations": 3, "mutation_rate": 0.0, "verbose": False},
    )
    initial = set(seen[:4])
    assert set(seen) == initial  # int crossover only swaps values, so nothing new can appear


def test_default_variable_type_is_float():
    """Variables are floats when variable_types is omitted."""
    result = run_imap_optimization(
        evaluate=lambda x: [x[0]], weights=[1.0], value_range=[(0.0, 1.0)],
        options={"population_size": 4, "generations": 1, "verbose": False},
    )
    assert all(isinstance(x[0], float) for x in result["archive"])
