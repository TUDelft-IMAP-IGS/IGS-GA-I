# IGS-GA-I: Preferendus (IMAP Inter-Generational Solver — GA-I)

**Preference-Based Optimisation — IMAP Inter-Generational Solver (IGS) Framework**

## About

`IGS-GA-I` is the **original IGS-GA algorithm** — the foundational *Inter-Generational
Solver* using a Genetic Algorithm (GA) backbone. It is a user-friendly implementation,
equivalent to the original version linked to all examples in the Open Design Systems
(ODESYS) book, and it integrates **IMAP** preference aggregation directly into standard
GA survival selection.

It is part of the **IMAP-IGS** family of preference-based optimisation solvers built on
the **IMAP (Integrative Maximisation of Aggregated Preferences)** aggregator:

| Repository | Backbone | Role |
|------------|----------|------|
| **IGS-GA-I** (this repo) | Genetic Algorithm | Foundational, user-friendly GA solver; ODESYS-book equivalent |
| IGS-GA-II | NSGA-II genetic loop + IMAP affine aggregation survival | Benchmarking against reference test suites (e.g. DAS-CMOP) |
| IGS-BRKGA | Biased Random-Key GA | Combinatorial / constraint-heavy problems (HVASP, MO-VRPTW) |
| IGS-SI | Swarm Intelligence (Ant Colony Optimisation) | Graph-based and routing optimisation |

## What is IMAP?

The IMAP preference function `P_{k,i}` maps an actor-independent performance function
vector `F` to a preference score in `[0, 100]` for actor `k`, acting component-wise on
the `i`-th performance function. The relatively worst and best performance levels are
assigned preference scores of `0` and `100` respectively — an arbitrary reference
scaling, since preferences possess only relative meaning within a one-dimensional affine
space.

In this solver, IMAP is realised through the **a-fine-aggregator** (`a_fine_aggregator`),
which for a pool of alternatives:

1. Transposes the preference matrix so rows are individuals and columns are criteria.
2. Z-score normalises each criterion (`z = (x − μ) / s`; if `s = 0`, uses `s = 1e-6`).
3. Computes the representative preference score `P* = Σ wᵢ · zᵢ` (with `Σ wᵢ = 1`).
4. Min-max scales `P*` to `[0, 100]` (returns `50` for all if every `P*` is identical).

The GA then maximises these aggregated preference scores across generations, so the
endpoints `0` and `100` represent the relative minimum and maximum within the alternative
set rather than absolute preference origins.

## Key features

- **Stakeholder-centric**: each stakeholder defines custom preference functions per
  objective, per-objective (local) weights, and an overall (global) weight.
- **IMAP affine aggregation** (the IGS method) integrated into GA survival selection.
- **Genetic Algorithm backbone**: DEAP `μ+λ` strategy with crossover, mutation, and
  Hall of Fame elitism — general GA mechanisms, not part of the IGS method itself.
- **Mixed variable support**: continuous (`float`) and discrete (`int`) decision variables.
- **Constraint handling**: infeasible solutions are penalised.
- **Decorator-based API**: wrap an existing simulation/performance function.

> **Terminology note (IMAP-IGS).** The *IGS method* is the **IMAP** preference
> aggregation (the `a_fine_aggregator`) integrated into GA survival selection. The
> supporting mechanisms — **Hall of Fame**, elitism, crossover, and mutation — are
> general **Genetic Algorithm** concepts inherited from the GA backbone and are *not*
> foundational IGS terms. This matches the IMAP-IGS terminology, which annotates Hall of
> Fame as a "general concept in GA's". Here it serves purely as the GA-backbone elitism /
> best-tracking device that carries elite individuals across generations; the preference
> ordering it relies on is always produced by the IMAP aggregator.

## Installation

```bash
# Development install
git clone https://github.com/TUDelft-IMAP-IGS/IGS-GA-I.git
cd IGS-GA-I
pip install -e .
```

## Quick start

```python
from igs_ga.optimization import Preferendus

# Define stakeholder preference functions (preference score, higher is better)
stakeholder_preferences = {
    1: {"cost": lambda c: max(0, 1 - c / 10000), "time": lambda t: max(0, 1 - t / 48)},
    2: {"cost": lambda c: max(0, 1 - c / 5000),  "time": lambda t: max(0, 1 - t / 24)},
}

@Preferendus(
    preference_functions=stakeholder_preferences,
    stakeholder_weights=[0.7, 0.3],
    variable_types=[float, int],
    value_range=[(100, 1000), (800, 30000)],
)
def run_simulation(x):
    """A simple performance function returning cost and time."""
    distance, num_products = x[0], x[1]
    cost = distance * 10 + num_products * 0.2
    time = distance * 0.05 + num_products * 0.001
    # Return prefix items (env, objects) followed by objective values
    return None, None, cost, time

result = run_simulation([500, 15000])
print(f"Optimal variables: {result['x_opt']}")
print(f"Utility achieved: {result['utility']}")
```

## Examples

The `examples/` folder contains worked Jupyter notebooks demonstrating the GA-I solver
on a range of multi-stakeholder decision problems (several are ODESYS-book aligned):

- `4_by_4_problem.ipynb`
- `Bridge_example.ipynb`
- `Shopping_mall_non_linear.ipynb`
- `Super_market_non_linear.ipynb`
- `South_Korean_Floating_WF.ipynb`
- `support_functions/` — shared objective and plotting helpers used by the notebooks.

Install the example dependencies and launch Jupyter:

```bash
pip install -e ".[examples]"
jupyter notebook examples
```

## Testing

```bash
pytest
```

## References

1. Wolfert, A.R.M. et al. (2023). *Open Design Systems (ODESYS)*. TU Delft.
   https://github.com/TUDelft-Odesys
2. Özer, E. and Wolfert, A.R.M. (2026). *Ordering Is Not Deciding: Towards
   Preference-Consistent Design–Decision Optimisation.* To be published.
3. Timp, L.J.K., Yorke-Smith, N. & Wolfert, A.R.M. (2026). *Solving highly constrained
   multi-objective decision problems with preference-guided metaheuristics: The IMAP-IGS
   framework.* To be published.
4. Wolfert, A.R.M. (2026). *Unique preference aggregation in design and decision making.*
   arXiv. https://doi.org/10.48550/arXiv.2601.19759
5. Wolfert, A.R.M. (2026). *Preference-Based Optimisation for Integrated Design and Group
   Decision-Making.* arXiv preprint arXiv:2603.19050. https://arxiv.org/abs/2603.19050

## Acknowledgements

This GA-I solver originates from the open-source **Preferendus** software used for the
IMAP demonstrators in the ODESYS book, migrated into the IMAP-IGS framework as its
foundational Genetic Algorithm instantiation.
