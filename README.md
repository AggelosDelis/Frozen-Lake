# FrozenLake Q-learning vs Value Iteration

## Files

- `common.py` — configuration, environment creation, printing, evaluation, and plotting helpers.
- `q_learning.py` — tabular Q-learning solution.
- `value_iteration.py` — dynamic-programming / value-iteration solution.
- `main.py` — experiment runner: trains, compares, changes map, retrains, and prints final summary.

## Run

```bash
source venv/bin/activate
python main.py
```

Plots are saved in the `plots/` folder.

## Important config

Edit these in `common.py`:

```python
IS_SLIPPERY = True      # or False
EPISODES = 500_000
EVAL_EPISODES = 10_000
SHOW_PLOTS = False
```
