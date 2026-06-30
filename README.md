# FrozenLake Reinforcment Learning vs Dynamic Programming

This project compares two approaches for solving the FrozenLake environment:

* **Q-learning**: a model-free reinforcement learning algorithm.
* **Value Iteration**: a dynamic-programming algorithm that uses the environment transition model.

The experiment trains a tabular Q-learning agent on an 8x8 FrozenLake map, compares it against value iteration.

## Implemented Algorithms

### Q-learning

Q-learning learns a Q-table:

```text
Q[state, action]
```

Each value estimates the expected future reward of taking an action from a given state. The learned policy is obtained by selecting the action with the highest Q-value in each state.

### Value Iteration

Value iteration is a dynamic-programming method. It uses the known transition model of the environment:

```text
P(next_state | state, action)
```

It repeatedly applies the Bellman optimality update until the value function converges, then extracts the greedy policy from the final value table.

## Project Structure

```text
frozenlake_project/
├── common.py
├── q_learning.py
├── value_iteration.py
├── main.py
└── plots/
```

## Files

* `common.py` — shared configuration, environment creation, map utilities, evaluation functions, printing helpers and plotting functions.
* `q_learning.py` — tabular Q-learning implementation.
* `value_iteration.py` — dynamic-programming / value-iteration implementation.
* `main.py` — experiment runner. It trains Q-learning, runs value iteration, compares both methods, changes the map, tests transfer, retrains, and prints the final summary.
* `plots/` — generated plots and map visualizations.

## Experiment Flow

The main experiment does the following:

1. Create the original 8x8 FrozenLake map.
2. Train Q-learning on the original map.
3. Compute the value-iteration policy for the same map.
4. Compare Q-learning and value iteration.
5. Generate a new random 8x8 FrozenLake map.
6. Test the old Q-learning policy on the new map without retraining.
7. Retrain Q-learning on the new map.
8. Compare the retrained Q-learning policy with value iteration on the new map.

## Run

Activate the virtual environment:

```bash
source venv/bin/activate
```

Run the experiment:

```bash
python main.py
```

Generated plots are saved in the `plots/` folder.

## Important Configuration

These values can be changed in `common.py`:

```python
IS_SLIPPERY = True      # True = stochastic environment, False = deterministic environment
EPISODES = 500_000      # Number of Q-learning training episodes
EVAL_EPISODES = 10_000  # Number of evaluation episodes
SHOW_PLOTS = False      # False = save plots without opening windows
```

## Notes

When `IS_SLIPPERY = False`, FrozenLake behaves like a deterministic path-planning problem. In this case, both Q-learning and value iteration can usually reach the goal in the shortest path after training.

When `IS_SLIPPERY = True`, the environment becomes stochastic. The agent may not move in the intended direction, so success rates are lower and the average number of steps is higher.

The Q-table is specific to the map on which it was trained. If the map changes, the old Q-table usually does not transfer well. This demonstrates a limitation of tabular reinforcement learning: it learns state-action values for a fixed environment, but it does not generalize to different map layouts without retraining.

## Current Conclusion

The current experiments show that:

* Q-learning can learn a good policy for a fixed FrozenLake map.
* Value iteration can solve a given map directly when the transition model is known.
* A learned Q-table does not generalize well to a different map.
* Retraining Q-learning on the new map improves performance again.
* The slippery version is a better stochastic MDP experiment, while the non-slippery version is useful as a deterministic sanity check.
* Similarities: Both algoritmhs use the Bellman idea:
    ```current value = reward now + discounted future value ```
    ```reward + gamma * future value```
    ```produce a policy by choosing the action with highest value```
