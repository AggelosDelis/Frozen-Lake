"""Common configuration, environment helpers, evaluation, and plotting."""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# ============================================================
# CONFIG
# ============================================================

ENV_NAME = "FrozenLake-v1"

# True = stochastic/slippery FrozenLake.
# False = deterministic FrozenLake.
IS_SLIPPERY = True

SEED = 42
GAMMA = 0.95

# Q-learning parameters
EPISODES = 500_000
MAX_STEPS = 200
LEARNING_RATE = 0.01
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = EPSILON_START / (EPISODES / 2)

# Optional reward shaping. Keep False for clean comparison with value iteration.
USE_REWARD_SHAPING = False
REWARD_SCHEDULE = (1.0, -1.0, -0.0001)  # goal, hole, frozen

# Evaluation
EVAL_EPISODES = 10_000

# Plotting
SHOW_PLOTS = False
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

ACTIONS = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP",
}

ACTION_ARROWS = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑",
}

def make_env(render_mode=None, map_name: Optional[str] = "8x8", desc=None, seed: int = SEED):
    """
    Create a FrozenLake environment.

    If desc is None, use map_name, for example "4x4" or "8x8".
    If desc is not None, use the custom/random map.
    """
    env = gym.make(
        ENV_NAME,
        map_name=map_name if desc is None else None,
        desc=desc,
        is_slippery=IS_SLIPPERY,
        render_mode=render_mode,
    )

    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

def get_map(env):
    """Return the FrozenLake map as normal strings instead of byte strings."""
    desc = env.unwrapped.desc
    return [[cell.decode("utf-8") for cell in row] for row in desc]

# State and coordinate conversion functions
def state_to_row_col(state: int, n_cols: int):
    return divmod(state, n_cols)

# Row and column to state conversion
def row_col_to_state(row: int, col: int, n_cols: int):
    return row * n_cols + col

# Get the tile at a given state
def get_tile(env, state: int) -> str:
    lake_map = get_map(env)
    n_cols = len(lake_map[0])
    row, col = state_to_row_col(state, n_cols)
    return lake_map[row][col]

# Apply optional reward shaping
def apply_optional_reward_shaping(env, next_state: int, reward: float) -> float:
    """
    Keep normal Gym reward by default.

    If USE_REWARD_SHAPING=True:
      goal   -> +1
      hole   -> -1
      frozen -> small step penalty
    """
    if not USE_REWARD_SHAPING:
        return reward

    goal_reward, hole_reward, frozen_reward = REWARD_SCHEDULE
    tile = get_tile(env, next_state)

    if tile == "G":
        return goal_reward
    if tile == "H":
        return hole_reward
    return frozen_reward

# Generate a new random map
def generate_new_random_map(size=8, frozen_probability=0.85, seed=123):
    """
    Generate a random FrozenLake map.

    Larger frozen_probability means fewer holes and an easier map.
    """
    return generate_random_map(size=size, p=frozen_probability, seed=seed)

# Printer functions for Q-table, value table, and policies
def print_map_from_desc(desc, title: str):
    print(f"\n================ {title} ================")
    for row in desc:
        print(" ".join(row))

def print_environment_info(env):
    print("\n================ ENVIRONMENT INFO ================")
    print(f"Environment: {ENV_NAME}")
    print(f"Slippery: {IS_SLIPPERY}")
    print(f"Number of states: {env.observation_space.n}")
    print(f"Number of actions: {env.action_space.n}")
    print("Actions:")
    for action_id, action_name in ACTIONS.items():
        print(f"  {action_id}: {action_name}")

    lake_map = get_map(env)
    print("\nMap:")
    for row in lake_map:
        print(" ".join(row))

    print("\nState numbering:")
    n_rows = len(lake_map)
    n_cols = len(lake_map[0])
    states = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
    print(states)


def print_policy(policy, env, title: str):
    print(f"\n================ {title} ================")

    lake_map = get_map(env)
    n_rows = len(lake_map)
    n_cols = len(lake_map[0])

    for row in range(n_rows):
        symbols = []
        for col in range(n_cols):
            state = row_col_to_state(row, col, n_cols)
            tile = lake_map[row][col]

            if tile in ["H", "G"]:
                symbols.append(tile)
            else:
                action = int(policy[state])
                symbols.append(ACTION_ARROWS[action])

        print(" ".join(symbols))

# Policy agreement function
def policy_agreement(policy_a, policy_b, env) -> float:
    """Fraction of non-terminal states where two policies choose the same action."""
    lake_map = get_map(env)
    n_rows = len(lake_map)
    n_cols = len(lake_map[0])

    comparable_states = []
    for row in range(n_rows):
        for col in range(n_cols):
            state = row_col_to_state(row, col, n_cols)
            tile = lake_map[row][col]
            if tile not in ["H", "G"]:
                comparable_states.append(state)

    same = sum(int(policy_a[state]) == int(policy_b[state]) for state in comparable_states)
    return same / len(comparable_states)

# Policy evaluation function
def evaluate_policy(env, policy, episodes: int = EVAL_EPISODES):
    """
    Monte Carlo evaluation of a fixed policy.

    This is not a solver. It only measures success rate, average reward,
    and average steps by executing an already-computed policy.
    """
    if len(policy) != env.observation_space.n:
        raise ValueError(
            f"Policy has length {len(policy)}, but environment has "
            f"{env.observation_space.n} states. Use a map with the same size."
        )

    total_rewards = []
    total_steps = []
    successes = 0

    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        steps = 0

        for _ in range(MAX_STEPS):
            action = int(policy[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward = apply_optional_reward_shaping(env, next_state, reward)
            done = terminated or truncated

            episode_reward += reward
            steps += 1
            state = next_state

            if done:
                if get_tile(env, state) == "G":
                    successes += 1
                break

        total_rewards.append(episode_reward)
        total_steps.append(steps)

    success_rate = successes / episodes
    average_reward = float(np.mean(total_rewards))
    average_steps = float(np.mean(total_steps))

    return success_rate, average_reward, average_steps

def run_policy_and_collect_steps(env, policy, episodes: int = EVAL_EPISODES, seed_start: int = 10_000):
    """Run a fixed policy and return steps plus success/failure for each episode."""
    if len(policy) != env.observation_space.n:
        raise ValueError(
            f"Policy has length {len(policy)}, but environment has "
            f"{env.observation_space.n} states."
        )

    steps_per_episode = []
    successes_per_episode = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed_start + episode)
        episode_steps = 0
        episode_success = 0

        for _ in range(MAX_STEPS):
            action = int(policy[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_steps += 1

            if done:
                if reward == 1:
                    episode_success = 1
                break

        steps_per_episode.append(episode_steps)
        successes_per_episode.append(episode_success)

    return np.array(steps_per_episode), np.array(successes_per_episode)

def moving_average(values, window: int):
    values = np.array(values)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")

def draw_lake(env, state=None, policy=None, values=None, title="FrozenLake", save_path=None, show: bool = SHOW_PLOTS):
    """Draw the FrozenLake grid, optionally with a state, policy arrows, and values."""
    lake_map = get_map(env)
    n_rows = len(lake_map)
    n_cols = len(lake_map[0])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(0, n_cols + 1, 1))
    ax.set_yticks(np.arange(0, n_rows + 1, 1))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for row in range(n_rows):
        for col in range(n_cols):
            tile = lake_map[row][col]
            s = row_col_to_state(row, col, n_cols)
            text = tile

            if values is not None:
                text += f"\n{values[s]:.2f}"
            if policy is not None and tile not in ["H", "G"]:
                text += f"\n{ACTION_ARROWS[int(policy[s])]}"
            if state is not None and s == state:
                text += "\nA"

            ax.text(col + 0.5, row + 0.5, text, ha="center", va="center", fontsize=12)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)

def plot_training_curve(rewards_per_episode, title="Q-learning training curve", save_path=None, show: bool = SHOW_PLOTS):
    """Plot moving average of training rewards."""
    window = 500
    rewards = np.array(rewards_per_episode)
    curve = moving_average(rewards, window)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Moving average reward")
    ax.plot(curve)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)

def plot_steps_comparison(
    env,
    q_policy,
    vi_policy,
    vi_iterations: int,
    title: str,
    episodes: int = EVAL_EPISODES,
    window: int = 200,
    seed_start: int = 10_000,
    save_path=None,
    show: bool = SHOW_PLOTS,
):
    """Plot number of steps per episode for Q-table policy and value-iteration policy."""
    q_steps, q_successes = run_policy_and_collect_steps(env, q_policy, episodes=episodes, seed_start=seed_start)
    vi_steps, vi_successes = run_policy_and_collect_steps(env, vi_policy, episodes=episodes, seed_start=seed_start)

    q_success_rate = float(np.mean(q_successes))
    vi_success_rate = float(np.mean(vi_successes))
    q_avg_steps = float(np.mean(q_steps))
    vi_avg_steps = float(np.mean(vi_steps))

    print(f"\n================ STEP PLOT: {title} ================")
    print(f"Value iteration iterations: {vi_iterations}")
    print("Q-table policy:")
    print(f"  Success rate: {q_success_rate:.3f}")
    print(f"  Average steps: {q_avg_steps:.2f}")
    print("Value-iteration policy:")
    print(f"  Success rate: {vi_success_rate:.3f}")
    print(f"  Average steps: {vi_avg_steps:.2f}")

    x = np.arange(1, episodes + 1)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(x, q_steps, s=8, alpha=0.15, color="tab:blue", label="Q-table steps")
    ax.scatter(x, vi_steps, s=8, alpha=0.15, color="tab:orange", label="Value iteration steps")

    q_ma = moving_average(q_steps, window)
    vi_ma = moving_average(vi_steps, window)
    ma_x = x if len(q_ma) == len(q_steps) else x[window - 1:]

    ax.plot(ma_x, q_ma, color="tab:blue", linewidth=2, label=f"Q-table moving avg ({window})")
    ax.plot(ma_x, vi_ma, color="tab:orange", linewidth=2, linestyle="--", label=f"Value iteration moving avg ({window})")

    ax.set_title(
        f"{title}\n"
        f"Q success={q_success_rate:.3f}, avg steps={q_avg_steps:.2f} | "
        f"VI success={vi_success_rate:.3f}, avg steps={vi_avg_steps:.2f}"
    )
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Number of steps")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    return {
        "q_steps": q_steps,
        "vi_steps": vi_steps,
        "q_success_rate": q_success_rate,
        "vi_success_rate": vi_success_rate,
        "q_avg_steps": q_avg_steps,
        "vi_avg_steps": vi_avg_steps,
        "vi_iterations": vi_iterations,
    }
