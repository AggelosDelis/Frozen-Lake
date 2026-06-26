"""Tabular Q-learning solution."""

from __future__ import annotations
import numpy as np
from common import (
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    EPISODES,
    GAMMA,
    LEARNING_RATE,
    MAX_STEPS,
    apply_optional_reward_shaping,
)

# Q-learning algorithm (may be replaced by deep Q-learning in the future)
def choose_action_epsilon_greedy(q_table, state, epsilon, env):
    """Choose an action using epsilon-greedy exploration."""
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return int(np.argmax(q_table[state]))

# Q-learning training function
def train_q_learning(env):
    """
    Train a tabular Q-learning agent.

    Returns:
        q_table: array with shape (n_states, n_actions)
        rewards_per_episode: list of episode returns
        steps_per_episode: list of episode lengths
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_table = np.zeros((n_states, n_actions))
    rewards_per_episode = []
    steps_per_episode = []
    epsilon = EPSILON_START
    training_successes = 0

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0

        # Loop for each step of the episode
        for _ in range(MAX_STEPS):
            action = choose_action_epsilon_greedy(q_table, state, epsilon, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward = apply_optional_reward_shaping(env, next_state, reward)
            done = terminated or truncated

            old_q_value = q_table[state, action]
            best_next_q_value = np.max(q_table[next_state])

            q_table[state, action] = old_q_value + LEARNING_RATE * (
                reward + GAMMA * best_next_q_value - old_q_value
            )

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                if reward == 1:
                    training_successes += 1
                break

        epsilon = max(EPSILON_END, epsilon * np.exp(-EPSILON_DECAY))
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        if (episode + 1) % 50_000 == 0:
            recent_avg_reward = float(np.mean(rewards_per_episode[-50_000:]))
            print(
                f"Episode {episode + 1}/{EPISODES} | "
                f"epsilon={epsilon:.4f} | recent avg reward={recent_avg_reward:.4f}"
            )

    print(f"Training successes: {training_successes} / {EPISODES}")
    return q_table, rewards_per_episode, steps_per_episode

# Evaluation function for the trained Q-learning agent 
def get_policy_from_q_table(q_table):
    """Return the greedy policy derived from a Q-table."""
    return np.argmax(q_table, axis=1)


def print_q_table(q_table):
    print("\n================ Q-TABLE ================")
    print("Columns: LEFT, DOWN, RIGHT, UP")
    np.set_printoptions(precision=4, suppress=True)
    print(q_table)
