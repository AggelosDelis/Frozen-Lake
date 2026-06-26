"""Dynamic-programming solution using value iteration."""
from __future__ import annotations
import numpy as np
from common import GAMMA, get_map


# value iteration algorithm
def value_iteration(env, theta=1e-8):
    """
    Solve FrozenLake using value iteration.

    This is the dynamic-programming solver. It uses env.unwrapped.P, which is the
    full transition model P(s' | s, a).
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    iteration_count = 0

    while True:
        # delta is the maximum change in the value function across all states in an iteration
        delta = 0.0

        # Update each state in the value function
        for state in range(n_states):
            old_value = V[state]
            action_values = np.zeros(n_actions)
            # Maximize over all possible actions to find the best action value
            for action in range(n_actions):
                for probability, next_state, reward, done in env.unwrapped.P[state][action]:
                    action_values[action] += probability * (reward + GAMMA * V[next_state])

            V[state] = np.max(action_values)
            delta = max(delta, abs(old_value - V[state]))

        iteration_count += 1
        if delta < theta:
            break

    policy = extract_policy_from_values(env, V)
    return V, policy, iteration_count

# Actual code for extracting the policy from the value function
def extract_policy_from_values(env, V):
    """Extract greedy policy from a solved value function."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = np.zeros(n_states, dtype=int)

    for state in range(n_states):
        action_values = np.zeros(n_actions)

        for action in range(n_actions):
            for probability, next_state, reward, done in env.unwrapped.P[state][action]:
                action_values[action] += probability * (reward + GAMMA * V[next_state])

        policy[state] = int(np.argmax(action_values))

    return policy

def print_value_table(V, env):
    print("\n================ VALUE ITERATION TABLE V(s) ================")
    lake_map = get_map(env)
    n_rows = len(lake_map)
    n_cols = len(lake_map[0])
    print(V.reshape(n_rows, n_cols))
