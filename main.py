"""Run experiments comparing Q-learning and value iteration on FrozenLake."""

from __future__ import annotations
import numpy as np

from common import (
    EVAL_EPISODES,
    PLOT_DIR,
    SEED,
    draw_lake,
    evaluate_policy,
    generate_new_random_map,
    make_env,
    plot_steps_comparison,
    plot_training_curve,
    policy_agreement,
    print_environment_info,
    print_map_from_desc,
    print_policy,
)
from q_learning import get_policy_from_q_table, print_q_table, train_q_learning
from value_iteration import value_iteration


def compare_on_environment(env, q_policy, title):
    """Compare an existing Q-learning policy with value iteration on one environment."""
    print("\n\n##################################################")
    print(f"# {title}")
    print("##################################################")

    print_environment_info(env)

    V, vi_policy, iterations = value_iteration(env)
    print(f"\nValue iteration converged after {iterations} iterations.")

    print_policy(q_policy, env, "Q-LEARNING POLICY BEING TESTED")
    print_policy(vi_policy, env, "VALUE ITERATION POLICY FOR THIS MAP")

    q_success, q_avg_reward, q_avg_steps = evaluate_policy(env, q_policy)
    vi_success, vi_avg_reward, vi_avg_steps = evaluate_policy(env, vi_policy)
    agreement = policy_agreement(q_policy, vi_policy, env)

    print("\n================ COMPARISON ================")
    print("\nQ-learning policy:")
    print(f"  Success rate:   {q_success:.3f}")
    print(f"  Average reward: {q_avg_reward:.3f}")
    print(f"  Average steps:  {q_avg_steps:.2f}")

    print("\nValue iteration policy:")
    print(f"  Success rate:   {vi_success:.3f}")
    print(f"  Average reward: {vi_avg_reward:.3f}")
    print(f"  Average steps:  {vi_avg_steps:.2f}")
    print(f"\nPolicy agreement with value iteration: {agreement:.3f}")

    return {
        "q_success": q_success,
        "q_avg_reward": q_avg_reward,
        "q_avg_steps": q_avg_steps,
        "vi_success": vi_success,
        "vi_avg_reward": vi_avg_reward,
        "vi_avg_steps": vi_avg_steps,
        "agreement": agreement,
        "vi_policy": vi_policy,
        "vi_iterations": iterations,
    }


def train_and_report(env, title_prefix: str, map_plot_name: str):
    """Train Q-learning on env and print/plot the learned policy."""
    print(f"\nTraining Q-learning on {title_prefix}...")
    q_table, rewards_per_episode, steps_per_episode = train_q_learning(env)
    print_q_table(q_table)

    q_policy = get_policy_from_q_table(q_table)
    print_policy(q_policy, env, f"Q-LEARNING POLICY TRAINED ON {title_prefix.upper()}")

    draw_lake(
        env,
        policy=q_policy,
        title=f"Q-learning policy trained on {title_prefix}",
        save_path=PLOT_DIR / f"{map_plot_name}_q_policy.png",
    )

    plot_training_curve(
        rewards_per_episode,
        title=f"Q-learning training curve: {title_prefix}",
        save_path=PLOT_DIR / f"{map_plot_name}_training_curve.png",
    )

    return q_table, q_policy, rewards_per_episode, steps_per_episode


def main():
    np.random.seed(SEED)

    # ============================================================
    # PART 1: TRAIN ON ORIGINAL 8x8 MAP
    # ============================================================
    print("\n\n")
    print("##################################################")
    print("# PART 1: TRAIN Q-LEARNING ON ORIGINAL 8x8 MAP")
    print("##################################################")

    train_env = make_env(map_name="8x8", desc=None, seed=SEED)
    print_environment_info(train_env)

    initial_state, _ = train_env.reset(seed=SEED)
    draw_lake(
        train_env,
        state=initial_state,
        title="Original 8x8 map",
        save_path=PLOT_DIR / "original_map.png",
    )

    q_table, q_policy, rewards_per_episode, steps_per_episode = train_and_report(
        train_env,
        title_prefix="original 8x8 map",
        map_plot_name="original_map",
    )

    # ============================================================
    # PART 2: COMPARE ON SAME MAP
    # ============================================================
    same_map_env = make_env(map_name="8x8", desc=None, seed=SEED + 1)
    same_map_results = compare_on_environment(
        same_map_env,
        q_policy,
        "PART 2: TEST TRAINED Q-POLICY VS VALUE ITERATION ON ORIGINAL 8x8 MAP",
    )

    plot_steps_comparison(
        same_map_env,
        q_policy,
        same_map_results["vi_policy"],
        same_map_results["vi_iterations"],
        title="After first training: original 8x8 map",
        episodes=EVAL_EPISODES,
        window=200,
        save_path=PLOT_DIR / "steps_after_first_training.png",
    )

    # ============================================================
    # PART 3: GENERATE NEW 8x8 MAP
    # ============================================================
    new_map_desc = generate_new_random_map(size=8, frozen_probability=0.85, seed=999)
    print_map_from_desc(new_map_desc, "NEW RANDOM 8x8 MAP")

    new_map_env = make_env(map_name=None, desc=new_map_desc, seed=SEED + 2)
    draw_lake(
        new_map_env,
        title="New random 8x8 map",
        save_path=PLOT_DIR / "new_random_map.png",
    )

    # ============================================================
    # PART 4: TEST OLD Q-POLICY ON NEW MAP WITHOUT RETRAINING
    # ============================================================
    new_map_results_without_retraining = compare_on_environment(
        new_map_env,
        q_policy,
        "PART 4: TEST OLD Q-POLICY ON NEW RANDOM MAP WITHOUT RETRAINING",
    )

    # ============================================================
    # PART 5: RETRAIN Q-LEARNING ON NEW MAP
    # ============================================================
    print("\n\n")
    print("##################################################")
    print("# PART 5: RETRAIN Q-LEARNING ON NEW RANDOM MAP")
    print("##################################################")

    retrain_env = make_env(map_name=None, desc=new_map_desc, seed=SEED + 3)
    new_q_table, new_q_policy, new_rewards_per_episode, new_steps_per_episode = train_and_report(
        retrain_env,
        title_prefix="new random 8x8 map",
        map_plot_name="new_random_map",
    )

    retrained_new_map_env = make_env(map_name=None, desc=new_map_desc, seed=SEED + 4)
    new_map_results_with_retraining = compare_on_environment(
        retrained_new_map_env,
        new_q_policy,
        "PART 5 RESULT: RETRAINED Q-POLICY VS VALUE ITERATION ON NEW MAP",
    )

    plot_steps_comparison(
        retrained_new_map_env,
        new_q_policy,
        new_map_results_with_retraining["vi_policy"],
        new_map_results_with_retraining["vi_iterations"],
        title="After second training: new random 8x8 map",
        episodes=EVAL_EPISODES,
        window=200,
        save_path=PLOT_DIR / "steps_after_second_training.png",
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n\n")
    print("================ FINAL SUMMARY ================")
    print("\n")
    print("Original map:")
    print(f"  Q-learning success:      {same_map_results['q_success']:.3f}")
    print(f"  Value iteration success: {same_map_results['vi_success']:.3f}")
    print("\n")
    print("New map, without retraining Q-learning:")
    print(f"  Old Q-policy success:    {new_map_results_without_retraining['q_success']:.3f}")
    print(f"  Value iteration success: {new_map_results_without_retraining['vi_success']:.3f}")
    print("\n")
    print("New map, after retraining Q-learning:")
    print(f"  New Q-policy success:    {new_map_results_with_retraining['q_success']:.3f}")
    print(f"  Value iteration success: {new_map_results_with_retraining['vi_success']:.3f}")

    train_env.close()
    same_map_env.close()
    new_map_env.close()
    retrain_env.close()
    retrained_new_map_env.close()


if __name__ == "__main__":
    main()
