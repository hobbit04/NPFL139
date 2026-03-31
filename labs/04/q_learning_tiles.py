#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import npfl139
npfl139.require_version("2526.4")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.3, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.15, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.004, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=5000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
parser.add_argument("--tiles", default=16, type=int, help="Number of tiles.")

def get_greedy_action(W, state):
    q_values = np.sum(W[state], axis=0)
    max_q = np.max(q_values)
    max_indices = np.where(np.isclose(q_values, max_q))[0]
    return np.random.choice(max_indices)

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.zeros([env.observation_space.nvec[-1], env.action_space.n])
    epsilon = args.epsilon

    episodes = 0
    max_episode = 10000
    while episodes < max_episode:
        # Perform episode
        state, done = env.reset()[0], False
        while not done:
            # TODO: Choose an action.
            if np.random.uniform() >= epsilon:  # Greedy case
                action = get_greedy_action(W, state)
            else:
                action = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TODO: Update the action-value estimates
            current_q_values = np.sum(W[state], axis=0)
            current_q = current_q_values[action]

            if terminated:
                target = reward
            else:
                next_q_values = np.sum(W[next_state], axis=0)
                target = reward + args.gamma * np.max(next_q_values)

            td_err = target - current_q
            alpha_scaled = args.alpha / args.tiles
            W[state, action] += alpha_scaled * td_err

            state = next_state

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])
        episodes += 1

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action
            action = get_greedy_action(W, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteMountainCarWrapper(gym.make("npfl139/MountainCar1000-v0"), tiles=main_args.tiles),
        main_args.seed, main_args.render_each,
    )

    main(main_env, main_args)
