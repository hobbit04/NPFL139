#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import npfl139
npfl139.require_version("2526.2")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.7, type=float, help="Exploration factor.")
parser.add_argument("--min_epsilon", default=0.001, type=float, help="Minimum exploration factor.")
parser.add_argument("--epsilon_discount", default=0.99999, type=float, help="Epsilon discount rate.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    # TODO: Variable creation and initialization
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    Q = np.zeros([num_states, num_actions])
    epsilon = args.epsilon

    # training = True
    episodes = 3300
    # while training:
    for _ in range(episodes):
        # Perform episode
        state, done = env.reset()[0], False

        epsilon = max(args.min_epsilon, epsilon * args.epsilon_discount)
        while not done:
            # TODO: Perform an action.
            if np.random.uniform() >= epsilon:  # Greedy case
                action = get_greedy_action(Q, state)
            else:
                action = env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TODO: Update the action-value estimates
            Q[state, action] += args.alpha * (reward + args.gamma * np.max(Q[next_state]) - Q[state, action])

            state = next_state

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose a greedy action
            action = get_greedy_action(Q, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


def get_greedy_action(Q, state):
    q_values = Q[state]
    max_q = np.max(q_values)
    max_indices = np.where(np.isclose(q_values, max_q))[0]
    return np.random.choice(max_indices)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteMountainCarWrapper(gym.make("npfl139/MountainCar1000-v0")),
        main_args.seed, main_args.render_each,
    )

    main(main_env, main_args)
