#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import npfl139

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--episodes", default=3000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Decay rate.")


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # TODO:
    # - Create Q, a zero-filled NumPy array with shape [number of states, number of actions],
    #   representing estimated Q value of a given (state, action) pair.
    # - Create C, a zero-filled NumPy array with the same shape,
    #   representing number of observed returns of a given (state, action) pair.

    Q = np.zeros([num_states, num_actions])
    C = np.zeros([num_states, num_actions])

    for _ in range(args.episodes):
        # TODO: Perform an episode, collecting states, actions and rewards.
        history = []

        state, done = env.reset()[0], False
        while not done:
            # TODO: Compute `action` using epsilon-greedy policy.
            if np.random.uniform() >= args.epsilon:  # Greedy case
                action = get_greedy_action(Q, state)
            else:
                action = env.action_space.sample()

            # Perform the action.
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            history.append( (state, action, reward) )

            state = next_state

        # TODO: Compute returns from the received rewards and update Q and C.
        G = 0
        for t in range(len(history)-1, -1, -1):
            s, a, r = history[t][0], history[t][1], history[t][2]
            G = args.gamma * G + r
            C[s, a] += 1
            Q[s, a] += (G - Q[s, a]) / C[s, a]


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
        npfl139.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), main_args.seed, main_args.render_each)

    main(main_env, main_args)
