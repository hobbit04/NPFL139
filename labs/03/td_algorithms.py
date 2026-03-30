#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import npfl139
npfl139.require_version("2526.3")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate alpha.")
parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration epsilon factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor gamma.")
parser.add_argument("--mode", default="sarsa", type=str, help="Mode (sarsa/expected_sarsa/tree_backup).")
parser.add_argument("--n", default=1, type=int, help="Use n-step method.")
parser.add_argument("--off_policy", default=False, action="store_true", help="Off-policy; use greedy as target")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=47, type=int, help="Random seed.")
# If you add more arguments, ReCodEx will keep them with your default values.


def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)


def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a fixed seed.
    generator = np.random.RandomState(args.seed)

    # Create the environment.
    env = npfl139.EvaluationEnv(gym.make("Taxi-v3"), seed=args.seed, report_each=min(200, args.episodes))

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # The next action is always chosen in the epsilon-greedy way.
    def choose_next_action(Q: np.ndarray) -> tuple[int, float]:
        greedy_action = argmax_with_tolerance(Q[next_state])
        next_action = greedy_action if generator.uniform() >= args.epsilon else env.action_space.sample()
        return next_action, args.epsilon / env.action_space.n + (1 - args.epsilon) * (greedy_action == next_action)

    # The target policy is either the behavior policy (if not `args.off_policy`),
    # or the greedy policy (if `args.off_policy`).
    def compute_target_policy(Q: np.ndarray) -> np.ndarray:
        target_policy = np.eye(env.action_space.n)[argmax_with_tolerance(Q, axis=-1)]
        if not args.off_policy:
            target_policy = (1 - args.epsilon) * target_policy + args.epsilon / env.action_space.n
        return target_policy

    # Run the TD algorithm
    for _ in range(args.episodes):
        next_state, done = env.reset()[0], False

        # Generate episode and update Q using the given TD method
        next_action, next_action_prob = choose_next_action(Q)

        # Buffer for n-step method
        states = [next_state]
        actions = [next_action]
        rewards = [0]  # First element of this list will not be used. (There is no R_0)
        action_probs = [next_action_prob]
        T = float("inf")
        t = 0
        while not done:
            action, action_prob, state = next_action, next_action_prob, next_state
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            states.append(next_state)
            rewards.append(reward)  
            if not done:
                next_action, next_action_prob = choose_next_action(Q)
                
                actions.append(next_action)
                action_probs.append(next_action_prob)
            else:
                T = t + 1

            # TODO: Perform the update to the state-action value function `Q`, using
            # a TD update with the following parameters:
            # - `args.n`: use `args.n`-step method
            # - `args.off_policy`:
            #    - if False, the epsilon-greedy behavior policy is also the target policy
            #    - if True, the target policy is the greedy policy
            #      - for SARSA (with any `args.n`) and expected SARSA (with `args.n` > 1),
            #        importance sampling must be used
            # - `args.mode`: this argument can have the following values:
            #   - "sarsa": regular SARSA algorithm
            #   - "expected_sarsa": expected SARSA algorithm
            #   - "tree_backup": tree backup algorithm
            #
            # Perform the updates as soon as you can -- whenever you have all the information
            # to update `Q[state, action]`, do it. For each `action`, use its corresponding
            # `action_prob` from the time of taking the `action` as the behavior policy probability,
            # and the `compute_target_policy(Q)` with the current `Q` (from the time of performing
            # the update) as the target policy.
            #
            # Do not forget that when `done` is True, bootstrapping on the
            # `next_state` is not used.
            #
            # Also note that when the episode ends and `args.n` > 1, there will
            # be several state-action pairs that also need to be updated. Perform
            # the updates in the order in which you encountered the state-action
            # pairs and during these updates, use the `compute_target_policy(Q)`
            # with the up-to-date value of `Q`.
            tau = t - args.n + 1
            if tau >= 0:
                rho = 1.0
                if args.off_policy:
                    rho_end = tau + args.n
                    if args.mode == "sarsa":
                        rho_end += 1
                    for i in range(tau + 1, min(rho_end, T)):
                        target_prob = compute_target_policy(Q)[states[i], actions[i]]
                        rho *= (target_prob / action_probs[i])

                G = 0.0
                if args.mode != "tree_backup":
                    for i in range(tau + 1, min(tau + args.n, T) + 1):
                        G += args.gamma ** (i - tau - 1) * rewards[i]
                    if tau + args.n < T:
                        if args.mode == "sarsa":
                            G += args.gamma ** args.n * Q[states[tau + args.n], actions[tau + args.n]]
                        elif args.mode == "expected_sarsa":
                            end_state = states[tau + args.n]
                            target_probs = compute_target_policy(Q)[end_state]
                            expected_q = np.sum(target_probs * Q[end_state])
                            G += (args.gamma ** args.n) * expected_q

                    error = G - Q[states[tau], actions[tau]]
                    Q[states[tau], actions[tau]] += args.alpha * rho * error
                else:  # mode == "tree_backup"
                    end_t = min(tau + args.n, T)
                    if end_t == T:
                        G = rewards[T]
                    else:
                        G = rewards[end_t]
                        end_state = states[end_t]
                        target_probs = compute_target_policy(Q)[end_state]
                        expected_q = np.sum(target_probs * Q[end_state])
                        G += args.gamma * expected_q
                    for k in range(end_t - 1, tau, -1):  # Loop backward
                        target_probs = compute_target_policy(Q)[states[k]]
                        temp = target_probs[actions[k]]
                        target_probs[actions[k]] = 0
                        expected_q = np.sum(target_probs * Q[states[k]])
                        G = rewards[k] + args.gamma * expected_q + args.gamma * temp * G

                    error = G - Q[states[tau], actions[tau]]
                    Q[states[tau], actions[tau]] += args.alpha * error
            t += 1

        # After done = True, handle the rest of the (s, a, r) in episode
        for tau in range(max(0, T - args.n + 1), T):
            rho = 1.0
            if args.off_policy:
                for i in range(tau + 1, min(tau + args.n + 1, T)):
                    target_prob = compute_target_policy(Q)[states[i], actions[i]]
                    rho *= (target_prob / action_probs[i])
                    
            G = 0.0
            for i in range(tau + 1, min(tau + args.n, T) + 1):
                G += (args.gamma ** (i - tau - 1)) * rewards[i]
            
            if args.mode != "tree_backup":
                if tau + args.n < T:
                    if args.mode == "sarsa":
                        G += args.gamma ** args.n * Q[states[tau + args.n], actions[tau + args.n]]
                    elif args.mode == "expected_sarsa":
                        end_state = states[tau + args.n]
                        target_probs = compute_target_policy(Q)[end_state]
                        expected_q = np.sum(target_probs * Q[end_state])
                        G += (args.gamma ** args.n) * expected_q

                error = G - Q[states[tau], actions[tau]]
                Q[states[tau], actions[tau]] += args.alpha * rho * error

            else:  # mode == "tree_backup"
                end_t = min(tau + args.n, T)
                if end_t == T:
                    G = rewards[T]
                else:
                    G = rewards[end_t]
                    end_state = states[end_t]
                    target_probs = compute_target_policy(Q)[end_state]
                    expected_q = np.sum(target_probs * Q[end_state])
                    G += args.gamma * expected_q
                for k in range(end_t - 1, tau, -1):  # Loop backward
                    target_probs = compute_target_policy(Q)[states[k]]
                    temp = target_probs[actions[k]]
                    target_probs[actions[k]] = 0
                    expected_q = np.sum(target_probs * Q[states[k]])
                    G = rewards[k] + args.gamma * expected_q + args.gamma * temp * G

                error = G - Q[states[tau], actions[tau]]
                Q[states[tau], actions[tau]] += args.alpha * error

    # Return the final action-value function for ReCodEx to validate.
    return Q


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
