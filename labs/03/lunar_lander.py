#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np

import npfl139
npfl139.require_version("2526.3")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=1000, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed.
    npfl139.startup(args.seed)

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent
        Q = np.load("q_table.npy")

        # Final evaluation
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                # TODO: Choose a greedy action
                action = get_greedy_action(Q, state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    # TODO: Implement a suitable RL algorithm and train the agent.
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    iter = 0
    num_of_iter = 3000
    training = True
    while training:
        # To generate an expert episode, you can use the following:
        #   episode = env.expert_episode()
        episode = env.expert_episode(seed=np.random.randint(1, 100))

        # Otherwise, you can generate a training episode the usual way:
        for i in range(len(episode) - 1):
            state, action, reward = episode[i]
            next_state, _, _ = episode[i + 1]
            bootstrap = np.max(Q[next_state])
            
            error = reward + args.gamma * bootstrap - Q[state, action]
            Q[state, action] += args.alpha * error

        iter += 1
        if iter > num_of_iter:
            break
    num_self_episodes = 5000  
    
    for _ in range(num_self_episodes):
        state, done = env.reset()[0], False
        
        while not done:
            # epsilon-greedy 행동 선택
            if np.random.uniform() < args.epsilon:
                action = env.action_space.sample() # 무작위 탐험
            else:
                action = np.argmax(Q[state])       # 배운 대로 행동
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Q-Learning 업데이트
            bootstrap = 0.0 if done else np.max(Q[next_state])
            error = reward + args.gamma * bootstrap - Q[state, action]
            Q[state, action] += args.alpha * error
            
            # 상태 업데이트 (다음 스텝을 위해)
            state = next_state
    
    # Save trained Q table as q_table.npy
    np.save("q_table.npy", Q)

def get_greedy_action(Q, state):
    q_values = Q[state]
    max_q = np.max(q_values)
    max_indices = np.where(np.isclose(q_values, max_q))[0]
    return np.random.choice(max_indices)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteLunarLanderWrapper(gym.make("LunarLander-v3")), main_args.seed, main_args.render_each)

    main(main_env, main_args)
