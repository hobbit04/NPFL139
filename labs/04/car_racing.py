#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import collections
import random
import os

import npfl139
npfl139.require_version("2526.4")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--continuous", default=0, type=int, help="Use continuous actions.")
parser.add_argument("--frame_skip", default=4, type=int, help="Frame skip.")


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions, dtype=np.int64), 
                np.array(rewards, dtype=np.float32), np.array(next_states), 
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

class Network:
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    def __init__(self, env, args):
        self._model = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 512), 
            torch.nn.ReLU(),
            torch.nn.Linear(512, env.action_space.n)
        ).to(self.device)

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        self._loss = torch.nn.MSELoss()


    @npfl139.typed_torch_function(device, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, q_values: torch.Tensor) -> None:
        self._model.train()
        states = states / 255.0
        predictions = self._model(states)
        loss = self._loss(predictions, q_values)
        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            states = states / 255.0
            return self._model(states)


    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    def load(self, path: str) -> None:
        if os.path.exists(path):
            self._model.load_state_dict(torch.load(path, map_location=self.device))
            self._model.eval()
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # If you want, you can wrap even the `npfl139.EvaluationEnv` with additional wrappers, like
    #   env = gym.wrappers.ResizeObservation(env, (64, 64))
    # or
    #   env = gym.wrappers.GrayscaleObservation(env)
    # However, if you do that, you can no longer call just `env.reset(start_evaluation=True)`;
    # instead, you need to pass the `start_evaluation` to the inner environment using
    #   env.reset(options={"start_evaluation": True})
    env = gym.wrappers.ResizeObservation(env, (64, 64))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)

    network = Network(env, args)
    model_path = "car_racing_model.pth"

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent
        network.load(model_path)
        # Final evaluation
        while True:
            while not done:
                # TODO: Choose a greedy action
                q_values = network.predict(np.expand_dims(state, axis=0))
                action = int(np.argmax(q_values[0]))

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        return

    # TODO: Implement a suitable RL algorithm and train the agent.
    #
    # If you want to create N multiprocessing parallel environments, use
    #   vector_env = gym.make_vec("npfl139/CarRacingFS-v3", N, gym.VectorizeMode.ASYNC,
    #                             frame_skip=args.frame_skip, continuous=args.continuous)
    #   vector_env.reset(seed=args.seed)  # The individual environments get incremental seeds
    #
    # There are several Autoreset modes available, see https://farama.org/Vector-Autoreset-Mode.
    # To change the autoreset mode to SAME_STEP from the default NEXT_STEP, pass
    #   vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}
    # as an additional argument to the above `gym.make_vec`.

    batch_size = 128
    gamma = 0.9
    epsilon_start = 0.8
    epsilon_end = 0.01
    epsilon_decay = 30000 
    target_update_freq = 500
    training_steps = 100000 

    replay_buffer = ReplayBuffer(capacity=50000)
    

    target_network = Network(env, args)
    target_network._model.load_state_dict(network._model.state_dict())
    target_network._model.eval()

    state, _ = env.reset()
    
    epsilon = epsilon_start
    episode_reward = 0

    train_freq = 4
    
    for step in range(1, training_steps + 1):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = network.predict(np.expand_dims(state, axis=0))
            action = int(np.argmax(q_values[0]))

        epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end) / epsilon_decay)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if done:
            print(f"Step: {step}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")
            state, _ = env.reset()
            episode_reward = 0

        if len(replay_buffer) > batch_size and step % train_freq == 0:
            b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(batch_size)
            
            # Numpy arrays to Torch tensors
            b_states_t = torch.tensor(b_states, device=network.device, dtype=torch.float32)
            b_next_states_t = torch.tensor(b_next_states, device=network.device, dtype=torch.float32)
            b_rewards_t = torch.tensor(b_rewards, device=network.device, dtype=torch.float32)
            b_dones_t = torch.tensor(b_dones, device=network.device, dtype=torch.float32)
            b_actions_t = torch.tensor(b_actions, device=network.device, dtype=torch.int64).unsqueeze(-1)

            with torch.no_grad():
                next_q_values = target_network._model(b_next_states_t / 255.0)
                max_next_q_values = next_q_values.max(dim=1)[0]

                target_q_values = b_rewards_t + gamma * max_next_q_values * (1 - b_dones_t)

            with torch.no_grad():
                current_q_values = network._model(b_states_t / 255.0)

            target_q_tensor = current_q_values.scatter(1, b_actions_t, target_q_values.unsqueeze(-1))
            network.train(b_states_t, target_q_tensor)

        if step % target_update_freq == 0:
            target_network._model.load_state_dict(network._model.state_dict())

        if step % 10000 == 0:
            network.save(model_path)
            print(f"Model saved at step {step}")

    network.save(model_path)
    print("Training finished!")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make("npfl139/CarRacingFS-v3", frame_skip=main_args.frame_skip, continuous=main_args.continuous),
        main_args.seed, main_args.render_each, evaluate_for=15, report_each=1)

    main(main_env, main_args)
