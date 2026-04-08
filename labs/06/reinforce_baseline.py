#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.6")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.995, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.014, type=float, help="Learning rate.")


class Agent:
    device = torch.device("cpu")
    # Use the following line instead to use GPU if available.
    # device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model of the policy and the value networks.
        #
        # In addition to the policy network defined in the `reinforce` assignment,
        # you need a value network for computing the baseline. It can be for example
        # another independent model with a single hidden layer and an output layer
        # with a single output and no activation. You can also experiment with just
        # a single shared model with two heads (the policy head and the value head),
        # but such a model is more difficult to train because of possible different
        # scales of the two losses.
        #
        # Using Adam optimizer with given `args.learning_rate` for both models
        # is a good default.
        input_size = env.observation_space.shape[0]

        self._policy = torch.nn.Sequential(
            torch.nn.Linear(input_size, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, env.action_space.n),
            torch.nn.Softmax(dim=-1),
        ).to(self.device)

        self._baseline = torch.nn.Sequential(
            torch.nn.Linear(input_size, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1),
        ).to(self.device)

        self._policy_optimizer = torch.optim.Adam(self._policy.parameters(), lr=args.learning_rate)
        self._baseline_optimizer = torch.optim.Adam(self._baseline.parameters(), lr=args.learning_rate)

        self._policy_loss = torch.nn.NLLLoss(reduction="none")
        self._baseline_loss = torch.nn.MSELoss()

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Define the training method.
        #
        # You should:
        # - compute the predicted baseline using the baseline model
        # - train the policy model, using `returns - predicted_baseline` as
        #   advantage estimate
        # - train the baseline model to predict `returns`
        baseline = self._baseline(states).squeeze(-1)
        advantage = returns - baseline.detach()
        # Normalize advantages to reduce variance
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        self._policy_optimizer.zero_grad()
        probs = self._policy(states)
        log_probs = torch.log(probs + 1e-8)
        per_sample_loss = self._policy_loss(log_probs, actions)
        # Entropy bonus to encourage exploration
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        policy_loss = (per_sample_loss * advantage).mean() - 0.01 * entropy
        policy_loss.backward()
        self._policy_optimizer.step()

        self._baseline_optimizer.zero_grad()
        baseline = self._baseline(states).squeeze(-1)
        baseline_loss = self._baseline_loss(baseline, returns)
        baseline_loss.backward()
        self._baseline_optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Define the prediction method returning policy probabilities.
        with torch.no_grad():
            return self._policy(states)


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the agent.
    agent = Agent(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset()[0], False
            while not done:
                # TODO(reinforce): Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `agent.predict` and current `state`.
                probs = agent.predict(state)
                action = np.random.choice(len(probs), p=probs)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO(reinforce): Compute returns by summing rewards (with discounting)
            returns = []
            G = 0
            for reward in reversed(rewards):
                G = reward + args.gamma * G
                returns.insert(0, G)

            # TODO(reinforce): Add states, actions and returns to the training batch
            batch_states += states
            batch_actions += actions
            batch_returns += returns

        # TODO(reinforce): Train using the generated batch.
        agent.train(batch_states, batch_actions, batch_returns)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO(reinforce): Choose a greedy action.
            probs = agent.predict(state)
            action = np.argmax(probs)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)
