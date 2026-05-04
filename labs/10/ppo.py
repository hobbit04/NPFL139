#!/usr/bin/env python3
import argparse
import json

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.10")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="npfl139/SingleCollect-v0", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--clip_epsilon", default=0.2, type=float, help="Clipping epsilon.")
parser.add_argument("--entropy_regularization", default=0.01, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=16, type=int, help="Workers during experience collection.")
parser.add_argument("--epochs", default=5, type=int, help="Epochs to train each iteration.")
parser.add_argument("--evaluate_each", default=20, type=int, help="Evaluate each given number of iterations.")
parser.add_argument("--evaluate_for", default=20, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.999, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=32, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--trace_lambda", default=0.95, type=float, help="Traces factor lambda.")
parser.add_argument("--worker_steps", default=128, type=int, help="Steps for each worker to perform.")


class Agent:
    # Use GPU if available.
    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self._args = args
        self._obs_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.n
        # TODO: Create an actor using a single hidden layer with `args.hidden_layer_size`
        # units and ReLU activation, produce a policy with `env.action_space.n` discrete actions.
        self._actor = torch.nn.Sequential(
            torch.nn.Linear(self._obs_dim, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, self._action_dim)
        ).to(self.device)

        # TODO: Create a critic (value predictor) consisting of a single hidden layer with
        # `args.hidden_layer_size` units and ReLU activation, and an output layer with a single output.
        self._critic = torch.nn.Sequential(
            torch.nn.Linear(self._obs_dim, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 1)
        ).to(self.device)

        self._optimizer = torch.optim.Adam(
            list(self._actor.parameters()) + list(self._critic.parameters()),
            lr=args.learning_rate
        )

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, action_probs: torch.Tensor,
              advantages: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Perform a single training step of the PPO algorithm.
        # For the policy model, the sum is the sum of:
        # - the PPO loss, where `self._args.clip_epsilon` is used to clip the probability ratio
        # - the entropy regularization with coefficient `self._args.entropy_regularization`.
        #   You can compute it for example using the `torch.distributions.Categorical` class.
        dist = torch.distributions.Categorical(logits=self._actor(states))
        new_log_probs = dist.log_prob(actions)
        old_log_probs = torch.log(action_probs + 1e-8)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1 - self._args.clip_epsilon, 1 + self._args.clip_epsilon)
        actor_loss = -torch.mean(torch.min(ratio * advantages, clipped * advantages))
        entropy_loss = -self._args.entropy_regularization * dist.entropy().mean()

        # TODO: The critic model is trained in a standard way, by using the MSE
        # error between the predicted value function and target returns.
        critic_loss = torch.nn.functional.mse_loss(self._critic(states).squeeze(-1), returns)

        self._optimizer.zero_grad()
        (actor_loss + entropy_loss + critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(list(self._actor.parameters()) + list(self._critic.parameters()), 0.5)

        self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted action probabilities.
        return torch.softmax(self._actor(states), dim=-1)

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return estimates of value function.
        return self._critic(states).squeeze(-1)

    # Serialization methods.
    def save_actor(self, path: str) -> None:
        torch.save(self._actor.state_dict(), path)

    def load_actor(self, path: str) -> None:
        self._actor.load_state_dict(torch.load(path, map_location=self.device))

    @staticmethod
    def save_args(path: str, args: argparse.Namespace) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(vars(args), file, ensure_ascii=False, indent=2)

    @staticmethod
    def load_args(path: str) -> argparse.Namespace:
        with open(path, "r", encoding="utf-8-sig") as file:
            args = json.load(file)
        return argparse.Namespace(**args)


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the agent.
    recodex = args.recodex
    if recodex:
        args = Agent.load_args("ppo.json")
    agent = Agent(env, args)
    if recodex:
        agent.load_actor("ppo.pt")

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            # TODO: Predict an action by using a greedy policy.
            action = np.argmax(agent.predict_actions([state])[0])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create an asynchronous vector environment for training.
    vector_env = gym.make_vec(args.env, args.envs, gym.VectorizeMode.ASYNC,
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})

    # Training
    state = vector_env.reset(seed=args.seed)[0]  # shape: [args.envs, state_dim]
    training, iteration = True, 0
    while training and not recodex:
        # Collect experience. Notably, we collect the following quantities
        # as tensors with the first two dimensions `[args.worker_steps, args.envs]`.
        states, actions, action_probs, rewards, dones, values = [], [], [], [], [], []
        for _ in range(args.worker_steps):
            # TODO: Choose `action`, which is a vector of `args.envs` actions, each
            # sampled from the corresponding policy generated by the `agent.predict`
            # executed on the vector `state`.
            probs = agent.predict_actions(state)  # shape: [args.envs, action_dim]
            action = np.array([np.random.choice(len(p), p=p) for p in probs])  # shape: [args.envs]

            value = agent.predict_values(state)
            # Perform the environment interaction.
            next_state, reward, terminated, truncated, _ = vector_env.step(action)
            done = terminated | truncated

            # TODO: Compute and collect the required quantities.
            states.append(state)
            actions.append(action)
            action_probs.append(probs[np.arange(args.envs), action])  # Store only selected action's prob
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            state = next_state
        # Stack the first elemenet of the experience list
        states = np.stack(states)
        actions = np.stack(actions)
        action_probs = np.stack(action_probs)
        rewards = np.stack(rewards)
        dones = np.stack(dones).astype(np.float32)

        values = np.stack(values)

        # TODO: Estimate `advantages` and `returns` (they differ only by the value function estimate)
        # using lambda-return with coefficients `args.trace_lambda` and `args.gamma`.
        # You need to handle both the cases that (a) the last episode is probably unfinished, and
        # (b) there are multiple episodes in the collected data.
        last_value = agent.predict_values(state)

        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            next_value = values[t+1] if t+1 < len(rewards) else last_value
            delta = rewards[t] + args.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + args.gamma * args.trace_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        # TODO: Train for `args.epochs` using the collected data. In every epoch,
        # you should randomly sample batches of size `args.batch_size` from the collected data.
        # A possible approach is to create a dataset of `(states, actions, action_probs, advantages, returns)`
        # quintuples using a single `torch.utils.data.StackDataset` and then use a dataloader.
        # Flatten worker_steps and envs dimensions into a single batch dimension
        N = args.worker_steps * args.envs
        dataset = torch.utils.data.StackDataset(
            states.reshape(N, -1), actions.reshape(N), action_probs.reshape(N),
            advantages.reshape(N), returns.reshape(N)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        for _ in range(args.epochs):
            for batch in loader:
                agent.train(*batch)

        # Periodic evaluation
        iteration += 1
        if iteration % args.evaluate_each == 0:
            eval_returns = [evaluate_episode() for _ in range(args.evaluate_for)]
            if np.mean(eval_returns) > 530:
                training = False
                print("-----Training Over-----")

    # Save the trained agent.
    if not recodex:
        agent.save_actor("ppo.pt")
        Agent.save_args("ppo.json", args)

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)



if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make(main_args.env), main_args.seed, main_args.render_each)

    main(main_env, main_args)
