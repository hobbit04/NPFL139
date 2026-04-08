#!/usr/bin/env python3
import argparse
import collections
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

import os

import npfl139
npfl139.require_version("2526.5")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verify", default=False, action="store_true", help="Verify the loss computation")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--atoms", default=51, type=int, help="Number of atoms.")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=2000, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=256, type=int, help="Target update frequency.")

class Network:
    device = torch.device("cpu")
    # Use the following line instead to use GPU if available.
    # device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model and store it as `self._model`. The model
        # should compute `args.atoms` logits for each action, so for input of shape
        # `[batch_size, *env.observation_space.shape]`, the output should have
        # the shape `[batch_size, env.action_space.n, args.atoms]`. The module
        # `torch.nn.Unflatten` might come handy.
        self._model = torch.nn.Sequential(
            torch.nn.Linear(np.prod(env.observation_space.shape), args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, int(env.action_space.n) * args.atoms),
            torch.nn.Unflatten(1, (int(env.action_space.n), args.atoms))
        )

        # Create `self._model.atoms` as uniform grid from 0 to 500 with `args.atoms` elements.
        # We create them as a buffer in `self._model` so they are automatically moved with `.to`.
        self._model.register_buffer("atoms", torch.linspace(0, 500, args.atoms))

        self._model.to(self.device)

        # Store the discount factor.
        self.gamma = args.gamma

        # TODO(q_network): Define a suitable optimizer from `torch.optim`.
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=args.learning_rate)

    @staticmethod
    def compute_loss(
        states_logits: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor,
        next_states_logits: torch.Tensor, atoms: torch.Tensor, gamma: float,
    ) -> torch.Tensor:
        # TODO: Implement the loss computation according to the C51 algorithm.
        # - The `states_logits` are current state logits of shape `[batch_size, actions, atoms]`.
        # - The `actions` are the integral actions taken in the states, of shape `[batch_size]`.
        # - The `rewards` are the rewards obtained after taking the actions, of shape `[batch_size]`.
        # - The `dones` are `torch.float32` indicating whether the episode ended, of shape `[batch_size]`.
        # - The `next_states_logits` are logits of the next states, of shape `[batch_size, actions, atoms]`.
        #   Because they should not be backpropagated through, use an appropriate `.detach()` call.
        # - The `atoms` are the atom values. Your implementation must handle any number of atoms. The
        #   `atoms[0]` is V_MIN (the minimum atom value), `atoms[-1]` is V_MAX (the maximum atom value),
        #   and use `atoms[1] - atoms[0]` as the distance between two consecutive atoms. You can
        #   assume that one of the atoms is always 0.
        # The resulting loss should be the mean of the cross-entropy losses of the individual batch examples.
        #
        # Your implementation most likely needs to be vectorized to pass ReCodEx time limits. Note that you
        # can add given values to a vector of (possibly repeating) tensor indices using `scatter_add_`.
        batch_size = states_logits.shape[0]
        num_atoms = atoms.shape[0]
        delta_z = atoms[1] - atoms[0]

        next_states_logits = next_states_logits.detach()
        next_probs = torch.softmax(next_states_logits, dim=-1)
        next_q_values = (next_probs * atoms).sum(dim=-1)
        next_actions = next_q_values.argmax(dim=-1)

        batch_idx = torch.arange(batch_size, device=states_logits.device)
        next_best_probs = next_probs[batch_idx, next_actions]

        Tz = rewards.unsqueeze(-1) + gamma * (1.0 - dones.unsqueeze(-1)) * atoms.unsqueeze(0)
        Tz = Tz.clamp(min=atoms[0], max=atoms[-1])
        
        b = (Tz - atoms[0]) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        exact_match = (l == u)
        dl = u.float() - b
        du = b - l.float()
        dl[exact_match] = 1.0
        du[exact_match] = 0.0

        m = torch.zeros(batch_size, num_atoms, device=states_logits.device)
        m.scatter_add_(1, l, next_best_probs * dl)
        m.scatter_add_(1, u, next_best_probs * du)

        log_probs = torch.log_softmax(states_logits, dim=-1)
        taken_log_probs = log_probs[batch_idx, actions]

        loss = -(m * taken_log_probs).sum(dim=-1).mean()
        return loss

    # The training function defers the computation to the `compute_loss` method.
    #
    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
              dones: torch.Tensor, next_states: torch.Tensor) -> None:
        self._model.train()
        # Pass all arguments to the `compute_loss` method.
        next_logits = self.target_network._model(next_states) if hasattr(self, 'target_network') else self._model(next_states)
        loss = self.compute_loss(
            self._model(states), actions, rewards, dones, next_logits, self._model.atoms, self.gamma
        )
        self._optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            # TODO: Return all predicted Q-values for the given states.
            logits = self._model(states)
            probs = torch.softmax(logits, dim=-1)
            q_values = (probs * self._model.atoms).sum(dim=-1)
            return q_values.cpu().numpy()

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    def copy_weights_from(self, other: "Network") -> None:
        self._model.load_state_dict(other._model.state_dict())


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> Callable | None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # When the `args.verify` is set, just return the loss computation function for validation.
    if args.verify:
        return Network.compute_loss

    # Construct the network
    network = Network(env, args)

    target_network = Network(env, args)
    target_network.copy_weights_from(network)
    network.target_network = target_network

    model_path = "dist_c51.pt"

    if os.path.exists(model_path):
        print(f"Found pre-trained model! Loading from {model_path}...")
        network._model.load_state_dict(torch.load(model_path, map_location=network.device, weights_only=True))
        training = False  # 파일이 있으면 학습 루프를 아예 건너뜀
    else:
        training = True   # 파일이 없으면 정상적으로 학습 진행

    # Replay memory; the `max_length` parameter is its maximum capacity.
    replay_buffer = npfl139.ReplayBuffer(max_length=1_000_000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    epsilon = args.epsilon

    step = 0
    recent_returns = collections.deque(maxlen=100)
    while training:
        # Perform episode
        state, done = env.reset()[0], False
        episode_return = 0.0
        while not done:
            # TODO(q_network): Choose an action.
            # You can compute the q_values of a given state by
            #   q_values = network.predict(state[np.newaxis])[0]
            step += 1
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = network.predict(state[np.newaxis])[0]
                action = np.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, terminated, next_state))

            # TODO: If the `replay_buffer` is large enough, perform training by
            # sampling a batch of `args.batch_size` uniformly randomly chosen transitions
            # and calling `network.train(states, actions, rewards, dones, next_states)`.
            #
            # The `replay_buffer` offers a method with signature
            #   sample(self, size, replace=True) -> NamedTuple
            # which returns uniformly selected batch of `size` transitions, either with
            # replacement (which is faster, and hence the default) or without.
            # The returned batch is a `Transition` named tuple, each field being
            # a NumPy array containing a batch of corresponding transition components.
            if len(replay_buffer) >= args.batch_size:
                batch = replay_buffer.sample(args.batch_size)
                network.train(batch.state, batch.action, batch.reward, batch.done, batch.next_state)

            if step % args.target_update_freq == 0:
                target_network.copy_weights_from(network)
            state = next_state

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

        recent_returns.append(episode_return)
        if len(recent_returns) == 100 and np.mean(recent_returns) >= 460:
            print(f"Saving trained model to {model_path}...")
            torch.save(network._model.state_dict(), model_path)
            break

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO(q_network): Choose (greedy) action
            q_values = network.predict(state[np.newaxis])[0]
            action = np.argmax(q_values).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(gym.make("CartPole-v1"), main_args.seed, main_args.render_each)

    result = main(main_env, main_args)
    if main_args.verify:
        np.testing.assert_allclose(result(
            states_logits=torch.tensor([[[-1.5, 1.2, -1.2], [-0.0, -1.8, -0.1]],
                                        [[-0.2, -0.3, 1.3], [0.5, -1.1, -0.7]],
                                        [[-0.1, 1.9, -0.0], [-0.3, -1.1, -0.1]]]),
            actions=torch.tensor([0, 1, 0]),
            rewards=torch.tensor([0.5, -0.2, 0.7]), dones=torch.tensor([1., 0., 0.]),
            next_states_logits=torch.tensor([[[1.1, 0.2, 0.3], [0.3, 1.1, 1.3]],
                                             [[-0.4, -0.5, -0.6], [2.0, 1.2, 0.4]],
                                             [[-0.3, -0.9, 2.3], [0.7, 0.7, -0.3]]]),
            atoms=torch.tensor([-2., -1., 0.]),
            gamma=0.3).numpy(force=True), 2.170941, atol=1e-5)

        np.testing.assert_allclose(result(
            states_logits=torch.tensor([[[0.1, 1.4, -0.5, -0.8], [0.3, -0.0, -0.2, -0.2]],
                                        [[1.2, -0.8, -1.4, -1.5], [0.1, -0.6, -2.1, -0.3]]]),
            actions=torch.tensor([0, 1]),
            rewards=torch.tensor([0.5, 0.6]), dones=torch.tensor([0., 0.]),
            next_states_logits=torch.tensor([[[0.8, 1.2, -1.2, 0.7], [0.3, 0.4, -1.2, 0.4]],
                                             [[-0.2, 1.0, -1.5, 0.2], [0.2, 0.5, 0.4, -0.9]]]),
            atoms=torch.tensor([-3., 0., 3., 6.]),
            gamma=0.2).numpy(force=True), 1.43398, atol=1e-5)
