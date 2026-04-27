#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.9")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--cards", default=4, type=int, help="Number of cards in the memory game.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=16, type=int, help="Number of episodes to train on.")
parser.add_argument("--evaluate_each", default=128, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=64, type=int, help="Evaluate for number of episodes.")
parser.add_argument("--hidden_layer", default=None, type=int, help="Hidden layer size; default 8*`cards`")
parser.add_argument("--memory_cells", default=None, type=int, help="Number of memory cells; default 2*`cards`")
parser.add_argument("--memory_cell_size", default=None, type=int, help="Memory cell size; default 3/2*`cards`")


class Agent:
    # device = torch.device("cpu")
    # Use the following line instead to use GPU if available.
    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        self.args = args
        self.env = env

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # TODO: Create suitable layers.
                input_size = sum(env.observation_space.nvec)  # == args.memory_cell_size
                hidden_size = args.hidden_layer
                memory_cell_size = args.memory_cell_size
                n_actions = env.action_space.n

                # Read key network: encoded_input -> hidden (ReLU) -> key (tanh)
                self.read_key_hidden = torch.nn.Linear(input_size, hidden_size)
                self.read_key_out = torch.nn.Linear(hidden_size, memory_cell_size)

                # Policy network: concat(encoded_input, read_value) -> hidden (ReLU) -> logits
                self.policy_hidden = torch.nn.Linear(input_size + memory_cell_size, hidden_size)
                self.policy_out = torch.nn.Linear(hidden_size, n_actions)

            def forward(self, memory, observation):
                # Encode the input observation, which is a (card, observation) pair,
                # by representing each element as one-hot and concatenating them, resulting
                # in a vector of length `sum(env.observation_space.nvec)`.
                encoded_input = torch.cat([torch.nn.functional.one_hot(torch.relu(observation[:, i]), dim).float()
                                           for i, dim in enumerate(env.observation_space.nvec)], dim=-1)

                # TODO: Generate a read key for memory read from the encoded input, by using
                # a ReLU hidden layer of size `args.hidden_layer` followed by a dense layer
                # with `args.memory_cell_size` units and `tanh` activation (to keep the memory
                # content in limited range).
                read_key = torch.tanh(self.read_key_out(torch.relu(self.read_key_hidden(encoded_input))))
                # read_key: (batch, memory_cell_size)

                # TODO: Read the memory using the generated read key. Notably, compute cosine
                # similarity of the key and every memory row, apply softmax to generate
                # a weight distribution over the rows, and finally take a weighted average of
                # the memory rows.
                key_norm = torch.nn.functional.normalize(read_key, dim=-1)   # (batch, cell_size)
                mem_norm = torch.nn.functional.normalize(memory, dim=-1)     # (batch, cells, cell_size)
                similarity = torch.bmm(mem_norm, key_norm.unsqueeze(-1)).squeeze(-1)  # (batch, cells)
                weights = torch.softmax(similarity, dim=-1)                  # (batch, cells)
                read_value = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)      # (batch, cell_size)

                # TODO: Using concatenated encoded input and the read value, use a ReLU hidden
                # layer of size `args.hidden_layer` followed by a dense layer with
                # `env.action_space.n` units to produce policy logits.
                policy_input = torch.cat([encoded_input, read_value], dim=-1)
                logits = self.policy_out(torch.relu(self.policy_hidden(policy_input)))

                # TODO: Perform memory write. For faster convergence, add directly
                # the `encoded_input` to the memory, i.e., prepend it as a first memory row
                # and drop the last memory row to keep memory size constant.
                new_memory = torch.cat([encoded_input.unsqueeze(1), memory[:, :-1, :]], dim=1)

                # TODO: Return the updated memory and the policy
                return new_memory, logits

        # Create the agent
        self._model = Model().to(self.device)

        # TODO: Create an optimizer and a loss function.
        self._optimizer = torch.optim.Adam(self._model.parameters())
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def zero_memory(self):
        # TODO: Return an empty memory. It should be a tensor
        # with shape `[self.args.memory_cells, self.args.memory_cell_size]` on `self.device`.
        memory = torch.zeros(self.args.memory_cells, self.args.memory_cell_size).to(self.device)
        return memory

    @npfl139.typed_torch_function(device, torch.int64, torch.int64, torch.int64)
    def _train(self, observations, targets, lengths):
        # TODO: Given a batch of sequences of `observations` (each being a (card, symbol) pair),
        # train the network to predict the required `targets`.
        #
        # Specifically, start with a batch of empty memories, and run the agent
        # sequentially as many times as necessary, using `targets` as gold labels.
        #
        # Note that the sequences can be of different length, so you need to pad them
        # to same length and then somehow indicate the length of the individual episodes
        # (one possibility is to add another parameter to `_train`).
        self._model.train()
        batch_size, max_seq_len = observations.shape[0], observations.shape[1]

        memories = torch.zeros(batch_size, self.args.memory_cells, self.args.memory_cell_size,
                               device=self.device)
        self._optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=self.device)

        for t in range(max_seq_len):
            mask = lengths > t                              # (batch,) bool
            obs_t = observations[:, t, :]                  # (batch, 2)
            target_t = targets[:, t]                       # (batch,)
            # Detach memories: encoded_input is from integer one-hot so gradients don't
            # propagate through the memory chain anyway; detaching keeps graphs short.
            memories, logits = self._model(memories.detach(), obs_t)
            if mask.any():
                total_loss = total_loss + self._loss_fn(logits[mask], target_t[mask])

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        self._optimizer.step()

    def train(self, episodes):
        # TODO: Given a list of episodes, prepare the arguments
        # of the self._train method, and execute it.
        lengths = np.array([sum(1 for _, a in ep if a is not None) for ep in episodes], dtype=np.int64)
        max_len = int(lengths.max())
        batch_size = len(episodes)

        observations = np.zeros((batch_size, max_len, 2), dtype=np.int64)
        targets = np.zeros((batch_size, max_len), dtype=np.int64)

        for i, episode in enumerate(episodes):
            for t, (obs, action) in enumerate(episode):
                if action is None:  # terminal state has no action
                    break
                observations[i, t] = obs
                targets[i, t] = action

        self._train(observations, targets, lengths)

    @npfl139.typed_torch_function(device, torch.float32, torch.int64)
    def predict(self, memory, observation):
        self._model.eval()
        with torch.no_grad():
            memory, logits = self._model(memory.unsqueeze(0), observation.unsqueeze(0))
            return memory.squeeze(0), torch.softmax(logits, dim=-1).squeeze(0)


def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Post-process arguments to default values if not overridden on the command line.
    if args.hidden_layer is None:
        args.hidden_layer = 8 * args.cards
    if args.memory_cells is None:
        args.memory_cells = 2 * args.cards
    if args.memory_cell_size is None:
        args.memory_cell_size = 3 * args.cards // 2
    assert sum(env.observation_space.nvec) == args.memory_cell_size

    # Construct the agent.
    agent = Agent(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        observation, memory = env.reset(start_evaluation=start_evaluation, logging=logging)[0], agent.zero_memory()
        rewards, done = 0, False
        while not done:
            # TODO: Find out which action to use.
            memory, probs = agent.predict(memory, observation)
            action = np.argmax(probs)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    model_path = f"memory_game_{args.cards}.pt"

    if args.recodex:
        agent._model.load_state_dict(torch.load(model_path, map_location=agent.device, weights_only=True))
    else:
        # Training
        training = True
        while training:
            # Generate required number of episodes
            for _ in range(args.evaluate_each // args.batch_size):
                episodes = []
                for _ in range(args.batch_size):
                    episodes.append(env.expert_episode())

                # Train the agent
                agent.train(episodes)

            # Periodic evaluation
            returns = [evaluate_episode() for _ in range(args.evaluate_for)]
            if np.mean(returns) > 1.3:
                training = False

        torch.save(agent._model.state_dict(), model_path)

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make("npfl139/MemoryGame-v0", cards=main_args.cards), main_args.seed, main_args.render_each)

    main(main_env, main_args)
