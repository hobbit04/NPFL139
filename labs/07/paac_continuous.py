#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import npfl139
npfl139.require_version("2526.7")

import itertools

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=0.01, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=32, type=int, help="Number of parallel environments.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--tiles", default=8, type=int, help="Tiles to use.")


class Agent:
    # Use GPU if available.
    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")

    def __init__(self, env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Analogously to paac, your model should contain two components:
        # - an actor, which predicts distribution over the actions, and
        # - a critic, which predicts the value function.
        #
        # The given states are tile encoded, so they are integer indices of
        # tiles intersecting the state. Therefore, you should convert them
        # to dense encoding (one-hot-like, with `args.tiles` ones); or you can
        # even use the `torch.nn.EmbeddingBag` layer.
        #
        # The actor computes `mus` and `sds`, each of shape `[batch_size, actions]`.
        # Compute each independently using states as input, adding a fully connected
        # layer with `args.hidden_layer_size` units and a ReLU activation. Then:
        # - For `mus`, add a fully connected layer with `actions` outputs.
        #   To avoid `mus` moving from the required range, you should apply
        #   properly scaled `torch.tanh` activation.
        # - For `sds`, add a fully connected layer with `actions` outputs
        #   and `torch.exp` or `torch.nn.functional.softplus` activation.
        #
        # The critic should be a usual one, passing states through one hidden
        # layer with `args.hidden_layer_size` ReLU units and then predicting
        # the value function.
        actions = env.action_space.shape[0]
        self.args = args
        num_tiles = int(env.observation_space.nvec.max())

        self._actor_mus = torch.nn.Sequential(
            torch.nn.EmbeddingBag(num_tiles, args.hidden_layer_size, mode="sum"),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(actions),
            torch.nn.Tanh(),
        ).to(self.device)
        self._actor_sds = torch.nn.Sequential(
            torch.nn.EmbeddingBag(num_tiles, args.hidden_layer_size, mode="sum"),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(actions),
            torch.nn.Softplus(),
        ).to(self.device)

        self._critic = torch.nn.Sequential(
            torch.nn.EmbeddingBag(num_tiles, args.hidden_layer_size, mode="sum"),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(1),
        ).to(self.device)

        # self.actor_mus_optimizer = torch.optim.Adam(self._actor_mus.parameters(), lr=args.learning_rate)
        # self.actor_sds_optimizer = torch.optim.Adam(self._actor_sds.parameters(), lr=args.learning_rate)
        self.actor_optimizer = torch.optim.Adam(
            itertools.chain(self._actor_mus.parameters(), self._actor_sds.parameters()),
            lr=args.learning_rate,
        )
        
        self.critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=args.learning_rate)

    # The `npfl139.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl139.typed_torch_function(device, torch.int64, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Run the model on given `states` and compute `sds`, `mus` and predicted values.
        # Then create `action_distribution` using `torch.distributions.Normal` class and
        # the computed `mus` and `sds`.
        #
        mus = self._actor_mus(states)
        sds = self._actor_sds(states)
        action_distribution = torch.distributions.Normal(mus, sds)
        # TODO: Train the actor using the sum of the following two losses:
        # - REINFORCE loss, i.e., the negative log likelihood of the `actions` in the
        #   `action_distribution` (using the `log_prob` method). You then need to sum
        #   the log probabilities of the action components in a single batch example.
        #   Finally, multiply the resulting vector by `(returns - baseline)`
        #   and compute its mean. Be sure to let the gradient flow only where it should.
        # - negative value of the distribution entropy (use `entropy` method of
        #   the `action_distribution`) weighted by `args.entropy_regularization`.
        #
        # Train the critic using mean square error of the `returns` and predicted values.
        values = self._critic(states).squeeze(-1)
        critic_loss = torch.nn.functional.mse_loss(values, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        with torch.no_grad():
            advantages = returns - self._critic(states).squeeze(-1)

        # REINFORCE loss
        log_probs = action_distribution.log_prob(actions).sum(dim=-1) 
        reinforce_loss = -(log_probs * advantages).mean()

        # Entropy bonus 
        entropy = action_distribution.entropy().sum(dim=-1) 
        entropy_loss = -self.args.entropy_regularization * entropy.mean()

        actor_loss = reinforce_loss + entropy_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            itertools.chain(self._actor_mus.parameters(), self._actor_sds.parameters()),
            max_norm=0.5
        )
        self.actor_optimizer.step() 


    @npfl139.typed_torch_function(device, torch.int64)
    def predict_actions(self, states: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return predicted action distributions (mus and sds).
        with torch.no_grad():
            mus = self._actor_mus(states)
            sds = self._actor_sds(states)

        return mus, sds
    
    @npfl139.typed_torch_function(device, torch.int64)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        # TODO: Return predicted state-action values.
        with torch.no_grad():
            values = self._critic(states)
        return values

def main(env: npfl139.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    # Construct the agent.
    agent = Agent(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state = env.reset(options={"start_evaluation": start_evaluation, "logging": logging})[0]
        rewards, done = 0, False
        while not done:
            # TODO: Predict an action using the greedy policy.
            mus, _ = agent.predict_actions([state])
            action = np.clip(mus[0], env.action_space.low, env.action_space.high)
        
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the vectorized environment, using the SAME_STEP autoreset mode.
    vector_env = gym.make_vec("MountainCarContinuous-v0", args.envs, gym.VectorizeMode.ASYNC,
                              wrappers=[lambda env: npfl139.DiscreteMountainCarWrapper(env, tiles=args.tiles)],
                              vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    states = vector_env.reset(seed=args.seed)[0]

    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Predict action distribution using `agent.predict_actions`
            # and then sample it using for example `np.random.normal`. Do not
            # forget to clip the actions to the `env.action_space.{low,high}`
            # range, for example using `np.clip`.
            mus, sds = agent.predict_actions(states)
            action_dist = np.random.normal(mus, sds)
            actions = np.clip(action_dist, env.action_space.low, env.action_space.high)
            
            # Perform steps in the vectorized environment
            next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
            dones = terminated | truncated

            # TODO(paac): Compute estimates of returns by one-step bootstrapping
            next_values = agent.predict_values(next_states)
            returns = rewards + args.gamma * next_values * (1 - dones.astype(np.float32))

            # TODO(paac): Train agent using current states, chosen actions and estimated returns.
            agent.train(states, actions, returns)


            states = next_states

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        if np.mean(returns) > 90:
            training = False
    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)

    # This time, I have to train the model in ReCodEx..! 

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        npfl139.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=main_args.tiles),
        main_args.seed, main_args.render_each)

    main(main_env, main_args)
