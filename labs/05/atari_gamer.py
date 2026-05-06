#!/usr/bin/env python3
import argparse
import json

import re

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import torch
import numpy as np

import npfl139
npfl139.require_version("2526.5")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--frame_skip", default=4, type=int, help="Frame skip.")
parser.add_argument("--frame_stack", default=4, type=int, help="Frame stack.")
parser.add_argument("--game", default="Pong", type=str, help="Game to play.")
parser.add_argument("--grayscale", default=True, action=argparse.BooleanOptionalAction, help="Grayscale obs.")
parser.add_argument("--screen_size", default=84, type=int, help="Screen size.")
parser.add_argument("--max_episode", default=50000, type=int, help="Number of vectorized training steps")
parser.add_argument("--cnn_hidden", default=64, type=int, help="Size of a cnn layer")
parser.add_argument("--lr", default=2.5e-4, type=float, help="Learning rate")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
parser.add_argument("--envs", default=16, type=int, help="Number of parallel environments.")
parser.add_argument("--evaluate_each", default=1000, type=int, help="Evaluate every N steps.")
parser.add_argument("--evaluate_for", default=5, type=int, help="Number of episodes for evaluation.")

class Agent:
    device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu")
    def __init__(self, env, args):
        self.args = args
        self.in_channels = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Define CNN backbone, Actor and Critic
        self._backbone = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, args.cnn_hidden, 8, 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(args.cnn_hidden, args.cnn_hidden, 4, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(args.cnn_hidden, args.cnn_hidden, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 8 * 8, 512),  # 64ch × 8×8 spatial (screen_size=84)
            torch.nn.ReLU(), 
        ).to(device=self.device)

        self._actor = torch.nn.Linear(512, self.action_dim).to(device=self.device)
        self._critic = torch.nn.Linear(512, 1).to(device=self.device)
        
        # Define optimizers 
        self._optimizer = torch.optim.Adam(
            list(self._backbone.parameters()) +
            list(self._actor.parameters()) +
            list(self._critic.parameters()),
            lr=args.lr
        )
    
    def _preprocess(self, state):
        return state / 255.0

    def forward(self, x):
        feature = self._backbone(x)
        action_logit = self._actor(feature)  # This is not prob dist! Need to pass softmax before using
        critic = self._critic(feature)

        return (action_logit, critic)

    @npfl139.typed_torch_function(device, torch.float32, torch.int64, torch.float32, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, next_states: torch.Tensor) -> None:
        states = self._preprocess(states)
        next_states = self._preprocess(next_states)

        action_logits, values = self.forward(states)
        values = values.squeeze(-1)

        with torch.no_grad():
            next_values = self._critic(self._backbone(next_states)).squeeze(-1)

        advantages = (rewards + self.args.gamma * next_values * (1 - dones)) - values

        critic_loss = (advantages ** 2).mean()
        dist = torch.distributions.Categorical(logits=action_logits)
        actor_loss = -(dist.log_prob(actions) * advantages.detach()).mean()

        total_loss = critic_loss + actor_loss

        torch.nn.utils.clip_grad_norm_(
            list(self._backbone.parameters()) + list(self._actor.parameters()) + list(self._critic.parameters()), 
            max_norm=0.5
        )
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_actions(self, states: torch.Tensor) -> np.ndarray:
        states = self._preprocess(states)
        x = self._backbone(states)
        return torch.softmax(self._actor(x), dim=-1)

    @npfl139.typed_torch_function(device, torch.float32)
    def predict_values(self, states: torch.Tensor) -> np.ndarray:
        x = self._backbone(states)
        return self._critic(x).squeeze(-1)

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

    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=args.frame_skip, grayscale_obs=args.grayscale, screen_size=args.screen_size)
    env = gym.wrappers.FrameStackObservation(env, stack_size=args.frame_stack)

    model_path = f"atari_gamer_{args.game}"

    agent = Agent(env, args if not args.recodex else Agent.load_args(model_path + ".json"))

    def evaluate_episode(start_evaluation: bool = False) -> float:
        state, done = env.reset(options={"start_evaluation": start_evaluation})[0], False
        total_reward = 0
        while not done:
            action = np.argmax(agent.predict_actions([state])[0])
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        return total_reward

    if args.recodex:
        agent.load_actor(model_path)
        while True:
            evaluate_episode(start_evaluation=True)

    # Create vectorized environment for training
    vector_env = ale_py.AtariVectorEnv(
        game=re.sub(r"(?<=[a-z])(?=[A-Z])", "_", args.game).lower(),
        num_envs=args.envs,
        frameskip=args.frame_skip, stack_num=args.frame_stack, grayscale=args.grayscale,
        img_height=args.screen_size, img_width=args.screen_size,
        use_fire_reset=False, reward_clipping=False, repeat_action_probability=0.25,
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )
    states = vector_env.reset(seed=args.seed)[0]

    for step in range(args.max_episode):
        probs = agent.predict_actions(states)
        actions = np.array([np.random.choice(len(p), p=p) for p in probs])

        next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
        dones = (terminated | truncated).astype(np.float32)

        agent.train(states, actions, rewards, dones, next_states)
        states = next_states

        if (step + 1) % args.evaluate_each == 0:
            ep_returns = [evaluate_episode() for _ in range(args.evaluate_for)]
            print(f"Step {step + 1}/{args.max_episode}, mean return: {np.mean(ep_returns):.1f}")

    # Save the agent
    agent.save_actor(model_path)
    agent.save_args(model_path + ".json", args)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    assert main_args.render_each in [0, 1], "Option render_each can be only 0 or 1 for Atari games"

    # Create the environment
    main_env = npfl139.EvaluationEnv(
        gym.make(f"ALE/{main_args.game}-v5", frameskip=1, render_mode="human" if main_args.render_each else None),
        main_args.seed)

    main(main_env, main_args)
