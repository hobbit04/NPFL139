#!/usr/bin/env python3
import argparse
import collections
import copy
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import torch

import npfl139
npfl139.require_version("2526.12")
from npfl139.board_games import Pisqorky
import board_game_cpp

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.3, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=512, type=int, help="Training batch size.")
parser.add_argument("--epsilon", default=0.25, type=float, help="MCTS exploration epsilon in root")
parser.add_argument("--evaluate_each", default=5, type=int, help="Evaluate each N iterations.")
parser.add_argument("--filters", default=64, type=int, help="Number of filters in residual blocks")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="pisqorky.pt", type=str, help="Model path")
parser.add_argument("--num_blocks", default=6, type=int, help="Number of residual blocks")
parser.add_argument("--num_simulations", default=50, type=int, help="MCTS simulations per move")
parser.add_argument("--replay_buffer_length", default=200_000, type=int, help="Replay buffer capacity")
parser.add_argument("--sampling_moves", default=15, type=int, help="Moves to sample stochastically")
parser.add_argument("--sim_games", default=50, type=int, help="Self-play games per iteration")
parser.add_argument("--train_for", default=100, type=int, help="Training steps per iteration")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="L2 regularisation")


#########
# Agent #
#########

class _ResBlock(torch.nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(filters, filters, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(filters),
            torch.nn.ReLU(),
            torch.nn.Conv2d(filters, filters, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(x + self.net(x))


class Agent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Network(torch.nn.Module):
        def __init__(self, args: argparse.Namespace):
            super().__init__()
            F = args.filters
            self.stem = torch.nn.Sequential(
                torch.nn.Conv2d(Pisqorky.C, F, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
            )
            self.body = torch.nn.Sequential(*[_ResBlock(F) for _ in range(args.num_blocks)])
            self.policy_head = torch.nn.Sequential(
                torch.nn.Conv2d(F, 2, 1, bias=False),
                torch.nn.BatchNorm2d(2),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(2 * Pisqorky.N * Pisqorky.N, Pisqorky.ACTIONS),
            )
            self.value_head = torch.nn.Sequential(
                torch.nn.Conv2d(F, 1, 1, bias=False),
                torch.nn.BatchNorm2d(1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(Pisqorky.N * Pisqorky.N, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
                torch.nn.Tanh(),
            )

        def forward(self, x: torch.Tensor):
            x = self.body(self.stem(x))
            return self.policy_head(x), self.value_head(x).squeeze(1)

    def __init__(self, args: argparse.Namespace):
        self._model = self.Network(args).to(self.device)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        )

    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> "Agent":
        agent = Agent(args)
        agent._model.load_state_dict(torch.load(path, map_location=agent.device))
        return agent

    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, boards: torch.Tensor, target_policies: torch.Tensor, target_values: torch.Tensor):
        # TODO: Train the model from boards [B,N,N,C], target_policies [B,ACTIONS], target_values [B].
        boards = boards.permute(0, 3, 1, 2)  # [B,C,N,N]
        self._model.train()
        self._optimizer.zero_grad()
        policy_logits, values = self._model(boards)
        log_p = torch.nn.functional.log_softmax(policy_logits, dim=1)
        policy_loss = -(target_policies * log_p).sum(dim=1).mean()
        value_loss = torch.nn.functional.mse_loss(values, target_values)
        (policy_loss + value_loss).backward()
        self._optimizer.step()
        return policy_loss, value_loss

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, board: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return (policy_probs [ACTIONS], value scalar) from a single board [N,N,C].
        board = board.unsqueeze(0).permute(0, 3, 1, 2)  # [1,C,N,N]
        self._model.eval()
        with torch.no_grad():
            policy_logits, value = self._model(board)
        return torch.softmax(policy_logits, dim=1).squeeze(0), value.squeeze()

    def board_features(self, game: Pisqorky) -> np.ndarray:
        # TODO: Return board features from current player's perspective (matching C++ board_features).
        if game.to_play == 0:
            return game.board_features
        return game.clone(swap_players=True).board_features


############
# Training #
############
ReplayBufferEntry = collections.namedtuple("ReplayBufferEntry", ["board", "policy", "outcome"])


def train(args: argparse.Namespace) -> "Agent":
    agent = Agent(args)
    best_agent = copy.deepcopy(agent)
    replay_buffer = npfl139.ReplayBuffer(max_length=args.replay_buffer_length)

    # Mutable reference so the evaluate closure always uses the current best agent.
    best_ref = [best_agent]

    def evaluate(boards: np.ndarray):
        """Batch evaluate: boards [B,N,N,C] → (policies [B,ACTIONS], values [B])."""
        t = torch.tensor(boards, dtype=torch.float32, device=best_ref[0].device)
        t = t.permute(0, 3, 1, 2)  # [B,C,N,N]
        best_ref[0]._model.eval()
        with torch.no_grad():
            policy_logits, values = best_ref[0]._model(t)
        return torch.softmax(policy_logits, dim=1).cpu().numpy(), values.cpu().numpy()

    board_game_cpp.simulated_games_start(
        threads=args.threads,
        num_simulations=args.num_simulations,
        sampling_moves=args.sampling_moves,
        epsilon=args.epsilon,
        alpha=args.alpha,
    )

    iteration = 0
    policy_loss = value_loss = float("nan")
    while True:
        iteration += 1

        # Collect self-play games using the C++ multi-threaded engine.
        for _ in range(args.sim_games):
            for board, policy, outcome in board_game_cpp.simulated_game(evaluate):
                replay_buffer.append(ReplayBufferEntry(board, policy, np.float32(outcome)))

        # Train from the replay buffer.
        if len(replay_buffer) >= args.batch_size:
            for _ in range(args.train_for):
                batch = replay_buffer.sample(args.batch_size)
                policy_loss, value_loss = agent.train(*batch)

        print(f"Iteration {iteration}: policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}", flush=True)

        if iteration % args.evaluate_each == 0:
            # Self-play: current vs best (use network-only for speed).
            eval_args = argparse.Namespace(num_simulations=0, alpha=args.alpha)
            self_score = npfl139.board_games.evaluate(
                Pisqorky,
                [Player(agent, eval_args), Player(best_agent, eval_args)],
                games=20, first_chosen=False, render=False, verbose=False,
            )
            print(f"  Self-play vs best: {100 * self_score:.1f}%")

            if self_score > 0.55:
                best_agent = copy.deepcopy(agent)
                best_ref[0] = best_agent
                print("  Best model updated.")

            # Evaluate against the heuristic player.
            score = npfl139.board_games.evaluate(
                Pisqorky,
                [Player(best_agent, eval_args),
                 Pisqorky.player_from_name("heuristic")(seed=args.seed)],
                games=20, first_chosen=False, render=False, verbose=False,
            )
            print(f"  vs heuristic: {100 * score:.1f}%", flush=True)

            # Save the best model periodically.
            best_agent.save(args.model_path)

    # Unreachable; training is stopped externally.
    board_game_cpp.simulated_games_stop()
    return best_agent


#############################
# BoardGamePlayer interface #
#############################
class Player(npfl139.board_games.BoardGamePlayer[Pisqorky]):
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args

    def play(self, game: Pisqorky) -> int:
        # TODO: Select the best valid action.
        if self.args.num_simulations == 0:
            policy, _ = self.agent.predict(self.agent.board_features(game))
        else:
            def evaluate(boards: np.ndarray):
                t = torch.tensor(boards, dtype=torch.float32, device=self.agent.device)
                t = t.permute(0, 3, 1, 2)
                self.agent._model.eval()
                with torch.no_grad():
                    policy_logits, values = self.agent._model(t)
                return torch.softmax(policy_logits, dim=1).cpu().numpy(), values.cpu().numpy()

            policy = board_game_cpp.mcts(
                game.board, game.to_play, evaluate,
                self.args.num_simulations, 0.0, self.args.alpha,
            )
        return max(game.valid_actions(), key=lambda a: policy[a])


########
# Main #
########
def main(args: argparse.Namespace) -> Player:
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()

    board_game_cpp.select_game("pisqorky")

    if args.recodex:
        agent = Agent.load(args.model_path, args)
    else:
        agent = train(args)

    return Player(agent, args)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    player = main(main_args)

    # Quick sanity evaluation against the heuristic.
    npfl139.board_games.evaluate(
        Pisqorky,
        [player, Pisqorky.player_from_name("heuristic")(seed=main_args.seed)],
        games=10, first_chosen=False, render=False, verbose=True,
    )
