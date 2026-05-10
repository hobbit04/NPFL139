#!/usr/bin/env python3
import argparse
import collections

import numpy as np
import torch

import npfl139
npfl139.require_version("2526.11.2")
from npfl139.board_games import AZQuiz

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.3, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=256, type=int, help="Number of game positions to train on.")
parser.add_argument("--epsilon", default=0.25, type=float, help="MCTS exploration epsilon in root")
parser.add_argument("--evaluate_each", default=1, type=int, help="Evaluate each number of iterations.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="az_quiz.pt", type=str, help="Model path")
parser.add_argument("--num_simulations", default=50, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--replay_buffer_length", default=50_000, type=int, help="Replay buffer max length.")
parser.add_argument("--sampling_moves", default=10, type=int, help="Sampling moves.")
parser.add_argument("--show_sim_games", default=False, action="store_true", help="Show simulated games.")
parser.add_argument("--sim_games", default=50, type=int, help="Simulated games to generate in every iteration.")
parser.add_argument("--train_for", default=20, type=int, help="Update steps in every iteration.")
parser.add_argument("--filters", default=16, type=int, help="Number of filters in CNN backbone")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="L2 Norm parameter")



#########
# Agent #
#########
class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class Network(torch.nn.Module):
        def __init__(self, args: argparse.Namespace):
            super().__init__()
            filters = args.filters

            self.backbone = torch.nn.Sequential(
                torch.nn.Conv2d(AZQuiz.C, filters, kernel_size=3, padding=1), torch.nn.ReLU(),
                torch.nn.Conv2d(filters, filters, kernel_size=3, padding=1), torch.nn.ReLU(),
                torch.nn.Conv2d(filters, filters, kernel_size=3, padding=1), torch.nn.ReLU(),
                torch.nn.Conv2d(filters, filters, kernel_size=3, padding=1), torch.nn.ReLU(),
                torch.nn.Conv2d(filters, filters, kernel_size=3, padding=1), torch.nn.ReLU(),
            )
            
            self.policy_head = torch.nn.Sequential(
                torch.nn.Conv2d(filters, 2, kernel_size=3, padding=1), torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(2 * AZQuiz.N * AZQuiz.N, AZQuiz.ACTIONS),
            )
            
            self.value_head = torch.nn.Sequential(
                torch.nn.Conv2d(filters, 2, kernel_size=3, padding=1), torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(2 * AZQuiz.N * AZQuiz.N, 1),
                torch.nn.Tanh(),
            )
        def forward(self, x):
            feature = self.backbone(x)
            policy = self.policy_head(feature)
            value = self.value_head(feature).squeeze(1)

            return policy, value
        
    def __init__(self, args: argparse.Namespace):
        # TODO: Define an agent network in `self._model`.
        #
        # A possible architecture known to work consists of
        # - 5 convolutional layers with 3x3 kernel and 15-20 filters,
        # - a policy head, which first uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens the representation, and finally uses a dense layer to produce
        #   the policy logits,
        # - a value head, which again uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens, and produces expected return using an output dense layer with
        #   `tanh` activation.
        self._model = self.Network(args=args)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> "Agent":
        # A static method returning a new Agent loaded from the given path.
        agent = Agent(args)
        agent._model.load_state_dict(torch.load(path, map_location=agent.device))
        return agent

    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    @npfl139.typed_torch_function(device, torch.float32, torch.float32, torch.float32)
    def train(self, boards: torch.Tensor, target_policies: torch.Tensor, target_values: torch.Tensor) -> tuple[np.ndarray, np.ndarray]: 
        # TODO: Train the model based on given boards, target policies and target values.
        # Note that the model returns logits.
        boards = boards.permute(0, 3, 1, 2)
        self._model.train()
        self._optimizer.zero_grad()
        policy_logits, values = self._model(boards)
        log_p = torch.nn.functional.log_softmax(policy_logits, dim=1)

        policy_loss = -(target_policies * log_p).sum(dim=1).mean()
        value_loss = torch.nn.functional.mse_loss(values, target_values)
        loss = policy_loss + value_loss
        loss.backward()
        self._optimizer.step()

        return policy_loss, value_loss

    @npfl139.typed_torch_function(device, torch.float32)
    def predict(self, boards: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return the predicted policy and the value function. Because the model
        # returns logits, you should apply softmax to return policy probabilities.
        boards = boards.unsqueeze(0)  
        boards = boards.permute(0, 3, 1, 2)
        self._model.eval()
        with torch.no_grad():
            policy_logits, value = self._model(boards)
        policy = torch.softmax(policy_logits, dim=1).squeeze(0)
        value = value.squeeze()
    
        return policy, value
    
    def board_features(self, game: AZQuiz) -> np.ndarray:
        # TODO: Generate the boards from the current game.
        #
        # The `game.board_features` returns a board representation, but you also
        # need to somehow indicate who is the current player. You can either
        # - change the game so that the current player is always the same one
        #   (i.e., always 0 or always 1; `swap_players` option of `AZQuiz.clone`
        #   method might come handy);
        # - indicate the current player by adding channels to the representation.
        if game.to_play == 0:
            return game.board_features
        else:
            return game.clone(swap_players=True).board_features 


########
# MCTS #
########
class MCTNode:
    def __init__(self, prior: float | None):
        self.prior = prior  # Prior probability from the agent.
        self.game = None    # If the node is evaluated, the corresponding game instance.
        self.children = {}  # If the node is evaluated, mapping of valid actions to the child `MCTNode`s.
        self.visit_count = 0
        self.total_value = 0

    def value(self) -> float:
        # TODO: Return the value of the current node, handling the
        # case when `self.visit_count` is 0.
       
        # total_value / visit_count 반환
        # 분모가 0인 경우 0 반환
        _value = 0 if self.visit_count == 0 else self.total_value / self.visit_count
        return _value

    def is_evaluated(self) -> bool:
        # A node is evaluated if it has non-zero `self.visit_count`.
        # In such case `self.game` is not None.
        return self.visit_count > 0

    def evaluate(self, game: AZQuiz, agent: Agent) -> None:
        # Each node can be evaluated at most once
        assert self.game is None
        self.game = game

        # TODO: Compute the value of the current game.
        # - If the game has ended, compute the value directly
        # - Otherwise, use the given `agent` to evaluate the current
        #   game. Then, for all valid actions, populate `self.children` with
        #   new `MCTNodes` with the priors from the policy predicted
        #   by the network.
        if game.outcome() is not None:
            value = -1.0 if game.outcome() == game.Outcome.WIN else 1.0
        else:
            # 신경망으로 (p, v) 계산
            policy, value = agent.predict(agent.board_features(game))
            # valid action에 대해서만 자식 노드 생성
            for action in game.valid_actions():
                self.children[action] = MCTNode(prior=policy[action]) 

        self.total_value = value
        self.visit_count = 1

    def add_exploration_noise(self, epsilon: float, alpha: float) -> None:
        # TODO: Update the children priors by exploration noise
        # Dirichlet(alpha), so that the resulting priors are
        #   epsilon * Dirichlet(alpha) + (1 - epsilon) * original_prior
        noise = np.random.dirichlet([alpha] * len(self.children))
        for i, child in enumerate(self.children.values()):
            child.prior = epsilon * noise[i] + (1 - epsilon) * child.prior

    def select_child(self) -> tuple[int, "MCTNode"]:
        # Select a child according to the PUCT formula.
        def ucb_score(child: "MCTNode"):
            # TODO: For a given child, compute the UCB score as
            #   Q(s, a) + C(s) * P(s, a) * (sqrt(N(s)) / (N(s, a) + 1)),
            # where:
            # - Q(s, a) is the estimated value of the action stored in the
            #   `child` node. However, the value in the `child` node is estimated
            #   from the view of the player playing in the `child` node, which
            #   is usually the other player than the one playing in `self`,
            #   and in that case the estimated value must be "inverted";
            # - C(s) in AlphaZero is defined as
            #     log((1 + N(s) + 19652) / 19652) + 1.25
            #   Personally I used 1965.2 to account for shorter games, but I do not
            #   think it makes any difference;
            # - P(s, a) is the prior computed by the agent;
            # - N(s) is the number of visits of state `s`;
            # - N(s, a) is the number of visits of action `a` in state `s`.
            C = np.log((1 + self.visit_count + 19652) / 19652) + 1.25
            Q = -child.value()  # 자식은 상대 플레이어 기준이므로 부호 반전
            U = C * child.prior * (np.sqrt(self.visit_count) / (child.visit_count + 1))
            return Q + U

        # TODO: Return the (action, child) pair with the highest `ucb_score`.
        return max(self.children.items(), key=lambda x: ucb_score(x[1]))

def mcts(game: AZQuiz, agent: Agent, args: argparse.Namespace, explore: bool) -> np.ndarray:
    # Run the MCTS search and return the policy proportional to the visit counts,
    # optionally including exploration noise to the root children.
    root = MCTNode(None)
    root.evaluate(game, agent)
    if explore:
        root.add_exploration_noise(args.epsilon, args.alpha)

    # Perform the `args.num_simulations` number of MCTS simulations.
    for _ in range(args.num_simulations):
        # TODO: Starting in the root node, traverse the tree using `select_child()`,
        # until a `node` without `children` is found.
        # 자식을 선택한 경로 즉 (action, child) pair 들을 저장해야 함. for Backpropagation step
        # select_child()는 직계 자손만 결정하는 함수. 끝까지 내려가려면 여기서 반복해야 함.

        mcts_path = [root]  # List for saving the path of the selection
        node = root
        while node.children:
            parent_action, node = node.select_child()
            mcts_path.append(node)

        # If the node has not been evaluated, evaluate it.
        if not node.is_evaluated():  # It means that it wasn't visited yet -> So we have to do Expansion part
            # TODO: Evaluate the `node` using the `evaluate` method. To that
            # end, create a suitable `AZQuiz` instance for this node by cloning
            # the `game` from its parent and performing a suitable action.
            parent = mcts_path[-2]
            
            game = parent.game.clone()
            game.move(parent_action)
            node.evaluate(game, agent)
            

        else:
            # TODO: If the node has been evaluated but has no children, the
            # game ends in this node. Update it appropriately.
            pass

        # Get the value of the node.
        value = node.value()

        # TODO: For all parents of the `node`, update their value estimate,
        # i.e., the `visit_count` and `total_value`.
        for parent_node in mcts_path:  # 순서는 상관 없음
            value = -value
            parent_node.total_value += value
            parent_node.visit_count += 1

    # TODO: Compute a policy proportional to visit counts of the root children.
    # Note that invalid actions are not the children of the root, but the
    # policy should still return 0 for them.
    policy = np.zeros(game.ACTIONS)
    for action, child in root.children.items():
        policy[action] = child.visit_count
    policy /= policy.sum()
    return policy


############
# Training #
############
ReplayBufferEntry = collections.namedtuple("ReplayBufferEntry", ["board", "policy", "outcome"])


def sim_game(agent: Agent, args: argparse.Namespace) -> list[ReplayBufferEntry]:
    # Simulate a game, return a list of `ReplayBufferEntry`s.
    game = AZQuiz()
    move_count = 0
    entries = []
    while not game.outcome():
        # TODO: Run the `mcts` with exploration.
        board = agent.board_features(game)  # 항상 player 0 기준으로 저장. 
        policy = mcts(game, agent, args, explore=True)
        entries.append((board, policy, game.to_play))

        # TODO: Select an action, either by sampling from the policy or greedily,
        # according to the `args.sampling_moves`.
        if move_count < args.sampling_moves:
            action = np.random.choice(len(policy), p=policy)
        else:
            action = np.argmax(policy)

        game.move(action)
        move_count += 1

    # TODO: Return all encountered game states, each consisting of
    # - the board (probably via `agent.board_features`),
    # - the policy obtained by MCTS,
    # - the outcome based on the outcome of the whole game.
    result = []
    for board, policy, player in entries:
        outcome = game.outcome(player=player)  # Game has ended
        result.append(
            ReplayBufferEntry(
                board, 
                policy, 
                1.0 if outcome==game.Outcome.WIN else -1.0  # There is no Draw in AZQuiz!
            )
        )

    return result

def train(args: argparse.Namespace) -> Agent:
    # Perform training
    agent = Agent(args)
    replay_buffer = npfl139.ReplayBuffer(max_length=args.replay_buffer_length)

    iteration = 0
    training = True
    while training:
        iteration += 1

        # Generate simulated games
        for _ in range(args.sim_games):
            game = sim_game(agent, args)
            replay_buffer.extend(game)

            # If required, show the generated game, as 8 very long lines showing
            # all encountered boards, each field showing as
            # - `XX` for the fields belonging to player 0,
            # - `..` for the fields belonging to player 1,
            # - percentage of visit counts for valid actions.
            if args.show_sim_games:
                log = [[] for _ in range(8)]
                for i, (board, policy, outcome) in enumerate(game):
                    log[0].append(f"Move {i}, result {outcome}".center(28))
                    action = 0
                    for row in range(7):
                        log[1 + row].append("  " * (6 - row))
                        for col in range(row + 1):
                            log[1 + row].append(
                                " XX " if board[row, col, 0] else
                                " .. " if board[row, col, 1] else
                                f"{policy[action] * 100:>3.0f} ")
                            action += 1
                        log[1 + row].append("  " * (6 - row))
                print(*["".join(line) for line in log], sep="\n")

        # Train
        for _ in range(args.train_for):
            # TODO: Perform training by sampling an `args.batch_size` of positions
            # from the `replay_buffer` and running `agent.train` on them.
            batch = replay_buffer.sample(args.batch_size, replace=False)
            policy_loss, value_loss = agent.train(*batch)  # Not 100% sure but...
        print(f"Iteration {iteration}: policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}")

        # Evaluate
        if iteration % args.evaluate_each == 0:
            # Run an evaluation on 2*56 games versus the simple heuristics,
            # using the `Player` instance defined below.
            # For speed, the implementation does not use MCTS during evaluation,
            # but you can of course change it so that it does.
            score = npfl139.board_games.evaluate(
                AZQuiz, [Player(agent, argparse.Namespace(num_simulations=0)),
                         AZQuiz.player_from_name("simple_heuristic")(seed=main_args.seed)],
                games=56, first_chosen=False, render=False, verbose=False,
            )
            print(f"Evaluation after iteration {iteration}: {100 * score:.1f}%", flush=True)
            if score > 0.9:
                print("Score passed 90%. Saving the model..")
                training = False
    agent.save(args.model_path)
    return agent


#############################
# BoardGamePlayer interface #
#############################
class Player(npfl139.board_games.BoardGamePlayer[AZQuiz]):
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args

    def play(self, game: AZQuiz) -> int:
        # Predict a best possible action.
        if self.args.num_simulations == 0:
            # TODO: If no simulations should be performed, use directly
            # the policy predicted by the agent on the current game board.
            policy, _ = self.agent.predict(self.agent.board_features(game))
        else:
            # TODO: Otherwise run the `mcts` without exploration and
            # utilize the policy returned by it.
            policy = mcts(game, self.agent, self.args, explore=False)

        # Now select a valid action with the largest probability.
        return max(game.valid_actions(), key=lambda action: policy[action])


########
# Main #
########
def main(args: argparse.Namespace) -> Player:
    # Set the random seed and the number of threads.
    npfl139.startup(args.seed, args.threads)
    npfl139.global_keras_initializers()  # Use Keras-style Xavier parameter initialization.

    if args.recodex:
        # Load the trained agent
        agent = Agent.load(args.model_path, args)
    else:
        # Perform training
        agent = train(args)

    return Player(agent, args)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    player = main(main_args)

    # Run an evaluation versus the simple heuristic with the same parameters as in ReCodEx.
    npfl139.board_games.evaluate(
        AZQuiz, [player, AZQuiz.player_from_name("simple_heuristic")(seed=main_args.seed)],
        games=56, first_chosen=False, render=False, verbose=True,
    )
