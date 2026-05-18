// This file is part of NPFL139 <http://github.com/ufal/npfl139/>.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "board_game.h"

template<BoardGame G>
using Policy = std::array<float, G::ACTIONS>;

template<BoardGame G>
using Evaluator = std::function<void(const G&, Policy<G>&, float&)>;

template<BoardGame G>
void mcts(const G& game, const Evaluator<G>& evaluator, int num_simulations, float epsilon, float alpha, Policy<G>& policy) {
  // TODO: Implement MCTS, returning the generated `policy`.
  //
  // To run the neural network, use the given `evaluator`, which returns a policy and
  // a value function for the given game.

  struct Node {
    float prior = 0.0f;
    int visit_count = 0;
    float total_value = 0.0f;
    bool evaluated = false;
    G game;
    std::array<int, G::ACTIONS> children;

    Node() { children.fill(-1); }

    float value() const {
      return visit_count == 0 ? 0.0f : total_value / visit_count;
    }

    bool has_children() const {
      for (auto c : children) if (c >= 0) return true;
      return false;
    }
  };

  // Pool-based node allocator: indices remain stable across push_back after reserve().
  std::vector<Node> pool;
  pool.reserve(num_simulations * 4 + G::ACTIONS + 1);
  pool.emplace_back();  // root at index 0

  // Evaluate a node: call the neural network (or detect terminal), populate children.
  auto evaluate_node = [&](int idx, const G& g) {
    pool[idx].evaluated = true;
    pool[idx].game = g;

    Outcome result = g.outcome();
    if (result != Outcome::UNFINISHED) {
      float val = (result == Outcome::WIN) ? 1.0f : (result == Outcome::LOSS) ? -1.0f : 0.0f;
      pool[idx].visit_count = 1;
      pool[idx].total_value = val;
    } else {
      Policy<G> net_policy{};
      float val = 0.0f;
      evaluator(g, net_policy, val);

      // Normalize prior over valid actions only.
      float sum = 0.0f;
      int valid_count = 0;
      for (int a = 0; a < G::ACTIONS; a++) {
        if (g.valid(a)) { sum += net_policy[a]; valid_count++; }
      }

      // Reserve before loop to avoid reallocation invalidating pool[idx].
      pool.reserve(pool.size() + valid_count);

      for (int a = 0; a < G::ACTIONS; a++) {
        if (g.valid(a)) {
          int child_idx = (int)pool.size();
          pool[idx].children[a] = child_idx;
          pool.emplace_back();
          pool.back().prior = (sum > 0.0f) ? net_policy[a] / sum : (1.0f / valid_count);
        }
      }

      pool[idx].visit_count = 1;
      pool[idx].total_value = val;
    }
  };

  // Evaluate root.
  evaluate_node(0, game);

  // Add Dirichlet exploration noise to root children.
  if (epsilon > 0.0f && alpha > 0.0f && pool[0].has_children()) {
    std::vector<int> valid_actions;
    for (int a = 0; a < G::ACTIONS; a++)
      if (pool[0].children[a] >= 0) valid_actions.push_back(a);

    std::gamma_distribution<float> gamma_dist(alpha, 1.0f);
    std::vector<float> noise(valid_actions.size());
    float noise_sum = 0.0f;
    for (auto& n : noise) { n = gamma_dist(*board_game_generator); noise_sum += n; }
    if (noise_sum > 0.0f) {
      for (size_t i = 0; i < valid_actions.size(); i++) {
        int child_idx = pool[0].children[valid_actions[i]];
        pool[child_idx].prior = epsilon * (noise[i] / noise_sum) + (1.0f - epsilon) * pool[child_idx].prior;
      }
    }
  }

  // Run MCTS simulations.
  for (int sim = 0; sim < num_simulations; sim++) {
    // Selection: follow best UCB child until reaching an unevaluated or terminal node.
    std::vector<int> path{0};
    int cur = 0;
    int last_action = -1;

    while (pool[cur].has_children()) {
      // PUCT formula: Q(s,a) + C(s) * P(s,a) * sqrt(N(s)) / (N(s,a) + 1)
      float C = std::log((1.0f + pool[cur].visit_count + 19652.0f) / 19652.0f) + 1.25f;
      float best_score = -std::numeric_limits<float>::infinity();
      int best_action = -1;

      for (int a = 0; a < G::ACTIONS; a++) {
        int child_idx = pool[cur].children[a];
        if (child_idx < 0) continue;

        // Child value is from the opponent's POV, so negate for current player.
        float Q = -pool[child_idx].value();
        float U = C * pool[child_idx].prior * std::sqrt((float)pool[cur].visit_count)
                  / (pool[child_idx].visit_count + 1.0f);

        if (Q + U > best_score) {
          best_score = Q + U;
          best_action = a;
        }
      }

      last_action = best_action;
      cur = pool[cur].children[best_action];
      path.push_back(cur);
    }

    // Expansion: evaluate the newly reached node, or update a terminal revisit.
    if (!pool[cur].evaluated) {
      int parent_idx = path[(int)path.size() - 2];
      G new_game = pool[parent_idx].game;
      new_game.move(last_action);
      evaluate_node(cur, new_game);
    } else {
      // Terminal node revisit: keep average unchanged.
      pool[cur].total_value += pool[cur].value();
      pool[cur].visit_count++;
    }

    // Backpropagation: update all ancestors above the leaf.
    float value = pool[cur].value();
    for (int i = (int)path.size() - 2; i >= 0; i--) {
      value = -value;  // alternate sign at each depth level
      pool[path[i]].total_value += value;
      pool[path[i]].visit_count++;
    }
  }

  // Build policy proportional to root children visit counts.
  policy.fill(0.0f);
  float total = 0.0f;
  for (int a = 0; a < G::ACTIONS; a++) {
    int child_idx = pool[0].children[a];
    if (child_idx >= 0) {
      policy[a] = (float)pool[child_idx].visit_count;
      total += policy[a];
    }
  }
  if (total > 0.0f) {
    for (auto& p : policy) p /= total;
  } else {
    // Fallback for num_simulations == 0: use network prior directly.
    int valid_count = 0;
    for (int a = 0; a < G::ACTIONS; a++) if (game.valid(a)) valid_count++;
    if (valid_count > 0)
      for (int a = 0; a < G::ACTIONS; a++)
        if (game.valid(a)) policy[a] = 1.0f / valid_count;
  }
}
