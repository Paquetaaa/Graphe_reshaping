"""Compare A* node expansions on the original weighted graph G versus
the reshaped unit-time graph G'.

Motivation: the reshape introduces intermediate nodes that make paths
longer (in number of nodes) by a factor of up to k = ceil(w_max / sigma).
This script quantifies the depth blow-up empirically: for each (start, goal)
pair of every agent in every benchmark instance, we run unconstrained A*
on both G and G' and report the mean expansion ratio per map.

Usage:
    python astar_expansions_comparison.py [--out astar_expansions.json]
"""

import argparse
import heapq
import json
import math
from collections import defaultdict
from datetime import datetime

import gym

import policy.Old_Policy.policy_CBS_Cardinal_Conflict_Picking as cbs_module
import problem.problems as problems


def instance_to_map_label(instance_id):
    if 1 <= instance_id <= 10:
        return "Map 1 (3x3 grid)"
    elif 11 <= instance_id <= 20:
        return "Map 2 (Aoba01)"
    elif 21 <= instance_id <= 30:
        return "Map 3 (Shibuya)"
    return f"Unknown ({instance_id})"


def euclidean(pos, u, v):
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def astar_unconstrained(graph, pos, start, goal):
    """Standard A* with Euclidean heuristic. Returns (path_length_in_nodes,
    expansions_count). path_length_in_nodes is the number of vertices in
    the discovered path including endpoints (None if unreachable)."""
    if start == goal:
        return 1, 0

    counter = 0
    h0 = euclidean(pos, start, goal)
    # (f, counter, node, g, parent)
    open_list = [(h0, counter, start, 0.0, None)]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()
    expansions = 0

    while open_list:
        f, _, node, g, parent = heapq.heappop(open_list)
        if node in closed:
            continue
        closed.add(node)
        expansions += 1
        if parent is not None:
            came_from[node] = parent
        if node == goal:
            # Reconstruct path length
            length = 1
            cur = node
            while cur in came_from:
                cur = came_from[cur]
                length += 1
            return length, expansions
        for nb in graph.neighbors(node):
            edge_w = graph[node][nb].get('weight', 1.0)
            tentative = g + edge_w
            if tentative < g_score.get(nb, float('inf')):
                g_score[nb] = tentative
                counter += 1
                heapq.heappush(open_list, (tentative + euclidean(pos, nb, goal),
                                          counter, nb, tentative, node))
    return None, expansions


def measure_instance(instance):
    """Build env, capture both G and G', run unconstrained A* per agent on both
    graphs, return per-agent expansion records."""
    map_name = instance["map"]
    drone_num = instance["drone_num"]
    start_arr = instance["start"]
    goal_arr = instance["goal"]
    instance_id = instance["id"]

    env = gym.make(
        "drp_env:drp-" + str(drone_num) + "agent_" + map_name + "-v2",
        state_repre_flag="onehot_fov",
        goal_array=goal_arr,
        start_ori_array=start_arr,
    )
    env.reset()

    # Original graph
    G_orig = env.G.copy()
    pos_orig = dict(env.pos)

    # Build reshaped graph
    G_reshaped = cbs_module.reshape_graph_from_G(env, G_orig, pos_orig)
    pos_reshaped = dict(env.pos)  # reshape_graph_from_G mutates env.pos

    records = []
    for agent in range(drone_num):
        s = start_arr[agent]
        g = goal_arr[agent]
        len_G, exp_G   = astar_unconstrained(G_orig,     pos_orig,     s, g)
        len_R, exp_R   = astar_unconstrained(G_reshaped, pos_reshaped, s, g)
        records.append({
            "instance_id": instance_id,
            "agent": agent,
            "start": s,
            "goal": g,
            "path_len_G":  len_G,
            "path_len_Gp": len_R,
            "expansions_G":  exp_G,
            "expansions_Gp": exp_R,
            "expansion_ratio": (exp_R / exp_G) if exp_G else None,
            "depth_ratio":     (len_R / len_G) if (len_G and len_R) else None,
        })

    del env
    return records


def aggregate_per_map(records):
    """Compute mean expansions on G vs G' and the mean ratio per map."""
    by_map = defaultdict(lambda: {
        "n_pairs": 0,
        "sum_exp_G": 0,
        "sum_exp_Gp": 0,
        "sum_len_G": 0,
        "sum_len_Gp": 0,
        "sum_exp_ratio": 0.0,
        "sum_depth_ratio": 0.0,
        "n_valid_ratio": 0,
    })
    for r in records:
        m = instance_to_map_label(r["instance_id"])
        by_map[m]["n_pairs"]  += 1
        by_map[m]["sum_exp_G"]  += r["expansions_G"]
        by_map[m]["sum_exp_Gp"] += r["expansions_Gp"]
        if r["path_len_G"] is not None:
            by_map[m]["sum_len_G"] += r["path_len_G"]
        if r["path_len_Gp"] is not None:
            by_map[m]["sum_len_Gp"] += r["path_len_Gp"]
        if r["expansion_ratio"] is not None:
            by_map[m]["sum_exp_ratio"]   += r["expansion_ratio"]
            by_map[m]["sum_depth_ratio"] += r["depth_ratio"]
            by_map[m]["n_valid_ratio"]   += 1

    rows = []
    for m, d in by_map.items():
        n = d["n_pairs"]
        nr = max(d["n_valid_ratio"], 1)
        rows.append({
            "map": m,
            "n_pairs":         n,
            "mean_exp_G":      d["sum_exp_G"]  / n,
            "mean_exp_Gp":     d["sum_exp_Gp"] / n,
            "mean_len_G":      d["sum_len_G"]  / n,
            "mean_len_Gp":     d["sum_len_Gp"] / n,
            "mean_exp_ratio":  d["sum_exp_ratio"]   / nr,
            "mean_depth_ratio": d["sum_depth_ratio"] / nr,
        })
    return rows


def print_summary(rows):
    print()
    print("=" * 95)
    print(f"{'Map':<22} {'#pairs':>7} {'exp G':>8} {'exp Gp':>8} "
          f"{'ratio':>7} {'len G':>7} {'len Gp':>8} {'depth':>7}")
    print("-" * 95)
    for r in rows:
        print(f"{r['map']:<22} {r['n_pairs']:>7} "
              f"{r['mean_exp_G']:>8.1f} {r['mean_exp_Gp']:>8.1f} "
              f"{r['mean_exp_ratio']:>7.2f} "
              f"{r['mean_len_G']:>7.1f} {r['mean_len_Gp']:>8.1f} "
              f"{r['mean_depth_ratio']:>7.2f}")
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="astar_expansions.json")
    args = parser.parse_args()

    all_records = []
    for instance in problems.instances:
        print(f"Instance {instance['id']} ({instance['map']}, "
              f"{instance['drone_num']} drones)")
        recs = measure_instance(instance)
        all_records.extend(recs)
        for r in recs:
            print(f"  agent {r['agent']}: G exp={r['expansions_G']} "
                  f"len={r['path_len_G']}  |  G' exp={r['expansions_Gp']} "
                  f"len={r['path_len_Gp']}  |  ratio={r['expansion_ratio']}")

    rows = aggregate_per_map(all_records)
    print_summary(rows)

    with open(args.out, "w") as f:
        json.dump({
            "scored_at": datetime.now().isoformat(),
            "per_agent": all_records,
            "per_map":   rows,
        }, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
