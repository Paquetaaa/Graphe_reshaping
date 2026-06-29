"""Evaluate CBS conflict-class statistics across all benchmark instances.

For each problem instance, runs CBS with a wall-clock timeout and collects:
- counts of cardinal / semi-cardinal / non-cardinal conflicts encountered
- number of CT iterations
- elapsed seconds
- termination reason ('solved' | 'timeout' | 'infeasible_root')

Then aggregates per map (Map 1 = 3x3 grid, Map 2 = Aoba01, Map 3 = Shibuya)
and prints a cardinal-conflict ratio table suitable for the paper.

Usage:
    python cbs_stats_evaluation.py [--timeout 30]

Claude genrated
"""



import argparse
import json
import gym
from collections import defaultdict
from datetime import datetime

import policy.Old_Policy.policy_CBS_Cardinal_Conflict_Picking as cbs_module
import problem.problems as problems


# Map instance_id -> map label. Adjust if benchmark layout changes.
def instance_to_map_label(instance_id):
    if 1 <= instance_id <= 10:
        return "Map 1 (3x3 grid)"
    elif 11 <= instance_id <= 20:
        return "Map 2 (Aoba01)"
    elif 21 <= instance_id <= 30:
        return "Map 3 (Shibuya)"
    else:
        return f"Unknown ({instance_id})"


def run_one_instance(instance, time_limit_seconds):
    """Build env, reshape, run CBS once with timeout, return stats dict."""
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

    # Preserve original graph and reshape (same preprocessing as policy.init)
    env.G_original = env.G.copy()
    env.pos_original = dict(env.pos)
    env.G = cbs_module.reshape_graph_from_G(env, env.G_original, env.pos_original)

    _, stats = cbs_module.cbs(env, time_limit_seconds=time_limit_seconds, return_stats=True)
    stats["instance_id"] = instance_id
    stats["map"] = instance_to_map_label(instance_id)
    stats["drone_num"] = drone_num

    del env
    return stats


def aggregate_per_map(all_stats):
    """Sum conflict counts per map and compute the cardinal ratio."""
    by_map = defaultdict(lambda: {
        "cardinal": 0,
        "semi-cardinal": 0,
        "non-cardinal": 0,
        "iterations": 0,
        "instances": 0,
        "solved": 0,
        "timeout": 0,
    })
    for s in all_stats:
        m = s["map"]
        by_map[m]["cardinal"]      += s.get("cardinal", 0)
        by_map[m]["semi-cardinal"] += s.get("semi-cardinal", 0)
        by_map[m]["non-cardinal"]  += s.get("non-cardinal", 0)
        by_map[m]["iterations"]    += s.get("iterations", 0)
        by_map[m]["instances"]     += 1
        if s.get("terminated") == "solved":
            by_map[m]["solved"] += 1
        elif s.get("terminated") == "timeout":
            by_map[m]["timeout"] += 1

    rows = []
    for m, d in by_map.items():
        total_conflicts = d["cardinal"] + d["semi-cardinal"] + d["non-cardinal"]
        ratio = d["cardinal"] / total_conflicts if total_conflicts > 0 else 0.0
        rows.append({
            "map": m,
            "instances": d["instances"],
            "solved": d["solved"],
            "timeout": d["timeout"],
            "total_iters": d["iterations"],
            "cardinal": d["cardinal"],
            "semi": d["semi-cardinal"],
            "non": d["non-cardinal"],
            "total_conflicts": total_conflicts,
            "cardinal_ratio": ratio,
        })
    return rows


def print_summary(rows):
    print()
    print("=" * 90)
    print(f"{'Map':<22} {'#inst':>6} {'solved':>7} {'TO':>4} {'iters':>8} "
          f"{'card':>7} {'semi':>7} {'non':>7} {'card%':>7}")
    print("-" * 90)
    for r in rows:
        print(f"{r['map']:<22} {r['instances']:>6} {r['solved']:>7} {r['timeout']:>4} "
              f"{r['total_iters']:>8} {r['cardinal']:>7} {r['semi']:>7} {r['non']:>7} "
              f"{100*r['cardinal_ratio']:>6.1f}%")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Wall-clock seconds per instance (default: 30)")
    parser.add_argument("--out", type=str, default="cbs_stats.json",
                        help="Output JSON path (default: cbs_stats.json)")
    args = parser.parse_args()

    print(f"Running CBS on {len(problems.instances)} instances "
          f"with timeout={args.timeout}s per instance.")

    all_stats = []
    for instance in problems.instances:
        print(f"\n--- Instance {instance['id']} ({instance['map']}, "
              f"{instance['drone_num']} drones) ---")
        s = run_one_instance(instance, args.timeout)
        all_stats.append(s)
        print(f"  -> {s.get('terminated')}, iters={s.get('iterations')}, "
              f"card={s.get('cardinal')}, semi={s.get('semi-cardinal')}, "
              f"non={s.get('non-cardinal')}, elapsed={s.get('elapsed', 0):.2f}s")

        # Incremental save
        with open(args.out, "w") as f:
            json.dump({
                "scored_at": datetime.now().isoformat(),
                "timeout_seconds": args.timeout,
                "completed": len(all_stats),
                "total": len(problems.instances),
                "per_instance": all_stats,
            }, f, indent=2)

    rows = aggregate_per_map(all_stats)
    print_summary(rows)

    # Final dump with aggregation
    with open(args.out, "w") as f:
        json.dump({
            "scored_at": datetime.now().isoformat(),
            "timeout_seconds": args.timeout,
            "per_instance": all_stats,
            "per_map": rows,
        }, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
