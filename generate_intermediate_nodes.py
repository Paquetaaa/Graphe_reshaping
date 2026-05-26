import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")  # MapMake creates a matplotlib figure; avoid needing a display.

from drp_env.EE_map import MapMake
from Graph_with_int_nodes import reshape_graph_from_G

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_DIR = os.path.join(THIS_DIR, "drp_env", "map")
OUT_DIR = os.path.join(THIS_DIR, "drp_env", "map_intermediate_nodes")
DEFAULT_SPEED = 5  # matches drp_env/__init__.py


class _Env:
    """Minimal stand-in for DrpEnv: reshape_graph_from_G only reads `speed` and writes `pos`."""

    def __init__(self, speed):
        self.speed = speed
        self.pos = None


def generate_for_map(map_name, speed=DEFAULT_SPEED):
    # MapMake reads ./map/<name>/{node,edge}.csv relative to drp_env/.
    mm = MapMake(agent_num=1, start_ori_array=[0], goal_array=[1], map_name=map_name)

    env = _Env(speed=speed)
    env.pos = dict(mm.pos)
    G_new = reshape_graph_from_G(env, mm.G, mm.pos)

    out_subdir = os.path.join(OUT_DIR, map_name)
    os.makedirs(out_subdir, exist_ok=True)
    out_path = os.path.join(out_subdir, "node.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "x", "y", "type"])
        for n, data in G_new.nodes(data=True):
            x, y = env.pos[n]
            writer.writerow([n, x, y, data.get("type", "original")])

    n_orig = sum(1 for _, d in G_new.nodes(data=True) if d.get("type") == "original")
    n_inter = sum(1 for _, d in G_new.nodes(data=True) if d.get("type") == "intermediate")
    return out_path, n_orig, n_inter


def main():
    if not os.path.isdir(MAP_DIR):
        sys.exit(f"map directory not found: {MAP_DIR}")

    map_names = sorted(
        name for name in os.listdir(MAP_DIR)
        if os.path.isdir(os.path.join(MAP_DIR, name)) and name.startswith("map_")
    )

    for map_name in map_names:
        try:
            out_path, n_orig, n_inter = generate_for_map(map_name)
            print(f"[ok] {map_name}: {n_orig} original + {n_inter} intermediate -> {out_path}")
        except Exception as e:
            print(f"[fail] {map_name}: {e}")


if __name__ == "__main__":
    main()
