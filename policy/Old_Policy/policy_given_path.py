"""Replay a recorded episode.

Loads a JSON trace (same format produced during RL training) and, at each call,
returns the `action_executed` recorded for the matching step. Lets you re-run
and visualise a past episode through `policy_tester.py`.

Trace path resolution order:
  1. environment variable TRACE_PATH
  2. sys.argv[1] when running a script that forwards CLI args
"""

import json
import os
import sys


def _resolve_trace_path():
    env_path = os.environ.get("TRACE_PATH")
    if env_path:
        return env_path
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json") and os.path.exists(sys.argv[1]):
        return sys.argv[1]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, "timeup_ep68222.json")


TRACE_PATH = _resolve_trace_path()
with open(TRACE_PATH) as f:
    _DATA = json.load(f)

# Metadata exposed so `policy_tester.py` can configure the env to match the trace.
MAP_NAME = _DATA["map_name"]
AGENT_NUM = _DATA["agent_num"]
STARTS = list(_DATA["starts"])
GOALS = list(_DATA["goals"])
EPISODE = _DATA.get("episode")

_ACTIONS = [step["action_executed"] for step in _DATA["trace"]]

_last_episode = None
_step_idx = 0


def policy(obs, env):
    """Return the next recorded joint action.

    Resets the step pointer at the start of each new episode (detected via
    `env.episode_account`). If the trace is exhausted, every agent waits in
    place by returning their current node.
    """
    global _last_episode, _step_idx

    if env.episode_account != _last_episode:
        _last_episode = env.episode_account
        _step_idx = 0

    if _step_idx < len(_ACTIONS):
        actions = list(_ACTIONS[_step_idx])
    else:
        actions = list(env.current_start)

    _step_idx += 1
    return actions
