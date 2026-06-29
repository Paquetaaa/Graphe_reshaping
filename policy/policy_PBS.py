import gym
import math
import networkx as nx
import heapq
import time

### submission information ####
TEAM_NAME = "GRENECHE Lucas"
##############################

# Global Variables 
paths = {} # Dict, key = Agent_id, value = A* path
last_episode = -1
path_idx = {} # Dict, key = Agent_id, value = current node index in paths
last_node = {} # To detect when an agent has reached its next waypoint

WAIT_COST = 1  # Cost of waiting one step


## UTILITY FUNCTIONS
def h_euclidian(pos, u, v):
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    return (math.sqrt((x1 - x2) ** 2 + (y1 - y2)**2))

def path_cost(env,path,goal):
    """Calculate the cost of a path, defined as the number of steps until reaching the goal, with a penalty for waiting (staying on the same node)."""
    cost = 0
    for t in range(len(path) - 1):
        if path[t] != path[t+1]:
            cost += 1
        else:
            cost += WAIT_COST

        if path[t+1] == goal:
            break # When goal is reached, we stop counting cost
        
    return cost


def path_translation(env,result):
    """Translate paths from the expanded graph back to the original graph."""
    translated_path = {}
    for agent, path in result.items():
        translated_path[agent] = [node for node in path if not isinstance(node, str)]
    return translated_path

def reshape_graph_from_G(env, G, pos):
    """Transform a Graph into a DiGraph with intermediate nodes, spaced by env.speed. This allows to use CBS with a discretization of 1 step = 1 edge, while still respecting the continuous nature of the problem and the speed constraint."""
    speed = env.speed
    G_new = nx.DiGraph()
    pos_new = dict(pos)

    def interpolate(p1, p2, alpha):
        return (
            round(p1[0] + alpha * (p2[0] - p1[0]),4),
            round(p1[1] + alpha * (p2[1] - p1[1]),4)
        )

    # Copy of original nodes
    for n in G.nodes():
        G_new.add_node(n, type="original")

    for u, v, data in G.edges(data=True):
        w = data['weight'] # distance between u and v
        k = math.ceil((w / speed) - 1e-9) # Number of intermediate nodes needed to ensure edge traversal takes at least 1 step
        #print(f"Reshaping edge ({u}, {v}) with weight {w}: needs {k} intermediate nodes")

        if k == 1: # No intermediate nodes needed, just copy the edge
            G_new.add_edge(u, v, weight=1)
            G_new.add_edge(v, u, weight=1)
            continue

        prev = u
        for i in range(1, k): # Create intermediate nodes, if k = 3 we will create 2 intermediate nodes at 1/3 and 2/3 of the way
            new_node = f"{u}_{v}_{i}"

            G_new.add_node(new_node, type="intermediate")

            #alpha = i / k
            alpha = (i * speed) / w  # au lieu de i / k
            pos_new[new_node] = interpolate(pos_new[u],pos_new[v], alpha)
            #print(f"Creating intermediate node {new_node} between {u} and {v} at position {pos_new[new_node]}")

            G_new.add_edge(prev, new_node, weight=1)
            prev = new_node

        G_new.add_edge(prev, v, weight=1)

        prev = v
        for i in range(1, k): # Create intermediate nodes, if k = 3 we will create 2 intermediate nodes at 1/3 and 2/3 of the way
            new_node = f"{v}_{u}_{i}"
            G_new.add_node(new_node, type="intermediate")
            #alpha = i / k
            alpha = (i * speed) / w  # au lieu de i / k

            pos_new[new_node] = interpolate(pos_new[v],pos_new[u], alpha)
            #print(f"Creating intermediate node {new_node} between {v} and {u} at position {pos_new[new_node]}")

            G_new.add_edge(prev, new_node, weight=1)
            prev = new_node

        G_new.add_edge(prev, u, weight=1)

    env.pos = pos_new
    return G_new

## Priority-Based Search (PBS) implementation
def priority_based_planning(env, max_horizon=500, max_attempts=2000):
    import random
    best_paths = None
    best_count = 0
    best_cost = float('inf')
    
    for attempt in range(max_attempts):

        if attempt == 0:
            print("Test descending order", flush=True)
            # First try, order by descending distance to goal (agents with longer paths are more likely to cause conflicts, so we plan them first)
            agent_order = sorted(range(env.agent_num),
                                 key=lambda a: -h_euclidian(env.pos,
                                                            env.current_start[a],
                                                            env.goal_array[a]))
        elif attempt == 1:
            print("Test ascending order", flush=True)
            # Second try, order by ascending distance to goal (agents with shorter paths are more likely to be flexible and adapt to constraints, so we plan them later)
            agent_order = sorted(range(env.agent_num),
                                 key=lambda a: h_euclidian(env.pos,
                                                            env.current_start[a],
                                                            env.goal_array[a]))
            
        elif attempt == 2:
            print("Test centrality order", flush=True)
            # Order by start node centrality (agents starting from more central nodes are more likely to cause conflicts, so we plan them first)
            centrality = env.centrality_cache
            agent_order = sorted(range(env.agent_num),
                                key=lambda a: -centrality.get(env.current_start[a], 0))

        elif attempt == 3:
            sp = env.shortest_paths_cache
            score = {}
            for a in range(env.agent_num):
                score[a] = sum(len(sp[a] & sp[b]) for b in range(env.agent_num) if b != a)
            agent_order = sorted(range(env.agent_num), key=lambda a: -score[a])

        # Goal criticality : agents whose goal node has higher degree (more edges) are more likely to cause conflicts at the end, so we plan them first
        elif attempt == 4:
            print("Test Goal criticality order", flush=True)
            agent_order = sorted(range(env.agent_num),
                                key=lambda a: -env.G.degree(env.goal_array[a]))

        # Start Criticality : agents whose start node has higher degree (more edges) are more likely to cause conflicts at the beginning, so we plan them first
        elif attempt == 5:
            print("Test Start criticality order", flush=True)
            agent_order = sorted(range(env.agent_num),
                                key=lambda a: -env.G.degree(env.current_start[a]))

        # Reverse of the original order
        elif attempt == 6:
            print("Test reverse order", flush=True)
            # déjà couvert par ascending mais avec une autre logique
            agent_order = list(range(env.agent_num))[::-1]

        # Alternating between the most and least critical agents according to distance to goal
        elif attempt == 7:
            print("Test alternating order", flush=True)
            by_dist = sorted(range(env.agent_num),
                            key=lambda a: -h_euclidian(env.pos,
                                                        env.current_start[a],
                                                        env.goal_array[a]))
            agent_order = []
            while by_dist:
                agent_order.append(by_dist.pop(0))
                if by_dist:
                    agent_order.append(by_dist.pop(-1))

        # Random order
        else:
            if attempt % 50 == 0:
                print("Test random order numero", attempt, flush=True)
            
            agent_order = list(range(env.agent_num))
            random.shuffle(agent_order)

        paths_pp = {}
        constraints = set()
        success_count = 0

        for agent in agent_order:                     
            p = a_star_constrained(env, agent,
                                   env.current_start[agent],
                                   env.goal_array[agent],
                                   constraints)
            if p is None:
                print(f"[fallback] attempt {attempt} agent {agent} infeasible, staying at start", flush=True)
                paths_pp[agent] = [env.current_start[agent]]
                continue
            success_count += 1
            paths_pp[agent] = p

            for t, node in enumerate(p):
                nearby = env.proximity_cache[node]    # ← LOOKUP O(1)
                for other in range(env.agent_num):
                    if other == agent:
                        continue
                    constraints.add((other, node, t, '-'))
                    # Blocage proximity
                    for near in nearby:
                        constraints.add((other, near, t, '-'))
            # Goal protection
            goal = env.goal_array[agent]
            arrival = len(p) - 1
            for t in range(arrival, arrival + max_horizon):
                for other in range(env.agent_num):
                    if other == agent:
                        continue
                    constraints.add((other, goal, t, '-'))

        
        if success_count == env.agent_num:
            # Cost comparison only if all agents reach their goal, otherwise we might favor attempts that are very good for some agents but leave others stuck (which would be unfair since we want all agents to reach their goals)
            attempt_cost = sum(path_cost(env, paths_pp[a], env.goal_array[a])
                                for a in range(env.agent_num))
            if attempt_cost < best_cost:
                best_cost = attempt_cost
                best_paths = paths_pp
                best_count = success_count
                print(f"[FALLBACK] attempt {attempt} success_count={success_count} "
                      f"cost={attempt_cost} (NEW BEST)", flush=True)
        elif success_count > best_count:
            best_count = success_count
            best_paths = paths_pp
            print(f"[FALLBACK] attempt {attempt} success_count={success_count} "
                  f"(partial best)", flush=True)

    
    return best_paths if best_paths is not None else {}

# A* with constraints for PBS
def a_star_constrained(env, agent, start, goal, constraints):
    """A* pathfinding in env.G using parent-pointer"""

    # Dijkstra Heuristic  
    h = env.h_table[goal].get(start, float('inf'))
    
    # A*
    counter = 0
    open_list = [(h, counter, start, 0, 0)]  # (f, counter, node, real_t, g)
    #visited = set()
    parent ={(start,0): None}   # dict : (node, real_t) → (parent_node, parent_real_t)

    max_real_t = env.max_real_t

    # Build per-agent constraint set (optim 3)
    my_negs = set()
    for c in constraints:
        if c[0] == agent and c[3] == '-':
            my_negs.add((c[1], c[2]))

    # Extract goal-times depuis my_negs (au lieu de re-parcourir constraints)
    goal_constraints_times = [t for (n, t) in my_negs if n == goal]
    min_goal_arrival = max(goal_constraints_times) + 1 if goal_constraints_times else 0

    while open_list:
        f, _, node, real_t, g = heapq.heappop(open_list)

        # END CONDITION
        if node == goal and real_t >= min_goal_arrival:
            # === PATH RECONSTRUCTION ===
            path = []
            cur = (node, real_t)
            while cur is not None:
                path.append(cur[0])
                cur = parent[cur]
            return path[::-1]
        
        #if (node, real_t) in visited:
        #    continue
        #visited.add((node, real_t))

        # Expansion
        if node != goal:
            for neighbor in env.neighbor_cache[node]: 
                edge_w = round(env.edge_weight_cache[(node, neighbor)])
                new_real_t = real_t + edge_w
                new_state = (neighbor, new_real_t)
                
                if new_state in parent:
                    continue

                if new_state not in my_negs and new_real_t <= max_real_t:
                    counter += 1
                    new_g = g + env.edge_weight_cache[(node, neighbor)]
                    new_f = new_g + env.h_table[goal].get(neighbor, float('inf'))
                    parent[new_state] = (node, real_t)   # PARENT POINTER
                    heapq.heappush(open_list, (new_f, counter, neighbor, new_real_t, new_g))

        # WAIT
        if env.node_is_original[node]: # Only allow waiting at original nodes

            wait_real_t = real_t + round(WAIT_COST)
            wait_state = (node, wait_real_t)

            if wait_state not in parent:
                if wait_state not in my_negs and wait_real_t <= max_real_t:
                        counter += 1
                        new_g = g + WAIT_COST
                        new_f = new_g + env.h_table[goal].get(node, float('inf'))
                        parent[wait_state] = (node, real_t)
                        heapq.heappush(open_list, (new_f, counter, node, wait_real_t, new_g))


    # Fallback: unconstrained A* so agent still has a valid path (CBS will detect conflicts and continue)
    if start == goal:
        return [start]
    return None




def init(env):
    global last_episode, paths, path_idx, last_node
    last_episode = env.episode_account # Initialization of the episode number
    paths.clear()
    path_idx.clear()
    last_node.clear()

    print(f"\n[{time.strftime('%H:%M:%S')}] [INSTANCE] episode={env.episode_account} "
      f"agents={env.agent_num}", flush=True)
    init_start = time.time()

    # Saving the original graph and positions.
    if not hasattr(env, "G_original"):
        env.G_original = env.G.copy()
        env.pos_original = dict(env.pos)

    # Reshape the graph to transform continous problem into discrete problem for CBS.
    env.G = reshape_graph_from_G(env, env.G_original, env.pos_original)


    # Caches for fast acces during A*
    env.neighbor_cache = {node: list(env.G.neighbors(node)) for node in env.G.nodes()}
    env.edge_weight_cache = {(u, v): env.G[u][v]['weight'] for u, v in env.G.edges()}
    env.node_is_original = {node: env.G.nodes[node]['type'] == 'original' for node in env.G.nodes()}

    # Heuristic precomputation for A* (one per goal node, since we have a lot of intermediate nodes but few original nodes):
    env.h_table = {}
    for goal in set(env.goal_array):
        env.h_table[goal] = nx.shortest_path_length(env.G, target=goal, weight='weight')

    env.max_edge_w = max((env.G[u][v]['weight'] for u, v in env.G.edges()), default=WAIT_COST)
    env.max_real_t = int(len(env.G.nodes) * env.max_edge_w * env.agent_num * 2)

    if not hasattr(env, "centrality_cache"):
        env.centrality_cache = nx.degree_centrality(env.G)

    if not hasattr(env, "shortest_paths_cache"):
        env.shortest_paths_cache = {}
        for a in range(env.agent_num):
            try:
                env.shortest_paths_cache[a] = set(nx.shortest_path(env.G,
                    env.current_start[a], env.goal_array[a]))
            except nx.NetworkXNoPath:
                env.shortest_paths_cache[a] = set()


    # Précompute proximity (une fois par épisode)
    if not hasattr(env, "proximity_cache"):
        env.proximity_cache = {}
        for node in env.G.nodes:
            nearby = [near for near in env.G.nodes
                      if near != node and h_euclidian(env.pos, near, node) <= env.speed]
            env.proximity_cache[node] = nearby

    # === PBS ===
    print(f"[{time.strftime('%H:%M:%S')}] [WARM START] running priority_based first...", flush=True)
    warm_solution = priority_based_planning(env)
    
    if warm_solution and len(warm_solution) == env.agent_num:
        # Cost calculation for warm solution
        warm_cost = sum(path_cost(env, warm_solution[a], env.goal_array[a])
                        for a in range(env.agent_num))
        # Goal rate calculation for warm solution
        all_reach_goal = all(warm_solution[a][-1] == env.goal_array[a]
                              for a in range(env.agent_num))
        paths = path_translation(env, warm_solution)

        if all_reach_goal:
            print(f"[PBS] cost={warm_cost} (used as upper bound)", flush=True)
            paths = path_translation(env, warm_solution)
        else:
            print(f"[PBS] partial (some agents stuck), no UB", flush=True)
    else:
        print(f"[PBS] failed", flush=True)
        paths = {a: [env.current_start[a]] for a in range(env.agent_num)} # fallback paths (agents stay at start)

    for agent in range(env.agent_num):
        path_idx[agent] = 0
        last_node[agent] = env.current_start[agent]

def policy(obs, env):
    global last_episode
    if env.episode_account != last_episode or len(paths) != env.agent_num:
        init(env)

    actions = []
    for agent in range(env.agent_num):
        if env.current_goal[agent] is not None:
            actions.append(env.current_goal[agent])
        else:
            # Agent is at a node, check if we need to advance the path index
            curr = env.current_start[agent]
            path = paths[agent]
            if path_idx[agent] < len(path) - 1 and curr == path[path_idx[agent]]:
                 path_idx[agent] += 1
            actions.append(path[path_idx[agent]])
    return actions
