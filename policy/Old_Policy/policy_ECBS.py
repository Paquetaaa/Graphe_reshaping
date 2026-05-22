import gym
import math
import networkx as nx
import heapq
import time

### submission information ####
TEAM_NAME = "GRENECHE Lucas_ECBS"
##############################

##############################
# Global Variables 
paths = {} # Dict, key = Agent_id, value = A* path
last_episode = -1
path_idx = {} # Dict, key = Agent_id, value = current node index in paths
last_node = {} # To detect when an agent has reached its next waypoint
cbs_success = True  # False if CBS failed and we fell back to A*

PROFILE = {
    "astar_calls": 0,
    "astar_time": 0.0,
    "detect_calls": 0,
    "detect_time": 0.0,
    "classify_calls": 0,
    "classify_time": 0.0,
}

USE_BYPASS = False
CARDINAL_SEARCH_TRESHOLD = 200
MAX_OPEN_SIZE = 1_000_000   # memory security for CBS (number of nodes in the open list)
MAX_ITER_CBS = 1_000_000   # time security for CBS (number of iterations in the main loop)
W_BOUND = 2.0  # weight bound for ECBS (1.0 = exact CBS, 1.5 = ≤1.5×OPT, 2.0 = ≤2×OPT)

FORCE_OPEN_EVERY = 5   # constante en haut


WAIT_COST = 1  # Cost of waiting one step

class CT_Node:
    __slots__ = ('constraints', 'solution', 'cost', 'costs', 'conflicts','popped','in_focal')
    def __init__(self, constraints, solution, cost, costs, conflicts):
        self.constraints = constraints 
        self.solution = solution
        self.cost = cost
        self.costs = costs
        self.conflicts = conflicts
        self.popped = False 
        self.in_focal = False


## UTILITY FUNCTIONS
def h_euclidian(pos, u, v):
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    return (math.sqrt((x1 - x2) ** 2 + (y1 - y2)**2))


def compute_sic(env, solution):
    sic = 0
    for agent in solution:
        path = solution[agent]
        goal = env.goal_array[agent]
        sic += path_cost(env,path,goal)
    return sic

def path_cost(env,path,goal):
    cost = 0
    for t in range(len(path) - 1):
        if path[t] != path[t+1]:
            cost += 1
        else:
            cost += WAIT_COST

        if path[t+1] == goal:
            break # When goal is reached, we stop counting cost
        
    return cost


# Utility functions for detect_conflict
def get_node_at(solution, agent_id, t):
    if agent_id not in solution:
        return None
    path = solution[agent_id]
    if t >= len(path):
        return path[-1]
    return path[t]

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



def conflicts_for_agent(solution, env, agent):
    """Return all conflicts where 'agent' is one of the two involved agents."""
    out = []
    path_agent = solution[agent]
    for other in solution:
        if other == agent:
            continue
        # On normalise pour que le tuple ait toujours (i, j) avec i < j
        i, j = (agent, other) if agent < other else (other, agent)
        path_i = solution[i]
        path_j = solution[j]
        max_len = max(len(path_i), len(path_j))

        # Vertex (au niveau original)
        for ki in range(max_len):
            pos_i = path_i[ki] if ki < len(path_i) else path_i[-1]
            pos_j = path_j[ki] if ki < len(path_j) else path_j[-1]
            if (pos_i == pos_j 
                and env.G.nodes[pos_i]['type'] == 'original' 
                and env.G.nodes[pos_j]['type'] == 'original'):
                out.append(('vertex', i, j, pos_i, ki, ki))

        # Edge swap
        for ki in range(max_len - 1):
            pos_i = path_i[ki] if ki < len(path_i) else path_i[-1]
            pos_j = path_j[ki] if ki < len(path_j) else path_j[-1]
            pos_i_next = path_i[ki+1] if ki+1 < len(path_i) else path_i[-1]
            pos_j_next = path_j[ki+1] if ki+1 < len(path_j) else path_j[-1]
            if pos_i == pos_j_next and pos_j == pos_i_next:
                out.append(('edge', i, j, pos_i, pos_j, ki, ki))

        # Proximity
        for ki in range(max_len):
            pos_i = path_i[ki] if ki < len(path_i) else path_i[-1]
            pos_j = path_j[ki] if ki < len(path_j) else path_j[-1]
            if h_euclidian(env.pos, pos_i, pos_j) <= env.speed and pos_i != pos_j:
                out.append(('proximity', i, j, pos_i, pos_j, ki))
    return out



def priority_based_planning(env, max_horizon=500):
    """Single-pass priority-based planning, matching the CBS/ICBS/DS fallback
    for methodological parity. Agents are planned sequentially in natural
    order [0, 1, ..., n-1]; each agent's A* respects time-stamped vertex
    constraints from higher-priority agents (no proximity blocking, to mirror
    the simpler CBS-family fallback)."""
    paths_pp = {}
    constraints = set()
    for agent in range(env.agent_num):
        p = a_star_constrained(env, agent,
                               env.current_start[agent],
                               env.goal_array[agent],
                               constraints)
        if p is None:
            print(f"[fallback] agent {agent} infeasible, skipping", flush=True)
            continue
        paths_pp[agent] = p
        # Block future agents from this agent's (node, t) pairs along the path
        for t, node in enumerate(p):
            for other in range(agent + 1, env.agent_num):
                constraints.add((other, node, t, '-'))
        # Goal protection: agent stays on its goal indefinitely
        goal = env.goal_array[agent]
        arrival = len(p) - 1
        for t in range(arrival, max_horizon):
            for other in range(agent + 1, env.agent_num):
                constraints.add((other, goal, t, '-'))
    return paths_pp

def apply_safety(env, actions):
    """Defensive net: prevent same-target and head-on-swap conflicts."""
    do = True
    while do:
        do = False
        for i in range(env.agent_num):
            if env.current_goal[i] is not None:
                continue   # agent en transit, on ne peut pas changer son action

            start_i = env.current_start[i]
            for j in range(env.agent_num):
                if j == i:
                    continue
                # Case 1: deux agents visent le même nœud
                if actions[i] == actions[j]:
                    actions[i] = start_i
                    do = True
                    break
                # Case 2: head-on swap
                if (actions[j] == env.current_start[i]
                        and actions[i] == env.current_start[j]):
                    actions[i] = start_i
                    do = True
                    break
    return actions

def a_star_constrained(env, agent, start, goal, constraints):
    """A* pathfinding in env.G using real time (cumulated edge weights) as constraint timestamps."""
    ## Mesuring Time
    PROFILE["astar_calls"] += 1
    _t0 = time.perf_counter()
    ## 

    h = h_euclidian(env.pos, start, goal)
    counter = 0
    open_list = [(h, counter, start, 0, [start], 0)]  # (f, counter, node, real_t, path, g)
    visited = set()
    max_edge_w = max((env.G[u][v]['weight'] for u, v in env.G.edges()), default=WAIT_COST)
    max_real_t = int(len(list(env.G.nodes())) * max_edge_w * env.agent_num * 2)

    goal_constraints_times = [c[2] for c in constraints
                          if c[0] == agent and c[1] == goal and c[3] == '-']
    
    min_goal_arrival = max(goal_constraints_times) + 1 if goal_constraints_times else 0


    positives_elsewhere_times = [c[2] for c in constraints
                              if c[0] == agent and c[3] == '+' and c[1] != goal]
    if positives_elsewhere_times:
        min_goal_arrival = max(min_goal_arrival, max(positives_elsewhere_times) + 1)


    # Pour cet agent: où il DOIT être à chaque temps t
    must_be_at = {}          # dict {t:node_obligatoire}
    for c in constraints:
        if c[0] == agent and c[3] == '+':
            must_be_at[c[2]] = c[1]

    # Nœuds/temps interdits à cet agent (positives des autres agents)
    forbidden = set()        # set {(node, t)}
    for c in constraints:
        if c[0] != agent and c[3] == '+':
            forbidden.add((c[1], c[2]))


    while open_list:
        f, _, node, real_t, path, g = heapq.heappop(open_list)
        if node == goal and real_t >= min_goal_arrival:
            PROFILE["astar_time"] += time.perf_counter() - _t0
            return path
        if (node, real_t) in visited:
            continue
        visited.add((node, real_t))

        # Do not allow leaving the goal once it has been reached.
        if node != goal:
            for neighbor in env.G.neighbors(node):
                edge_w = round(env.G[node][neighbor]['weight'])
                new_real_t = real_t + edge_w
                if (agent, neighbor, new_real_t,'-') not in constraints and (neighbor, new_real_t) not in forbidden and (new_real_t not in must_be_at or must_be_at[new_real_t] == neighbor) and new_real_t <= max_real_t:
                    counter += 1
                    new_g = g + env.G[node][neighbor]['weight']
                    new_f = new_g + h_euclidian(env.pos, neighbor, goal)
                    #new_f = new_g + env.h_table[goal].get(neighbor, float('inf'))
                    heapq.heappush(open_list, (new_f, counter, neighbor, new_real_t, path + [neighbor], new_g))

        
        if env.G.nodes[node]['type'] == 'original': # Only allow waiting at original nodes

            wait_real_t = real_t + round(WAIT_COST)
            if ((agent, node, wait_real_t, '-') not in constraints
        and (node, wait_real_t) not in forbidden
        and (wait_real_t not in must_be_at or must_be_at[wait_real_t] == node)
        and wait_real_t <= max_real_t):

                counter += 1
                new_g = g + WAIT_COST
                new_f = new_g + h_euclidian(env.pos, node, goal)
                #env.h_table[goal].get(node, float('inf'))
                heapq.heappush(open_list, (new_f, counter, node, wait_real_t, path + [node], new_g))

    # Fallback: unconstrained A* so agent still has a valid path (CBS will detect conflicts and continue)
    if start == goal:
        # print(f"[A* FALLBACK cas 1] agent={agent} start={start} goal={goal}")
        PROFILE["astar_time"] += time.perf_counter() - _t0
        return [start]
    # print(f"[A* NONE] agent={agent} start={start} goal={goal} "
    #   f"neg={sum(1 for c in constraints if c[3]=='-')} "
    #   f"pos={sum(1 for c in constraints if c[3]=='+')}")
    PROFILE["astar_time"] += time.perf_counter() - _t0
    return None



def pick_best_conflict(env,ct_node):
    """ From all conflicts pick the best one - Cardinal, Semi-Cardinal or Non-Cardinal -"""
    conflicts = ct_node.conflicts
    if not conflicts:
        return None
    
    best_semi = None
    best_non = None

    for i,conflict in enumerate(conflicts):
        result = classify_conflict_spliting(env, ct_node, conflict)
        cls = result[0]
        if cls == "cardinal":
            return (conflict, result)           # EARLY-EXIT
        elif cls == "semi-cardinal" and best_semi is None:
            best_semi = (conflict,result)
        elif cls == "non-cardinal" and best_non is None:
            best_non = (conflict,result)
        if i+1 > CARDINAL_SEARCH_TRESHOLD and best_semi is not None:
            ### this avoid to scan all the conflict in case no cardinal is found
            print("Threshold atteint", flush=True)
            return best_semi
    return best_semi if best_semi else best_non

def update_conflicts(parent_conflicts, solution, env, changed_agent):
    """Drop parent's conflicts involving changed_agent, add fresh ones."""
    # 1. Conflicts not concerned by changed_agent don't changes
    kept = [c for c in parent_conflicts 
            if c[1] != changed_agent and c[2] != changed_agent]
    # 2. Recompute conflicts where changed_agent is involved
    new_for_agent = conflicts_for_agent(solution, env, changed_agent)
    return kept + new_for_agent


def detect_all_conflicts(solution,env):
    """Return all conflicts in the solution"""

    ## Measuring time
    PROFILE["detect_calls"] += 1
    _t0 = time.perf_counter()


    all_conflicts = []
    agents = list(solution.keys())
    for i_idx, i in enumerate(agents):
        path_i = solution[i]
        for j in agents[i_idx+1:]:
            path_j = solution[j]

            max_len = max(len(path_i), len(path_j)) 
            # Only check vertex conflicts at original nodes, edge conflicts are implicitly handled by the fact that intermediate nodes are unique to each edge and can't be shared without a vertex conflict at an original node.
            for ki in range(max_len):
                pos_i = path_i[ki] if ki < len(path_i) else path_i[-1]
                pos_j = path_j[ki] if ki < len(path_j) else path_j[-1]

                if (pos_i == pos_j and env.G.nodes[pos_i]['type'] == 'original' and env.G.nodes[pos_j]['type'] == 'original'): # Only consider vertex conflicts at original nodes
                    #print(f"Vertex conflict detected between agent {i} and agent {j} at node {pos_i} at time {ki}") ## DEBUG
                    all_conflicts.append(('vertex', i, j, pos_i, ki, ki))

            for ki in range(max_len - 1):
                pos_i = path_i[ki] if ki < len(path_i) else path_i[-1]
                pos_j = path_j[ki] if ki < len(path_j) else path_j[-1]
                pos_i_next = path_i[ki+1] if ki+1 < len(path_i) else path_i[-1]
                pos_j_next = path_j[ki+1] if ki+1 < len(path_j) else path_j[-1]

                # Edge conflicts: opposite traversal of the same edge, overlapping times
                if (pos_i == pos_j_next and pos_j == pos_i_next): 
                    #print(f"Edge conflict detected between agent {i} and agent {j} at nodes {pos_i} and {pos_j} at time {ki}") ## DEBUG
                    all_conflicts.append(('edge', i, j, pos_i, pos_j, ki, ki))
                
            # Proximity conflicts: two agents at different nodes but closer than env.speed at the same time step    
            for ki in range(max_len):
                pos_i = path_i[ki] if ki < len(path_i) else path_i[-1]
                pos_j = path_j[ki] if ki < len(path_j) else path_j[-1]

                if h_euclidian(env.pos, pos_i, pos_j) <= env.speed and pos_i != pos_j:
                    #print(f"Proximity conflict detected between agent {i} and agent {j} at nodes {pos_i} and {pos_j} at time {ki}") ## DEBUG
                    all_conflicts.append(('proximity', i, j, pos_i, pos_j, ki))

    #print(f"All conflicts: {all_conflicts}") ## DEBUG
    PROFILE["detect_time"] += time.perf_counter() - _t0
    return all_conflicts




def detect_conflict(solution,env):
    """Return the first conflict detected"""
    cs = detect_all_conflicts(solution,env)
    return cs[0] if cs else None


def cleanup_heap(heap, value_index=2):
    """Remove nodes marked as popped from the heap.
    value_index = position of the CT_Node in the heap tuple.
    2 in focal_list
    3 in open_list
    """
    while heap and heap[0][value_index].popped:
        heapq.heappop(heap)

def make_root_node(env):
    """Initialisation of the constraints tree with the root node."""
    costs = {}
    for agent in range(env.agent_num):
        p = a_star_constrained(env, agent, env.current_start[agent], env.goal_array[agent], set())
        if p is None:
            return None
        paths[agent] = p
        costs[agent] = path_cost(env, p, env.goal_array[agent])
    solution = paths.copy()
    sic = sum(costs.values())
    conflicts = detect_all_conflicts(solution, env)    # ← full scan, one time
    return CT_Node(set(), solution, sic, costs, conflicts)

def classify_conflict_spliting(env, ct_node, conflict):
    PROFILE["classify_calls"] += 1
    _t0 = time.perf_counter()

    agent1 = conflict[1]
    agent2  = conflict[2]

    if conflict[0] == 'vertex':
        # Disjoint splitting, chosen = agent / smaller cost

        if ct_node.costs[agent1] < ct_node.costs[agent2]:
            chosen = agent1
            other = agent2
        else:
            chosen = agent2
            other = agent1
        c_A, c_B = build_disjoint_constraints(conflict, chosen)
        replan_A_agent = other     # côté A : positive sur chosen → replan other
        replan_B_agent = chosen    # côté B : negative sur chosen → replan chosen
    else:
        chosen = agent1
        other = agent2
        # Edge ou proximity : split symétrique standard
        c_A, c_B = build_constraints(conflict)
        replan_A_agent = chosen    # côté A (négatif sur chosen = ex-c1) → replan chosen
        replan_B_agent = other     # côté B (négatif sur other = ex-c2) → replan other

    old_cost_A = ct_node.costs[replan_A_agent]
    old_cost_B = ct_node.costs[replan_B_agent]

    new_path_A = a_star_constrained(env, replan_A_agent,
                                     env.current_start[replan_A_agent],
                                     env.goal_array[replan_A_agent],
                                     ct_node.constraints | c_A)
    new_cost_A = float('inf') if new_path_A is None else \
                 path_cost(env, new_path_A, env.goal_array[replan_A_agent])

    new_path_B = a_star_constrained(env, replan_B_agent,
                                     env.current_start[replan_B_agent],
                                     env.goal_array[replan_B_agent],
                                     ct_node.constraints | c_B)
    new_cost_B = float('inf') if new_path_B is None else \
                 path_cost(env, new_path_B, env.goal_array[replan_B_agent])

    inc_A = new_cost_A > old_cost_A
    inc_B = new_cost_B > old_cost_B

    if inc_A and inc_B:    cls = 'cardinal'
    elif inc_A or inc_B:   cls = 'semi-cardinal'
    else:                  cls = 'non-cardinal'

    PROFILE["classify_time"] += time.perf_counter() - _t0
    return (cls, replan_A_agent, replan_B_agent,
            new_path_A, new_cost_A,
            new_path_B, new_cost_B,
            c_A, c_B)
    
def build_disjoint_constraints(conflict, chosen):
    """Build constraint based on the conflict type, with the disjoint splitting method."""

    conflict_type = conflict[0]

    if conflict_type == 'vertex':
        node, t = conflict[3], conflict[4]
        c_pos = {(chosen, node, t, '+')}
        c_neg = {(chosen, node, t, '-')}
        

    elif conflict_type == 'edge':
        u, v, t = conflict[3], conflict[4], conflict[5]
        # Si chosen == agent1, il était sur u au temps t (départ du swap)
        # Si chosen == agent2, il était sur v
        # On garde ta convention actuelle (vertex constraint sur le départ)
        agent1 = conflict[1]
        chosen_node = u if chosen == agent1 else v
        c_pos = {(chosen, chosen_node, t, '+')}
        c_neg = {(chosen, chosen_node, t, '-')}
    

    else:  # proximity
        node_i, node_j, t = conflict[3], conflict[4], conflict[5]
        agent1 = conflict[1]
        chosen_node = node_i if chosen == agent1 else node_j
        c_pos = {(chosen, chosen_node, t, '+')}
        c_neg = {(chosen, chosen_node, t, '-')}


    return c_pos, c_neg

def has_conflict_pair(path_i, path_j, env):
    """True s'il y a vertex, edge swap ou proximity entre les deux chemins."""
    max_len = max(len(path_i), len(path_j))
    for ki in range(max_len):
        pos_i = path_i[ki] if ki < len(path_i) else path_i[-1]
        pos_j = path_j[ki] if ki < len(path_j) else path_j[-1]
        # vertex (any node, original ou intermediate)
        if pos_i == pos_j:
            return True
        # proximity
        if h_euclidian(env.pos, pos_i, pos_j) <= env.speed:
            return True
    for ki in range(max_len - 1):
        pos_i = path_i[ki] if ki < len(path_i) else path_i[-1]
        pos_j = path_j[ki] if ki < len(path_j) else path_j[-1]
        pos_i_next = path_i[ki+1] if ki+1 < len(path_i) else path_i[-1]
        pos_j_next = path_j[ki+1] if ki+1 < len(path_j) else path_j[-1]
        # edge swap
        if pos_i == pos_j_next and pos_j == pos_i_next:
            return True
    return False


def build_constraints(conflict):
    """Return (c1,c2) the two constraints that resolve a given conflict."""

    conflict_type = conflict[0]
    agent1 = conflict[1]
    agent2 = conflict[2]

    if conflict_type == 'vertex':
        node, t = conflict[3], conflict[4]
        c1 = {(agent1,node,t,'-')}
        c2 = {(agent2,node,t,'-')}

    elif conflict_type == 'edge':  # edge conflict
        u, v, t = conflict[3], conflict[4], conflict[5]
        c1 = {(agent1,u,t,'-')}
        c2 = {(agent2,v,t,'-')}

    else:
            # Proximity conflict : Agent are not on the exact same node but closer than env.speed, treated as a vertex conflict.
        node_i = conflict[3]
        node_j = conflict[4]
        t = conflict[5]
        c1 = {(agent1, node_i, t,'-')}
        c2 = {(agent2, node_j, t,'-')}

    return c1, c2

# def local_search_improve(env, best_paths, n_iter=50):
    import random
    """Try replanning one agent at a time, keep if improves total cost."""
    current = dict(best_paths)
    current_cost = sum(path_cost(env, current[a], env.goal_array[a])
                       for a in range(env.agent_num))
    
    for it in range(n_iter):
        # Pick random agent to replan
        agent = random.randint(0, env.agent_num - 1)
        # Build constraints from OTHER agents' current paths
        constraints = set()
        for other in range(env.agent_num):
            if other == agent:
                continue
            for t, node in enumerate(current[other]):
                constraints.add((agent, node, t, '-'))
                # proximity
                for near in env.G.nodes:
                    if near != node and h_euclidian(env.pos, near, node) <= env.speed:
                        constraints.add((agent, near, t, '-'))
        # Replan agent
        new_path = a_star_constrained(env, agent,
                                       env.current_start[agent],
                                       env.goal_array[agent],
                                       constraints)
        if new_path is None:
            continue
        new_cost_agent = path_cost(env, new_path, env.goal_array[agent])
        old_cost_agent = path_cost(env, current[agent], env.goal_array[agent])
        if new_cost_agent < old_cost_agent:
            # Vérifie qu'on n'introduit pas de conflit avec les autres
            safe = True
            for other in range(env.agent_num):
                if other == agent:
                    continue
                if has_conflict_pair(new_path, current[other], env):
                    safe = False
                    break
            
            if safe:
                current[agent] = new_path
                current_cost = current_cost - old_cost_agent + new_cost_agent
                improvements += 1
                print(f"[LOCAL SEARCH] iter {it}: agent {agent} {old_cost_agent}→{new_cost_agent}, total={current_cost}", flush=True)
            else:
                # Rejeté car création de conflit
                pass

    
    return current, current_cost


def cbs(env, upper_bound=float('inf'), warm_solution=None, w=W_BOUND):
    ## Creation of the root node
    start_time = time.time()
    TIME_OUT = 60.0

    print(f"[CBS START] {time.strftime('%H:%M:%S')} agents={env.agent_num} nodes={len(env.G.nodes)}", flush=True)
    loop_start = time.time()
    last_log = loop_start


    root = make_root_node(env)
    if root is None:
        print("Racine impossible", flush=True)
        return None
    root.in_focal = True  # root is in focal list by definition 
    
    open_list = []
    focal_list = []  
    last_min_cost = root.cost
    counter = 0
    iter_count = 0
    bypass_count = 0

    cnt_card = 0
    cnt_semi = 0
    cnt_non  = 0


    # Profil counter for analysis
    for k in PROFILE:
        PROFILE[k] = 0

    # Push the root node in the queue
    heapq.heappush(open_list, (root.cost, len(root.conflicts), counter, root))

    # Push the root node in the focal list (ECBS)
    heapq.heappush(focal_list,(len(root.conflicts), counter, root))
    


    while open_list and iter_count < MAX_ITER_CBS: 
        # GARDE-FOU MÉMOIRE
        if len(open_list) > MAX_OPEN_SIZE:
            print(f"[{time.strftime('%H:%M:%S')}] [CBS] MAX_OPEN_SIZE hit "
                  f"({len(open_list)}), aborting", flush=True)
            return None    # init() utilisera warm_solution

        if time.time() - start_time > TIME_OUT:
            print(f"[CBS] timeout après {iter_count} itérations")
            return None

        # Cleanup popped en tête d'open
        cleanup_heap(open_list, value_index=3)
        if not open_list:
            break

        # Si le min(open) a augmenté, élargir focal
        current_min = open_list[0][0]
        if current_min > last_min_cost:
            f_bound = w * current_min
            for entry in open_list:
                cost = entry[0]
                node = entry[3]
                if not node.popped and not node.in_focal and cost <= f_bound:
                    heapq.heappush(focal_list, (entry[1], entry[2], node))
                    node.in_focal = True
            last_min_cost = current_min

        # Cleanup popped en tête de focal
        cleanup_heap(focal_list, value_index=2)
        if not focal_list:
            break


        now = time.time()

        if iter_count % 1000 == 0 or now - last_log > 30:
            elapsed = now - loop_start
            its_per_sec = iter_count / max(elapsed, 0.001)
            min_cost = open_list[0][0] if open_list else 0
            print(f"[{time.strftime('%H:%M:%S')}] iter={iter_count} elapsed={elapsed:.0f}s "
                f"open={len(open_list)} min_cost={min_cost:.0f} its/s={its_per_sec:.1f} "
                f"card={cnt_card} semi={cnt_semi} astar={PROFILE['astar_calls']}", flush=True)
            last_log = now
            print(f"  open={len(open_list)} focal={len(focal_list)}", flush=True)


        iter_count += 1

        # Dans la boucle, remplace la pop logic :
        if focal_list and (iter_count % FORCE_OPEN_EVERY != 0):
            cleanup_heap(focal_list, value_index=2)
            if focal_list:
                (_, _, ct_node) = heapq.heappop(focal_list)
            else:
                # focal vide après cleanup, fallback open
                cleanup_heap(open_list, value_index=3)
                if not open_list:
                    break
                (_, _, _, ct_node) = heapq.heappop(open_list)
        else:
            # Force pop depuis open pour faire progresser min_cost
            cleanup_heap(open_list, value_index=3)
            if not open_list:
                break
            (_, _, _, ct_node) = heapq.heappop(open_list)

        ct_node.popped = True


        # ÉLAGAGE PAR UPPER BOUND
        if ct_node.cost >= upper_bound:
            # Aucun descendant ne peut faire mieux que UB → on drop
            continue

        solution = ct_node.solution
        #print("Current solution:", solution)
        #print("Current SIC cost:", ct_node.cost)
        #time.sleep(1)


        picked_conflict = pick_best_conflict(env, ct_node)
        
        if picked_conflict is None:
            # Solution conflict-free trouvée — c'est l'optimal sous notre UB
            print(f"[{time.strftime('%H:%M:%S')}] [CBS SOLUTION] cost={ct_node.cost} "
                  f"after {iter_count} iter (UB was {upper_bound})", flush=True)
            print(f"[CBS] {iter_count} iterations, {bypass_count} bypasses "
                f"({100*bypass_count/max(iter_count,1):.1f}%)", flush=True)
            print(f"Total iteration : {iter_count}", flush=True)
            print(f"[CBS] {cnt_card} cardinal conflicts, {cnt_semi} semi-cardinal conflicts, {cnt_non} non-cardinal conflicts)", flush=True)

            total_classify_a_star = PROFILE["classify_calls"] * 2  # 2 A* par classify
            print(f"[PROFILE]", flush=True)
            print(f"  A* calls: {PROFILE['astar_calls']} ({PROFILE['astar_time']:.2f}s, "
                f"{1000*PROFILE['astar_time']/max(PROFILE['astar_calls'],1):.2f}ms/call)", flush=True)
            print(f"  detect_all_conflicts calls: {PROFILE['detect_calls']} "
                f"({PROFILE['detect_time']:.2f}s)", flush=True)
            print(f"  classify_conflict calls: {PROFILE['classify_calls']} "
                f"({PROFILE['classify_time']:.2f}s)", flush=True)
            print(f"  iterations: {iter_count}", flush=True)

            return ct_node.solution

        
        conflict, (cls, agent_A, agent_B,
           new_path_A, new_cost_A,
           new_path_B, new_cost_B,
           c_A, c_B) = picked_conflict
        

        # Child A
        if new_path_A is not None:
            childA_constraints = ct_node.constraints | c_A
            solutionA = dict(ct_node.solution)
            solutionA[agent_A] = new_path_A
            costsA = dict(ct_node.costs)
            costsA[agent_A] = new_cost_A
            sicA = sum(costsA.values())
            if sicA < upper_bound:
                childA_conflicts = update_conflicts(ct_node.conflicts, solutionA, env, agent_A)
                childA = CT_Node(childA_constraints, solutionA, sicA, costsA, childA_conflicts)
                counter += 1
                heapq.heappush(open_list, (childA.cost, len(childA.conflicts), counter, childA))
                if childA.cost <= w * last_min_cost:
                    heapq.heappush(focal_list, (len(childA.conflicts), counter, childA))
                    childA.in_focal = True


        # Child B
        if new_path_B is not None:
            childB_constraints = ct_node.constraints | c_B
            solutionB = dict(ct_node.solution)
            solutionB[agent_B] = new_path_B
            costsB = dict(ct_node.costs)
            costsB[agent_B] = new_cost_B
            sicB = sum(costsB.values())
            if sicB < upper_bound:
                childB_conflicts = update_conflicts(ct_node.conflicts, solutionB, env, agent_B)
                childB = CT_Node(childB_constraints, solutionB, sicB, costsB, childB_conflicts)
                counter += 1
                heapq.heappush(open_list, (childB.cost, len(childB.conflicts), counter, childB))
                if childB.cost <= w * last_min_cost:
                    heapq.heappush(focal_list, (len(childB.conflicts), counter, childB))
                    childB.in_focal = True


        
        if cls == 'cardinal':
            cnt_card += 1
        elif cls == 'semi-cardinal':
            cnt_semi += 1
        else:
            cnt_non += 1


    return None

def path_translation(env,result):
    """Translate paths from the expanded graph back to the original graph."""
    translated_path = {}
    for agent, path in result.items():
        translated_path[agent] = [node for node in path if not isinstance(node, str)]
    return translated_path


def init(env):
    global last_episode, paths, path_idx, last_node, cbs_success
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

    # Précompute proximity (une fois par épisode)
    if not hasattr(env, "proximity_cache"):
        env.proximity_cache = {}
        for node in env.G.nodes:
            nearby = [near for near in env.G.nodes
                      if near != node and h_euclidian(env.pos, near, node) <= env.speed]
            env.proximity_cache[node] = nearby



    # === WARM START : on lance d'abord le fallback ===
    print(f"[{time.strftime('%H:%M:%S')}] [WARM START] running priority_based first...", flush=True)
    warm_solution = priority_based_planning(env)
    
    if warm_solution and len(warm_solution) == env.agent_num:
        # Calcule le coût de la solution warm
        warm_cost = sum(path_cost(env, warm_solution[a], env.goal_array[a])
                        for a in range(env.agent_num))
        # Vérifie que tous les agents atteignent leur goal (sinon UB est artificiellement bas)
        all_reach_goal = all(warm_solution[a][-1] == env.goal_array[a]
                              for a in range(env.agent_num))
        if all_reach_goal:
            UB = warm_cost
            print(f"[WARM START] cost={warm_cost} (used as upper bound)", flush=True)
        else:
            UB = float('inf')
            print(f"[WARM START] partial (some agents stuck), no UB", flush=True)
    else:
        warm_solution = None
        UB = float('inf')
        print(f"[WARM START] failed", flush=True)
   
    # === CBS avec UB et fallback ===
    result = cbs(env, upper_bound=UB, warm_solution=warm_solution)
    
    if result is not None:
        paths = path_translation(env, result)
        cbs_success = True
        print(f"[{time.strftime('%H:%M:%S')}] CBS solution used", flush=True)
    elif warm_solution is not None:
        paths = path_translation(env, warm_solution)
        cbs_success = False
        print(f"[{time.strftime('%H:%M:%S')}] CBS failed, using warm start", flush=True)
    else:
        # Cas extrême : ni CBS ni warm start
        paths = {a: [env.current_start[a]] for a in range(env.agent_num)}
        cbs_success = False
        print(f"[{time.strftime('%H:%M:%S')}] No solution, all agents stay at start", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] [INSTANCE END] cbs_success={cbs_success} "
          f"total_time={time.time()-init_start:.1f}s", flush=True)

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

    return apply_safety(env,actions)
