import gym
import math
import networkx as nx
import heapq
import time

### submission information ####
TEAM_NAME = "GRENECHE Lucas"
##############################

##############################
# Global Variables 
paths = {} # Dict, key = Agent_id, value = A* path
last_episode = -1
path_idx = {} # Dict, key = Agent_id, value = current node index in paths
last_node = {} # To detect when an agent has reached its next waypoint
cbs_success = True  # False if CBS failed and we fell back to A*



USE_BYPASS = False

WAIT_COST = 1  # Coût d'attente pour 1 step (équivalent à la vitesse par défaut)
## A_STAR PATHFINDING

class CT_Node:
    """Class for the nodes in constraints tree"""
    def __init__(self, constraints, solution, cost, costs):
        self.constraints = constraints 
        self.solution = solution
        self.cost = cost
        self.costs = costs



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
            break # On arrête de compter une fois le but atteint
        
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

def priority_based_planning(env, max_horizon=500):
    paths_pp = {}
    constraints = set()
    for agent in range(env.agent_num):
        p = a_star_constrained(env, agent,
                               env.current_start[agent],
                               env.goal_array[agent],
                               constraints)
        
        if p is None:
            print(f"[fallback] agent {agent} infeasible, skipping")
            continue
        paths_pp[agent] = p
        # Bloquer chaque (agent_futur, node, t) le long du chemin de cet agent
        for t, node in enumerate(p):
            for other in range(agent + 1, env.agent_num):
                constraints.add((other, node, t,'-'))
        # Goal protection : agent reste sur son goal indéfiniment
        goal = env.goal_array[agent]
        arrival = len(p) - 1                # temps d'arrivée
        for t in range(arrival, max_horizon):
            for other in range(agent + 1, env.agent_num):
                constraints.add((other, goal, t,'-'))
    return paths_pp


def a_star_constrained(env, agent, start, goal, constraints):
    """A* pathfinding in env.G using real time (cumulated edge weights) as constraint timestamps."""
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
        return [start]
    # print(f"[A* NONE] agent={agent} start={start} goal={goal} "
    #   f"neg={sum(1 for c in constraints if c[3]=='-')} "
    #   f"pos={sum(1 for c in constraints if c[3]=='+')}")
    return None



def pick_best_conflict(env,ct_node,stats=None):
    """ From all conflicts pick the best one - Cardinal, Semi-Cardinal or Non-Cardinal -"""
    conflicts = detect_all_conflicts(ct_node.solution, env)
    if not conflicts:
        return None
    best_semi = None
    best_non = None
    for conflict in conflicts:
        result = classify_conflict_spliting(env, ct_node, conflict)
        cls = result[0]
        if stats is not None:
            stats[cls] = stats.get(cls, 0) + 1
        if cls == "cardinal":
            return (conflict, result)           # EARLY-EXIT
        elif cls == "semi-cardinal" and best_semi is None:
            best_semi = (conflict,result)
        elif cls == "non-cardinal" and best_non is None:
            best_non = (conflict,result)
    return best_semi if best_semi else best_non


def detect_all_conflicts(solution,env):
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

                if h_euclidian(env.pos, pos_i, pos_j) < env.speed and pos_i != pos_j:
                    #print(f"Proximity conflict detected between agent {i} and agent {j} at nodes {pos_i} and {pos_j} at time {ki}") ## DEBUG
                    all_conflicts.append(('proximity', i, j, pos_i, pos_j, ki))

    #print(f"All conflicts: {all_conflicts}") ## DEBUG
    return all_conflicts




def detect_conflict(solution,env):
    cs = detect_all_conflicts(solution,env)
    return cs[0] if cs else None

def make_root_node(env):
    costs = {}
    for agent in range(env.agent_num):
        p = a_star_constrained(env, agent, env.current_start[agent], env.goal_array[agent], set())
        if p is None:
            return None    # problème infaisable d'emblée
        paths[agent] = p
        costs[agent] = path_cost(env, p, env.goal_array[agent])
    solution = paths.copy()
    sic = sum(costs.values())
    return CT_Node(set(), solution, sic, costs)


def count_conflicts(solution, env):
    return len(detect_all_conflicts(solution, env))


def classify_conflict_spliting(env, ct_node, conflict):
    chosen = conflict[1]
    other  = conflict[2]

    if conflict[0] == 'vertex':
        # Disjoint splitting
        c_A, c_B = build_disjoint_constraints(conflict, chosen)
        replan_A_agent = other     # côté A : positive sur chosen → replan other
        replan_B_agent = chosen    # côté B : negative sur chosen → replan chosen
    else:
        # Edge ou proximity : split symétrique standard
        c_A, c_B = build_constraints(conflict)   # ton ancienne fonction qui retourne 2 négatives
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

    return (cls, replan_A_agent, replan_B_agent,
            new_path_A, new_cost_A,
            new_path_B, new_cost_B,
            c_A, c_B)


def classify_conflict(env, ct_node, conflict):
    """Take curent CT Node, return ('cardinal' | 'semi-cardinal' | 'non-cardinal',
        new_path1, new_cost1,
        new_path2, new_cost2)"""
    
    agent1 = conflict[1]
    agent2 = conflict[2]
    c1, c2 = build_constraints(conflict)

    old_cost1 = ct_node.costs[agent1]
    old_cost2 = ct_node.costs[agent2]

    new_path1 = a_star_constrained(env, agent1, env.current_start[agent1],
                                env.goal_array[agent1],
                                ct_node.constraints | c1)
    new_cost1 = path_cost(env, new_path1, env.goal_array[agent1])
    
    new_path2 = a_star_constrained(env, agent2, env.current_start[agent2],
                                env.goal_array[agent2],
                                ct_node.constraints | c2)
    new_cost2 = path_cost(env, new_path2, env.goal_array[agent2])

    inc1 = new_cost1 > old_cost1
    inc2 = new_cost2 > old_cost2

    if inc1 and inc2:
        return ('cardinal', new_path1, new_cost1, new_path2, new_cost2)
    elif inc1 or inc2:
        return ('semi-cardinal', new_path1, new_cost1, new_path2, new_cost2)
    else:
        return ('non-cardinal', new_path1, new_cost1, new_path2, new_cost2)
    
def build_disjoint_constraints(conflict, chosen):
    """Pour le 'chosen' agent, retourne (c_pos, c_neg).
       c_pos sera ajouté au child A, c_neg au child B."""

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

def cbs(env, time_limit_seconds=None, return_stats=False):
    """Run CBS with cardinal-conflict prioritization.

    Args:
        env: drp environment (with reshaped graph already in env.G)
        time_limit_seconds: optional wall-clock budget. If exceeded, CBS
            returns the best solution found at the root or None if none.
        return_stats: when True, returns (solution, stats_dict). Default
            False preserves the legacy single-return API for init().
    """
    start_time = time.time()

    ## Creation of the root node
    root = make_root_node(env)
    if root is None:
        print("Racine impossible")
        if return_stats:
            return None, {'terminated': 'infeasible_root', 'iterations': 0,
                          'elapsed': 0.0, 'cardinal': 0, 'semi-cardinal': 0,
                          'non-cardinal': 0}
        return None
    open_list = []
    counter = 0
    max_iter = 5000000 # Increased for better results
    iter_count = 0
    bypass_count = 0

    cnt_card = 0
    cnt_semi = 0
    cnt_non  = 0
    stats = {}


    # Push the root node in the queue
    heapq.heappush(open_list, (root.cost, counter, root))


    while open_list and iter_count < max_iter:
        # Wall-clock timeout check
        if time_limit_seconds is not None and (time.time() - start_time) > time_limit_seconds:
            print(f"[CBS TIMEOUT] {time_limit_seconds:.1f}s exceeded at iter {iter_count}")
            elapsed = time.time() - start_time
            if return_stats:
                return None, {'terminated': 'timeout', 'iterations': iter_count,
                              'elapsed': elapsed,
                              'cardinal':     stats.get('cardinal', 0),
                              'semi-cardinal': stats.get('semi-cardinal', 0),
                              'non-cardinal': stats.get('non-cardinal', 0)}
            return None

        if iter_count % 500 == 0:
            print(f"Iteration {iter_count}, open list size: {len(open_list)}")

        iter_count += 1
        #print(f"Iteration {iter_count}")
        # We pop the node with the lowest cost
        (_,__,ct_node) = heapq.heappop(open_list)
        solution = ct_node.solution
        #print("Current solution:", solution)
        #print("Current SIC cost:", ct_node.cost)
        #time.sleep(1)


        picked_conflict = pick_best_conflict(env, ct_node,stats)

        if picked_conflict is None:
            elapsed = time.time() - start_time
            print(f"[CBS] {iter_count} iterations, {bypass_count} bypasses "
                f"({100*bypass_count/max(iter_count,1):.1f}%)")
            print(f"Total iteration : {iter_count}")
            print(f"[CBS] {cnt_card} cardinal conflicts, {cnt_semi} semi-cardinal conflicts, {cnt_non} non-cardinal conflicts)")
            print(stats)
            if return_stats:
                return ct_node.solution, {
                    'terminated': 'solved',
                    'iterations': iter_count,
                    'elapsed': elapsed,
                    'cardinal':     stats.get('cardinal', 0),
                    'semi-cardinal': stats.get('semi-cardinal', 0),
                    'non-cardinal': stats.get('non-cardinal', 0)
                }
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
            childA = CT_Node(childA_constraints, solutionA, sicA, costsA)
            counter += 1
            heapq.heappush(open_list, (childA.cost, counter, childA))

        # Child B (même logique avec agent_B)
        if new_path_B is not None:
            childB_constraints = ct_node.constraints | c_B
            solutionB = dict(ct_node.solution)
            solutionB[agent_B] = new_path_B
            costsB = dict(ct_node.costs)
            costsB[agent_B] = new_cost_B
            sicB = sum(costsB.values())
            childB = CT_Node(childB_constraints, solutionB, sicB, costsB)
            counter += 1
            heapq.heappush(open_list, (childB.cost, counter, childB))

        
        # print(f"[ITER {iter_count}] cls={cls} chosen={chosen} other={other}")
        # print(f"   path_A is None: {new_path_A is None}, path_B is None: {new_path_B is None}")

        
        if cls == 'cardinal':
            cnt_card += 1
        elif cls == 'semi-cardinal':
            cnt_semi += 1
        else:
            cnt_non += 1

        # agent1 = conflict[1]
        # agent2 = conflict[2]

        # ## - BYPASS 
        # if USE_BYPASS and cls != 'cardinal':
        #     old_count = count_conflicts(ct_node.solution, env)
        #     old_cost1 = ct_node.costs[agent1]
        #     old_cost2 = ct_node.costs[agent2]

        #     # Côté 1 : peut bypass si même coût ET strictement moins de conflits
        #     if new_cost1 == old_cost1:
        #         candidate = dict(ct_node.solution)
        #         candidate[agent1] = new_path1
        #         if count_conflicts(candidate, env) < old_count:
        #             ct_node.solution = candidate
        #             ct_node.costs[agent1] = new_cost1
        #             counter += 1
        #             bypass_count += 1
        #             # cost reste identique car cost1 inchangé
        #             # On NE re-pousse PAS dans open_list, on REPREND la boucle sur le même nœud
        #             heapq.heappush(open_list, (ct_node.cost, counter, ct_node))
        #             continue

        #     # Côté 2 : pareil
        #     if new_cost2 == old_cost2:
        #         candidate = dict(ct_node.solution)
        #         candidate[agent2] = new_path2
        #         if count_conflicts(candidate, env) < old_count:
        #             ct_node.solution = candidate
        #             ct_node.costs[agent2] = new_cost2
        #             counter += 1
        #             bypass_count += 1
        #             heapq.heappush(open_list, (ct_node.cost, counter, ct_node))
        #             continue


        ## - NO BYPASS

        ## Version sans Disjoint Spliting 

        ## c1,c2 = build_constraints(conflict) 

        # ## CHILD1
    
        # ## Update constraints
        # child1_constraints = ct_node.constraints | c1
        # #new_path1 = a_star_constrained(env, agent1, env.current_start[agent1], env.goal_array[agent1], child1_constraints)
        # solution1 = dict(ct_node.solution)
        # solution1[agent1] = new_path1

        # ## Update des costs 
        # costs1 = dict(ct_node.costs)
        # #costs1[agent1] = path_cost(env,new_path1,env.goal_array[agent1])
        # costs1[agent1] = new_cost1
        # sic1 = sum(costs1.values())

        # #Création Child
        # child1 = CT_Node(child1_constraints, solution1, sic1, costs1)
        # counter += 1

        # # Ajout dans l'arbre
        # heapq.heappush(open_list, (child1.cost, counter, child1))

        # ## CHILD2

        # # Update Constraints
        # child2_constraints = ct_node.constraints | c2
        # #new_path2 = a_star_constrained(env, agent2, env.current_start[agent2], env.goal_array[agent2], child2_constraints)
        # solution2 = dict(ct_node.solution)
        # solution2[agent2] = new_path2
        
        # # Update des costs
        # costs2 = dict(ct_node.costs)
        # #costs2[agent2] = path_cost(env,new_path2,env.goal_array[agent2])
        # costs2[agent2] = new_cost2
        # sic2 = sum(costs2.values())

        # #Création Child
        # child2 = CT_Node(child2_constraints, solution2, sic2,costs2)
        # counter += 1

        # # Ajout dans l'arbre
        # heapq.heappush(open_list, (child2.cost, counter, child2))
        ## CHILD A : positive sur chosen, replan de 'other'

        ### Fin version sans Disjoint Spliting


        # ### Version Disjoint Spliting 
        # if new_path_A is not None:
        #     childA_constraints = ct_node.constraints | c_pos
        #     solutionA = dict(ct_node.solution)
        #     solutionA[other] = new_path_A
        #     costsA = dict(ct_node.costs)
        #     costsA[other] = new_cost_A
        #     sicA = sum(costsA.values())
        #     childA = CT_Node(childA_constraints, solutionA, sicA, costsA)
        #     counter += 1
        #     heapq.heappush(open_list, (childA.cost, counter, childA))

        # ## CHILD B : negative sur chosen, replan de 'chosen'
        # if new_path_B is not None:
        #     childB_constraints = ct_node.constraints | c_neg
        #     solutionB = dict(ct_node.solution)
        #     solutionB[chosen] = new_path_B
        #     costsB = dict(ct_node.costs)
        #     costsB[chosen] = new_cost_B
        #     sicB = sum(costsB.values())
        #     childB = CT_Node(childB_constraints, solutionB, sicB, costsB)
        #     counter += 1
        #     heapq.heappush(open_list, (childB.cost, counter, childB))

        # ## Fin version disjoint spliting

    # Open list exhausted or max_iter reached without finding a conflict-free solution.
    elapsed = time.time() - start_time
    if return_stats:
        return None, {
            'terminated': 'exhausted' if not open_list else 'max_iter',
            'iterations': iter_count,
            'elapsed': elapsed,
            'cardinal':     stats.get('cardinal', 0),
            'semi-cardinal': stats.get('semi-cardinal', 0),
            'non-cardinal': stats.get('non-cardinal', 0)
        }
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

    # Saving the original graph and positions.
    if not hasattr(env, "G_original"):
        env.G_original = env.G.copy()
        env.pos_original = dict(env.pos)

    # Reshape the graph to transform continous problem into discrete problem for CBS.
    env.G = reshape_graph_from_G(env, env.G_original, env.pos_original)
    
    ## Compute exact heuristic
    # env.h_table = {}
    # for goal in set(env.goal_array):
    #     env.h_table[goal] = nx.shortest_path_length(env.G, target=goal, weight='weight')

   
    ## CBS
    result = cbs(env)
    if result is not None:
        print("resultat de CBS : ")
        print(result)
        paths = path_translation(env,result)
        print("resultat du path_translation")
        print(paths)
   
        cbs_success = True
        print("CBS found a solution.")

    else:
        # Fallback: Standard Priority-based A* (more robust than simple A*)
        cbs_success = False
        print("CBS failed to find a solution, falling back to priority-based A*.")

        result = priority_based_planning(env)
        paths = path_translation(env, result)

        # blocked = set()
        # for agent in range(env.agent_num):
        #     p = a_star_constrained(env, agent, env.current_start[agent], env.goal_array[agent],
        #                            {(agent, b, t) for b in blocked for t in range(100)})
            
        #     paths[agent] = p
        #     blocked.add(env.goal_array[agent])

        # paths = path_translation(env,paths)
        
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
