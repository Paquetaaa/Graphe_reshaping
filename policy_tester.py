import gym
#from policy.policy_CBS_20_04 import policy
#from policy.Old_Policy.policy_astar_visited import policy
# from example.policy_prfl import policy
from policy.Old_Policy import policy_given_path
from policy.Old_Policy.policy_given_path import policy
import time

def policy_evaluation(policy, drone_num, map_name, reward_list, start, goal, render):
    if not start or goal:

        assert drone_num == len(start) and drone_num == len(
            goal
        ), "The number of elements in start and goal list does not match with drone_num."
        assert not any(
            element in start for element in goal
        ), "The elements of goal and start must not match."
    print("drp_env:drp-" + str(drone_num) + "agent_" + map_name + "-v2")
    env = gym.make(
        "drp_env:drp-" + str(drone_num) + "agent_" + map_name + "-v2",
        state_repre_flag="onehot_fov",
        reward_list=reward_list,
        goal_array=goal,
        start_ori_array=start,
    )
    obs = env.reset()
    #print(f"observation_space:{env.observation_space}")
    #print(f"action_space:{env.action_space}")

    done_all = False
    while not done_all:
        if render == True:
            env.render()
            time.sleep(0.2)
        #print(f"obs:{obs}")  # current global observation
        actions = policy(obs, env)  # policy:input n_obs,env return each drone's action
        obs, reward, done, info = env.step(
            actions
        )  # transfer to next state once joint action is taken
        #print(f"obs:{obs}, actions:{actions}, reward:{reward}, done:{done},info:{info}")
        done_all = all(done)
        env.get_obs()


if __name__ == "__main__":
    # When replaying a recorded trace, pull map / agent_num / starts / goals from it
    # so the environment matches what was logged.
    drone_num = policy_given_path.AGENT_NUM
    map_name = policy_given_path.MAP_NAME
    start = policy_given_path.STARTS
    goal = policy_given_path.GOALS

    reward_list = {
        "goal": 100,
        "collision": -10,
        "wait": -10,
        "move": -1,
    }

    render = True  # Choose whether to visualize

    """
    policy_evaluation() function is used to evaluate the "policy" developed by participants
    participants are expected to develop "policy",
    which is essentially a mapping from input(global observation) to output(joint action) at each step
    """
    policy_evaluation(
        policy=policy,  # this is an example policy
        drone_num=drone_num,
        map_name=map_name,
        reward_list=reward_list,
        goal=goal,
        start=start,
        render=render,
    )
