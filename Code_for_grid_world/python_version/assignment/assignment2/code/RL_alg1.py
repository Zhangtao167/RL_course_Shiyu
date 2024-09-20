import sys
import os
from grid_world import GridWorld
import random
import numpy as np
import imageio
import pandas as pd
from typing import Union
import argparse

parser = argparse.ArgumentParser("Grid World Environment")

## ==================== User settings ====================
# specify the number of columns and rows of the grid world
parser.add_argument("--env-size", type=Union[list, tuple, np.ndarray], default=(4,4) )   

# specify the start state
parser.add_argument("--start-state", type=Union[list, tuple, np.ndarray], default=(0,0))

# specify the target state
parser.add_argument("--target-state", type=Union[list, tuple, np.ndarray], default=(2,2))

# specify the forbidden states
parser.add_argument("--forbidden-states", type=list, default=[ (2, 1), (1, 2), (2, 3)] )

# specify the reward when reaching target
parser.add_argument("--reward-target", type=float, default = 1)

# specify the reward when entering into forbidden area
parser.add_argument("--reward-forbidden", type=float, default = -1)

# specify the reward for each step
parser.add_argument("--reward-step", type=float, default = -1)

# discount factor
parser.add_argument("--gamma", type=float, default = 0.9)
## ==================== End of User settings ====================

## ==================== Advanced Settings ====================
parser.add_argument("--action-space", type=list, default=[(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)] )  # down, right, up, left, stay           
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--animation-interval", type=float, default = 0.2)
parser.add_argument("--results_path", type=str, default = "./results")
parser.add_argument("--exp_name", type=str, default = "assignment2")
parser.add_argument("--exp_date", type=str, default = "2024-09-13")
parser.add_argument("--policy_type", type=str, help = "deterministic or stochastic",default="deterministic")
parser.add_argument("--plot_policy", type=int, default = 1)
parser.add_argument("--length_simulation", type=int, default = 50)
parser.add_argument("--is_save_gif", type=int, default = 0)
parser.add_argument("--V_cal_method", type=str, help = " null or iter or closed_form", default="null")
## ==================== End of Advanced settings ====================

def get_policy_matrix_deterministic(env):
    # Read the deterministic policy matrix from Excel file
    policy_df = pd.read_excel('../policy_matrix_deterministic_designed.xlsx', header=None)
    policy_matrix = policy_df.to_numpy()
    return policy_matrix[1:]

def get_policy_matrix_stochastic(env):
    # Initialize stochastic policy from deterministic policy
    # Then modify it into a stochastic policy
    policy_df = pd.read_excel('../policy_matrix_deterministic_designed.xlsx', header=None)
    policy_matrix_deterministic = policy_df.to_numpy()[1:]
    num_states, num_actions = policy_matrix_deterministic.shape

    policy_matrix = np.zeros((num_states, num_actions))

    for s in range(num_states):
        # Identify the deterministic action (the one with probability 1)
        deterministic_action = np.argmax(policy_matrix_deterministic[s])
        # Assign a high probability to the deterministic action
        high_prob = 0.7  # You can adjust this value as needed
        # Assign equal probabilities to other actions
        remaining_prob = 1.0 - high_prob
        other_actions = [a for a in range(num_actions) if a != deterministic_action]
        num_other_actions = len(other_actions)
        if num_other_actions > 0:
            prob_other_actions = remaining_prob / num_other_actions
            # Assign probabilities
            policy_matrix[s, deterministic_action] = high_prob
            for a in other_actions:
                policy_matrix[s, a] = prob_other_actions
        else:
            # If there's only one possible action, assign probability 1
            policy_matrix[s, deterministic_action] = 1.0
        # Ensure the sum of probabilities is 1
        assert abs(np.sum(policy_matrix[s]) - 1.0) < 1e-6, f"Probabilities do not sum to 1 for state {s}"

    return policy_matrix


def compute_P_pi_and_r_pi(env, policy_matrix):
    num_states = env.num_states
    num_actions = len(env.action_space)
    P_pi = np.zeros((num_states, num_states))
    r_pi = np.zeros(num_states)
    
    for s in range(num_states):
        x = s % env.env_size[0]
        y = s // env.env_size[0]
        state = (x, y)
        
        for a in range(num_actions):
            action = env.action_space[a]
            pi_a_s = policy_matrix[s, a]
            if pi_a_s == 0:
                continue
            # Compute next state s' and reward r(s,a)
            next_state, reward = env._get_next_state_and_reward(state, action)
            s_prime = next_state[1] * env.env_size[0] + next_state[0]
            P_pi[s, s_prime] += pi_a_s
            r_pi[s] += pi_a_s * reward
    return P_pi, r_pi

def policy_evaluation(env, policy_matrix, gamma, theta=1e-6):
    V = np.zeros(env.num_states)
    delta = float('inf')
    while delta > theta:
        delta = 0
        for s in range(env.num_states):
            v = V[s]
            x = s % env.env_size[0]
            y = s // env.env_size[0]
            state = (x, y)
            v_new = 0
            for a in range(len(env.action_space)):
                action = env.action_space[a]
                pi_a_s = policy_matrix[s, a]
                if pi_a_s == 0:
                    continue
                next_state, reward = env._get_next_state_and_reward(state, action)
                s_prime = next_state[1] * env.env_size[0] + next_state[0]
                v_new += pi_a_s * (reward + gamma * V[s_prime])
            V[s] = v_new
            delta = max(delta, abs(v - V[s]))
    return V

if __name__ == "__main__": 
    args = parser.parse_args()   
    args.results_path = os.path.join(args.results_path, args.policy_type)
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)            
    if  args.policy_type in ['deterministic', 'stochastic']:
        # args.policy_type = policy_type
        env = GridWorld(args)
        state = env.reset()    
        _ = env.render()           

        # Add policy
        if args.policy_type == "deterministic":
            policy_matrix = get_policy_matrix_deterministic(env)
        elif args.policy_type == "stochastic":
            policy_matrix = get_policy_matrix_stochastic(env)

        # save policy_matrix as excel table
        # import pdb;pdb.set_trace()
        df = pd.DataFrame(policy_matrix)
        df.to_excel(args.results_path + f'/policy_matrix_{args.policy_type}.xlsx', index=False)
        env.add_policy(policy_matrix)
        env.render(animation_interval=2)
        # Compute P_pi and r_pi
        P_pi, r_pi = compute_P_pi_and_r_pi(env, policy_matrix)
        # Save P_pi and r_pi
        np.savetxt(args.results_path + f'/P_pi_{args.policy_type}.csv', P_pi, delimiter=',', fmt='%.2f')
        np.savetxt(args.results_path + f'/r_pi_{args.policy_type}.csv', r_pi, delimiter=',', fmt='%.2f')
        if args.V_cal_method == "closed_form":
            # Closed-form solution
            V_closed_form = np.linalg.inv(np.eye(env.num_states) - args.gamma * P_pi) @ r_pi
            # Plot the state values
            env.add_state_values(V_closed_form, precision=2)
            env.render(animation_interval=2)
            # Save V_closed_form
            np.savetxt(args.results_path + f'/V_closed_form_{args.policy_type}.csv', V_closed_form, delimiter=',')
        elif args.V_cal_method == "iter":
            # Iterative algorithm
            V_iterative = policy_evaluation(env, policy_matrix, args.gamma)
            # Plot the state values
            env.add_state_values(V_iterative, precision=2)
            env.render(animation_interval=2)
            # Save V_iterative
            np.savetxt(args.results_path + f'/V_iterative_{args.policy_type}.csv', V_iterative, delimiter=',')
        print(f"Results for {args.policy_type} policy have been saved.")

