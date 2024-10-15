# RL_alg1.py

import sys
import os
from grid_world import GridWorld
import numpy as np
import imageio
import pandas as pd
from typing import Union
import argparse
import matplotlib.pyplot as plt

# Set up argument parser
parser = argparse.ArgumentParser("Grid World Environment")

## ==================== User settings ====================
# Specify the number of columns and rows of the grid world
parser.add_argument("--env-size", type=tuple, default=(4, 4))

# Specify the start state
parser.add_argument("--start-state", type=tuple, default=(0, 0))

# Specify the target state
parser.add_argument("--target-state", type=tuple, default=(2, 2))

# Specify the forbidden states
parser.add_argument("--forbidden-states", type=list, default=[(2, 1), (1, 2), (2, 3)])

# Specify the reward when reaching target
parser.add_argument("--reward-target", type=float, default=1)

# Specify the reward when entering into forbidden area
parser.add_argument("--reward-forbidden", type=float, default=-1)

# Specify the reward for each step
parser.add_argument("--reward-step", type=float, default=-0.5)

# Discount factor
parser.add_argument("--gamma", type=float, default=0.9)
## ==================== End of User settings ====================

## ==================== Advanced Settings ====================
parser.add_argument(
    "--action-space",
    type=list,
    default=[(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)],
)  # down, right, up, left, stay
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--animation-interval", type=float, default=0.2)
parser.add_argument("--results_path", type=str, default="../results")
parser.add_argument("--exp_name", type=str, default="assignment3")
parser.add_argument("--exp_date", type=str, default="2024-10-15")
parser.add_argument(
    "--policy_type", type=str, help="deterministic or stochastic", default="deterministic"
)
parser.add_argument("--plot_policy", type=int, default=1)
parser.add_argument("--length_simulation", type=int, default=50)
parser.add_argument("--is_save_gif", type=int, default=0)
parser.add_argument(
    "--V_cal_method", type=str, help="null or iter or closed_form", default="null"
)
## ==================== End of Advanced settings ====================

# Define the value iteration function
def value_iteration(env, gamma, theta=1e-6):
    num_states = env.num_states
    V = np.zeros(num_states)
    delta = float('inf')
    iteration = 0
    V_history = []  # To store the history of V for plotting
    while delta > theta:
        delta = 0
        V_old = V.copy()
        V_history.append(V_old.copy())
        for s in range(num_states):
            x = s % env.env_size[0]
            y = s // env.env_size[0]
            state = (x, y)
            if state in env.forbidden_states:
                V[s] = 0  # Ensure forbidden states have zero value
                continue
            max_value = float('-inf')
            for a, action in enumerate(env.action_space):
                next_state, reward = env._get_next_state_and_reward(state, action)
                s_prime = next_state[1] * env.env_size[0] + next_state[0]
                value = reward + gamma * V[s_prime]
                if value > max_value:
                    max_value = value
            delta = max(delta, abs(max_value - V[s]))
            V[s] = max_value
        iteration += 1
    return V, V_history

# Extract the optimal policy from the optimal value function
def extract_policy(env, V, gamma):
    num_states = env.num_states
    num_actions = len(env.action_space)
    policy_matrix = np.zeros((num_states, num_actions))
    for s in range(num_states):
        x = s % env.env_size[0]
        y = s // env.env_size[0]
        state = (x, y)
        if state in env.forbidden_states:
            continue
        action_values = []
        for a, action in enumerate(env.action_space):
            next_state, reward = env._get_next_state_and_reward(state, action)
            s_prime = next_state[1] * env.env_size[0] + next_state[0]
            value = reward + gamma * V[s_prime]
            action_values.append(value)
        best_action = np.argmax(action_values)
        policy_matrix[s, best_action] = 1
    return policy_matrix

if __name__ == "__main__":
    args = parser.parse_args()
    args.results_path = os.path.join(args.results_path, 'value_iteration')
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    # Initialize the environment
    env = GridWorld(args)
    state, _ = env.reset()
    _ = env.render(animation_interval=0.1)

    # Perform Value Iteration
    V_optimal, V_history = value_iteration(env, args.gamma)

    # Extract Optimal Policy
    optimal_policy_matrix = extract_policy(env, V_optimal, args.gamma)

    # Save the optimal value function and policy
    np.savetxt(args.results_path + '/V_optimal.csv', V_optimal, delimiter=',')
    np.savetxt(args.results_path + '/optimal_policy_matrix.csv', optimal_policy_matrix, delimiter=',')

    # Plot the optimal policy and state values
    # env.ax.clear()  # Clear the previous plot
    env = GridWorld(args)
    state = env.reset()    
    _ = env.render()  
    env.add_policy(optimal_policy_matrix)
    env.add_state_values(V_optimal, precision=2)
    env.render(index_V_his="optimal",animation_interval=2)


    # Plot the evolution of values
    num_iterations = len(V_history)
    if num_iterations > 10:
        indices_to_plot = list(range(5)) + list(range(num_iterations - 5, num_iterations))
    else:
        indices_to_plot = list(range(num_iterations))
    for idx in indices_to_plot:
        V = V_history[idx]
        env = GridWorld(args)
        state = env.reset()    
        _ = env.render() 
        env.add_state_values(V, precision=2)
        optimal_policy_matrix = extract_policy(env, V, args.gamma)
        env.add_policy(optimal_policy_matrix)
        env.render(index_V_his=idx,animation_interval=2)

    print("Value Iteration completed. Results have been saved.")