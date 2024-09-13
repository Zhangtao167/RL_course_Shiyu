
import sys
import os
from grid_world import GridWorld
import random
import numpy as np
import imageio
import pdb
# Example usage:

from typing import Union
import numpy as np
import argparse

parser = argparse.ArgumentParser("Grid World Environment")

## ==================== User settings ===================='''
# specify the number of columns and rows of the grid world
parser.add_argument("--env-size", type=Union[list, tuple, np.ndarray], default=(5,5) )   

# specify the start state
parser.add_argument("--start-state", type=Union[list, tuple, np.ndarray], default=(0,0))

# specify the target state
parser.add_argument("--target-state", type=Union[list, tuple, np.ndarray], default=(4,4))

# sepcify the forbidden states
parser.add_argument("--forbidden-states", type=list, default=[ (2, 1), (3, 3), (1, 3)] )

# sepcify the reward when reaching target
parser.add_argument("--reward-target", type=float, default = 10)

# sepcify the reward when entering into forbidden area
parser.add_argument("--reward-forbidden", type=float, default = -5)

# sepcify the reward for each step
parser.add_argument("--reward-step", type=float, default = -1)

# discount factor
parser.add_argument("--gamma", type=float, default = 0.9)
## ==================== End of User settings ====================


## ==================== Advanced Settings ====================
parser.add_argument("--action-space", type=list, default=[(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)] )  # down, right, up, left, stay           
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--animation-interval", type=float, default = 0.2)
parser.add_argument("--results_path", type=str, default = "/storage/zhangtao/project2024/RL_course_Shiyu/Code for grid world/python_version/assignment/assignment1/results")
parser.add_argument("--exp_name", type=str, default = "exp_deterministic")
parser.add_argument("--exp_date", type=str, default = "2024-09-13")
parser.add_argument("--policy_type", type=str, help = "deterministic or stochastic",default="deterministic")
parser.add_argument("--plot_policy", type=bool, default = False)
parser.add_argument("--length_simulation", type=int, default = 50)
parser.add_argument("--is_save_gif", type=bool, default = True)
## ==================== End of Advanced settings ====================


if __name__ == "__main__": 
    args = parser.parse_args()   
    args.results_path = args.results_path + "/" + args.exp_name + "/" + args.exp_date + "/" + args.policy_type
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)            
    env = GridWorld(args)
    state = env.reset()               
    for t in range(1):
        _ = env.render()
        action = (0,0)
        next_state, reward, done, info = env.step(action)
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        # if done:
        #     break
    
    # Add policy
    if args.policy_type == "deterministic":
        policy_matrix = np.zeros((env.num_states, len(env.action_space)))
        for state_index in range(env.num_states):
            x = state_index % env.env_size[0]
            y = state_index // env.env_size[0]

            if (x, y) == env.target_state:
                action_index = 4  # stay
            elif x==2 and y==3:
                action_index = 0
            else:
                delta_x = env.target_state[0] - x
                delta_y = env.target_state[1] - y

                if abs(delta_x) > abs(delta_y):
                    if delta_x > 0:
                        action_index = 1  # right
                    else:
                        action_index = 3  # left
                else:
                    if delta_y > 0:
                        action_index = 0  # down
                    else:
                        action_index = 2  # up

            policy_matrix[state_index, action_index] = 1

    elif args.policy_type == "stochastic":
        policy_matrix = np.zeros((env.num_states, len(env.action_space)))
        for state_index in range(env.num_states):
            x = state_index % env.env_size[0]
            y = state_index // env.env_size[0]

            if (x, y) == env.target_state:
                policy_matrix[state_index, 4] = 1  # stay
            else:
                distances = []
                for action_index in range(len(env.action_space)):
                    action = env.action_space[action_index]
                    dx, dy = action
                    x_new = x + dx
                    y_new = y + dy

                    if (0 <= x_new < env.env_size[0] and
                        0 <= y_new < env.env_size[1] and
                        (x_new, y_new) not in env.forbidden_states):
                        distance = abs(x_new - env.target_state[0]) + abs(y_new - env.target_state[1])
                        distances.append(-distance)
                    else:
                        distances.append(float('-inf'))

                exp_values = np.exp(distances)
                probabilities = exp_values / np.sum(exp_values)
                policy_matrix[state_index, :] = probabilities
    else:
        raise ValueError("Invalid policy_type. Choose 'deterministic' or 'stochastic'.")

    env.add_policy(policy_matrix)

    # Render the environment
    _ = env.render(animation_interval=2)
    if not args.plot_policy:
        env = GridWorld(args)
        state = env.reset()  
        if args.policy_type == "deterministic":
            img_list = []
            return_dis = 0
            for t in range(args.length_simulation):
                # save the image list and draw .gif
                img = env.render()
                img_list.append(img)
                # import pdb; pdb.set_trace()
                # based on the policy matrix, select the action
                try:
                    index_p = state[0][1] * env.env_size[0] + state[0][0]
                except:
                    index_p = state[1] * env.env_size[0] + state[0]
                action = np.argmax(policy_matrix[index_p])
                action = env.action_space[action]
                next_state, reward, done, info = env.step(action)
                state = next_state
                print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
                return_dis += reward*(args.gamma**t)
            if args.is_save_gif:
                imageio.mimsave(args.results_path+'/traj.gif', img_list, fps=5)
        elif args.policy_type == "stochastic":
            index = [0,1,2,3,4]
            img_list = []
            return_dis = 0
            for t in range(args.length_simulation):
                img = env.render()
                img_list.append(img)
                # based on the policy matrix, select the action
                try:
                    index_p = state[0][1] * env.env_size[0] + state[0][0]
                except:
                    index_p = state[1] * env.env_size[0] + state[0]
                action = np.random.choice(index, p=policy_matrix[index_p])
                action = env.action_space[action]
                next_state, reward, done, info = env.step(action)
                state = next_state
                return_dis += reward*(args.gamma**t)
                print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
            if args.is_save_gif:
                imageio.mimsave(args.results_path+'/traj.gif', img_list, fps=5)
    print("The return of the traj is: ", return_dis)
    print("Simulation finished, results are saved in ", args.results_path)
        
