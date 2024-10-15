# Assignment 3 Report

**Course:** Reinforcement Learning  
**Student Number:** 20241203425

---

## 1. Software Description 

- **Programming Language:** Python 3.12
- **Libraries Used:** NumPy, Matplotlib, etc. More details can be found in [requirements.txt](./code/requirements.txt).
- **Custom Code:** RL_alg1.py, etc.

---

## 2. Problem setup

Describe the reward setting and discount rate used in the task.

- **Task:** 
•  1 target area, 3 forbidden areas
• 13 states, each of which has 5 actions
- **Reward setting:**  
  - \( r_{\text{step}} = -0.5 \)  
  - \( r_{\text{forbidden}} = -1 \) 
  - \( r_{\text{boundary}} = -1 \) 
  - \( r_{\text{target}} = 1 \)
  
- **Discount rate (\(\gamma\)):** 0.9

---

## 3. Value iteration
### 3.1 My understanding of value iteration
The **Value Iteration Algorithm** is a dynamic programming technique used to find the optimal policy for a Markov Decision Process (MDP). It computes the optimal value function and the corresponding optimal policy by iteratively updating the value of each state based on the Bellman optimality equation. Here's a brief breakdown of how it works:

1. **Value Function**: The value function \( V(s) \) represents the expected cumulative reward that an agent can achieve starting from state \( s \) and following an optimal policy.

2. **Bellman Optimality Equation**:
   The core of the algorithm is the Bellman equation, which defines the value of a state as the maximum expected reward from all possible actions:
   \[
   V(s) = \max_{a} \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V(s') \right]
   \]
   - \( R(s, a) \) is the immediate reward for taking action \( a \) in state \( s \).
   - \( P(s'|s, a) \) is the probability of transitioning to state \( s' \) given state \( s \) and action \( a \).
   - \( \gamma \) is the discount factor, which determines how much future rewards are valued compared to immediate rewards.

3. **Iteration Process**:
   - The algorithm starts by initializing the value of all states to arbitrary values, typically zeros.
   - In each iteration, the value function is updated by applying the Bellman equation to every state, taking into account all possible actions and their outcomes.
   - This process is repeated until the value function converges, meaning the change in values between iterations is smaller than a predefined threshold \( \theta \).

4. **Optimal Policy Extraction**:
   Once the value function converges, the optimal policy \( \pi^*(s) \) can be extracted by selecting the action that maximizes the expected cumulative reward from each state:
   \[
   \pi^*(s) = \arg\max_{a} \left[ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V(s') \right]
   \]

5. **Convergence**:
   The algorithm converges when the maximum difference between values in successive iterations (denoted as \( \delta \)) becomes smaller than a predefined small threshold \( \theta \), indicating that further updates will not significantly change the values.

### 3.2 My implementation of value iteration

```python
def value_iteration(env, gamma, theta=1e-6):
  '''
  Value Iteration Algorithm
  args: 
    env: GridWorld object
    gamma: discount factor
    theta: convergence threshold
  return:
    V: optimal state values
    V_history: history of state values during iterations
  '''
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
```

### Key Points of the Implementation:
1. **Initialization**: 
   - The value function `V` is initialized to zeros, and `delta` starts as infinity.
   - `V_history` stores the value function at each iteration for later analysis or plotting.

2. **Main Iteration Loop**: 
   - The loop continues until the maximum change (`delta`) between iterations is less than `theta`.
   - For each state, the algorithm checks all possible actions and computes the expected value using the Bellman equation.
   - The maximum expected value across all actions is chosen as the new value for the state.

3. **Convergence Check**: 
   - The value function is updated for each state, and the algorithm tracks the largest change (`delta`) between iterations.
   - Once `delta` is smaller than `theta`, the loop stops, and the function returns the optimal value function and its history.


### 3.3 Plot of the  obtained optimal policy and optimal state values
**Optimal Policy**
![optimal policy](./results/value_iteration/iteration_optimal.png)
### 3.4 The evolvement of the value
**The first five**
![iter 0](./results/value_iteration/iteration_0.png)
![iter 1](./results/value_iteration/iteration_1.png)
![iter 2](./results/value_iteration/iteration_2.png)
![iter 3](./results/value_iteration/iteration_3.png)
![iter 4](./results/value_iteration/iteration_4.png)

**The last five**
![iter 128](./results/value_iteration/iteration_128.png)
![iter 129](./results/value_iteration/iteration_129.png)
![iter 130](./results/value_iteration/iteration_130.png)
![iter 131](./results/value_iteration/iteration_131.png)
![iter 132](./results/value_iteration/iteration_132.png)

## 4.  Observations from Value Iteration Experiments

1. **Convergence of the Value Function**:
   - The value function starts at zero and gradually updates during each iteration. Initially, states closer to the target show higher values, and over time, this effect spreads to farther states.
   - The algorithm converges when the value updates become very small, indicating stability in the state values.

2. **Influence of Forbidden States**:
   - Forbidden states act as barriers, and the agent avoids them. The value of states near forbidden states is lower due to the penalty, causing the agent to seek safer paths.
   - In environments with multiple forbidden states, the optimal path becomes more complex as the agent navigates around obstacles.

3. **Policy Behavior**:
   - The policy extracted from the value function directs the agent toward the target state while avoiding forbidden states. In simple environments, the policy follows a direct path; in more complex grids, the agent takes longer, safer routes.

4. **Value Evolution**:
   - As iterations proceed, the value of states updates progressively, starting from the target state and propagating throughout the grid. This evolution shows how the agent's strategy improves over time.

5. **Overall Performance**:
   - The value iteration algorithm efficiently finds the optimal policy, and the number of iterations required depends on the grid’s complexity, the discount factor, and the presence of forbidden states.