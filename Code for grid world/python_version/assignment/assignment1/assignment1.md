# Assignment 1 Report

**Course:** Reinforcement Learning  
**Student Number:** [Your Student Number]

---

## 1. Software Description (5%)

Describe the software that you use, such as the programming language, libraries, and any custom tools or code used for this assignment.

Example:
- **Programming Language:** Python
- **Libraries Used:** NumPy, Matplotlib, etc.
- **Custom Code:** Custom grid-world simulation, policy plotting, etc.

---

## 2. Reward Setting and Discount Rate (5%)

Describe the reward setting and discount rate used in the task.

- **Reward setting:**  
  - \( r_{\text{boundary}} = -1 \)  
  - \( r_{\text{forbidden}} = -1 \)  
  - \( r_{\text{target}} = 1 \)
  
- **Discount rate (\(\gamma\)):** 0.9

---

## 3. Policy Design

### 3.1 Deterministic Policy (10%)

#### 3.1.1 Policy Description (Table)

Provide the table or array representation of the deterministic policy.

Example:

| State | Action |
|-------|--------|
| s1    | Right  |
| s2    | Down   |
| s3    | Left   |

#### 3.1.2 Policy Plot (10%)

Plot the policy using arrows to indicate actions at each state.  
(Include a figure with the policy visualized on the grid.)

#### 3.1.3 Trajectory and Discounted Return (10%)

- **Starting State:** (Choose an arbitrary starting state, e.g., s1)
- **Trajectory Length:** 50 steps
- **Discounted Return:** Calculate and display the return for this trajectory.

---

### 3.2 Stochastic Policy (10%)

#### 3.2.1 Policy Description (Table)

Provide the table or array representation of the stochastic policy.

Example:

| State | Action Probabilities             |
|-------|----------------------------------|
| s1    | [Right: 0.8, Down: 0.2]          |
| s2    | [Down: 0.7, Right: 0.3]          |

#### 3.2.2 Policy Plot (10%)

Plot the stochastic policy using arrows, showing probabilities of each action.  
(Include a figure with the stochastic policy visualized on the grid.)

#### 3.2.3 Trajectory and Discounted Return (10%)

- **Starting State:** (Choose an arbitrary starting state, e.g., s1)
- **Trajectory Length:** 50 steps
- **Discounted Return:** Calculate and display the return for this trajectory.

---

## 4. Policy Comparison (10%)

Discuss your observations about the two policies. Focus on their performance, differences in the trajectories, and discounted returns.

---

## 5. Key Parts of the Code

Include and explain the key parts of your code used for plotting, calculating the trajectory, and discounted return.

---

## 6. Conclusion

Summarize the results and the key takeaways from the experiments.

---