# SARSA Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Train agent with SARSA in Gym environment, making sequential decisions for maximizing cumulative rewards.

## SARSA LEARNING ALGORITHM
### Step 1:
Initialize the Q-table with random values for all state-action pairs.

### Step 2:
Initialize the current state S and choose the initial action A using an epsilon-greedy policy based on the Q-values in the Q-table.

### Step 3:
Repeat until the episode ends and then take action A and observe the next state S' and the reward R.

### Step 4:
Update the Q-value for the current state-action pair (S, A) using the SARSA update rule.

### Step 5:
Update State and Action and repeat the step 3 untill the episodes ends.

## SARSA LEARNING FUNCTION
Developed by: MIDHUN AZHAHU RAJA P

Register no:212222240066
```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):

    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False
        action = select_action(state, Q, epsilons[e])

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = select_action(next_state, Q, epsilons[e])

            td_target = reward + gamma * Q[next_state, np.argmax(Q[next_state])] * (1 - done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alphas[e] * td_error

            state = next_state
            action = next_action

        Q_track[e] = Q.copy()
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    pi = lambda s: np.argmax(Q[s])

    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
## value function
<img width="867" height="378" alt="image" src="https://github.com/user-attachments/assets/f2e4c408-b10a-4775-963f-67f67ca863cf" />


##  optimal policy.
<img width="1154" height="638" alt="image" src="https://github.com/user-attachments/assets/a14bd2f8-61f2-4fa8-8a4e-c443aa86e71a" />

<img width="1280" height="166" alt="image" src="https://github.com/user-attachments/assets/e494d70b-9b2b-428c-8097-f7ba73ebb08d" />

<img width="777" height="138" alt="image" src="https://github.com/user-attachments/assets/d55ddfe8-eef1-4797-83a3-1795e8c5f614" />






## Include plot comparing the state value functions of Monte Carlo method and SARSA learning.
<img width="1648" height="486" alt="image" src="https://github.com/user-attachments/assets/9aeefc5f-d86b-4fb6-b3af-815e6bfd5c7e" />
<img width="1664" height="496" alt="image" src="https://github.com/user-attachments/assets/0da90076-4cfd-4f96-9db5-52794b4895a0" />


## RESULT:

Thus to develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method has been implemented successfully
