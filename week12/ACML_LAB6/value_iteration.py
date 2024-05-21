import torch

num_states = 9  # 3x3 grid
num_actions = 4  # actions: left, up, right, down
action_dict = {0: 'left', 1: 'up', 2: 'right', 3: 'down'}
states = torch.arange(num_states).reshape(3, 3)

# Initialize rewards and transitions
R = torch.full((num_states, num_actions), -1.0)  # General penalty for moving
goal_state = 8  # State s22 (3x3 grid flattened index)
R[7, 2] = 100  # Reward from s12 to s22 with action 'right'

# Initialize state transition probabilities
P = torch.zeros((num_states, num_actions, num_states))

# Define grid transitions
for i in range(3):
    for j in range(3):
        s = states[i, j]
        if j > 0:  # Can move left
            P[s, 0, states[i, j-1]] = 1
        else:
            P[s, 0, s] = 1  # Stay if off-grid left

        if i > 0:  # Can move up
            P[s, 1, states[i-1, j]] = 1
        else:
            P[s, 1, s] = 1  # Stay if off-grid up

        if j < 2:  # Can move right
            P[s, 2, states[i, j+1]] = 1
        else:
            P[s, 2, s] = 1  # Stay if off-grid right

        if i < 2:  # Can move down
            P[s, 3, states[i+1, j]] = 1
        else:
            P[s, 3, s] = 1  # Stay if off-grid down

# Barriers
P[4, 2, :] = 0  # s11 cannot move right to s12
P[4, 2, 4] = 1  # Stays at s11
P[7, 2, :] = 0  # s21 cannot move right to s22
P[7, 2, 7] = 1  # Stays at s21

# Setting rewards: Checking where no transition occurs
for s in range(num_states):
    for a in range(num_actions):
        if P[s, a, s] == 1 and (P[s, a].sum() == 1):  # No transition, only self-loops
            R[s, a] = -5  # Penalty for staying in place (off-grid or hitting a barrier)

gamma = 0.95  # Discount factor
epsilon = 1e-6  # Convergence threshold
V = torch.zeros(num_states)

# Value Iteration
while True:
    delta = 0
    new_V = torch.clone(V)
    for s in range(num_states):
        temp_values = torch.tensor([torch.sum(P[s, a, :] * (R[s, a] + gamma * V)) for a in range(num_actions)])
        new_V[s] = torch.max(temp_values)
        delta = max(delta, torch.abs(new_V[s] - V[s]))
    V = new_V
    if delta < epsilon:
        break

# Extract policy
policy = torch.zeros(num_states, dtype=torch.int32)
for s in range(num_states):
    policy[s] = torch.argmax(torch.tensor([torch.sum(P[s, a, :] * (R[s, a] + gamma * V)) for a in range(num_actions)]))

# Display results
print("Optimal Value Function V:")
print(V.reshape(3, 3))
print("\nOptimal Policy π:")
policy_names = [action_dict[a.item()] for a in policy]
for i in range(3):
    print(' | '.join(policy_names[i*3:(i+1)*3]))
