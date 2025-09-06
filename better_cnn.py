# import os
# import gymnasium as gym
# from gymnasium.envs.toy_text.frozen_lake import generate_random_map
# import numpy as np
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque
# import time

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# def state_to_grid(state, env):
#     """ The 3-channel state representation for the CNN. """
#     grid_size = env.unwrapped.nrow
#     agent_grid = np.zeros((grid_size, grid_size))
#     hole_grid = np.zeros((grid_size, grid_size))
#     goal_grid = np.zeros((grid_size, grid_size))
#     row, col = state // grid_size, state % grid_size
#     agent_grid[row, col] = 1
#     desc = env.unwrapped.desc
#     for r in range(grid_size):
#         for c in range(grid_size):
#             if desc[r, c] == b'H': hole_grid[r, c] = 1
#             elif desc[r, c] == b'G': goal_grid[r, c] = 1
#     grid = np.stack([agent_grid, hole_grid, goal_grid], axis=0)
#     return grid.astype(np.float32)

# class CNN_DQN(nn.Module):
#     """ The CNN architecture that matches your saved model. """
#     def __init__(self, in_channels, action_dim):
#         super(CNN_DQN, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=2),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=2),
#             nn.ReLU()
#         )
#         grid_size = 4
#         conv_out_size = 64 * (grid_size - 2) * (grid_size - 2)
#         self.fc = nn.Sequential(
#             nn.Linear(conv_out_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, action_dim)
#         )
#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)


# def get_valid_actions(state, grid_size=4):
#     """ Returns a boolean tensor indicating valid actions for a given state. """
#     row, col = state // grid_size, state % grid_size
#     actions = [True, True, True, True] 
#     if col == 0: actions[0] = False
#     if row == grid_size - 1: actions[1] = False
#     if col == grid_size - 1: actions[2] = False
#     if row == 0: actions[3] = False
#     return torch.tensor(actions, dtype=torch.bool, device=device)

# grid_size = 4
# in_channels = 3
# action_dim = 4
# model_file = "better_cnn_dqn_frozenlake_masked.pth" 

# q_net = CNN_DQN(in_channels, action_dim).to(device)
# target_net = CNN_DQN(in_channels, action_dim).to(device)
# target_net.load_state_dict(q_net.state_dict())

# optimizer = optim.Adam(q_net.parameters(), lr=0.0005)
# loss_fn = nn.MSELoss()
# memory = deque(maxlen=50000)

# batch_size = 128
# gamma = 0.99
# epsilon = 1.0
# epsilon_min = 0.01
# epsilon_decay = 0.99995
# num_maps = 200
# episodes_per_map = 100
# target_update_freq = 500

# start_time = time.time()
# total_steps = 0
# wins_in_last_100_maps = 0

# if os.path.exists(model_file):
#     print("Loading existing model...")
#     checkpoint = torch.load(model_file, map_location=device)
#     q_net.load_state_dict(checkpoint['q_net'])
#     target_net.load_state_dict(checkpoint['target_net'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     epsilon = checkpoint.get('epsilon', epsilon)

# for map_idx in range(num_maps):
#     desc = generate_random_map(size=grid_size)
#     env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False)
#     map_wins = 0

#     for epi in range(episodes_per_map):
#         state_int, _ = env.reset()
#         state = torch.from_numpy(state_to_grid(state_int, env)).unsqueeze(0).to(device)
#         done = False

#         while not done:
#             valid_actions = get_valid_actions(state_int)
#             if random.random() < epsilon:
#                 valid_action_indices = torch.where(valid_actions)[0]
#                 action = random.choice(valid_action_indices.tolist())
#             else:
#                 with torch.no_grad():
#                     q_values = q_net(state)
#                     masked_q_values = q_values.clone()
#                     masked_q_values[0, ~valid_actions] = -float('inf')
#                     action = torch.argmax(masked_q_values).item()

#             next_state_int, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
            
#             is_win = terminated and reward == 1.0
#             if is_win:
#                 map_wins += 1

#             if is_win: reward = 10.0
#             elif terminated and reward == 0.0: reward = -1.0
#             else: reward = -0.01

#             next_state = torch.from_numpy(state_to_grid(next_state_int, env)).unsqueeze(0).to(device)
#             memory.append((state.cpu(), action, reward, next_state.cpu(), done))
            
#             state = next_state
#             state_int = next_state_int

#             if len(memory) >= batch_size:
#                 batch = random.sample(memory, batch_size)
#                 states_t = torch.cat([s for s,a,r,ns,d in batch]).to(device)
#                 actions_t = torch.LongTensor([a for s,a,r,ns,d in batch]).unsqueeze(1).to(device)
#                 rewards_t = torch.FloatTensor([r for s,a,r,ns,d in batch]).unsqueeze(1).to(device)
#                 next_states_t = torch.cat([ns for s,a,r,ns,d in batch]).to(device)
#                 dones_t = torch.FloatTensor([d for s,a,r,ns,d in batch]).unsqueeze(1).to(device)

#                 q_values = q_net(states_t).gather(1, actions_t)
#                 with torch.no_grad():
#                     max_next_q = target_net(next_states_t).max(1)[0].unsqueeze(1)
#                     q_target = rewards_t + gamma * max_next_q * (1 - dones_t)

#                 loss = loss_fn(q_values, q_target)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 total_steps += 1
#                 if total_steps % target_update_freq == 0:
#                     target_net.load_state_dict(q_net.state_dict())

#         epsilon = max(epsilon_min, epsilon * epsilon_decay)

#     if map_wins > 0:
#         wins_in_last_100_maps += 1
#     if (map_idx + 1) % 10 == 0:
#         win_rate = (wins_in_last_100_maps / 10) * 100 if map_idx > 0 else map_wins
#         wins_in_last_100_maps = 0
#         elapsed_time = time.time() - start_time
#         print(f"Map Block [{map_idx-9}-{map_idx+1}], Win Rate (per map): {win_rate:.1f}%, Epsilon: {epsilon:.3f}, Time: {elapsed_time:.1f}s")
        
#         torch.save({'q_net': q_net.state_dict()}, model_file)
#         print(f"Model saved after Map Block {map_idx+1}")


import os
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def state_to_grid(state, env):
    """ The 3-channel state representation for the CNN. """
    grid_size = env.unwrapped.nrow
    agent_grid = np.zeros((grid_size, grid_size))
    hole_grid = np.zeros((grid_size, grid_size))
    goal_grid = np.zeros((grid_size, grid_size))
    row, col = state // grid_size, state % grid_size
    agent_grid[row, col] = 1
    desc = env.unwrapped.desc
    for r in range(grid_size):
        for c in range(grid_size):
            if desc[r, c] == b'H': 
                hole_grid[r, c] = 1
            elif desc[r, c] == b'G': 
                goal_grid[r, c] = 1
    grid = np.stack([agent_grid, hole_grid, goal_grid], axis=0)
    return grid.astype(np.float32)

class CNN_DQN(nn.Module):
    """ A slightly deeper but still CPU-friendly CNN. """
    def __init__(self, in_channels, action_dim):
        super(CNN_DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=2),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),          
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2),          
            nn.ReLU()
        )
        grid_size = 4
        conv_out_size = 64 * (grid_size - 3) * (grid_size - 3)  # 64*1*1 = 64
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),   
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def get_valid_actions(state, grid_size=4):
    """ Returns a boolean tensor indicating valid actions for a given state. """
    row, col = state // grid_size, state % grid_size
    actions = [True, True, True, True] 
    if col == 0: actions[0] = False
    if row == grid_size - 1: actions[1] = False
    if col == grid_size - 1: actions[2] = False
    if row == 0: actions[3] = False
    return torch.tensor(actions, dtype=torch.bool, device=device)

grid_size = 4
in_channels = 3
action_dim = 4
model_file = "better_cnn_dqn_frozenlake_masked.pth" 

q_net = CNN_DQN(in_channels, action_dim).to(device)
target_net = CNN_DQN(in_channels, action_dim).to(device)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()
memory = deque(maxlen=80000)

batch_size = 128
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99995
num_cycles = 8           
num_maps_per_cycle = 300 
episodes_per_map = 100
target_update_freq = 500

start_time = time.time()
total_steps = 0
wins_in_block = 0

if os.path.exists(model_file):
    print("Loading existing model...")
    checkpoint = torch.load(model_file, map_location=device)
    q_net.load_state_dict(checkpoint['q_net'])
    target_net.load_state_dict(checkpoint['target_net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epsilon = checkpoint.get('epsilon', epsilon)

for cycle in range(num_cycles):
    print(f"\nCYCLE {cycle+1}/{num_cycles}")
    wins_in_block = 0

    for map_idx in range(num_maps_per_cycle):
        desc = generate_random_map(size=grid_size, p=0.8)
        env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False)
        map_wins = 0

        for epi in range(episodes_per_map):
            state_int, _ = env.reset()
            state = torch.from_numpy(state_to_grid(state_int, env)).unsqueeze(0).to(device)
            done = False

            while not done:
                valid_actions = get_valid_actions(state_int)
                if random.random() < epsilon:
                    valid_action_indices = torch.where(valid_actions)[0]
                    action = random.choice(valid_action_indices.tolist())
                else:
                    with torch.no_grad():
                        q_values = q_net(state)
                        masked_q_values = q_values.clone()
                        masked_q_values[0, ~valid_actions] = -float('inf')
                        action = torch.argmax(masked_q_values).item()

                next_state_int, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                is_win = terminated and reward == 1.0
                if is_win:
                    map_wins += 1

                if is_win: reward = 10.0
                elif terminated and reward == 0.0: reward = -1.0
                else: reward = -0.01

                next_state = torch.from_numpy(state_to_grid(next_state_int, env)).unsqueeze(0).to(device)
                memory.append((state.cpu(), action, reward, next_state.cpu(), done))

                state = next_state
                state_int = next_state_int

                if len(memory) >= batch_size:
                    batch = random.sample(memory, batch_size)
                    states_t = torch.cat([s for s,a,r,ns,d in batch]).to(device)
                    actions_t = torch.LongTensor([a for s,a,r,ns,d in batch]).unsqueeze(1).to(device)
                    rewards_t = torch.FloatTensor([r for s,a,r,ns,d in batch]).unsqueeze(1).to(device)
                    next_states_t = torch.cat([ns for s,a,r,ns,d in batch]).to(device)
                    dones_t = torch.FloatTensor([d for s,a,r,ns,d in batch]).unsqueeze(1).to(device)

                    q_values = q_net(states_t).gather(1, actions_t)
                    with torch.no_grad():
                        max_next_q = target_net(next_states_t).max(1)[0].unsqueeze(1)
                        q_target = rewards_t + gamma * max_next_q * (1 - dones_t)

                    loss = loss_fn(q_values, q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_steps += 1
                    if total_steps % target_update_freq == 0:
                        target_net.load_state_dict(q_net.state_dict())

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if map_wins > 0:
            wins_in_block += 1

        if (map_idx + 1) % 50 == 0:
            win_rate = (wins_in_block / 50) * 100
            wins_in_block = 0
            elapsed_time = time.time() - start_time
            print(f"[Cycle {cycle+1}] Maps {map_idx-49}-{map_idx+1}, Win Rate: {win_rate:.1f}%, Epsilon: {epsilon:.3f}, Time: {elapsed_time:.1f}s")

            torch.save({
                'q_net': q_net.state_dict(),
                'target_net': target_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epsilon': epsilon
            }, model_file)
            print(f"Model saved after Cycle {cycle+1}, Map Block {map_idx+1}")

# After running 5 and a half cycles (which took like 1o hrs) and testing, I've come to the realisation that, 
# CNN might be an overkill for this simple frozen lake environment. And by talking to Gemini I also came to know
# about the "echo-chamber" or "positive bias" problem, which can be solved through a Double-DQN...? 
# Still not very clear about it but I've asked for a new code with DDQN implementation and it will be the last iteration
# of this Frozen Lake programs. Will understand more about the problem and the DDQN solution, will execute the code and see the results..