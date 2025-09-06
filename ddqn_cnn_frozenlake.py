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
print(f"Using device: {device}")

def state_to_grid(state, env):
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
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, grid_size, grid_size)
            conv_out_size = self.conv(dummy_input).flatten().shape[0]
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

def get_valid_actions_mask(states_int, grid_size=4, device='cpu'):
    batch_size = len(states_int)
    masks = torch.ones((batch_size, 4), dtype=torch.bool, device=device)
    for i, state in enumerate(states_int):
        row, col = state // grid_size, state % grid_size
        if col == 0: masks[i, 0] = False
        if row == grid_size - 1: masks[i, 1] = False
        if col == grid_size - 1: masks[i, 2] = False
        if row == 0: masks[i, 3] = False
    return masks

grid_size = 4
in_channels = 3
action_dim = 4
model_file = "ddqn_cnn_frozenlake.pth" 

q_net = CNN_DQN(in_channels, action_dim).to(device)
target_net = CNN_DQN(in_channels, action_dim).to(device)
target_net.load_state_dict(q_net.state_dict())
target_net.eval() 

optimizer = optim.Adam(q_net.parameters(), lr=0.0001) 
loss_fn = nn.MSELoss()
memory = deque(maxlen=75000) 

batch_size = 128
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.99995 
total_maps = 800  
episodes_per_map = 75 
target_update_freq = 2000 

start_time = time.time()
total_steps = 0
wins_in_block = 0

if os.path.exists(model_file):
    print("Loading existing model to resume training...")
    checkpoint = torch.load(model_file, map_location=device)
    q_net.load_state_dict(checkpoint['q_net'])
    target_net.load_state_dict(checkpoint['target_net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epsilon = checkpoint.get('epsilon', epsilon)
    total_steps = checkpoint.get('total_steps', 0)

num_cycles = 2

for cycle in range(num_cycles):
    print(f"\n=== Starting Cycle {cycle+1}/{num_cycles} ===")
    print(f"Initial Epsilon: {epsilon:.4f}, Steps: {total_steps}")

    for map_idx in range(total_maps):
        desc = generate_random_map(size=grid_size, p=0.8)
        if desc[0][0] == 'H' or desc[grid_size - 1][grid_size - 1] == 'H':
            continue
        env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False)
        map_wins = 0

        for epi in range(episodes_per_map):
            state_int, _ = env.reset()
            done = False
            while not done:
                state_grid = torch.from_numpy(state_to_grid(state_int, env)).unsqueeze(0).to(device)
                valid_actions_mask = get_valid_actions_mask([state_int], device=device)[0]
                if random.random() < epsilon:
                    valid_action_indices = torch.where(valid_actions_mask)[0]
                    action = random.choice(valid_action_indices.tolist())
                else:
                    with torch.no_grad():
                        q_values = q_net(state_grid)                  
                        q_values[0, ~valid_actions_mask] = -float('inf')
                        action = torch.argmax(q_values).item()
                next_state_int, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if terminated and reward == 1.0:
                    final_reward = 1.0
                    map_wins += 1
                elif terminated:
                    final_reward = -1.0
                else:
                    final_reward = -0.075 
                next_state_grid = torch.from_numpy(state_to_grid(next_state_int, env)).unsqueeze(0)
                memory.append((state_grid.cpu(), action, final_reward, next_state_grid.cpu(), done, next_state_int))
                state_int = next_state_int
                if len(memory) >= batch_size:
                    batch = random.sample(memory, batch_size)
                    states_t = torch.cat([s[0] for s in batch]).to(device)
                    actions_t = torch.LongTensor([s[1] for s in batch]).unsqueeze(1).to(device)
                    rewards_t = torch.FloatTensor([s[2] for s in batch]).unsqueeze(1).to(device)
                    next_states_t = torch.cat([s[3] for s in batch]).to(device)
                    dones_t = torch.FloatTensor([s[4] for s in batch]).unsqueeze(1).to(device)
                    next_states_int_t = [s[5] for s in batch]
                    q_values = q_net(states_t).gather(1, actions_t)
                    next_valid_actions_mask = get_valid_actions_mask(next_states_int_t, device=device)
                    next_q_values_online = q_net(next_states_t).clone()
                    next_q_values_online[~next_valid_actions_mask] = -float('inf') 
                    best_next_actions = next_q_values_online.argmax(1).unsqueeze(1)
                    with torch.no_grad():
                        next_q_from_target = target_net(next_states_t).gather(1, best_next_actions)
                        q_target = rewards_t + gamma * next_q_from_target * (1 - dones_t)
                    loss = loss_fn(q_values, q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_steps += 1
                    if total_steps % target_update_freq == 0:
                        print(f"--- Step {total_steps}: Updating target network ---")
                        target_net.load_state_dict(q_net.state_dict())
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if map_wins > 0:
            wins_in_block += 1
        if (map_idx + 1) % 50 == 0:
            win_rate = (wins_in_block / 50) * 100
            wins_in_block = 0
            elapsed_time = time.time() - start_time
            print(f"Cycle {cycle+1} | Maps {map_idx-49}-{map_idx+1}/{total_maps} | Win Rate: {win_rate:.1f}% | Epsilon: {epsilon:.4f} | Steps: {total_steps} | Time: {elapsed_time:.1f}s")
            torch.save({
                'q_net': q_net.state_dict(),
                'target_net': target_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epsilon': epsilon,
                'total_steps': total_steps
            }, model_file)
            print(f"Model saved to {model_file}")

print("\nTraining complete.")

# Okay so wayyy better results from this execution compared to DQN...
# The main difference betw DQN and DDQN is that, in DQN the targ_net itself chose the next best action and evaluated itself,
# which, when the estimations are noisy would always choose the highest rewarding action and stay there, so positive bias...
# How DDQN differs is that, here q_net is the one choosing the action and sending it to targ_net to evaluate, so even if the current
# q_values are noisy, the more stable targ_net would make the correct estimations as it has knowledge base that is a little older.

# DQN: Qtarget​(s,a)=r+γa′max​Qtarget_net​(s′,a′)
# DDQN: Qtarget​(s,a)=r+γQtarget_net​(s′,arga′max​Qq_net​(s′,a′))