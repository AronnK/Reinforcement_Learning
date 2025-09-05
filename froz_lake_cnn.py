import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gymnasium import RewardWrapper
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def state_to_grid(state, env):
    """
    Return a 3-channel grid representation:
    0 - agent position
    1 - holes
    2 - goal
    Shape: (3, grid_size, grid_size)
    """
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
    
    grid = np.stack([agent_grid, hole_grid, goal_grid], axis=0)  # (C, H, W)
    return grid.astype(np.float32)

class ShapedFrozenLake(RewardWrapper):
    def __init__(self, env, max_steps=8):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)

    def reward(self, reward):
        self.current_step += 1
        if reward == 1.0:
            return 10.0
        elif self.env.unwrapped.desc[
            self.env.unwrapped.s // self.env.unwrapped.ncol,
            self.env.unwrapped.s % self.env.unwrapped.ncol
        ] == b'H':
            return -1.0
        elif self.current_step > self.max_steps:
            return -0.5
        else:
            return 0.01

class CNN_DQN(nn.Module):
    def __init__(self, in_channels, action_dim):
        super(CNN_DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=2),  # output: (32, grid-1, grid-1)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),           # output: (64, grid-2, grid-2)
            nn.ReLU()
        )
        conv_out_size = 64 * (grid_size - 2) * (grid_size - 2)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

grid_size = 4
in_channels = 3
action_dim = 4
model_file = "cnn_dqn_frozenlake.pth"

q_net = CNN_DQN(in_channels, action_dim).to(device)
target_net = CNN_DQN(in_channels, action_dim).to(device)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()
memory = deque(maxlen=50000)

batch_size = 128
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99995
episodes_per_map = 500
num_maps = 50
target_update = 25

if os.path.exists(model_file):
    print("Loading existing model...")
    checkpoint = torch.load(model_file, map_location=device)
    q_net.load_state_dict(checkpoint['q_net'])
    target_net.load_state_dict(checkpoint['target_net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epsilon = checkpoint.get('epsilon', epsilon)

for map_idx in range(num_maps):
    print(f"\n=== Map {map_idx+1}/{num_maps} ===")
    desc = generate_random_map(size=grid_size)
    env = ShapedFrozenLake(gym.make("FrozenLake-v1", desc=desc, is_slippery=False))

    for epi in range(episodes_per_map):
        state, _ = env.reset()
        state = state_to_grid(state, env)
        state = torch.from_numpy(state).unsqueeze(0).to(device)  # (1, C, H, W)
        done = False
        total_reward = 0
        running_loss, updates = 0, 0

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(state)
                    action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_grid = state_to_grid(next_state, env)
            next_state_tensor = torch.from_numpy(next_state_grid).unsqueeze(0).to(device)

            memory.append((state, action, reward, next_state_tensor, done))
            state = next_state_tensor
            total_reward += reward

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states).float().to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.cat(next_states).float().to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                q_values = q_net(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                    q_target = rewards + gamma * max_next_q * (1 - dones)

                loss = loss_fn(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                updates += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if epi % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        if epi % 100 == 0:
            avg_loss = running_loss / updates if updates > 0 else 0
            print(f"Episode {epi}, reward: {total_reward:.2f}, epsilon: {epsilon:.2f}, loss: {avg_loss:.4f}")

    torch.save({
        'q_net': q_net.state_dict(),
        'target_net': target_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epsilon': epsilon
    }, model_file)
    print(f"Model saved after Map {map_idx+1}")
 
 # After one training cycle, loss: 0.0346
 # The model became a pessimistic expert, it did predict failing correctly...
 # After second training cycle, loss: 0.01423, but way more wins than before.
 # Model is now trying to win and so loss is a little high than before before its not predicting failure but rather winning.
 # So way more wins...

