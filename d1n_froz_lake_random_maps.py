import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

def state_to_coords(state, grid_size=4):
    row = state // grid_size
    col = state % grid_size
    return np.array([row, col], dtype=np.float32) / (grid_size - 1)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

grid_size = 4
state_dim = 2  # row, col
action_dim = 4  # up, down, left, right

q_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

memory = deque(maxlen=5000)

batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 2000
target_update = 50

for epi in range(episodes):
    # create a fresh random map each episode
    env = gym.make("FrozenLake-v1", desc=generate_random_map(size=grid_size), is_slippery=False)
    state, _ = env.reset()
    state = state_to_coords(state, grid_size)
    done = False
    total_reward = 0
    running_loss, updates = 0, 0

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(torch.FloatTensor(state))
                action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = state_to_coords(next_state, grid_size)

        memory.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

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
        print(f"Episode {epi}, reward: {total_reward}, epsilon: {epsilon:.2f}, loss: {avg_loss:.4f}")
