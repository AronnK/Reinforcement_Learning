# import gymnasium as gym
# from gymnasium.envs.toy_text.frozen_lake import generate_random_map
# import numpy as np
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque

# def state_to_coords(state, grid_size=4):
#     row = state // grid_size
#     col = state % grid_size
#     return np.array([row, col], dtype=np.float32) / (grid_size - 1)

# class DQN(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(DQN, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_dim)
#         )
#     def forward(self, x):
#         return self.fc(x)

# grid_size = 4
# state_dim = 2  # row, col
# action_dim = 4  # up, down, left, right

# q_net = DQN(state_dim, action_dim)
# target_net = DQN(state_dim, action_dim)
# target_net.load_state_dict(q_net.state_dict())

# optimizer = optim.Adam(q_net.parameters(), lr=0.001)
# loss_fn = nn.MSELoss()

# memory = deque(maxlen=5000)

# batch_size = 64
# gamma = 0.99
# epsilon = 1.0
# epsilon_min = 0.01
# epsilon_decay = 0.995
# episodes = 2000
# target_update = 50

# for epi in range(episodes):
#     # create a fresh random map each episode
#     env = gym.make("FrozenLake-v1", desc=generate_random_map(size=grid_size), is_slippery=False)
#     state, _ = env.reset()
#     state = state_to_coords(state, grid_size)
#     done = False
#     total_reward = 0
#     running_loss, updates = 0, 0

#     while not done:
#         if random.random() < epsilon:
#             action = env.action_space.sample()
#         else:
#             with torch.no_grad():
#                 q_values = q_net(torch.FloatTensor(state))
#                 action = torch.argmax(q_values).item()

#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
#         next_state = state_to_coords(next_state, grid_size)

#         memory.append((state, action, reward, next_state, done))

#         state = next_state
#         total_reward += reward

#         if len(memory) >= batch_size:
#             batch = random.sample(memory, batch_size)
#             states, actions, rewards, next_states, dones = zip(*batch)

#             states = torch.FloatTensor(states)
#             actions = torch.LongTensor(actions).unsqueeze(1)
#             rewards = torch.FloatTensor(rewards).unsqueeze(1)
#             next_states = torch.FloatTensor(next_states)
#             dones = torch.FloatTensor(dones).unsqueeze(1)

#             q_values = q_net(states).gather(1, actions)

#             with torch.no_grad():
#                 max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
#                 q_target = rewards + gamma * max_next_q * (1 - dones)

#             loss = loss_fn(q_values, q_target)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             updates += 1

#     epsilon = max(epsilon_min, epsilon * epsilon_decay)

#     if epi % target_update == 0:
#         target_net.load_state_dict(q_net.state_dict())

#     if epi % 100 == 0:
#         avg_loss = running_loss / updates if updates > 0 else 0
#         print(f"Episode {epi}, reward: {total_reward}, epsilon: {epsilon:.2f}, loss: {avg_loss:.4f}")

# In prev code, the model kept learning on the same map which kinda makes it memorize the map...
# So here we moved to a more dynamic hole-reward situation, and changed the OHE to co-ordinates for better learning...
# Lowest loss of 0.0002 was achieved, but the reward is never really 1, the model just learns to take steps thats all...

# Solution: Add a negative reward when stepped on holes, and positive reward when reached target.

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gymnasium import RewardWrapper

def state_to_onehot(state, num_states=16):
    onehot = np.zeros(num_states, dtype=np.float32)
    onehot[state] = 1.0
    return onehot

# class ShapedFrozenLake(RewardWrapper):
#     def __init__(self, env):
#         super().__init__(env)

#     def reward(self, reward):
#         if reward == 1.0:  # reached goal
#             return 10.0
#         elif self.env.unwrapped.desc[
#             self.env.unwrapped.s // self.env.unwrapped.ncol,
#             self.env.unwrapped.s % self.env.unwrapped.ncol
#         ] == b'H':
#             return -1.0  # fell into hole
#         else:
#             return 0.001  # small step reward

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
        if reward == 1.0:  # goal
            return 10.0
        elif self.env.unwrapped.desc[
            self.env.unwrapped.s // self.env.unwrapped.ncol,
            self.env.unwrapped.s % self.env.unwrapped.ncol
        ] == b'H':  # hole
            return -1.0
        elif self.current_step > self.max_steps:  # step limit penalty
            return -0.5
        else:
            return 0.01  # small step reward


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

grid_size = 4
num_states = grid_size * grid_size
state_dim = num_states 
action_dim = 4  # up, down, left, right

q_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()

memory = deque(maxlen=50000)

batch_size = 128
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.99995
episodes = 50000
target_update = 25

for epi in range(episodes):
    env = ShapedFrozenLake(gym.make("FrozenLake-v1", desc=generate_random_map(size=grid_size), is_slippery=False))
    # env = ShapedFrozenLake(gym.make("FrozenLake-v1", is_slippery=False))
    state, _ = env.reset()
    state = state_to_onehot(state, num_states)
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
        next_state_oh = state_to_onehot(next_state, num_states)

        memory.append((state, action, reward, next_state_oh, done))

        state = next_state_oh
        total_reward += reward

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.from_numpy(np.array(states)).float()
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.from_numpy(np.array(next_states)).float()
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

    if epi % 1000 == 0:
        avg_loss = running_loss / updates if updates > 0 else 0
        print(f"Episode {epi}, reward: {total_reward}, epsilon: {epsilon:.2f}, loss: {avg_loss:.4f}")
