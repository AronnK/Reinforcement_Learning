import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

env = gym.make("FrozenLake-v1")
state_space_size = env.observation_space.n
action_space_size = env.action_space.n

def one_hot_state(state):
    s = np.zeros(state_space_size)
    s[state] = 1
    return s

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

q_net = DQN(state_space_size, action_space_size)
target_net = DQN(state_space_size, action_space_size)
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
    state, _ = env.reset()
    state = one_hot_state(state)
    done = False
    total_reward = 0
    running_loss = 0
    updates = 0

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(torch.FloatTensor(state))
                action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = one_hot_state(next_state)

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

            # Q(s,a)
            q_values = q_net(states).gather(1, actions)

            # Q target
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
