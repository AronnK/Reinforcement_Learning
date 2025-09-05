import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import torch
import torch.nn as nn
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def state_to_grid(state, env):
    """ The 3-channel state representation your model was trained on. """
    grid_size = env.unwrapped.nrow
    agent_grid = np.zeros((grid_size, grid_size))
    hole_grid = np.zeros((grid_size, grid_size))
    goal_grid = np.zeros((grid_size, grid_size))
    row, col = state // grid_size, state % grid_size
    agent_grid[row, col] = 1
    desc = env.unwrapped.desc
    for r in range(grid_size):
        for c in range(grid_size):
            if desc[r, c] == b'H': hole_grid[r, c] = 1
            elif desc[r, c] == b'G': goal_grid[r, c] = 1
    grid = np.stack([agent_grid, hole_grid, goal_grid], axis=0)
    return grid.astype(np.float32)

class CNN_DQN(nn.Module):
    """ The CNN architecture that matches your saved model. """
    def __init__(self, in_channels, action_dim):
        super(CNN_DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU()
        )
        grid_size = 4
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

def get_valid_actions(state, grid_size=4):
    """ Returns a boolean tensor indicating valid actions for a given state. """
    row, col = state // grid_size, state % grid_size
    # Actions: [Left, Down, Right, Up] -> Indices [0, 1, 2, 3]
    actions = [True, True, True, True] 
    if col == 0: actions[0] = False
    if row == grid_size - 1: actions[1] = False
    if col == grid_size - 1: actions[2] = False
    if row == 0: actions[3] = False
    return torch.tensor(actions, dtype=torch.bool, device=device)

model_path = r"D:\reinforcement_learning\cnn_dqn_frozenlake.pth" 
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

in_channels = 3
action_dim = 4

q_net = CNN_DQN(in_channels, action_dim).to(device)
checkpoint = torch.load(model_path, map_location=device)
q_net.load_state_dict(checkpoint['q_net'])
q_net.eval()
print("Model loaded successfully!")

test_map = generate_random_map(size=4)
env = gym.make("FrozenLake-v1", desc=test_map, is_slippery=False, render_mode="human")

state_int, _ = env.reset()
done = False
action_map = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

print("\n--- Starting new game on a random map ---")
time.sleep(1)

while not done:
    state_tensor = torch.from_numpy(state_to_grid(state_int, env)).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = q_net(state_tensor)

        valid_actions = get_valid_actions(state_int)

        masked_q_values = q_values.clone()
        masked_q_values[0, ~valid_actions] = -float('inf') # Use ~ to invert the mask

        action = torch.argmax(masked_q_values).item()

    next_state_int, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state_int = next_state_int

    print(f"Action Taken: {action_map[action]}")
    time.sleep(0.5)

print(f"\n--- Game Over! ---")
if reward > 0:
    print("Agent reached the goal! 🎉")
else:
    print("Agent did not reach the goal. 🤔")

env.close()