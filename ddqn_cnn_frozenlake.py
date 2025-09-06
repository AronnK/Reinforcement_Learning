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

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    """ A CNN architecture for the 4x4 grid. """
    def __init__(self, in_channels, action_dim):
        super(CNN_DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=2),  # conv1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),          # conv2
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2),          # conv3
            nn.ReLU()
        )
        grid_size = 4
        # Calculate the output size of the conv layers automatically
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
        x = x.view(x.size(0), -1) # Flatten the output
        return self.fc(x)

def get_valid_actions_mask(states_int, grid_size=4, device='cpu'):
    """ 
    Returns a batch of boolean masks indicating valid actions for a batch of states.
    Actions: 0:Left, 1:Down, 2:Right, 3:Up
    """
    batch_size = len(states_int)
    masks = torch.ones((batch_size, 4), dtype=torch.bool, device=device)
    for i, state in enumerate(states_int):
        row, col = state // grid_size, state % grid_size
        if col == 0: masks[i, 0] = False
        if row == grid_size - 1: masks[i, 1] = False
        if col == grid_size - 1: masks[i, 2] = False
        if row == 0: masks[i, 3] = False
    return masks

# ================== HYPERPARAMS ==================
grid_size = 4
in_channels = 3
action_dim = 4
# *** CHANGED MODEL NAME ***
model_file = "ddqn_cnn_frozenlake.pth" 

q_net = CNN_DQN(in_channels, action_dim).to(device)
target_net = CNN_DQN(in_channels, action_dim).to(device)
target_net.load_state_dict(q_net.state_dict())
target_net.eval() # Target network is only for evaluation

optimizer = optim.Adam(q_net.parameters(), lr=0.0001) # Slightly lower learning rate for stability
loss_fn = nn.MSELoss()
memory = deque(maxlen=50000) # Adjusted memory size

# *** HYPERPARAMS FOR SHORTER TRAINING ***
batch_size = 128
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9999 # Faster decay for shorter training
total_maps = 500  # Number of random maps to train on
episodes_per_map = 20 # Number of episodes per map
target_update_freq = 2000 # Update target network less frequently

# ================== TRAINING ==================
start_time = time.time()
total_steps = 0
wins_in_block = 0

# Load model and state if it exists to resume training
if os.path.exists(model_file):
    print("Loading existing model to resume training...")
    checkpoint = torch.load(model_file, map_location=device)
    q_net.load_state_dict(checkpoint['q_net'])
    target_net.load_state_dict(checkpoint['target_net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epsilon = checkpoint.get('epsilon', epsilon)
    total_steps = checkpoint.get('total_steps', 0)

print(f"Starting training loop... Initial Epsilon: {epsilon:.4f}, Initial Steps: {total_steps}")

for map_idx in range(total_maps):
    desc = generate_random_map(size=grid_size, p=0.8)
    # Skip impossible maps where start or goal is a hole
    if desc[0, 0] == b'H' or desc[grid_size - 1, grid_size - 1] == b'H':
        continue
    
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False)
    map_wins = 0

    for epi in range(episodes_per_map):
        state_int, _ = env.reset()
        done = False

        while not done:
            state_grid = torch.from_numpy(state_to_grid(state_int, env)).unsqueeze(0).to(device)
            valid_actions_mask = get_valid_actions_mask([state_int], device=device)[0]

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                valid_action_indices = torch.where(valid_actions_mask)[0]
                action = random.choice(valid_action_indices.tolist())
            else:
                with torch.no_grad():
                    q_values = q_net(state_grid)
                    # Apply mask before choosing the best action
                    q_values[0, ~valid_actions_mask] = -float('inf')
                    action = torch.argmax(q_values).item()

            next_state_int, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Reward shaping
            if terminated and reward == 1.0:
                final_reward = 1.0
                map_wins += 1
            elif terminated:
                final_reward = -1.0
            else:
                final_reward = -0.01 # Small penalty for each step

            next_state_grid = torch.from_numpy(state_to_grid(next_state_int, env)).unsqueeze(0)
            
            # Store the integer state for future masking
            memory.append((state_grid.cpu(), action, final_reward, next_state_grid.cpu(), done, next_state_int))

            state_int = next_state_int

            # Start training only when we have enough samples
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                # Unpack batch and send to device
                states_t = torch.cat([s[0] for s in batch]).to(device)
                actions_t = torch.LongTensor([s[1] for s in batch]).unsqueeze(1).to(device)
                rewards_t = torch.FloatTensor([s[2] for s in batch]).unsqueeze(1).to(device)
                next_states_t = torch.cat([s[3] for s in batch]).to(device)
                dones_t = torch.FloatTensor([s[4] for s in batch]).unsqueeze(1).to(device)
                next_states_int_t = [s[5] for s in batch]

                # Get current Q values for chosen actions
                q_values = q_net(states_t).gather(1, actions_t)

                # --- *** DDQN & MASKING IMPLEMENTATION *** ---
                # 1. Get valid actions mask for the batch of next states
                next_valid_actions_mask = get_valid_actions_mask(next_states_int_t, device=device)

                # 2. Select best actions using the ONLINE network, respecting the mask
                next_q_values_online = q_net(next_states_t).clone()
                next_q_values_online[~next_valid_actions_mask] = -float('inf') # Mask invalid actions
                best_next_actions = next_q_values_online.argmax(1).unsqueeze(1)
                
                # 3. Evaluate those actions using the TARGET network
                with torch.no_grad():
                    next_q_from_target = target_net(next_states_t).gather(1, best_next_actions)
                    # Calculate the DDQN target
                    q_target = rewards_t + gamma * next_q_from_target * (1 - dones_t)
                # --- *** END OF DDQN & MASKING *** ---

                # Calculate loss and perform backpropagation
                loss = loss_fn(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_steps += 1
                
                # Update target network periodically
                if total_steps % target_update_freq == 0:
                    print(f"--- Step {total_steps}: Updating target network ---")
                    target_net.load_state_dict(q_net.state_dict())

        # Decay epsilon after each episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if map_wins > 0:
        wins_in_block += 1

    # Reporting and saving checkpoint
    if (map_idx + 1) % 50 == 0:
        win_rate = (wins_in_block / 50) * 100
        wins_in_block = 0
        elapsed_time = time.time() - start_time
        print(f"Maps {map_idx-49}-{map_idx+1}/{total_maps} | Win Rate: {win_rate:.1f}% | Epsilon: {epsilon:.4f} | Steps: {total_steps} | Time: {elapsed_time:.1f}s")

        torch.save({
            'q_net': q_net.state_dict(),
            'target_net': target_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epsilon': epsilon,
            'total_steps': total_steps
        }, model_file)
        print(f"Model saved to {model_file}")

print("\nTraining complete.")