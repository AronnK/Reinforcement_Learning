import numpy as np
import gymnasium
import random 
from IPython.display import clear_output

env = gymnasium.make("FrozenLake-v1")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))

# print(q_table)

n_epi = 10000
max_steps_per_epi = 100

lr = 0.2
disc_r = 0.99
expl_r = 1
max_er = 1
min_er = 0.01
exp_decay_r = 0.005

rewards = []

for epi in range(n_epi):
    state,_ = env.reset()
    done = False
    rewards_current = 0

    for step in range(max_steps_per_epi):
        er_threshold = random.uniform(0,1)
        if er_threshold> expl_r:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
        
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        q_table[state,action] = q_table[state,action] * (1-lr) + lr * (reward + disc_r * np.max(q_table[new_state, :])) 

        state = new_state
        rewards_current += reward

        if done == True:
            break

    expl_r = min_er + (max_er - min_er) * np.exp(-exp_decay_r * epi)

    rewards.append(rewards_current)

rewards_per_1k = np.split(np.array(rewards), n_epi/1000)
count = 1000
print("Avg reward per 1k epis: ")
for r in rewards_per_1k:
    print(count, ": ", str(sum(r) / 1000))
    count += 1000

print(q_table)


