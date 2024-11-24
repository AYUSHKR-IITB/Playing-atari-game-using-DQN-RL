import matplotlib.pyplot as plt
import time
from collections import deque
import numpy as np
import torch
import the_agent
from environment import make_env, play_episode

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

name = "ALE/Pong-v5"
agent = the_agent.Agent(possible_actions=[0,2,3], 
              starting_mem_len=5000,
              max_mem_len=75000,
              starting_epsilon=1, 
              learn_rate=.00025)

env = make_env(name, agent)

last_100_avg = [-21]
scores = deque(maxlen=100)
max_score = -21

""" If testing:
agent.load_weights('recent_weights.pt')
agent.epsilon = 0.0
"""

env.reset()

for i in range(1000):
    timesteps = agent.total_timesteps
    timee = time.time()
    score = play_episode(name, env, agent, debug=True)  # set debug to true for rendering
    scores.append(score)
    
    if score > max_score:
        max_score = score

    print('\nEpisode:', i)
    print('Steps:', agent.total_timesteps - timesteps)
    print('Duration:', time.time() - timee)
    print('Score:', score)
    print('Max Score:', max_score)
    print('Epsilon:', agent.epsilon)

    if i % 100 == 0 and i != 0:
        last_100_avg.append(sum(scores)/len(scores))
        plt.plot(np.arange(0, i+1, 100), last_100_avg)
        plt.show()

    # Save weights periodically
    if i % 100 == 0:
        agent.save_weights('recent_weights.pt')
        print('Weights saved!')
