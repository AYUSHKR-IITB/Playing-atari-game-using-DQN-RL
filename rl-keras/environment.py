import gymnasium as gym
import preprocess_frame as ppf
import numpy as np
import ale_py

def initialize_new_game(name, env, agent):
    """We don't want an agents past game influencing its new game, so we add in some dummy data to initialize"""
    
    env.reset()
    starting_frame = ppf.resize_frame(env.step(0)[0])

    dummy_action = 0
    dummy_reward = 0
    dummy_done = False
    for i in range(3):
        agent.memory.add_experience(starting_frame, dummy_reward, dummy_action, dummy_done)

def make_env(name, agent):
    
    env = gym.make(name, render_mode= 'human')
    return env

def take_step(name, env, agent, score, debug):
    # 1 and 2: Update timesteps and save weights
    agent.total_timesteps += 1
    if agent.total_timesteps % 50000 == 0:
        agent.model.save_weights('recent.weights.h5')
        print('\nWeights saved!')

    # 3: Take action
    next_frame, next_frames_reward, terminated, truncated, info = env.step(agent.memory.actions[-1])

    # 4: Get next state
    next_frame = ppf.resize_frame(next_frame)
    new_state = [agent.memory.frames[-3], agent.memory.frames[-2], agent.memory.frames[-1], next_frame]
    new_state = np.moveaxis(new_state, 0, 2) / 255  # Prepare data for Keras input format
    new_state = np.expand_dims(new_state, 0)

    # 5: Get next action, using next state
    next_action = agent.get_action(new_state)

    # 6: Check if the game is over
    done = terminated or truncated
    if done:
        agent.memory.add_experience(next_frame, next_frames_reward, next_action, done)
        return (score + next_frames_reward), True

    # 7: Add the next experience to memory
    agent.memory.add_experience(next_frame, next_frames_reward, next_action, done)

    # 8: If debugging, render the environment
    if debug:
        env.render()

    # 9: If the memory threshold is met, make the agent learn
    if len(agent.memory.frames) > agent.starting_mem_len:
        agent.learn(debug)

    return (score + next_frames_reward), False


def play_episode(name, env, agent, debug = True):
    initialize_new_game(name, env, agent)
    done = False
    score = 0
    while True:
        score,done = take_step(name,env,agent,score, debug)
        if done:
            break
    return score
