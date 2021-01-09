# Q Learning (with table)
import math
import gym
import numpy as np

PROBLEM = "MountainCar-v0"
env = gym.make(PROBLEM)

# Constants
EPISODES = 10000
MAX_TIMESTEPS = 200
ACTION_CNT = env.action_space.n
# Discount factor
GAMMA = 0.99

# Initialize state variable ranges 
state_range = tuple(5 for i in range(2))

# Collection environment space bounds (may need to be motified)
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))

# Initialize Q table 
q_table = np.zeros(state_range + (ACTION_CNT,))

# Lambda functions for constants
# Returns epsilon
get_epsilon = lambda i: 0.3 
# Returns learning rate
get_lr = lambda i: 0.001 

# Policy
def get_action(epsilon, state):
    global q_table
    if np.random.random_sample() < epsilon:
        # (Exploration)
        return env.action_space.sample() 
    else: 
        # (Exploitation)
        return np.argmax(q_table[state])

def get_state(observations):
    global state_bounds, state_range
    state = [0] * len(observations)
    for i, env_val in enumerate(observations):
        lower, upper = state_bounds[i][0], state_bounds[i][1]
        if (env_val <= lower):
            state[i] = 0
        elif (env_val >= upper):
            state[i] = state_range[i] - 1
        else:
            diff = upper - lower
            state[i] = int((env_val - lower) / diff * state_range[i])
    # Makes the state immutable
    return tuple(state)

def update_table(state, new_state, action_taken, l_rate, reward):
    global q_table, GAMMA
    q_next_max = np.argmax(q_table[new_state])
    q_table[state + (action_taken,)] += l_rate * (reward + GAMMA * q_next_max - q_table[state + (action_taken,)])

def get_reward(observations, timestep):
    pos, vel = observations
    r1 = abs(pos + 0.5)
    r2 = abs(vel * 100.0)
    r3 = 0 # math.sqrt((200 - timestep))/20.0 
    # print(pos, vel, timestep, r1, r2, r3)
    return r1 + r2 + r3

for episode in range(EPISODES):
    # Observation = [position, velocity]
    observations = env.reset()
    rewards = 0
    epsilon = get_epsilon(episode)
    l_rate = get_lr(episode)
    state = get_state(observations)
    for timestep in range(MAX_TIMESTEPS):
        env.render()

        # Action is either: 0 (push left), 1 (no push), 2 (push right)
        action = get_action(epsilon, state)
        observations, reward, done, info = env.step(action)
        # print(observations)
        # reward = get_reward(observations, timestep)
        rewards += reward
        # print(reward)
        new_state = get_state(observations)

        update_table(state, new_state, action, l_rate, reward)

        state = new_state

        if done or timestep == MAX_TIMESTEPS - 1:
            print("Episode {} is over after {} timesteps. Total reward: {}".format(episode, timestep + 1, rewards))
            break
        
env.close()
