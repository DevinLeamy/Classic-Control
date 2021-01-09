import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import gym

class Net(nn.Module):
    def __init__(self, states_cnt, hidden_cnt, actions_cnt):
        super(Net, self).__init__()

        self.l1 = nn.Linear(states_cnt, hidden_cnt)
        self.l2 = nn.Linear(hidden_cnt, hidden_cnt)
        self.l3 = nn.Linear(hidden_cnt, actions_cnt)
    
    def forward(self, state):
        x = self.l1(state)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        action_values = self.l3(x)
        return action_values


env = gym.make("CartPole-v0")

states_cnt = env.observation_space.shape[0]
hidden_cnt = 64
actions_cnt = env.action_space.n

model = Net(states_cnt, hidden_cnt, actions_cnt)
parameters = torch.load("models/CartPoleModel.pt")
model.load_state_dict(parameters)

total_rewards = 0
episodes = 100

for episode in range(episodes):
    done = False
    rewards = 0
    state = env.reset()
    while not done:
        env.render()

        # Greedy Policy 
        state = torch.FloatTensor(state)
        action_values = model(state)
        action = torch.argmax(action_values).item()
        
        new_state, reward, done, info = env.step(action)
        rewards += reward

        state = new_state
        
    total_rewards += rewards
    print("Episode: {} \t Rewards: {}".format(episode, rewards))

env.close()

average = round(total_rewards/episodes, 1)
print("Average reward over {} episodes: {}".format(episodes, average))


