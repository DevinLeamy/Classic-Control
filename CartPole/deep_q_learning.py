# Source: https://pyliaorachel.github.io/blog/tech/python/2018/06/14/deep-q-learning.html

import gym
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, states_cnt, actions_cnt, hidden_cnt):
        super(Net, self).__init__()
        
        self.l1 = nn.Linear(states_cnt, hidden_cnt)
        self.l2 = nn.Linear(hidden_cnt, actions_cnt)
    
    def forward(self, state):
        x = self.l1(state)
        x = F.relu(x)
        actions_value = self.l2(x)
        return actions_value


class DQN(object):
    def __init__(self, states_cnt, actions_cnt, hidden_cnt, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):
        self.eval_net, self.target_net = Net(states_cnt, actions_cnt, hidden_cnt), Net(states_cnt, actions_cnt, hidden_cnt)
        
        # Shape of memory is I x J where J is states_cnt * 2 + 2 given 
        # that the memory has to accomidate the current state (state_cnt), the next state (state_cnt)
        # a reward (+ 1), and an action (+ 1).
        self.memory = np.zeros((memory_capacity, states_cnt * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0

        self.states_cnt = states_cnt
        self.actions_cnt = actions_cnt
        self.hidden_cnt = hidden_cnt
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity
    
    # Determines actions based on epsilon-greedy policy
    def policy(self, state):
        # Transforms state into torch.Tensor/torch.FloatTensor object
        x = torch.Tensor(state)

        # epsilon greedy
        if np.random.uniform() < self.epsilon:
            # Random action is taken
            action = np.random.randint(0, self.actions_cnt)
        else:
            # Best current action is returned
            actions_value = self.eval_net.forward(x)
            action = torch.argmax(actions_value).item()
        return action

    # Stores experience in memory
    def store_transition(self, state, action, reward, next_state):
        # Object to hold experience. Shape = (10,)
        # transition = [four-state-values, action, reward, four-new-state-values]
        transition = np.hstack((state, [action, reward], next_state))

        # Index where transition will be stored
        index = self.memory_counter % self.memory_capacity
        self.memory[index] = transition
        self.memory_counter += 1

    def learn(self):
        # Returns array of batch_size # of random indices
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)

        # Returns an array containing all elements at the indices given by sample_index
        b_memory = self.memory[sample_index, :]
        
        # Returns a tensor of the states from memory batch. Shape = (32, 4)
        b_state = torch.FloatTensor(b_memory[:, :self.states_cnt])

        # Returns a tensor of actions from memory batch. Shape = (32, 1)
        b_action = torch.LongTensor(b_memory[:, self.states_cnt:self.states_cnt+1].astype(int))

        # Returns a tensor of rewards from memory batch. Shape = (32, 1)
        b_reward = torch.FloatTensor(b_memory[:, self.states_cnt+1:self.states_cnt+2])

        # Returns a tensor of next_states from the memory batch. Shape = (32, 4)
        b_next_state = torch.FloatTensor(b_memory[:, -self.states_cnt:])
        
        # Returns a tensor of the values Q(s, a) for all elements of batch
        q_eval = self.eval_net(b_state).gather(1, b_action)

        # Returns a tensor with Q(s, a) for all actions for all elements in the batch
        # .detach() is called to remove the reference to the graph node that created the tensor
        q_next = self.target_net(b_next_state).detach()

        # Returns array of max future reward all next_state's in batch
        # .view() is called the reshape the tensor into shape=(32,1)
        q_next_max = q_next.max(1)[0].view(self.batch_size, 1)
        
        # TD-Target (Temporal difference target)
        q_target = b_reward + self.gamma * q_next_max 

        # Calculates loss function  
        loss = self.loss_func(q_eval, q_target)

        # Backpropagation (done on eval_net/q_net)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Increments learn counter and replaces target_net parameters 
        # (if target_replace_iter transitons have passed)
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())


def get_reward(state):
    global env
    pos, v, ang, rot = state
    r1 = (env.x_threshold - abs(pos)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(ang)) / env.theta_threshold_radians - 0.5
    return r1 + r2

PROBLEM = 'CartPole-v0'
env = gym.make(PROBLEM)

# Environment parameters
actions_cnt = env.action_space.n
states_cnt = env.observation_space.shape[0]

# Hyper parameters
hidden_cnt = 50
batch_size = 32
lr = 0.01
epsilon = 0.1
gamma = 0.9
target_replace_iter = 100
memory_capacity = 2000
episodes_cnt = 4000

# Deeq Q Learning Network
dqn = DQN(states_cnt, actions_cnt, hidden_cnt, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)

for episode in range(episodes_cnt):
    t = 0
    rewards = 0
    state = env.reset()
    while True:
        env.render()

        # Get action
        action = dqn.policy(state)
        next_state, reward, done, info = env.step(action)

        reward = get_reward(next_state)

        # Store experience
        dqn.store_transition(state, action, reward, next_state)

        # Collect reward
        rewards += reward

        if dqn.memory_counter > memory_capacity:
            dqn.learn()

        state = next_state

        if done:
            print('Episode {} finished after {} timesteps. Total reward: {}'.format(episode, t + 1, rewards))
            break

        t += 1

env.close()
