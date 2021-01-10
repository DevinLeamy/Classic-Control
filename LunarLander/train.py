import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

np.random.seed(1024)

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

class DQN():
    def __init__(self, states_cnt, hidden_cnt, actions_cnt, lr=0.0001, gamma=0.99, batch_sz=64, memory_sz=5000, update_fq=200):
        self.q_net = Net(states_cnt, hidden_cnt, actions_cnt)
        self.target_net = Net(states_cnt, hidden_cnt, actions_cnt)
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.memory_idx = 0
        self.update_cntr = 0
        
        self.states_cnt = states_cnt
        self.hidden_cnt = hidden_cnt
        self.actions_cnt = actions_cnt

        self.gamma = gamma
        self.lr = lr

        self.batch_sz = batch_sz
        self.update_fq = update_fq
        
        self.memory_sz = memory_sz
        self.memory = np.zeros((self.memory_sz, 2 * self.states_cnt + 3))

    def store_transition(self, state, action, reward, new_state, done):
        transition = np.hstack((state, [action, reward], new_state, [done]))
        self.memory[self.memory_idx % self.memory_sz] = transition
        self.memory_idx += 1
        self.update_cntr += 1

    def policy(self, observation, epsilon):
        state = torch.FloatTensor(observation)
        p = np.random.uniform(0, 1, 1)[0]
        if p < epsilon:
            action = np.random.randint(0, self.actions_cnt)
        else:
            action_values = self.q_net(state)
            action = torch.argmax(action_values).item()
        return action

    def learn(self):
        if self.memory_idx < self.memory_sz:
            return 

        mem_idxs = np.random.choice(self.memory_sz, self.batch_sz)

        transitions = self.memory[mem_idxs]

        states = torch.FloatTensor(transitions[:, :self.states_cnt])
        actions = torch.LongTensor(transitions[:, self.states_cnt:self.states_cnt+1].astype(int))
        rewards = torch.FloatTensor(transitions[:, self.states_cnt+1:self.states_cnt+2])
        new_states = torch.FloatTensor(transitions[:, self.states_cnt+2:2*self.states_cnt + 2])
        dones = transitions[:, -1]

        q_given = self.q_net(states).gather(1, actions)

        q_next_values = self.target_net(new_states).detach()
        q_next_max = q_next_values.max(1)[0].view(self.batch_sz, 1)
        td_target = rewards + self.gamma * q_next_max

        for i in range(self.batch_sz):
            if dones[i]:
                td_target[i] = q_given[i]

        loss = self.loss_func(q_given, td_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.update_cntr % self.update_fq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.update_cntr = 0
    def store_model(self):
        torch.save(self.q_net.state_dict(), "models/LunarLanderModel.pt")
        print("Model Saved")

def train(problem, episodes, gamma, epsilon, epsilon_decay, lr, threshold_value):
    env = gym.make(problem)
    actions_cnt = env.action_space.n
    states_cnt = env.observation_space.shape[0]
    hidden_cnt = 64
    records = []
    
    dqn = DQN(states_cnt, hidden_cnt, actions_cnt)
    
    for episode in range(episodes):
        state = env.reset()
        rewards = 0
        _iter = 0 
        done = False
        while not done:
            env.render()

            action = dqn.policy(state, epsilon)
            epsilon = max(0.001, epsilon * epsilon_decay)

            new_state, reward, done, info = env.step(action)

            dqn.store_transition(state, action, reward, new_state, done)

            rewards += reward
            
            state = new_state

            dqn.learn()

            records.append([episode, _iter, rewards])
            _iter += 1

        print("Episode: {}\tTimesteps: {}\tRewards: {}".format(episode, _iter, rewards))
    env.close()
    dqn.store_model()
    return records

records = train("LunarLander-v2", 2000, 0.9, 1.0, 0.9, 0.001, 200)
for episode, iteration, rewards in records:
    print("Episode: {}\tIterations: {}\tRewards: {}".format(episode, iteration, rewards))



