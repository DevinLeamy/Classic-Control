# Classic-Control
Using Deep Q Learning to solve classic control problems.
## Notes
1. Both environments (CartPole-v0 and MountainCar-v0) have been sourced from OpenAI's Gym API (https://gym.openai.com/envs/#classic_control). 
2. Q Learning was facilitated through use of two artifically neural network (a Q Net and a Target Net). The functions of both are layed out in this article: https://towardsdatascience.com/self-learning-ai-agents-part-i-markov-decision-processes-baf6b8fc4c5f.
3. Testing figures are drawn from model parameters found in /models of each respective environment (these parameters were collected following training).

# CartPole (CartPole-v0)
The agent in this problem is tasked to keep a pole balance on a cart.<br/><br/>
**Hyperparameters**:
* learning rate = 0.01
* gamma = 0.99
* batch size = 64
* memory size = 1000
* target net update frequency = 200 (timesteps) <br/>

**Agent after training:**<br/>
![Agent Balancing Pole](results/CartPole.gif)

**Results:**
+ First successful episode: 185
+ Desired reward: 200.0
+ Average reward during testing (100 episodes): 200.0

# MountainCar (MountainCar-v0)
The agent in this problem is a car tasked with making its way to a flag perched on a hill. The car cannot make it up the hill and, thus, must learn the generate the momentum required to do so. <br/><br/>
**Hyperparameters**:
* learning rate = 0.0001
* gamma = 0.99
* batch size = 64
* memory size = 100000
* target net update frequency = 200 (timesteps) <br/>

**Agent after training:**<br/>
![Agent Climbing Hill](results/MountainCar.gif)

**Results:**
+ First successful episode: 1042
+ Desired reward: >= -100
+ Average reward during testing (100 episodes): -147.1 (successful roughly 3/5th of the time) 

## Play around with it!
**Requirements**:<br/>
```bash
pip3 install gym
pip3 install numpy
pip3 install torch
```

**Clone the repo**:<br/>
```bash
git clone https://github.com/DevinLeamy/Classic-Control.git
```

**Train an agent (MountainCar in this example)**:<br/>
```bash
cd MountainCar
python3 train.py
```

**Test your agent**:<br/>
```bash
python3 test.py
```

