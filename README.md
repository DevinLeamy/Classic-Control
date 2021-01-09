# Classic-Control
Using Deep Q Learning to solve classic control problems <br/> <br/>
## Notes
1. Both environments (CartPole-v0 and MountainCar-v0) have been sourced from OpenAi's Gym API. 
2. Additionally, for both environments Q Learning is facilitated through use of two artifically neural network (a Q Net and a Target Net). The functions of both are layed out in this article: https://towardsdatascience.com/self-learning-ai-agents-part-i-markov-decision-processes-baf6b8fc4c5f.
3. Testing figures are drawn from the models with model parameters found in the /models folder (these parameters were collected following training).

# Results 
### CartPole
The agent in this problem is tasked to keep a pole balance on a cart.<br/>
**Hyperparameters**:
* learning rate = 0.01
* gamma = 0.99
* batch size = 64
* memory size = 1000
* target net update frequency = 200 (timesteps) <br/>

**Agent after training**
Video of agent balancing pole

+ First successful episode: 185
+ Desired reward: 200.0
+ Average reward during testing (100 episodes): 200.0

### MountainCar
The agent in this problem is a car tasked with making its way to a flag perched on a hill. The car cannot make it up the hill and, thus, must learn the generate the momentum required to do so. <br/>
**Hyperparameters**:
* learning rate = 0.0001
* gamma = 0.99
* batch size = 64
* memory size = 100000
* target net update frequency = 200 (timesteps) <br/>

**Agent after training**
Video of mountain car

+ First successful episode: 1042
+ Desired reward: >= -100
+ Average reward during testing (100 episodes): -147.1 (successful roughly 3/5th of the time) 



