# Classic-Control
Using Deep Q Learning to solve classic control problems <br/> <br/>
## Notes
1. Both environments (CartPole-v0 and MountainCar-v0) have been sourced from OpenAi's Gym API. 
2. Additionally, for both environments Q Learning is facilitated through use of two artifically neural network (a Q Net and a Target Net). The functions of both are layed out in this article: https://towardsdatascience.com/self-learning-ai-agents-part-i-markov-decision-processes-baf6b8fc4c5f.

# Problems
### CartPole
The agent in this problem is tasked to keep a pole balance on a cart.<br/>
Hyperparameters:
1. learning rate = 0.01
2. Gamma = 0.99
3. Batch size = 64
4. Memory size = 1000
5. Target net update frequency = 200 (timesteps) <br/>

