import numpy as np
import gym

env = gym.make('CartPole-v0')

def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        env.render()
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

"""
# Random Search to select weights/parameters 
# to receive the highest amout of average reward
bestparams = None
bestreward = 0
for _ in range(10000):
    parameters = np.random.rand(4) * 2 - 1
    reward = run_episode(env, parameters)
    if reward > bestreward:
        bestreward = reward
        bestparams = parameters
        # considered solved if the agent lasts 200 timesteps
        if reward == 200:
            break
"""

# Hill-Climbing, start with some randomly chosen initial weights.
# Every episode, add some noise to the weights
# and keep the new weights if the agen improves
noise_scaling = 0.1
parameters = np.random.rand(4) * 2 - 1
bestreward = 0
for _ in range(10000):
    newparams = parameters + (np.random.rand(4) * 2 - 1) * noise_scaling
    reward = 0
    run = run_episode(env, newparams)
    if reward > bestreward:
        bestreward = reward
        parameters = newparams
        if reward == 200:
            break