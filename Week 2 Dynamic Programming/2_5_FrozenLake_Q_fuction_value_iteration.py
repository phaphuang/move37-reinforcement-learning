import numpy as np
import time
import gym
"""
Args:
    policy: [S,A] shaped matrix representing the policyself.
    env: OpenAI gym envselfself.
    render: boolean to turn rendering on/off
"""
# Execution
def execute(env, policy, gamma=1.0, render=False):
    start = env.reset()
    totalReward = 0
    stepIndex = 0
    while True:
        if render:
            env.render()
        start, reward, done, _ = env.step(int(policy[start]))
        totalReward += (gamma ** stepIndex * reward)
        stepIndex += 1
        if done:
            break
    return totalReward

# Evaluation
def evaluatePolicy(env, policy, gamma=1.0, n=100):
    scores = [execute(env, policy, gamma=gamma, render=False) for _ in range(n)]
    return np.mean(scores)

# Choosing the policy given a value-Function
def calculatePolicy(v, gamma=1.0):
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy

# Value Iteration Algorithm
def valueIteration(env, gamma=1.0):
    value = np.zeros(env.env.nS)
    max_iterations = 10000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(value)
        for s in range(env.env.nS):
            q_sa = [sum([p * (r + prev_v[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.env.nA)]
            value[s] = max(q_sa)
        if(np.sum(np.fabs(prev_v - value)) <= eps):
            print("Value-iteration converged at # %d" % (i + 1))
            break
    return value

if __name__ == '__main__':
    gamma = 1.0
    env = gym.make('FrozenLake-v0')
    # Policy search
    optimalValue = valueIteration(env, gamma)
    startTime = time.time()
    policy = calculatePolicy(optimalValue, gamma)
    policy_score = evaluatePolicy(env, policy, gamma, n=1000)
    endTime = time.time()
    print("Best score = %.2f. Time taken = %4.4f seconds" % (np.max(policy_score), endTime-startTime))
