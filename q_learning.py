import gym
import numpy as np
import matplotlib.pyplot as plt
import random

episodes = 100000
epsilon = 0.8
lr = 0.1
gamma = 0.6

env = gym.make('Taxi-v3')
state = env.reset() #initial

state_space_size = env.observation_space.n
action_space_size = env.action_space.n

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

rewards, penalties, timesteps = [], [], []
epsilons = []

for episode in range(episodes):
    state = env.reset()

    total_penalty = 0
    total_reward = 0
    total_timestep = 0

    epsilons.append(epsilon)
    
    #epsilon = (1 - episode/episodes) * epsilon
    dead = False

    while not dead:
        
        #env.render()
        total_timestep += 1
        
        #Choose action : using epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state,:])
        
        #Perform action
        new_state, reward, dead, info = env.step(action)

        #Update Q-table
        next_state_q_value = np.max(q_table[new_state,:])
        old_state_q_value = q_table[state, action]
        q_table[state, action] = (1-lr)*old_state_q_value + lr*(reward + gamma*next_state_q_value) 

        #Update state
        state = new_state

        #Accumulate rewards and penalties
        if reward == -10:
            total_penalty += 1
        total_reward += reward


    penalties.append(total_penalty)
    rewards.append(total_reward)
    timesteps.append(total_timestep)

plt.subplot(2,2,1)
plt.title('Rewards')
plt.plot(rewards)
plt.subplot(2,2,2)
plt.title('Penalties')
plt.plot(penalties)
plt.subplot(2,2,3)
plt.title('Timesteps')
plt.plot(timesteps)
plt.subplot(2,2,4)
plt.title('Epsilon')
plt.plot(epsilons)

plt.show()

"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs
    print(penalties, epochs)

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")



