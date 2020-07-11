import gym
import time
import cv2
import numpy as np

env = gym.make('CarRacing-v0')
env.reset()

total_reward = 0

for _ in range(1000):
    env.render()
    
    action = env.action_space.sample()
    env_state, reward, dead, info = env.step(action)
    total_reward += reward

    if dead:
    	env.close()
    	print(reward)

    print(env_state, reward, dead, info)
    #time.sleep(0.1)

