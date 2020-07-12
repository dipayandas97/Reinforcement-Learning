import gym
import random
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop

from collections import deque


def get_network(input_shape, action_space):
    X_input = Input(input_shape)
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X, name='CartPole DQN model')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    return model


class Agent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.env.reset()
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000

        # create main model
        self.model = get_network(input_shape=(self.state_size,), action_space = self.action_size)

    def remember(self, state, action, reward, next_state, dead):
        self.memory.append((state, action, reward, next_state, dead))

        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
    def replay(self):
        if len(self.memory) < self.train_start:
            return
        
        #randomly sample mini-batch
        mini_batch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        #separate data of the mini_batch
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []
        for i in range(self.batch_size):
            state[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            next_state[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])
        
        #prepare target
        target = self.model.predict(state)
        Q_s_prime = self.model.predict(next_state)
        
        #Form target = cumulative reward        
        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma*(np.max(Q_s_prime[i]))
        
        #Train model
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)
    
    def get_action(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def run(self):

        for e in range(self.EPISODES):
            print('Epiode: ',e)

            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            dead = False
            i = 0

            while not dead:
                #self.env.render()
                
                action = self.get_action(state)

                next_state, reward, dead, info = self.env.step(action)
                next_state = np.expand_dims(next_state, axis=0)

                if not dead or i == self.env._max_episode_steps-1: 
                    reward = reward
                else:
                    reward = -100
                
                self.remember(state, action, reward, next_state, dead)
                state = next_state
                i += 1

                if dead:
                    print('Frames lived:',i)
                    if i == 500:
                        self.model.save('CartPole_model.h5')
                        #return

                self.replay()

    def evaluate(self):
        self.model = load_model('CartPole_model.h5')

        for e in range(20):
            state = self.env.reset()
            state = np.expand_dims(state, axis=0)
            dead = False
            i = 0

            while not dead:
                action = np.argmax(self.model.predict(state))

                next_state, reward, dead, info = self.env.step(action)
                state = np.expand_dims(next_state, axis=0)
                i+=1

                if dead:
                    print('Frames lived:',i)
                    brea


agent = Agent()
agent.run()

agent.evaluate()