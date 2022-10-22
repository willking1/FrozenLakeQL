import numpy as np
import gym
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import time
from IPython import display

# SETUP

class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if(env.desc[next_state//8][next_state%8] == b"H") : reward = -15
        elif(env.desc[next_state//8][next_state%8] == b"G") : reward = 100
        else : reward = -1
        return next_state, reward, done, info
     
env = BasicWrapper(gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True))

numStates = env.observation_space.n
numActions = env.action_space.n

qTable = np.zeros((numStates, numActions))

# HYPER PARAMETERS

learningRate = 0.05
discountRate = 0.95

exploreRate = 0.9
baseExploreRate = exploreRate
decay = 1e-5
eps = 20000
maxEpSteps = 750

# EVAL

def evaluation() :
    wins = 0
    stepList = []
    for i in range(100) :
        done = False
        state = env.reset()
        step = 0

        for j in range(maxEpSteps) :
            action = np.argmax(qTable[state])
            newState, reward, done, info = env.step(action)
            state = newState
            step += 1
            if(done) : break

        if(state == 63) : 
            wins += 1
            stepList.append(step)
    
    if(len(stepList)==0) :
        return 0, 0
    else :   
        avgSteps = np.mean(stepList)
        return wins, avgSteps

# TRAINING LOOP

exploreRates = []
winsList = []
avgStepsList = []
for i in tqdm(range(eps)) :
    state = env.reset()
    for j in range(maxEpSteps) :
        if random.uniform(0,1) <= exploreRate :
            action = env.action_space.sample()
        else :
            action = np.argmax(qTable[state])
        newState, reward, done, info = env.step(action)
        qTable[state][action] = (1-learningRate) * qTable[state][action] + learningRate * (discountRate*np.max(qTable[newState]) + reward)
        state = newState
        if(done) : break
    exploreRate = baseExploreRate * np.exp(-decay*i)
    
    if i%(eps/10) == 0 :
        exploreRates.append(exploreRate)
        wins, avgSteps = evaluation()
        winsList.append(wins)
        avgStepsList.append(avgSteps)
        
# WATCH SOLUTION

done = False
state = env.reset()
while(not done) :
    action = np.argmax(qTable[newState])
    newState, reward, done, info = env.step(action)
    state = newState
    time.sleep(0.25)
    display.clear_output(wait=True)
    env.render()

# PLOTS

plt.plot(winsList, label="Wins")
plt.plot(avgStepsList, label="Avg. Steps")
plt.legend()

plt.plot(exploreRates)
