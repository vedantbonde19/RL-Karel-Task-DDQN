import gym
#!pip3 install box2d
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import gym_examples
from dqn_agent import Agent

agent = Agent(state_size=88, action_size=6, seed = 0)

def dqn_easy(n_episodes=30000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    mean_scores = []
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        # state, info  = env.reset()

        i = np.random.randint(0, 3999)

        path = "C:/Users/vedan/Downloads/Bonde_Project/datasets/data_easy/train/train/task/"
        fname = str(i) + "_task.json"
        path = path + fname

        env = gym.make("gym_examples/GridWorldMarker-v0", jsondata = path)

        state, info = env.reset()

        state = state  + np.random.rand(1,88)/10.0 #D
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, trunc, _ = env.step(action)
            next_state = next_state + np.random.rand(1,88)/10.0 #D
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            #mean_scores.append(np.mean(scores_window))
        if i_episode % 200 == 0:
            #print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            mean_scores.append(np.mean(scores_window))
        
    return scores, mean_scores

print("==========================================")
print("Starting the training process on task_easy")
print("==========================================")

scores, mean_scores = dqn_easy()

print("\n============================================")
print("Starting the training process on task_medium")
print("============================================")

def dqn_medium(n_episodes=30000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    mean_scores = []
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes):
        # state, info  = env.reset()

        i = np.random.randint(0, 23999)

        path = "C:/Users/vedan/Downloads/Bonde_Project/datasets/data_medium/train/train/task/"
        fname = str(i_episode) + "_task.json"
        path = path + fname

        env = gym.make("gym_examples/GridWorldMarker-v0", jsondata = path)

        state, info = env.reset()

        state = state  + np.random.rand(1,88)/10.0 #D
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, trunc, _ = env.step(action)
            next_state = next_state + np.random.rand(1,88)/10.0 #D
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            #mean_scores.append(np.mean(scores_window))
        if i_episode % 200 == 0:
            #print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            mean_scores.append(np.mean(scores_window))
        
    return scores, mean_scores

scores, mean_scores = dqn_medium()

print("\n===========================================")
print("Starting the training process on final task")
print("===========================================")

def dqn_final(n_episodes=30000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    mean_scores = []
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes):
        # state, info  = env.reset()

        i = np.random.randint(0, 23999)

        path = "C:/Users/vedan/Downloads/Bonde_Project/datasets/data/train/train/task/"
        fname = str(i_episode) + "_task.json"
        path = path + fname

        env = gym.make("gym_examples/GridWorldMarker-v0", jsondata = path)

        state, info = env.reset()

        state = state  + np.random.rand(1,88)/10.0 #D
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, trunc, _ = env.step(action)
            next_state = next_state + np.random.rand(1,88)/10.0 #D
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            #mean_scores.append(np.mean(scores_window))
        if i_episode % 200 == 0:
            #print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            mean_scores.append(np.mean(scores_window))
        
    return scores, mean_scores


scores, mean_scores = dqn_final()

print("\n==============================")
print("Finished the training process!")
print("==============================")


torch.save(agent.qnetwork_local.state_dict(), 'C:/Users/vedan/Downloads/Bonde_Project/checkpoint.pth')

print("Saved Training Model!")
print("==============================")






