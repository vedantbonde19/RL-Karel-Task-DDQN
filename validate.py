
import gym
#!pip3 install box2d
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import gym_examples
from dqn_agent import Agent
import json
from dqn_agent import Agent

agent = Agent(state_size=88, action_size=6, seed = 0)

agent.qnetwork_local.load_state_dict(torch.load('C:/Users/vedan/Downloads/Bonde_Project/best_1000_iter_ddqn_checkpoint_final.pth'))

jdata = None

if (jdata != None):
    with open(jsondata) as f:
        data = json.load(f)

else:
    with open("C:/Users/vedan/Downloads/Bonde_Project/datasets/data/val/val/seq/100000_seq.json") as f:
        data = json.load(f)

seqar = data["sequence"]
#print("Length of the list is : ", str(len(seqar)))

tot_true = 0
tot_solved = 0

for i in range(2400): 
    tot_count = 0
    path = "C:/Users/vedan/Downloads/Bonde_Project/datasets/data/val/val/task/"
    seqpath = "C:/Users/vedan/Downloads/Bonde_Project/datasets/data/val/val/seq/"
    fname = str(100000 + i) + "_task.json"
    fnameseq  = str(100000 + i) + "_seq.json"
    path = path + fname
    seqpath = seqpath + fnameseq

    env = gym.make("gym_examples/GridWorldMarker-v0", jsondata = path)
    state, info = env.reset()
    state = state + np.random.rand(1,88)/10.0 #D

    tot_reward = 0
    is_done = False

    for j in range(20):
        action = agent.act(state)
        tot_count = tot_count + 1
        state, reward, done, trunc, _ = env.step(action)
        state = state + np.random.rand(1,88)/10.0 #D
        tot_reward = tot_reward + reward

        if done: 
            is_done = True
            #print(tot_count)
            break

    with open(seqpath) as f:
        data = json.load(f)
    
    seqar = data["sequence"]
    val_seqlength = len(seqar)

    if (val_seqlength == tot_count):
        tot_true = tot_true + 1

    if (tot_reward >= 70 and is_done):
        tot_solved = tot_solved + 1



print("Total correctly solved values are: ", str(tot_solved))
print("Total correctly predicted minimal-size sequence values are: ", str(tot_true))

    

