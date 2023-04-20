import torch
import numpy as np
import sys
from q_learning import Q_learning
from gridworld.environments import GridWalk
from model import NN_Paramters
from util.parameters import Algo_Param, Save_Paths, Load_Paths


q_param = NN_Paramters(state_dim=2, action_dim=5, hidden_layer_dim=[10, 10], non_linearity=torch.tanh, device=torch.device("cpu"), l_r=0.05)
algo_param = Algo_Param()
algo_param.gamma = 0.9

max_episodes = 100

grid_size = 10
num_z = 2

env = GridWalk(grid_size, False)
env2 = GridWalk(grid_size, False)


Q = Q_learning(env, q_param, algo_param,)

grid = [[0 for i in range(10)] for j in range(10)]


update_interval = 100
save_interval = 1000
eval_interval = 1000
state = Q.initalize()

for i in range(10000):

    Q.train()
    state = Q.step(state)
    grid[state[1]][state[0]] += 1
    if i%update_interval == 0:
        Q.hard_update()
    if i%save_interval == 0:
        print("saving")
        Q.save("2.1/results/with_eps_greedy/2/q0", "2.1/results/with_eps_greedy/3/target_q0")
    if i%eval_interval == 0:
        s = env2.reset()
        i_s = s
        rew = 0
        for j in range(max_episodes):
            
            a = Q.get_action(s)
            s, r, d, _ = env2.step(a)
            rew += r
            if j == max_episodes-1:
                d = True
            if d == True:
                break
        print("reward at itr " + str(i) + " = " + str(rew) + " at state: " + str(s) + " starting from: " + str(i_s) + " espislon = "+ str(Q.epsilon))
print(grid)
torch.save(grid, "2.1/results/with_eps_greedy/3/occu")