import argparse
import gym
import numpy as np
from itertools import count
from tqdm import tqdm, trange
import pandas as pd
from numpy import int32

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import matplotlib
import matplotlib.pyplot as plt

from IPython.display import clear_output


#ABLE TO USE MATPLOT FOR THE GRAPHING
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


#SETTING VALUES TO CALL FROM LATER
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--gamma_value",
                    type=float,
                    default=0.99,)
parser.add_argument("--env",
                    type=str,
                    default="CartPole-v0",)
parser.add_argument("--n-episode",
                    type=int,
                    default=1000,)
parser.add_argument("--batch-size",
                    type=int,
                    default=128,)
parser.add_argument("--hidden-dim",
                    type=int,
                    default=12,)
parser.add_argument("--capacity",
                    type=int,
                    default=50000,)
parser.add_argument("--max-episode",
                    type=int,
                    default=50,)
parser.add_argument("--min-eps",
                    type=float,
                    default=0.01,)
parser.add_argument("--learning_rate",
                    type=float,
                    default=1e-2,)
parser.add_argument('--seed', type=int, default=543, metavar='N',)
parser.add_argument('--log-interval', type=int, default=5, metavar='N',)


#SETTING THE ARGS
args = parser.parse_args()
FLAGS = parser.parse_args()

#SETTING THE ENVIRONMENT
env = gym.make('CartPole-v1').unwrapped
env.seed(parser.parse_args().seed)
torch.manual_seed(args.seed)

#POLICY DECLERATION USING A HIDDEN LAYER OF 128 AND DROP OUT OF 0.6
class Policy(nn.Module):
    def __init__ (self, gamma_value: int, learning_rate: int ):
        super(Policy, self).__init__()

        print(gamma_value)

        # self.affine1 = nn.Linear(4, 128)
        # self.affine2 = nn.Linear(128, 2)

        # self.dropout = nn.Dropout(p=0.6)

        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        print(self.state_space)
        print(self.action_space)
#DECLERATIONS FOR THE POLICIES, REWARDS, AND LOSSES
        self.gamma = gamma_value

        self.saved_log_probs = []

        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []

        self.reward_history = []
        self.loss_history = []

        self.arr = []

#SETTING THE MODEL
    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)

#USING "ADAM" AS THE OPTIMIZER WITH THE LR DECLARED ABOVE
policy = Policy(args.gamma_value,args.learning_rate)
optimizer = optim.Adam(policy.parameters(), lr= args.learning_rate)
eps = np.finfo(np.float32).eps.item()

#SELECTING THE ACTION BASED ON 0 OR 1 BY LEARNING FROM THE MODEL
def select_action(state):

    state = torch.from_numpy(state).float().unsqueeze(0)
    action = Categorical(policy(state)).sample()
    policy.saved_log_probs.append(Categorical(policy(state)).log_prob(action))
    return action.item()

#UPDATING THE POLICIES
def finish_episode():
    
    R = 0
    policy_loss = []
    returns = []
    for r in policy.reward_history[::-1]:
        R = r + args.gamma_value * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.reward_history[:]
    del policy.saved_log_probs[:]

#MAIN METHOD, RUNNING THROUGH THE EPISODES
def main():
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, args.n_episode):
            action = select_action(state)
            state, reward, done, _ = env.step(action)

            # if i_episode % 10 ==0:
            #     env.render()

            if t % 10 ==0:
                 env.render()

            policy.reward_history.append(reward)
            ep_reward += reward
            if done:
                break

        #CALCULATING CURRENT RUNNING REWARD
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        policy.arr.append(running_reward)
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('EPISODE {}\tLAST REWARD: {:.2f}\tAVG: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("SOLVED RUNNING TIME IS {}"
                  "LAST EPISODE{} TS!".format(running_reward, t))

            plot_durations(policy.arr, i_episode)
            break

#PLOT GRAPHING
def plot_durations(array,i_episode):

    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(array, dtype=torch.float)

    plt.xlabel('EPISODES')
    plt.ylabel('REWARDS')

    plt.plot(durations_t.numpy())

    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())

    plt.pause(100)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.pause(100) #WAIT SO IT CAN DISPLAY

if __name__ == '__main__':
    main()
