import math
import math
import numpy as np

import torch
import torchvision
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import Box2D
from Box2D.b2 import fixtureDef
from Box2D.b2 import polygonShape
from Box2D.b2 import contactListener

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.utils import seeding, EzPickle

import pyglet
from pyglet import gl

from gym import error, spaces, utils
from gym.utils import seeding

env = gym.make('CarRacing-v0')
env.reset()

class Agent():

    def __init__(self):
        resnet18 = models.resnet18(pretrained=True)
        new_lin = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        resnet18.fc = new_lin
        print(resnet18)
        self.optimizer = optim.Adam(resnet18.parameters(), lr = 0.0001)
        loss_func = nn.BCELoss()
        self.optimizer.zero_grad()

        for _ in range(20000):
            env.render()
            env.step(env.action_space.sample()) #random action
            #self.optimizer.step(self)
        env.close()

if __name__ == "__main__":
    Agent()